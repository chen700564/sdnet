import torch
import numpy as np
import tqdm
from model import T5model
from fewshotreader import FSNERreader
import os
import sys
import copy
import gc
import math
from transformers import AutoTokenizer
from evalue import evalue

class datainfo():
    def __init__(self,dataset=None,target_classes=None,labels=[None],typing=False,pred=False,sample=[5,10],withoutconcepts=False):
        self.dataset = dataset
        self.target_classes = target_classes
        self.labels = labels
        self.pred = pred
        self.sample = sample
        self.typing = typing
        self.withoutconcepts = withoutconcepts

def typingdecoder(generated,spans):
    start = 0
    result = []
    for span in spans:
        text = span[0] + ' is '
        if text in generated[start:]:
            start = start + generated[start:].index(text) + len(text)
            if '.' in generated[start:]:
                end = start + generated[start:].index('.')
                label = generated[start:end].strip()
                result.append([span,label])
    return result

def obtainwordscore(spanmaps,mapping,other='other thing'):
    id2label = [i for i in mapping.keys()]
    voacb = []
    wordscore = []
    othernum = [0] * len(id2label)
    for spanmap in spanmaps:
        for i in spanmap:
            finelabels = i[1].split(',')
            typeindex = id2label.index(i[0][1])
            for label in finelabels:
                label = label.strip()
                if label == other:
                    othernum[typeindex] += 1
                if len(label) > 0 and label != other:
                    if label not in voacb:
                        voacb.append(label)
                        wordscore.append([0] * len(id2label))
                    wordindex = voacb.index(label)
                    wordscore[wordindex][typeindex] += 1
    othernum = wordscore + [othernum]
    wordscore = np.array(wordscore)
    othernum = np.array(othernum)
    return id2label,voacb,wordscore,othernum

def filterchild(spanmaps,mapping,k):
    id2label,voacb,wordscore,othernum = obtainwordscore(spanmaps,mapping)
    score1 = copy.deepcopy(wordscore)
    score1 = score1/(np.sum(score1,axis=0)+1e-16)
    score2 = copy.deepcopy(wordscore)
    score2 = score2/(np.sum(score2,axis=1,keepdims=True)+1e-16)
    otherscore = othernum/(np.sum(othernum,axis=0)+1e-16)
    allscore = score1 * score2
    nums = np.sum(copy.deepcopy(wordscore),axis=0)
    spanlabel = np.argmax(allscore,axis=-1)
    spanlabel = [id2label[i] for i in spanlabel]
    for spanmap in spanmaps:
        for i in spanmap:
            finelabels = i[1].split(',')
            labelpred = []
            for label in finelabels:
                label = label.strip()
                if len(label) > 0 and label != 'other thing':
                    typeindex = id2label.index(i[0][1])
                    wordindex = voacb.index(label)
                    value = [label,allscore[wordindex][typeindex]]
                    labelpred.append([wordindex,spanlabel[wordindex]])
            for label in labelpred:
                value = [voacb[label[0]],allscore[label[0]][typeindex]]
                if value not in mapping[i[0][1]]:
                    mapping[i[0][1]].append(value)
    for label in mapping:
        mapping[label] = sorted(mapping[label],key=lambda k:k[1],reverse=True)
        if len(mapping[label]) > 0:
            # if k > 100:
            # mapping[label] = mapping[label][:10]
            mapping[label] = [i[0] for i in mapping[label]]
            if otherscore[-1][id2label.index(label)] > 0.5:
                mapping[label] = []
    return mapping

def supportsetpred(model,reader,targetlabel,support,tokenizer):
    filterdata = reader.obtaindatawithtype(support,targetlabel)
    info = datainfo(dataset=filterdata,target_classes=targetlabel,pred=True,typing=True)
    reader.setinfo(info)
    typingdata = reader.read('file')
    model.eval()
    print('FET for support set')
    result = []
    with torch.no_grad():
        for data in typingdata:
            y = model.forward_on_instance(data)
            spans = data['span'].as_tensor(data['span'].get_padding_lengths())
            generated = tokenizer.decode(y['output'])
            generated = generated.replace('<pad>','')
            generated = generated.replace('</s>','')
            generated = generated.strip()
            result.append(typingdecoder(generated,spans))
    return result


def finetuning(model, reader, testreader, lr, batch_size, num_epochs, tokenizer, k, cuda_device=-1, modelfile = None, supports=None, mode='prompt',maxpromptnum=12,seed=0,force=False):
    print(mode)
    from allennlp.data.dataloader import PyTorchDataLoader
    from transformers import AdamW
    from allennlp.training.trainer import GradientDescentTrainer
    from allennlp.training.learning_rate_schedulers.polynomial_decay import PolynomialDecay
    cpumodel = model
    mappingfile = 'support_mapping'
    if maxpromptnum != 12:
        mappingfile += ('_promptnum' + str(maxpromptnum))
    mappingfile += '_'
    if supports is not None:
        instancenum = len(supports)
    resultfile = modelfile + '/result.txt'
    recordfile = modelfile + '/record.txt'
    with open(resultfile,'w') as f1:
        with open(recordfile,'w') as f2:
            for i in range(instancenum):
                print('step: '+str(i))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic=True

                model = copy.deepcopy(cpumodel)
                model = model.cuda(cuda_device)
                
                result = {
                    'support_idx':[],
                    'target_classes':[],
                    'pred':[],
                    'start':[]
                }

                support = supports[i]['support']
                target_classes = supports[i]['target_label']
                result['target_classes'] = supports[i]['target_label']
                if mode == 'prompt':
                    if os.path.exists(modelfile + '/support_preds_' + str(i) + '.json'):
                        supportresult = []
                        with open(modelfile +'/support_preds_' + str(i) + '.json') as f:
                            for j in f:
                                supportresult.append(json.loads(j))
                    else:
                        supportresult = supportsetpred(model,testreader,target_classes,support,tokenizer)
                        with open(modelfile + '/support_preds_' + str(i) + '.json','w') as f:
                            for j in supportresult:
                                f.write(json.dumps(j)+'\n')
                    if not os.path.exists(modelfile + '/' + mappingfile + str(i) + '.json') or force:
                        supportmapping = {}
                        for label in target_classes:
                            supportmapping[label] = []
                        supportmapping = filterchild(supportresult,supportmapping,k)
                        json.dump(supportmapping,open(modelfile + '/' + mappingfile + str(i) + '.json','w'))
                    else:
                        supportmapping = json.load(open(modelfile + '/' + mappingfile + str(i) + '.json'))
                    labels = []
                    for label in supportmapping:
                        if label in target_classes:
                            finelabel = supportmapping[label]
                            random.shuffle(finelabel)
                            labels.append([label,finelabel])
                    testlabels = copy.deepcopy(labels)
                    if len(testlabels) > maxpromptnum:
                        splitnum = math.ceil(len(testlabels)/8)
                        parentnum = math.ceil(len(testlabels)/splitnum)
                        newtestlabels = []
                        repeatnum = 2
                        for rep in range(repeatnum):
                            random.shuffle(testlabels)
                            for randomnum in range(splitnum):
                                newtestlabels.append(testlabels[randomnum*parentnum:(randomnum+1)*parentnum])
                        testlabels = newtestlabels
                    else:
                        testlabels = [testlabels]
                else:
                    labels = None
                    testlabels = [None]
                if num_epochs > 0:
                    model.train()
                    num = len(target_classes)
                    batches_per_epoch = math.ceil(len(support)/ batch_size )
                    info = datainfo(dataset=support,target_classes=target_classes,labels=[labels],pred=False,sample=[min(num//2,5),min(num,10)])
                    reader.setinfo(info)
                    support_set = reader.read('file')
                    data_loader = PyTorchDataLoader(support_set,batch_size,batches_per_epoch=batches_per_epoch)
                    parameters_to_optimize = list(model.named_parameters())
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    parameters_to_optimize = [
                            {'params': [p for n, p in parameters_to_optimize
                                        if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
                            {'params': [p for n, p in parameters_to_optimize
                                        if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
                        ]
                    optimizer = AdamW(parameters_to_optimize, lr=lr, correct_bias=False)
                    warm = int(num_epochs * batches_per_epoch * 0.06)
                    learning_rate_scheduler = PolynomialDecay(optimizer,num_epochs,batches_per_epoch,1,warm,0)
                    trainer = GradientDescentTrainer(
                                model=model,
                                optimizer=optimizer,
                                data_loader=data_loader,
                                num_epochs=num_epochs,
                                cuda_device=cuda_device,
                                learning_rate_scheduler=learning_rate_scheduler
                            )
                    print('finetuning')
                    trainer.train()
                    del trainer
                    del optimizer
                    del parameters_to_optimize
                    print('pred')
                model.eval()
                testbatch = 4
                j = 0
                info = datainfo(target_classes=target_classes,labels=testlabels,pred=True)
                testreader.setinfo(info)
                query_set = testreader.read('file')
                if len(labels) > maxpromptnum:
                    result['query_idx'] = []
                with torch.no_grad():
                    for i in tqdm.tqdm(range(math.ceil(len(query_set)/testbatch))):
                        y = model.forward_on_instances(query_set[i*testbatch:(i+1)*testbatch])
                        outputs = [output['output'] for output in y]
                        inputtext = [data['inputid'].as_tensor(data['inputid'].get_padding_lengths()).long().tolist() for data in query_set[i*testbatch:(i+1)*testbatch]]
                        outputs = tokenizer.batch_decode(outputs)
                        inputtext = tokenizer.batch_decode(inputtext)
                        if len(labels) > maxpromptnum:
                            result['query_idx'] += [data['index'].as_tensor(data['index'].get_padding_lengths()) for data in query_set[i*testbatch:(i+1)*testbatch]]
                        for index,data in enumerate(query_set[i*testbatch:(i+1)*testbatch]):
                            gold = data['generated'].as_tensor(data['generated'].get_padding_lengths())
                            text = inputtext[index]
                            generated = outputs[index]
                            generated = generated.replace('<pad>','')
                            generated = generated.replace('</s>','')
                            generated = generated.strip()
                            result['pred'].append(generated)
                            f1.write(str(j)+"\n"+text+"\n"+gold+"\n"+generated+"\n")
                            j += 1
                model.model = None
                model.to("cpu")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                f2.write(json.dumps(result)+"\n")

def obtainevalue(modelfile,entitymap):
    recordfile = modelfile + '/record.txt'
    error,result,f1s,wrongmap = evalue(recordfile,testreader,entitymap)
    result['f1s'] = f1s
    result['meanf1'] = sum([i['f1'] for i in f1s])/len(f1s)
    result['meanp'] = sum([i['p'] for i in f1s])/len(f1s)
    result['meanr'] = sum([i['r'] for i in f1s])/len(f1s)
    sys.stdout.write('meanp:{0:.4f}, meanr:{1:.4f}, meanf1: {2:.4f}'.format(result['meanp'],result['meanr'],result['meanf1']) + '\r')
    sys.stdout.write('\n')
    print('\n')
    filename = ['/f1','/error','/predresult','/wrongmap']
    json.dump(result,open(modelfile+filename[0]+'.json','w'))
    json.dump(wrongmap,open(modelfile+filename[3]+'.json','w'))
    with open(modelfile+filename[2]+'.txt','w') as f:
        for e in error:
            f.write(json.dumps(e))
    with open(modelfile+filename[1]+'.txt','w') as f:
        for e in error:
            if len(e['wrongspan']) > 0 or len(e['unpred']) > 0 or len(e['wronglabel']) > 0 or len(e['wrongmargin']) > 0 or len(e['allspanerror']) > 0:
                f.write(str(e['index'])+'\n')
                f.write(e['text']+'\n')
                f.write('[Gold]:'+str(e['gold'])+'\n')
                f.write('[Pred]:'+str(e['pred'])+'\n')
                if len(e['unpred']) > 0:
                    f.write('[Unpredicted]:'+str(e['unpred'])+'\n')
                if len(e['wrongspan']) > 0:
                    f.write('[wrongSpan]:'+str(e['wrongspan'])+'\n')
                if len(e['wronglabel']) > 0:
                    f.write('[wrongLabel]:'+str(e['wronglabel'])+'\n')
                if len(e['wrongmargin']) > 0:
                    f.write('[wrongMargin]:'+str(e['wrongmargin'])+'\n') 
                if len(e['allspanerror']) > 0:
                    f.write('[otherspan]:'+str(e['allspanerror'])+'\n') 


import argparse
import random
import torch
import numpy as np
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-K", "--K", type=int, default=5)
    parser.add_argument("-SEED", "--SEED", type=int, default=2333)
    parser.add_argument("-lr", "--lr", type=float, default=1e-4)
    parser.add_argument("-epoch", "--epoch", type=int, default=50)
    parser.add_argument("-tag", "--tag", type=int, default=1)
    parser.add_argument("-cuda", "--cuda", type=int, default=0)

    parser.add_argument("-batch", "--batch", type=int, default=4)
    parser.add_argument("-sdnet", "--sdnet", action='store_true')

    parser.add_argument("-mapping", "--mapping", type=str, default='mapping')
    parser.add_argument("-dataset", "--dataset", type=str, default='WNUT17')


    parser.add_argument("-full", "--full", action='store_true')

    parser.add_argument("-evalue", "--evalue", action='store_true')
    parser.add_argument("-force", "--force", action='store_true')


    args = parser.parse_args()
    K = args.K
    SEED = args.SEED
    tag = args.tag
    batch = args.batch
    dataset = args.dataset
    num_epochs = args.epoch
    sdnet = args.sdnet
    lr = args.lr
    mapping = args.mapping
    e = args.evalue
    full = args.full
    cuda = args.cuda
    force = args.force

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True

    if sdnet:
        pretrainfile = 'sdnet'
    else:
        pretrainfile = 't5base'

    if not os.path.exists("tmp/" + dataset):
        os.mkdir("tmp/" + dataset)

    modelfile = "tmp/" + dataset + '/' + pretrainfile

    if not os.path.exists(modelfile):
        os.mkdir(modelfile)
    
    pretrain = 't5-base'

    if full:
        modelfile = modelfile + '/' +'full_' + pretrain + '_' + str(num_epochs)
    else:
        modelfile = modelfile + '/' + str(K) + '_' + pretrain + '_' + str(num_epochs)
    if lr != 1e-4:
        modelfile =  modelfile + '_' + str(lr)
    if mapping != 'mapping':
        modelfile = modelfile+ '_' + mapping
    if batch != 4:
        modelfile = modelfile + '_batch' + str(batch)
    modelfile = modelfile + '_' + str(tag)

    if not os.path.exists(modelfile):
        os.mkdir(modelfile)

    testfile = 'data/' + dataset + '/test.json'
    mapping = json.load(open('data/'+ dataset +'/' + mapping + '.json'))

    tokenizer = AutoTokenizer.from_pretrained(pretrain, use_fast=True)

    if sdnet:
        mode = 'prompt'
    testreader = FSNERreader(file=testfile, pretrainedfile=pretrain, mapping=mapping, mode=mode, lazy=False)
    

    batch_size = batch

    print('modelfile:'+modelfile)
    if e:
        print('evalue')
        entitymap = {}
        for i in mapping:
            entitymap[mapping[i]] = i
        obtainevalue(modelfile,entitymap)
    else:
        model = T5model(pretrain)
        if sdnet:
            pretrainfile = 'sdnetpretrain/sdnet.th'

            with open(modelfile+'/pretrain.txt','w') as f:
                f.write(pretrainfile)
            model_dict=model.state_dict()
            pretrained_dict=torch.load(pretrainfile,map_location='cuda:'+str(cuda))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict(torch.load(pretrainfile,map_location='cuda:0'))
            print(pretrainfile)

        reader = FSNERreader(file=None, pretrainedfile=pretrain, mapping=mapping, mode=mode, lazy=True)
        supports = []

        
        if full:
            K = 1000
            file = 'data/'+dataset+'/'+'full.json'
        else:
            file = 'data/'+dataset+'/'+str(K)+'shot.json'
        with open(file) as f:
            for line in f:
                supports.append(json.loads(line))
        
        finetuning(model, reader, testreader, lr, batch_size, num_epochs, tokenizer, K, cuda_device=cuda, modelfile = modelfile, supports=supports, mode=mode,seed=SEED,force=force)





