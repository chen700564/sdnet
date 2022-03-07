from typing import List
from allennlp.data import DatasetReader
from allennlp.data.fields import LabelField, ArrayField, ListField, MetadataField
from allennlp.data.instance import Instance
import json
from transformers import AutoTokenizer
import numpy as np
import random
import copy


class FSNERreader(DatasetReader):
    def __init__(self, file=None,pretrainedfile=None, mapping=None, mode='prompt', lazy=False) -> None:
        super().__init__(lazy)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrainedfile)
        self.mapping = mapping
        self.mode = mode
        if file is not None:
            self.init(file)

    def entity2text(self,entitys,labelmap=None):
        generated = []
        for entity in entitys:
            type = entity['type']
            if self.target_classes is None or type in self.target_classes:
                type = labelmap[type]
                generated.append( entity['text'] + ' is' + ' ' + type + '.')
        return generated

    def prefix_generator(self,instances,typing=False,labelmap=None):
        '''
        instances = labels / spans
        labels: [[Label1,[finelabel1,finelabel2],[Label2,[finelabel3,finelabel4]]
        spans: [span1,span2]
        FET: {Typing | &span1 &span2}
        NER: {NER | & Label1: finelabel1, finelabel2 & Label2: finelabel3, finelabel4}
        { <extra_id_0> 32099
        } <extra_id_1> 32098
        | <extra_id_2> 32097
        : <extra_id_3> 32096
        , <extra_id_4> 32095
        & <extra_id_5> 32094
        NER <extra_id_6> 32093
        FET <extra_id_7> 32092
        '''
        prefix = [32099]
        if typing:
            prefix.append(32092)
            prefix.append(32097)
            for span in instances:
                spanid = [32094] + self.tokenizer.encode(span,add_special_tokens=False)
                prefix = prefix + spanid
        else:
            prefix.append(32093)
            prefix.append(32097)
            for label in instances:
                if self.info.withoutconcepts:
                    labelid = [32094] + self.tokenizer.encode(labelmap[label],add_special_tokens=False)
                    prefix = prefix + labelid
                    continue
                labelid = [32094] + self.tokenizer.encode(labelmap[label[0]],add_special_tokens=False) + [32096]
                labeltoken = [[self.tokenizer.encode(finelabel,add_special_tokens=False),index] for index,finelabel in enumerate(label[1])]
                labeltoken = sorted(labeltoken, key=lambda k:len(k[0]))
                cum = 0
                finelabelids = []
                for finelabel in labeltoken:
                    if len(labelid) + cum + min(len(finelabelids) - 1, 0) + len(finelabel[0]) <= 15:
                        cum += len(finelabel[0])
                        finelabelids.append(finelabel)
                    else:
                        break
                finelabelids = sorted(finelabelids,key = lambda k:k[1])
                finelabelids = [finelabel[0] for finelabel in finelabelids]
                for index,finelabel in enumerate(finelabelids):
                    if index > 0:
                        labelid.append(32095)
                    labelid = labelid + finelabel
                prefix = prefix + labelid
        prefix.append(32098)
        return prefix

    def getinstance(self,data,pred=False,labels=None,typing=False,index=0):
        if (not pred) and self.mode == 'prompt':
            if self.info.withoutconcepts:
                self.target_classes = [i for i in labels]
            else:
                self.target_classes = [i[0] for i in labels]
        text = ' '.join(data['tokens'])
        labelmap = copy.deepcopy(self.mapping)
        if typing:
            entities = data['entity']
            spans = [entity['text'] for entity in entities]
            prefix = self.prefix_generator(spans,True)
            generated = self.entity2text(entities,labelmap)
            generated = ' '.join(generated)
        else:     
            generated = self.entity2text(data['entity'],labelmap)
            generated = ' '.join(generated)
            if self.mode == 'prompt':
                prefix = self.prefix_generator(labels,labelmap=labelmap)
        
        if self.mode == 'prompt' or typing:
            inputid = self.tokenizer.encode(text,max_length=511-len(prefix),truncation=True)
            inputid = prefix + inputid
        else:
            inputid = self.tokenizer.encode(text,max_length=511,truncation=True)
        outputid = [self.tokenizer.pad_token_id] + self.tokenizer.encode(generated,max_length=511,truncation=True)
        labels = outputid[1:]
        inputmask = [1] * len(inputid)
        generated = self.tokenizer.decode(outputid)
        generated = generated.replace('<pad>','')
        generated = generated.replace('</s>','')
        generated = generated.strip()
        outputmask = [1] * (len(outputid) - 1)
        field = {
            'inputid':ArrayField(np.array(inputid)),
            'mask':ArrayField(np.array(inputmask)),
            'outputid':ArrayField(np.array(outputid[:-1])),
            'outmask':ArrayField(np.array(outputmask)),
            'text':MetadataField(text),
            'tokens':MetadataField(data['tokens']),
            'generated':MetadataField(generated),
            'entity':MetadataField(data['entity']),
        }
        if not pred:
            field['label'] = ListField([LabelField(int(i),skip_indexing=True) for i in labels])
        else:
            field['index'] = MetadataField(index)
        if typing:
            field['span'] = MetadataField([[self.tokenizer.decode(self.tokenizer.encode(i['text'],add_special_tokens=False)),i['type']] for i in data['entity']])
        return Instance(field)
    
    def obtaindatawithtype(self,dataset,target_classes):
        newdataset = []
        for data in dataset:
            entities = []
            for entity in data['entity']:
                if entity['type'] in target_classes:
                    entities.append(entity)
            if len(entities) > 0:
                newdataset.append({'tokens':data['tokens'],'entity':entities})
        return newdataset

    def sampleOneEpoch(self,dataset,typing=False,labels=None,pred=False):
        results = []
        for i in range(len(dataset)):
            if typing:
                results.append(self.getinstance(dataset[i],pred=True,typing=True,index=i))
            else:
                if pred:
                    for singlelabels in labels:
                        results.append(self.getinstance(dataset[i],pred=True,labels=singlelabels,index=i))
                else:
                    for singlelabels in labels:
                        sampledlabel = copy.deepcopy(singlelabels)
                        num = random.choice(list(range(self.info.sample[0],self.info.sample[1] + 1)))
                        sampledlabel = random.sample(sampledlabel,num)
                        results.append(self.getinstance(dataset[i],pred=False,labels=sampledlabel,index=i))
                    random.shuffle(results)
        return results

    def init(self,file):
        self.dataset = []
        with open(file) as f:
            for line in f:
                line = json.loads(line)
                self.dataset.append(line)
    
    def setinfo(self,info):
        self.info = info
    
    def _read(self,file):
        if self.info.dataset is not None:
            dataset = copy.deepcopy(self.info.dataset)
        else:
            dataset = copy.deepcopy(self.dataset)
        self.target_classes = self.info.target_classes
        if self.info.pred:
            epochdata = self.sampleOneEpoch(dataset,typing=self.info.typing,labels=self.info.labels,pred=True)
            for data in epochdata:
                yield data
        else:
            while True:
                epochdata = self.sampleOneEpoch(dataset,typing=self.info.typing,labels=self.info.labels,pred=False)
                for data in epochdata:
                    yield data
