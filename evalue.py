import json
import sys


def geterror(pred,gold,fullgold,length,wrongmap):
    '''
    margin error: one of the token in span is true 
    label error: span is correct, label is wrong
    all span error: all span should be other
    span error: one of the token in span is other label 
    '''
    predlabel = [0] * length
    goldlabel = [0] * length
    wrongSpan = []
    wrongmargin = []
    allSpanerror = []
    unpred = []
    for entity in pred:
        for i in range(entity['offset'][0],entity['offset'][1]):
            predlabel[i] = entity['type']
    for entity in fullgold:
        for i in range(entity['offset'][0],entity['offset'][1]):
            goldlabel[i] = entity['type']
    for entity in pred:
        if entity not in gold:
            type = entity['type']
            if type not in wrongmap:
                wrongmap[type] = {}
            error = 'allspan'
            for i in range(entity['offset'][0],entity['offset'][1]):
                if goldlabel[i] == type:
                    error = 'margin'
                    if 'margin' not in wrongmap[type]:
                        wrongmap[type]['margin'] = 1
                    else:
                        wrongmap[type]['margin'] += 1
                    break
                elif goldlabel[i] != 0:
                    error = 'span'
                    if goldlabel[i] not in wrongmap[type]:
                        wrongmap[type][goldlabel[i]] = 1
                    else:
                        wrongmap[type][goldlabel[i]] += 1
                    break
            if error == 'span':
                wrongSpan.append(entity)
            elif error == 'margin':
                wrongmargin.append(entity)
            else:
                if 'other' not in wrongmap[type]:
                    wrongmap[type]['other'] = 1
                else:
                    wrongmap[type]['other'] += 1
                allSpanerror.append(entity)
    for entity in gold:
        if entity not in pred:
            type = entity['type']
            error = 'span'
            for i in range(entity['offset'][0],entity['offset'][1]):
                if predlabel[i] != 0:
                    error = 'margin'
                    break
            if error == 'span':
                unpred.append(entity)
    return wrongSpan,wrongmargin,unpred,allSpanerror,wrongmap

def text2entity(text):
    entitys = []
    indexs = [i for i in range(len(text)) if text.startswith('.', i)]
    start = 0
    for i in indexs:
        subtext = text[start:i].split(' ')
        if 'is' in subtext and subtext[-1] != 'is':
            indexs2 =  [i for i,a in enumerate(subtext) if a=='is']
            index = indexs2[-1]
            entity = [' '.join(subtext[:index]).strip(),' '.join(subtext[index+1:]).strip()]
            entitys.append(entity)
            start = i + 1
    return entitys

def decode(tokenizer,oritokens,generated,entitymap,targetclass):
    results = text2entity(generated)
    tokens = [tokenizer.decode(tokenizer.encode(token,add_special_tokens=False)).replace(' ','') for token in oritokens]
    tokenindex = []
    l = 0
    for token in tokens:
        for i in range(len(token)):
            tokenindex.append(l)
        l += 1
    tokenindex.append(l)
    tokens = ''.join(tokens) 
    start = 0
    preds = []
    index = 0
    for result in results:
        result[0] = result[0].replace(' ','')
        while True:
            if result[0] in tokens[start:] and len(result[0]) > 0:
                index = tokens[start:].index(result[0])
                offset = [tokenindex[start + index],tokenindex[start + index + len(result[0])]]
                if offset[0] == offset[1]:
                    start = start + index + len(result[0])
                else:
                    if result[1] in entitymap and entitymap[result[1]] in targetclass:
                        preds.append({'text':' '.join(oritokens[offset[0]:offset[1]]),'offset':offset,'type':entitymap[result[1]]})
                        start = start + index + len(result[0])
                    break
            else:
                break
    return preds



def getmetric(prednum,goldnum,tt,wrongspan=None,wrongmargin=None,wronglabel=None,unpred=None,aspan=None,classprednum=None,classgoldnum=None,classtt=None):
    p = 0
    r = 0
    f1 = 0
    if prednum > 0:
        p = tt/prednum
    if goldnum > 0:
        r = tt/goldnum
    if p > 0 and r > 0:
        f1 = 2*p * r / (p+r)
    result = {'p':p,'r':r,'f1':f1}
    if wrongspan is not None:
        result['unpred'] = unpred
        result['wrongspan'] = wrongspan
        result['wrongmargin'] = wrongmargin
        result['wronglabel'] = wronglabel
        result['allspan'] = aspan
    if classprednum is not None:
        result['typef1'] = {}
        for label in classprednum:
            p = 0
            r = 0
            f1 = 0
            if classprednum[label] > 0:
                p = classtt[label]/classprednum[label]
            if classgoldnum[label] > 0:
                r = classtt[label]/classgoldnum[label]
            if p > 0 and r > 0:
                f1 = 2*p * r / (p+r)
            result['typef1'][label] = {'p':p,'r':r,'f1':f1}
    return result

def filterpred(preds):
    textoffset = []
    if len(preds) == 1:
        return preds[0]
    offsetnum = {}
    for pred in preds:
        offset = []
        for entity in pred:
            entitytype = entity['type']
            if entitytype not in offsetnum:
                offsetnum[entitytype] = {}
            for i in range(entity['offset'][0],entity['offset'][1]):
                offset.append(i)
                if i not in offsetnum[entitytype]:
                    offsetnum[entitytype][i] = 1
                else:
                    offsetnum[entitytype][i] += 1
    newpred = []
    for pred in preds:
        for entity in pred:
            entitytype = entity['type']
            maxthisnum = 0
            maxothernum = 0
            for i in range(entity['offset'][0],entity['offset'][1]):
                if i in textoffset:
                    maxothernum = 5
                    break
                maxthisnum = max(maxthisnum,offsetnum[entitytype][i])
                for j in offsetnum:
                    if j != entitytype and i in offsetnum[j]:
                        maxothernum = max(maxothernum,offsetnum[j][i])
            if maxthisnum > maxothernum:
                newpred.append(entity)
                for i in range(entity['offset'][0],entity['offset'][1]):
                    try:
                        assert i not in textoffset
                    except:
                        print(preds)
                        print(i)
                        print(entity)
                    textoffset.append(i)
    return newpred


def evalue(file,reader,entitymap):
    prednum = 0
    goldnum = 0
    tt = 0
    classprednum = {}
    classgoldnum = {}
    classtt = {}
    wrongmap = {}
    ws = 0
    aspan = 0
    wm = 0
    wl = 0
    up = 0
    step = 0
    gindex = 0
    f1s = []
    dataset = reader.dataset
    with open(file) as f:    
        errors = []
        for line in f:
            f1 = [0,0,0]
            line = json.loads(line)
            for label in line['target_classes']:
                if label not in classtt:
                    classprednum[label] = 0
                    classgoldnum[label] = 0
                    classtt[label] = 0
            query_idx = []
            if 'query_idx' in line:
                query_idx = line['query_idx']
            else:
                query_idx = list(range(len(dataset)))
            query_idx.append(-1)
            lastpred = []
            lastindex = 0
            for i in range(len(query_idx)):
                if query_idx[i] == -1:
                    pred = []
                else:
                    data = dataset[query_idx[i]]
                    targetclass = line['target_classes']
                    generated = line['pred'][i]
                    pred = decode(reader.tokenizer,data['tokens'],generated,entitymap,targetclass)
                    
                    if lastindex == query_idx[i]:
                        lastpred.append(pred)
                        continue
                lastpred = filterpred(lastpred)    
                data = dataset[lastindex]          
                error = {
                    "index": str(gindex) + '_' + str(lastindex),
                    "text": ' '.join(data['tokens']),
                    "pred":[],
                    "gold":[],
                    "unpred":[],
                    'wronglabel':[]
                }
                error['pred'] = lastpred
                prednum += len(lastpred)
                f1[0] += len(lastpred)
                for entity in lastpred:
                    classprednum[entity['type']] += 1
                golds = []
                for entity in data['entity']:
                    if entity['type'] in targetclass:
                        golds.append(entity)
                        classgoldnum[entity['type']] += 1

                error['gold'] = data['entity']
                goldnum += len(golds)
                f1[1] += len(golds)

                goldentity = [j['text'] for j in golds]
                for p in lastpred:
                    if p in golds:
                        tt += 1
                        classtt[p['type']] += 1
                        f1[2] += 1
                    else:
                        if p['text'] in goldentity:
                            error['wronglabel'].append(p)
                            wl += 1
                wrongspan,wrongmargin,unpred,allspanerror,wrongmap = geterror(lastpred,golds,error['gold'],len(data['tokens']),wrongmap)
                error['wrongspan'] = wrongspan
                error['wrongmargin'] = wrongmargin
                error['unpred'] = unpred
                error['allspanerror'] = allspanerror
                ws += len(wrongspan)
                wm += len(wrongmargin)
                up += len(unpred)
                aspan += len(allspanerror)
                d = getmetric(prednum,goldnum,tt,ws,wm,wl,up,aspan)
                sys.stdout.write('test step: {0}, p:{1:.4f}, r:{2:.4f}, f1: {3:.4f}, unpred: {4:.4f}, span: {5:.4f}, margin: {6:.4f}, label: {7:.4f}, allspan: {8:.4f}'.format(
                    step,d['p'],d['r'], d['f1'], d['unpred'], d['wrongspan'],d['wrongmargin'],d['wronglabel'],d['allspan']) + '\r')
                # sys.stdout.write('\n')
                step += 1
                errors.append(error)
                lastpred = [pred]
                lastindex = query_idx[i]
            f1s.append(getmetric(f1[0],f1[1],f1[2]))
            gindex += 1
    sys.stdout.write('\n')
    print('\n')
    result = getmetric(prednum,goldnum,tt,ws,wm,wl,up,aspan,classprednum,classgoldnum,classtt)
    newrongmap = {}
    for label in wrongmap:
        newrongmap[label] = sorted(wrongmap[label].items(), key=lambda k:k[1], reverse=True)
    return errors,result,f1s,newrongmap