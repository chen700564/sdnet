# SDNet

- An implementation for ``Few-shot Named Entity Recognition with Self-describing Networks``

## Quick links

* [Environment](#Environment)
* [Dataset](#Dataset)
* [Fewshot Fine-tuning](#Fewshot-Fine-tuning)
* [Model Evaluation](#Model-Evaluation)

### Environment

```bash
conda create -n sdnet python=3.8.5
conda activate sdnet
bash env.sh
```

### Dataset

Dataset is be putted in data folder:

```text
data/DATASET/
├── test.json
├── kshot.json/full.json
└── mapping.json
```

test.json is data file, and each line is a Dict. 

Instance format: Each instance is a Dict, containing `tokens` and `entity` fields, in which `tokens` is the list of tokens, and `entity` is the list of entity mentions.

```text
{
    "tokens": [token1,token2,...],
    "entity": [
        [
            {"text":mention1, "type": type1, "offset":[startindex1,endindex1]},
            {"text":mention2, "type": type2, "offset":[startindex2,endindex2]},
            ...
        ]
},
```

kshot.json: the data file for k-shot fine-tuning, each line is a Dict, containing `support` and `target_label` fields, in which `support` is the list of instances in support set, and `target_label` is the list of target novel entity types.
full.json: the data file for full shot fine-tuning, each line is a Dict, containing `support` and `target_label` fields, in which `support` is the list of instances in support set (full trraining set), and `target_label` is the list of target novel entity types.
test.json: each line is an instance.
mapping.json: a Dict mapping, the key is label name, the value is mapping words for each label (is commonly label name ). 

### Pretrained SDNet
The pretrained model should be put in path sdnetpretrain/

You can download the pretrained file in this link: 

### Fewshot Fine-tuning

run:

```bash
python main.py -dataset DATASET -K 5 -sdnet -cuda DEVICE
```

+ -dataset: DATASET is the dataset name in path data/DATASET 
+ -sdnet: finetuning with our pre-trained SDNet, if not added, using t5-base
+ -K: control the shot number, default is 5
+ -full: if added, using the full training set to fine-tune
+ -cuda: is the GPU id

The model checkpoint is saved in tmp/dataset/...

### Model Evaluation

just add -evalue:

```bash
python main.py -dataset DATASET -sdnet -K 5 -evalue
```
