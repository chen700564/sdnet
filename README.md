# SDNet

- An implementation for ACL 2022 paper [Few-shot Named Entity Recognition with Self-describing Networks](https://arxiv.org/abs/2203.12252)

## Implementation

### Quick links

* [Environment](#Environment)
* [Dataset](#Pretrained-Data)
* [Dataset](#Dataset)
* [Pretrained SDNet](#Pretrained-SDNet)
* [Fewshot Fine-tuning](#Fewshot-Fine-tuning)
* [Model Evaluation](#Model-Evaluation)

### Environment

```bash
conda create -n sdnet python=3.8.5
conda activate sdnet
bash env.sh
```

### Pretrained Data
Part of pretrianed data is in the pretrain_data folder, the file includes 200k instances.

### Dataset

Dataset is in the data folder:

```text
data/DATASET/
├── test.json
├── kshot.json/full.json
└── mapping.json
```

+ test.json is data file, and each line is an Instance. 

**Instance format**: Each instance is a Dict, containing `tokens` and `entity` fields, in which `tokens` is the list of tokens, and `entity` is the list of entity mentions.

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

+ kshot.json/full.json: the data file for k-shot fine-tuning, each line is a Dict, containing `support` and `target_label` fields, in which `support` is the list of instances in support set (full training set in full.json), and `target_label` is the list of target novel entity types.

+ mapping.json: a Dict mapping, the key is label name, the value is mapping words for each label (is commonly label name). 

### Pretrained SDNet

The pretrained SDNet (sdnet.th) should be putted in folder `sdnetpretrain`

You can download the pretrained SDNet in this [link](https://1drv.ms/u/s!Apx2f2KG2lXYglzYgrNd479FaoLS). 

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

The predicted result is saved in tmp/dataset/...

### Model Evaluation

just add -evalue:

```bash
python main.py -dataset DATASET -K 5 -sdnet -cuda DEVICE -evalue
```

## License

The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg