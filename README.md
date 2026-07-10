# DiGTex
# DiGTex: A Consistency-Driven Distillation Framework for Transferring LLM Knowledge to GNNs


## Install requirements in  `requirements.txt`

## Download TAG datasets



| Dataset | Description |
| ----- |  ---- |

| ogbn-products (subset) |  Download the dataset [here](https://drive.google.com/file/d/1F9D9NauJMlmwGcmLxhwbyAhfguWEZApA/view?usp=drive_link), unzip and move them to `datasets/products`.|
| arxiv_2023 |  Download the dataset [here](https://drive.google.com/file/d/1ekG96JHNPWqeQdb6D_GZoM28OGRLdcS_/view?usp=drive_link), unzip and move it to `datasets/arxiv_2023`.|
| PubMed | Download the dataset [here](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing), unzip and move it to `datasets/pubmed`.|
| Photo | Download the dataset (photo.pt) [here](https://drive.google.com/drive/folders/1bSRCZxt0c11A3717DYDjO112fo_zC8Ec), unzip and move it to `datasets/photo`.|

## step1: create embeddings of raw node texts

```
python lm1.py 'cora' 
```
## step2: employs an LLM to generate pseudo-labels 

```
python llm1_1.py 'cora' 'Llama3:8b' 'Minimalist'
```
## step3: employs an LLM to generate reasons

```
python llm1_2.py 'cora' 'Llama3:8b' path_pseudo_Label
```
## Training and then save embeddings for BiGTex and ogbn-arxiv

```
python main.py 'arxiv' 'BiGTex'
```




