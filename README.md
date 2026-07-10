# DiGTex
# DiGTex: A Consistency-Driven Distillation Framework for Transferring LLM Knowledge to GNNs


## Install requirements in  `requirements.txt`

## Download TAG datasets



| Dataset | Description |
| ----- |  ---- |
| Cora  | Download the dataset [here](https://drive.google.com/file/d/1tSepgcztiNNth4kkSR-jyGkNnN7QDYax/view?usp=sharing), move it to `datasets/cora`.|
| ogbn-products (subset) |  Download the dataset [here](https://drive.google.com/file/d/1F9D9NauJMlmwGcmLxhwbyAhfguWEZApA/view?usp=drive_link), unzip and move them to `datasets/products`.|
| arxiv_2023 |  Download the dataset [here](https://drive.google.com/file/d/1ekG96JHNPWqeQdb6D_GZoM28OGRLdcS_/view?usp=drive_link), unzip and move it to `datasets/arxiv_2023`.|
| PubMed | Download the dataset [here](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing), unzip and move it to `datasets/pubmed`.|


## step1: create embeddings of raw node texts

```
python lm1.py 'cora' 
```
## step2: employ a LLM to generate pseudo-labels 

```
python llm1_1.py 'cora' 'Llama3:8b' 'Minimalist'
```
## step3: employ a LLM to generate reasons

```
python llm1_2.py 'cora' 'Llama3:8b' path_pseudo_Label
```
## step4: calculate Alpha and find S*

```
python Alpha.py 'cora' 
```
## Final step: Apply DS and train GNN

```
python main.py 'cora' --Model 'DiGTex' --GNN 'GCN' --LLM 'Llama3:8b'
```




