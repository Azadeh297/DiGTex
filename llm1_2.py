from load_data import * 
import argparse
import torch
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import csv
import math
from label_map import *


reasoning_prompt = """
You are a concise AI assistant.

Given the following text and its label, write a short, numbered list of reasoning steps explaining why this label fits the text.

- Be concise (each step under 20 words).
- Do not repeat the label or the text.
- Only output the steps (start with "1."), no explanation before or after.

Text:
{text}

Label:
{label}

Reasoning steps:
"""

import re

def extract_reasoning(texts):
    reasoning_sections = []
    
    pattern = re.compile(r"(\d+\..*?)(?=\n\d+\.|\Z)", re.DOTALL)

    for text in texts:
        if isinstance(text, str):
            steps = pattern.findall(text)
            if steps:
                reasoning_sections.append(steps)

    return reasoning_sections
# تابع اصلی
def generate_reasoning_and_embeddings(texts, labels,LLM, batch_size, dataset_name):
    assert len(texts) == len(labels), "Texts and labels must be the same length."
    output_file="reasoning_embeddings_"+dataset_name+".csv"
    # مدل‌ها
    llm = Ollama(model=LLM, temperature=0,           
                                num_predict=150,         
                                num_ctx=1024,          
                                num_thread=8)  
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")  

    template = PromptTemplate.from_template(reasoning_prompt)
    total_batches = math.ceil(len(texts) / batch_size)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "reasoning", "embedding"])

        for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_labels = labels[start:end]

     
            reasonings = []
           
            prompts = [template.format(text=text, label=label) 
                        for text, label in zip(batch_texts, batch_labels)]
            reasonings = llm.batch(prompts)
            reasonings= extract_reasoning(reasonings)

            
            embeddings = embedding_model.embed_documents(reasonings)
            embeddings = [[f"{val:.4f}" for val in emb] for emb in embeddings]

            for i, (reasoning, embedding) in enumerate(zip(reasonings, embeddings)):
                global_id = start + i
                embedding_str = "[" + ", ".join(embedding) + "]"  
                writer.writerow([global_id, reasoning, embedding_str])

    

    print(f"Saved {len(texts)} reasoning and embeddings to {output_file}")

#----------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='َ  To convert text to embedding')
    
    # اضافه کردن آرگومان‌های مورد نیاز
    parser.add_argument('dataset_name',default='cora', help='dataset name')
    parser.add_argument('LLM',default='Llama3:8b',  help='name of llm')
    parser.add_argument('path_pseudo_Label',default='labels/cora/predicted_labels_cora_0', help='path_Pseudo_Label')
    parser.add_argument('--batch_size',default=32, type=int, help='size of batch')
  
    args = parser.parse_args()
    batch_size=args.batch_size
    LLM=args.LLM
    path_pseudo_Label= args.path_pseudo_Label

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name=args.dataset_name
    #------------------------- LOAD DATASET
    if dataset_name=='products-subset':
        data, title, content=load_products_subset() 
        _, _, _, labels = load_ogb_products()
        # labels = label_map_products(data)

    elif dataset_name=='products':
        data, title, content, labels = load_ogb_products()
        
    elif dataset_name  in ['arxiv', 'arxiv_sim']:
        data, title, content=load_arxiv(dataset_name=dataset_name)
        
    elif dataset_name=='cora':
        data, title, content=load_cora()
        
    elif dataset_name=='citeseer':
        data, title, content=load_citeseer()
       
    elif dataset_name=='pubmed':
        data, title, content=load_pubmed()
        
    elif dataset_name=='arxiv_2023':
        data, title, content=load_arxiv_2023()
       
   
    texts = [t + " - " + c for t, c in zip(title, content)]
    #------------------------------------------
    num_nodes = data.x.shape[0]
    csv_path = path_pseudo_Label
    labels = torch.full((num_nodes,), 0, dtype=torch.long)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                node_id = int(row[0])
                
                label_id = int(row[2])
                labels[node_id] = label_id
            except Exception as e:
                print(f"Error in row {row}: {e}")
    print("labels is loaded...")
    labels= labels.to(device)
    print("labeeeeeeeeelssss",len(texts))
    generate_reasoning_and_embeddings(content, labels,LLM, batch_size, dataset_name)
    
    

if __name__ == '__main__':
    main()
