from load_data import * 
from Make_prompt import make
from label_map import *
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

import re
label_map_cora1 = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
    }
label_map_photo1 = {
0: "Video Surveillance",
1: "Accessories",
2: "Binoculars & Scopes",
3: "Video",
4: "Lighting & Studio",
5: "Bags & Cases",
6: "Tripods & Monopods",
7: "Flashes",
8: "Digital Cameras",
9: "Film Photography",
10: "Lenses",
11: "Underwater Photography",
}
label_map_citeseer1 ={
    0:'Agents', 1:'ML', 2:'IR', 3:'DB', 4:'HCI', 5:'AI'
}
label_map_citeseer_={
0: ['Agents','Multi-Agent Systems'],
1: ['ML','Machine Learning'],
2: ['IR','Information Retrieval'],
3: ['DB','Database'],
4: ['HCI','Human-Computer Interaction'],
5: ['AI','Artificial Intelligence']
}
label_map_products1={
    0: 'Home & Kitchen',
    1: 'Health & Personal Care',
    2: 'Beauty',
    3: 'Sports & Outdoors',
    4: 'Books',
    5: 'Patio, Lawn & Garden',
    6: 'Toys & Games',
    7: 'CDs & Vinyl',
    8: 'Cell Phones & Accessories',
    9: 'Grocery & Gourmet Food',
    10: 'Arts, Crafts & Sewing',
    11: 'Clothing, Shoes & Jewelry',
    12: 'Electronics',
    13: 'Movies & TV',
    14: 'Software',
    15: 'Video Games',
    16: 'Automotive',
    17: 'Pet Supplies',
    18: 'Office Products',
    19: 'Industrial & Scientific',
    20: 'Musical Instruments',
    21: 'Tools & Home Improvement',
    22: 'Magazine Subscriptions',
    23: 'Baby Products',
    24: 'label 25',
    25: 'Appliances',
    26: 'Kitchen & Dining',
    27: 'Collectibles & Fine Art',
    28: 'All Beauty',
    29: 'Luxury Beauty',
    30: 'Amazon Fashion',
    31: 'Computers',
    32: 'All Electronics',
    33: 'Purchase Circles',
    34: 'MP3 Players & Accessories',
    35: 'Gift Cards',
    36: 'Office & School Supplies',
    37: 'Home Improvement',
    38: 'Camera & Photo',
    39: 'GPS & Navigation',
    40: 'Digital Music',
    41: 'Car Electronics',
    42: 'Baby',
    43: 'Kindle Store',
    44: 'Buy a Kindle',
    45: 'Furniture & decoration'}
label_map_arxiv1  = {
    0: "cs.NA",
    1: "cs.MM",
    2: "cs.LO",
    3: "cs.CY",
    4: "cs.CR",
    5: "cs.DC",
    6: "cs.HC",
    7: "cs.CE",
    8: "cs.NI",
    9: "cs.CC",
    10: "cs.AI",
    11: "cs.MA",
    12: "cs.GL",
    13: "cs.NE",
    14: "cs.SC",
    15: "cs.AR",
    16: "cs.CV",
    17: "cs.GR",
    18: "cs.ET",
    19: "cs.SY",
    20: "cs.CG",
    21: "cs.OH",
    22: "cs.PL",
    23: "cs.SE",
    24: "cs.LG",
    25: "cs.SD",
    26: "cs.SI",
    27: "cs.RO",
    28: "cs.IT",
    29: "cs.PF",
    30: "cs.CL",
    31: "cs.IR",
    32: "cs.MS",
    33: "cs.FL",
    34: "cs.DS",
    35: "cs.OS",
    36: "cs.GT",
    37: "cs.DB",
    38: "cs.DL",
    39: "cs.DM"
}
label_map_arxiv_1  = {
    0: ["cs.NA", "Numerical Analysis", "NA"],
    1: ["cs.MM", "Multimedia", "MM"],
    2: ["cs.LO", "Logic in Computer Science", "LO"],
    3: ["cs.CY", "Computers and Society", "CY"],
    4: ["cs.CR", "Cryptography and Security", "CR"],
    5: ["cs.DC", "Distributed, Parallel, and Cluster Computing", "DC"],
    6: ["cs.HC", "Human-Computer Interaction", "HC"],
    7: ["cs.CE", "Computational Engineering, Finance, and Science", "CE"],
    8: ["cs.NI", "Networking and Internet Architecture", "NI"],
    9: ["cs.CC", "Computational Complexity", "CC"],
    10: ["cs.AI", "Artificial Intelligence", "AI"],
    11: ["cs.MA", "Multiagent Systems", "MA"],
    12: ["cs.GL", "General Literature", "GL"],
    13: ["cs.NE", "Neural and Evolutionary Computing", "NE"],
    14: ["cs.SC", "Symbolic Computation", "SC"],
    15: ["cs.AR", "Hardware Architecture", "AR"],
    16: ["cs.CV", "Computer Vision and Pattern Recognition", "CV"],
    17: ["cs.GR", "Graphics", "GR"],
    18: ["cs.ET", "Emerging Technologies", "ET"],
    19: ["cs.SY", "Systems and Control", "SY"],
    20: ["cs.CG", "Computational Geometry", "CG"],
    21: ["cs.OH", "Other Computer Science", "OH"],
    22: ["cs.PL", "Programming Languages", "PL"],
    23: ["cs.SE", "Software Engineering", "SE"],
    24: ["cs.LG", "Machine Learning", "LG"],
    25: ["cs.SD", "Sound", "SD"],
    26: ["cs.SI", "Social and Information Networks", "SI"],
    27: ["cs.RO", "Robotics", "RO"],
    28: ["cs.IT", "Information Theory", "IT"],
    29: ["cs.PF", "Performance", "PF"],
    30: ["cs.CL", "Computation and Language", "CL"],
    31: ["cs.IR", "Information Retrieval", "IR"],
    32: ["cs.MS", "Mathematical Software", "MS"],
    33: ["cs.FL", "Formal Languages and Automata Theory", "FL"],
    34: ["cs.DS", "Data Structures and Algorithms", "DS"],
    35: ["cs.OS", "Operating Systems", "OS"],
    36: ["cs.GT", "Computer Science and Game Theory", "GT"],
    37: ["cs.DB", "Databases", "DB"],
    38: ["cs.DL", "Digital Libraries", "DL"],
    39: ["cs.DM", "Discrete Mathematics", "DM"]
}
reverse_label_map_arxiv_1  = {
    "cs.NA": 0, "Numerical Analysis": 0, "NA": 0,
    "cs.MM": 1, "Multimedia": 1, "MM": 1,
    "cs.LO": 2, "Logic in Computer Science": 2, "LO": 2,
    "cs.CY": 3, "Computers and Society": 3, "CY": 3,
    "cs.CR": 4, "Cryptography and Security": 4, "CR": 4,
    "cs.DC": 5, "Distributed, Parallel, and Cluster Computing": 5, "DC": 5,
    "cs.HC": 6, "Human-Computer Interaction": 6, "HC": 6,
    "cs.CE": 7, "Computational Engineering, Finance, and Science": 7, "CE": 7,
    "cs.NI": 8, "Networking and Internet Architecture": 8, "NI": 8,
    "cs.CC": 9, "Computational Complexity": 9, "CC": 9,
    "cs.AI": 10, "Artificial Intelligence": 10, "AI": 10,
    "cs.MA": 11, "Multiagent Systems": 11, "MA": 11,
    "cs.GL": 12, "General Literature": 12, "GL": 12,
    "cs.NE": 13, "Neural and Evolutionary Computing": 13, "NE": 13,
    "cs.SC": 14, "Symbolic Computation": 14, "SC": 14,
    "cs.AR": 15, "Hardware Architecture": 15, "AR": 15,
    "cs.CV": 16, "Computer Vision and Pattern Recognition": 16, "CV": 16,
    "cs.GR": 17, "Graphics": 17, "GR": 17,
    "cs.ET": 18, "Emerging Technologies": 18, "ET": 18,
    "cs.SY": 19, "Systems and Control": 19, "SY": 19,
    "cs.CG": 20, "Computational Geometry": 20, "CG": 20,
    "cs.OH": 21, "Other Computer Science": 21, "OH": 21,
    "cs.PL": 22, "Programming Languages": 22, "PL": 22,
    "cs.SE": 23, "Software Engineering": 23, "SE": 23,
    "cs.LG": 24, "Machine Learning": 24, "LG": 24,
    "cs.SD": 25, "Sound": 25, "SD": 25,
    "cs.SI": 26, "Social and Information Networks": 26, "SI": 26,
    "cs.RO": 27, "Robotics": 27, "RO": 27,
    "cs.IT": 28, "Information Theory": 28, "IT": 28,
    "cs.PF": 29, "Performance": 29, "PF": 29,
    "cs.CL": 30, "Computation and Language": 30, "CL": 30,
    "cs.IR": 31, "Information Retrieval": 31, "IR": 31,
    "cs.MS": 32, "Mathematical Software": 32, "MS": 32,
    "cs.FL": 33, "Formal Languages and Automata Theory": 33, "FL": 33,
    "cs.DS": 34, "Data Structures and Algorithms": 34, "DS": 34,
    "cs.OS": 35, "Operating Systems": 35, "OS": 35,
    "cs.GT": 36, "Computer Science and Game Theory": 36, "GT": 36,
    "cs.DB": 37, "Databases": 37, "DB": 37,
    "cs.DL": 38, "Digital Libraries": 38, "DL": 38,
    "cs.DM": 39, "Discrete Mathematics": 39, "DM": 39
}

label_map_pubmed1 = {
    0: 'Diabetes Mellitus, Experimental',
    1: 'Diabetes Mellitus Type 1', 
    2: 'Diabetes Mellitus Type 2'
    }

classification_prompt = """
You are an expert classifier.

Given the following node text and  its neighbors and a list of possible labels, choose the **single most appropriate label** for the text.

- Only return the selected label, with no explanation or formatting. ONLY RETURN THE SELECTED LABEL.
- Do not repeat the input.
- Choose only from the provided labels WITHOUT ANY CHANGES.

Choose only one label from the following list: 

- Agents: 'Multi-Agent Systems',
- ML: 'Machine Learning',
- IR :'Information Retrieval',
- DB: 'Database',
- HCI: 'Human-Computer Interaction',
- AI': 'Artificial Intelligence'


Respond with only the label. Do not explain or add anything else.

Node text:
{text}


Selected label:
"""
# classification_prompt = """
# You are a medical text classifier specializing in diabetes research literature.

# Your task is to classify research papers into ONE of three diabetes categories based on their title and abstract.

# LABEL DEFINITIONS:
# - Diabetes_Mellitus_Type_1: Research focusing on Type 1 diabetes (T1DM), an autoimmune condition where the pancreas produces little or no insulin. Usually develops in children/young adults. Keywords include: autoimmune, beta cells, insulin-dependent, IDDM, juvenile diabetes, pancreatic islets, autoantibodies.

# - Diabetes_Mellitus_Type_2: Research focusing on Type 2 diabetes (T2DM), a metabolic disorder characterized by insulin resistance and relative insulin deficiency. Most common form, often associated with obesity and lifestyle factors. Keywords include: insulin resistance, metabolic syndrome, obesity, NIDDM, lifestyle, metformin, cardiovascular risk.

# - Diabetes_Mellitus_Experimental: Laboratory research, animal studies, or experimental models of diabetes. Includes in vitro studies, animal models (mice, rats), experimental treatments, novel therapeutic approaches, or basic research mechanisms. Keywords include: mouse model, in vitro, experimental, animal study, novel therapy, laboratory, preclinical.

# CLASSIFICATION RULES:
# 1. Read the title and abstract carefully
# 2. Identify key terms and research context
# 3. If the study involves human patients with a specific diabetes type, classify accordingly
# 4. If the study uses animal models or experimental systems, choose Diabetes_Mellitus_Experimental
# 5. If multiple types are mentioned, choose based on the PRIMARY focus
# 6. When uncertain between Type 1 and Type 2, look for age of onset, treatment mentions, or pathophysiology clues

# Choose exactly ONE label from: [Diabetes_Mellitus_Type_1, Diabetes_Mellitus_Type_2, Diabetes_Mellitus_Experimental]

# Node text (title and abstract):
# {text}

# IMPORTANT: Return only the exact label name (e.g., "Diabetes_Mellitus_Type_1"), not a number or ID.

# Selected label:"""
classification_prompt = """
You are an expert classifier.
Given the node text, its neighbors, find the most appropriate label from the labels list. and tell ONLY THE LABEL NAME


Think through these steps internally (don't write them out):
1. What is the main topic?
2. What do neighbors suggest?
3. Which label fits best?
4. Double-check: is this really the best choice?
5. Final decision

Return ONLY the exact label name from the list. No explanation, no analysis, no other text.

Node text and its Neighbors : {text}


Labels List:
{label_list}


Return ONLY the exact label name from the list. No explanation, no analysis, no other text.

 Selected label(ONLY THE LABEL NAME):

"""



def generate_sudo_labels(prompt, label_list, reverse_label_map, batch_size, dataset_name):
    
    output_file="predicted_labels_"+dataset_name+".csv"
    
    llm = Ollama(model="llama3:instruct",         
                                num_predict=50,          
                                num_ctx=1024,           
                                num_thread=8)  
    

    template = PromptTemplate.from_template(classification_prompt)
    total_batches = math.ceil(len(prompt) / batch_size)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "result", "label"])
        # results=[]
        for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(prompt))
            batch_texts = prompt[start:end]
          
            
            prompts = [template.format(text=text, label_list=label_list) 
                        for text in batch_texts]
            results = llm.batch(prompts)
           
            label_ids = []
            for label in results:
                clean_label = label.strip("'\"") 
                clean_label=clean_label.strip()
                label_id = reverse_label_map.get(clean_label, -1)  
                label_ids.append(label_id)
            

            for i, (result, label_id) in enumerate(zip(results, label_ids)):
                global_id = start + i
                print(f"Wrote row: {global_id}, {result}, {label_id}")

                writer.writerow([global_id, result, label_id])
            csvfile.flush()
            

    print(f"Saved {len(prompt)} result and label to {output_file}")
#----------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='َ  To convert text to embedding')
    
   
    parser.add_argument('dataset_name',default='cora', help='dataset name')
    parser.add_argument('--batch_size',default=32, type=int, help='size of batch')
  
    args = parser.parse_args()
    batch_size=args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name=args.dataset_name
    #------------------------- LOAD DATASET
    if dataset_name=='products-subset':
        data, title, content=load_products_subset() 
        _, _, _, labels = load_ogb_products()
        label_list = list(label_map_products1.values())
        reverse_label_map = {v: k for k, v in label_map_products1.items()}
    elif dataset_name=='products':
        data, title, content, labels = load_ogb_products()
        label_list = list(label_map_products1.values())
        reverse_label_map = {v: k for k, v in label_map_products1.items()}
    elif dataset_name  in ['arxiv', 'arxiv_sim']:
        data, title, content=load_arxiv(dataset_name=dataset_name)
        label_list = list(label_map_arxiv1.values())
        reverse_label_map = {v: k for k, v in label_map_arxiv1.items()}
        labels= label_map_arxiv(data)
    elif dataset_name=='cora':
        data, title, content=load_cora()
        label_list = list(label_map_cora1.values())
        reverse_label_map = {v: k for k, v in label_map_cora1.items()}
        labels= label_map_cora(data)
    elif dataset_name=='citeseer':
        data, title, content=load_citeseer()
        label_list = list(label_map_citeseer1.values())
        reverse_label_map = {v: k for k, v in label_map_citeseer1.items()}
        labels= label_map_citeseer(data)
    elif dataset_name=='pubmed':
        data, title, content=load_pubmed()
        label_list = list(label_map_pubmed1.values())
        reverse_label_map = {v: k for k, v in label_map_pubmed1.items()}
        labels= label_map_pubmed(data)
    elif dataset_name=='arxiv_2023':
        data, title, content=load_arxiv_2023()
        label_list = list(label_map_arxiv1.values())
        reverse_label_map ={v: k for k, v in label_map_arxiv1.items()}
        labels= label_map_arxiv(data)

    prompt= make(data, content, content,labels,  dataset_name)
    print("prompt=",len(prompt['prompt'].tolist()))
    prompt=prompt['prompt'].tolist()
 
    texts = [t + " - " + c for t, c in zip(title, content)]


    
    generate_sudo_labels(prompt, label_list, reverse_label_map, batch_size, dataset_name)
    

if __name__ == '__main__':
    main()
