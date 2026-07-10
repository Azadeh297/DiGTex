import torch
import pandas as pd
import os
import random
from torch_geometric.data import Data
from collections import defaultdict
from tqdm import tqdm

def build_adj_list(edge_index):
    print("edge_index",edge_index.shape)
    adj = defaultdict(set)  # حتماً set نه dict یا int
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    for i in range(edge_index.shape[1]):
        src = int(src_nodes[i])
        dst = int(dst_nodes[i])
        adj[src].add(dst)
        adj[dst].add(src)  # اگر گراف جهت‌دار نیست
    return adj

def get_n_hop_neighbors(adj, node, n):
    visited = set()
    current = {node}
    for _ in range(n):
        next_nodes = set()
        for u in current:
            next_nodes |= adj[u]
        visited |= current
        current = next_nodes - visited
    return list(current)

def generate_prompts(data, title, content, label_list, dataset_name ):
    output_file='node_prompts_'+dataset_name+'.csv'
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Reading from file.")
        return pd.read_csv(output_file)

    edge_index = data.edge_index
    adj = build_adj_list(edge_index)
    print("adj=",len(adj))
    print("adj[0]=",adj[0])
    

    prompts = []
    for node in tqdm(data.n_id):
        node=node.item()
        title_text = title[node]
        content_text = content[node]

        one_hop = list(adj[node])
        # print("one_hop=",one_hop)
        one_hop_title = title[random.choice(one_hop)] if one_hop else ""
        

        two_hop = get_n_hop_neighbors(adj, node, 2)
        two_hop_title = title[random.choice(two_hop)] if two_hop else ""
        

        three_hop = get_n_hop_neighbors(adj, node, 3)
        three_hop_title = title[random.choice(three_hop)] if three_hop else ""
        

        prompt = f"""
title: {title_text}
content: {content_text}

title for 1-hop neighbor : {one_hop_title}, 


title for 2-hop neighbor : {two_hop_title}, 


title for 3-hop neighbor : {three_hop_title}, 


"""
        prompts.append({'node': node, 'prompt': prompt})

    df = pd.DataFrame(prompts)
    df.to_csv(output_file, index=False)
    print(f"Prompts saved to {output_file}")
    return df

# فرض بر این است که load_cora به شکل زیر عمل می‌کند:
# data: از نوع torch_geometric.data.Data
# title و content: دیکشنری‌هایی با کلید node index و مقدار string
def make(data, title, content,  label_list,  dataset_name):
    df_prompts = generate_prompts(data, title, content, label_list,  dataset_name)

    return df_prompts
