from load_data import * 
import argparse
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

def text_embedding(texts, device, dataset_name, batch_size):
    # بارگذاری مدل و توکنایزر
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    
    # تنظیمات batch
    # batch_size = 32
    embeddings = []
    ids = []

    # پردازش به صورت batch
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[start_idx:start_idx + batch_size]
            encoded_batch = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**encoded_batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # گرفتن [CLS]
            cls_embeddings = [[f"{val:.4f}" for val in emb] for emb in cls_embeddings]
            # انتقال به CPU و ذخیره
            for i, emb in enumerate(cls_embeddings):
                emb = "[" + ", ".join(emb) + "]"
                embeddings.append(emb)
                ids.append(start_idx + i)

    # تبدیل به دیتافریم و ذخیره به CSV
    # df = pd.DataFrame(embeddings)
    df = pd.DataFrame({'embedding': embeddings})
    df.insert(0, 'id', ids)
    df.to_csv('lm1_embeddings_'+dataset_name+'.csv', index=False)












#----------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='َ  To convert text to embedding')
    
    # اضافه کردن آرگومان‌های مورد نیاز
    parser.add_argument('dataset_name',default='cora', help='dataset name')
    parser.add_argument('--batch_size',default=32, type=int, help='size of batch')
  
    args = parser.parse_args()
    batch_size=args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name=args.dataset_name
    #------------------------- LOAD DATASET
    if dataset_name=='products-subset':
        _, title, content=load_products_subset() 
    elif dataset_name=='products':
        _, title, content, _= load_ogb_products()
    elif dataset_name  in ['arxiv', 'arxiv_sim']:
        _, title, content=load_arxiv(dataset_name=dataset_name)
    elif dataset_name=='cora':
        _, title, content =load_cora()
    elif dataset_name == 'photo':
        _, title, content =load_photo()
    elif dataset_name == 'citeseer':
        _, title, content =load_citeseer()
    elif dataset_name=='pubmed':
        _, title, content=load_pubmed()
    elif dataset_name=='arxiv_2023':
        _, title, content=load_arxiv_2023()
    # فرض: title و content دو لیست هم‌طول هستند
    # print(title[0])
    # print(len(title[0]))

    # texts = [t + " - " + c for t, c in zip(title, content)]

    #--------------------------------------------------
    text_embedding(content, device, dataset_name,  batch_size)
    

if __name__ == '__main__':
    main()
