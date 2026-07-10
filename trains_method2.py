import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import NeighborLoader
import torch.nn as nn
import pandas as pd
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from ogb.nodeproppred import Evaluator
from load_data import * 

import torch
import torch.nn.functional as F
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.utils import to_dense_adj
import random



def contrastive_loss_sampled(h, reasoning_emb, num_neg=5, temperature=0.1):
    """
    Contrastive loss using sampled negatives.
    
    Args:
        h: Tensor of shape (B, D) - anchor embeddings
        reasoning_emb: Tensor of shape (B, D) - positive embeddings
        num_neg: number of negative samples per anchor
        temperature: scaling factor for similarity

    Returns:
        Mean contrastive loss over the batch.
    """
    B = h.size(0)
    loss = 0.0

    for i in range(B):
        pos_sim = F.cosine_similarity(h[i], reasoning_emb[i], dim=0) / temperature

        # نمونه‌های منفی: به صورت تصادفی از سایر ایندکس‌ها انتخاب می‌شن
        neg_indices = random.sample([j for j in range(B) if j != i], min(num_neg, B - 1))
        neg_sims = F.cosine_similarity(h[i].unsqueeze(0), reasoning_emb[neg_indices], dim=1) / temperature

        # numerator = exp(similarity with positive)
        # denominator = exp(similarity with positive) + sum(exp(similarity with negatives))
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sims))

        # InfoNCE loss
        loss_i = -torch.log(numerator / (denominator + 1e-8))  # to avoid log(0)
        loss += loss_i

    return loss / B  # میانگین برای batch


def compute_enhanced_loss(h, logits,reason_emb, llm_labels,edge_index=None, alpha=1, beta=1.0, gamma=1.0):
         
    #                     
    contrastive= contrastive_loss_sampled(h, reason_emb, num_neg=5, temperature=0.1)   #LOSS 2
    contrastive = contrastive / torch.log(torch.tensor(5 + 1.0))



    # LLM supervision loss
    valid_llm_mask = llm_labels >= 0
    if valid_llm_mask.sum() > 0:
        llm_loss = F.cross_entropy(logits[valid_llm_mask], llm_labels[valid_llm_mask])    #LOSS 3
        # llm_loss=loss = (loss * confidence).mean()
    else:
        llm_loss = torch.tensor(0.0, device=h.device)


    #-------------------- soft cross entropy
    def soft_cross_entropy(pred_logits, soft_targets):
        # print("pred_logits=",pred_logits.shape)
        log_probs = F.log_softmax(pred_logits, dim=1)   # (N, C)
        # print("soft_targets=",soft_targets.shape)
        # print("log_probs=",log_probs.shape)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss
    loss= soft_cross_entropy(logits,llm_labels)

    total_loss =  0.7* loss + 0.3* contrastive 

    return total_loss

def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    return correct.item() / len(labels)

#-----------------------------------SAVE
import torch
import pandas as pd
from tqdm import tqdm


def save_all_node_embeddings_from_existing_loaders(
    model,
    data,
    train_loader,
    valid_loader,
    test_loader,
    text_embeddings,
    device,
    save_path="node_embeddings.csv"
):
    """
    Use existing NeighborLoaders (train/val/test) to extract
    embeddings for ALL nodes (seed-only) and save to CSV.
    """

    model.eval()
    model.to(device)

    all_embeddings = None
    visited = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

    def process_loader(loader, desc):
        nonlocal all_embeddings, visited

        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)

            batch_text_emb = text_embeddings[batch.n_id]
            h, _ = model(batch.x, batch.edge_index, batch_text_emb)

            batch_size = batch.batch_size
            seed_h = h[:batch_size]
            seed_ids = batch.n_id[:batch_size]

            if all_embeddings is None:
                hidden_dim = seed_h.size(1)
                all_embeddings = torch.zeros(
                    (data.num_nodes, hidden_dim),
                    device=device
                )

            # ذخیره فقط اگر قبلاً ذخیره نشده
            mask = ~visited[seed_ids]
            all_embeddings[seed_ids[mask]] = seed_h[mask]
            visited[seed_ids] = True

    with torch.no_grad():
        process_loader(train_loader, "Extracting train embeddings")
        process_loader(valid_loader, "Extracting val embeddings")
        process_loader(test_loader, "Extracting test embeddings")

    # sanity check
    assert visited.all(), "⚠️ Some nodes were never processed as seeds!"

    # CSV
    all_embeddings = all_embeddings.cpu().numpy()
    df = pd.DataFrame(
        all_embeddings,
        columns=[f"dim_{i}" for i in range(all_embeddings.shape[1])]
    )
    df.insert(0, "node_id", range(data.num_nodes))
    df.to_csv(save_path, index=False)

    print(f"✅ Node embeddings saved to {save_path}")
#-----------------------------------------------------------------------------
def train_enhanced_model(model,data, epochs, train_loader, valid_loader, test_loader, 
                        text_embeddings,reasoning_embeddings, llm_labels, device, dataset_name):
                        
    llm_labels=torch.tensor(llm_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, threshold=1e-3,  min_lr=1e-6)
    # --- normalize train/val/test indices ---
    def _to_tensor(idx):
        if isinstance(idx, torch.Tensor):
            return idx
        return torch.tensor(idx, dtype=torch.long)

    train_idx = _to_tensor(data.train_idx).to(device)

   
    if dataset_name=='products':
        evaluator = Evaluator(name='ogbn-products')
    else:
        evaluator = Evaluator(name='ogbn-arxiv')

    best_val_acc = 0
    best_model_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            batch_text_emb = text_embeddings[batch.n_id]
            batch_reason_emb = reasoning_embeddings[batch.n_id]
            batch_llm_labels = llm_labels[batch.n_id]

            #--------------- for seed and other train node in batch
            # mask نودهایی از batch که جزو train set هستند
            # batch_train_mask = torch.isin(
            #     batch.n_id,
            #     data.train_idx.to(device)
            # )
            batch_train_mask = torch.isin(batch.n_id, train_idx)



            h, logits = model(batch.x, batch.edge_index, batch_text_emb)
            #
            #----------------- for ONLY SEEDS
            batch_size = batch.batch_size
            seed_h = h[:batch_size]
            seed_logits=logits[:batch_size]
            seed_labels = batch_llm_labels[:batch_size]
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            seed_reason= batch_reason_emb[:batch_size]

            loss = compute_enhanced_loss(
                seed_h, seed_logits, seed_reason, seed_labels, edge_index=batch.edge_index)
            
             #
            #----------------- for ONLY SEEDS


            #
            #------------------ both SEDDS and other training nodes in batch
            # batch_size = batch.batch_size

            # # seed mask
            # seed_mask = torch.zeros_like(batch_train_mask)
            # seed_mask[:batch_size] = True

            # # final loss mask: seed + train neighbors
            # loss_mask = seed_mask | batch_train_mask

            # loss_h = h[loss_mask]
            # loss_logits = logits[loss_mask]
            # loss_labels = batch_llm_labels[loss_mask]
            # loss_reason = batch_reason_emb[loss_mask]

            # loss = compute_enhanced_loss(
            #     loss_h,
            #     loss_logits,
            #     loss_reason,
            #     loss_labels,
            #     edge_index=batch.edge_index
            # )
            #
            #------------------ both SEEDS and other training nodes in batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_acc_total, val_count = 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                batch = batch.to(device)
                batch_llm_labels = llm_labels[batch.n_id]
                batch_text_emb = text_embeddings[batch.n_id]

                h, logits = model(batch.x, batch.edge_index, batch_text_emb)

                batch_size = batch.batch_size
                seed_h = h[:batch_size]
                seed_logits=logits[:batch_size]
                seed_labels = batch_llm_labels[:batch_size]
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]

                y_pred = torch.argmax(seed_logits, dim=1)
                y_true= torch.argmax(seed_labels, dim=1)

                y_pred = y_pred.cpu().long()
                # y_true = batch.y[:batch_size].cpu().long()
                y_true= y_true.cpu().long()
   
                y_true = y_true.squeeze()
                y_pred = y_pred.squeeze()

                # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
                acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
        
                val_acc_total += acc['acc'] * batch_size
                val_count += batch_size
            
            val_acc = val_acc_total / val_count
        scheduler.step(val_acc)

        # Test
        model.eval()
        test_acc_total, test_count = 0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(device)
                batch_llm_labels = llm_labels[batch.n_id]
                batch_text_emb = text_embeddings[batch.n_id]
                h, logits = model(batch.x, batch.edge_index, batch_text_emb)

                batch_size = batch.batch_size
                seed_h = h[:batch_size]
                seed_logits=logits[:batch_size]
                seed_labels = batch_llm_labels[:batch_size]
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]
 
                y_pred = torch.argmax(seed_logits, dim=1)
                y_pred = y_pred.cpu().long()
                y_true = batch.y[:batch_size].cpu().long()

                # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})  # for others
                acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred.unsqueeze(1)}) # for products
                # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})            # for others
                test_acc_total += acc['acc'] * batch_size
                test_count += batch_size

            test_acc = test_acc_total / test_count


        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"[Epoch {epoch:03d}] Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_all_node_embeddings_from_existing_loaders(
            model=model,
            data=data,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            text_embeddings=text_embeddings,
            device=device,
            save_path=f"{dataset_name}_node_embeddings.csv"
        )

    model_name=dataset_name+"_model.pt"
    torch.save(model.state_dict(), model_name)
    return best_val_acc, test_acc



#----------------------------------------------------------------------------------------------------
def train_GNN(model, epochs, train_loader, valid_loader, test_loader, 
                        text_embeddings, labels, device, dataset_name ):
                        
    labels=torch.tensor(labels).to(device)
    labels = labels.long().to(device)

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    
    if dataset_name=='products':
        evaluator = Evaluator(name='ogbn-products')
    else:
        evaluator = Evaluator(name='ogbn-arxiv')

    best_val_acc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        

        total_original_loss = 0
        total_graph_reg_loss = 0

        for batch in tqdm(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            batch_text_emb = text_embeddings[batch.n_id]
            batch_labels = labels[batch.n_id]
           

           
            h, logits = model(batch.x, batch.edge_index, batch_text_emb)
            
            
            


            batch_size = batch.batch_size
            # print("batch_size=",batch_size)
            seed_h = h[:batch_size]
            # print("seed_h=",seed_h.shape)
            seed_logits=logits[:batch_size]
            # print("seed_logits=",seed_logits.shape)
            seed_labels = batch_labels[:batch_size]
            # print("seed_labels",seed_labels.shape)
                            # ✅ بررسی سایز
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            # print("seed_labels",seed_labels.shape)
            


            valid_mask = (seed_labels >= 0) & (seed_labels < seed_logits.shape[1])
            valid_mask = valid_mask.view(-1)
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(seed_logits[valid_mask], seed_labels[valid_mask].view(-1))
            else:
                loss = torch.tensor(0.0, device=logits.device)

            # loss = F.cross_entropy(logits[batch.n_id], llm_labels[batch.n_id])
            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_acc_total, val_count = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                batch_labels = labels[batch.n_id]
                # print("batch.shape",len(batch))
                batch_text_emb = text_embeddings[batch.n_id]
                h, logits = model(batch.x, batch.edge_index, batch_text_emb)
                # print("logits=", logits.shape)
                # print("logits=", logits)
                # Use true labels for validation
                # print("batch.y",(batch.y).shape)
                # acc = compute_accuracy(logits, batch.y)
                # پیش‌بینی کلاس (اندیسی با بیشترین logit)
                # y_pred = torch.argmax(logits, dim=1)

                # # evaluator expects both tensors on CPU and long dtype
                # y_pred = y_pred.cpu().long()
                # # y_true = batch.y.cpu().long()
                # y_true= batch_llm_labels.cpu().long()

                # y_true = y_true.view(-1)
                # y_pred = y_pred.view(-1)
                # print(y_true.shape, y_pred.shape)

                # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
                batch_size = batch.batch_size
                # print("batch_size=",batch_size)
                seed_h = h[:batch_size]
                # print("seed_h=",seed_h.shape)
                seed_logits=logits[:batch_size]
                # print("seed_logits=",seed_logits.shape)
                seed_labels = batch_labels[:batch_size]
                # print("seed_labels",seed_labels.shape)
                                # ✅ بررسی سایز
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]



                # print("seed_labels",seed_labels.shape)
                y_pred = torch.argmax(seed_logits, dim=1).cpu().long()
                y_true = seed_labels.cpu().long()

                y_true = y_true.view(-1, 1)
                y_pred = y_pred.view(-1, 1)

                acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

                # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
        
                val_acc_total += acc['acc'] * batch_size
                val_count += batch_size
            
            val_acc = val_acc_total / val_count
        scheduler.step(val_acc)
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"[Epoch {epoch:03d}] Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Test with best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_acc_total, test_count = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_labels = labels[batch.n_id]
            batch_text_emb = text_embeddings[batch.n_id]
            h, logits = model(batch.x, batch.edge_index, batch_text_emb)


            batch_size = batch.batch_size
            # print("batch_size=",batch_size)
            seed_h = h[:batch_size]
            # print("seed_h=",seed_h.shape)
            seed_logits=logits[:batch_size]
            # print("seed_logits=",seed_logits.shape)
            seed_labels = batch_labels[:batch_size]
            # print("seed_labels",seed_labels.shape)
                            # ✅ بررسی سایز
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            # acc = compute_accuracy(logits, batch.y)
            y_pred = torch.argmax(seed_logits, dim=1)

            # evaluator expects both tensors on CPU and long dtype
            y_pred = y_pred.cpu().long()
            # y_true = batch.y.cpu().long()
            y_true= seed_labels.cpu().long()
            y_true = y_true.view(-1, 1)
            y_pred = y_pred.view(-1, 1)

            acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
            # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
            # acc = evaluator.eval({'y_true': batch, 'y_pred': logits})
            test_acc_total += acc['acc'] * batch_size
            test_count += batch_size

        test_acc = test_acc_total / test_count
    model_name=dataset_name+"_model_GNN.pt"
    torch.save(model.state_dict(), model_name)
    return best_val_acc, test_acc



# === 1. تابع smoothing با propagation ساده (مثل GCN-style)
def smooth(logits, adj_t, num_propagations=50, alpha=0.5):
    smoothed = logits
    for _ in range(num_propagations):
        smoothed = alpha * smoothed + (1 - alpha) * adj_t.matmul(smoothed)
    return smoothed

from torch_sparse import SparseTensor
def correct_and_smooth(model, data, text_embeddings, device, confidence_threshold=0.7, correct_prop=50, smooth_prop=50):
    model.eval()
    with torch.no_grad():
        text_embeddings = text_embeddings.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        logits_all = torch.zeros((data.num_nodes, model.classifier[-1].out_features), device=device)

        h, logits = model(x, edge_index, text_embeddings)
        logits_all = logits

    probs = F.softmax(logits_all, dim=1)
    confidence, pseudo_labels = probs.max(dim=1)
    confident_mask = confidence > confidence_threshold
    print(f"C&S: {confident_mask.sum().item()} confident nodes for correction")

    # ساختن ماتریس مجاورتی sparse
    adj_t = SparseTensor.from_edge_index(data.edge_index).to(device)
    adj_t = adj_t.set_diag().to_symmetric()
    deg = adj_t.sum(dim=1).clamp(min=1)
    adj_t = adj_t / deg.view(-1, 1)  # normalize

    # === مرحله 1: Correction
    one_hot = F.one_hot(pseudo_labels, num_classes=logits_all.size(1)).float()
    error = (one_hot - probs) * confident_mask.unsqueeze(1).float()
    error = smooth(error, adj_t, num_propagations=correct_prop)
    logits_corrected = logits_all + error

    # === مرحله 2: Smoothing
    logits_smoothed = smooth(logits_corrected, adj_t, num_propagations=smooth_prop)

    return logits_smoothed, pseudo_labels, confidence



   
# -------------------------------------------------FINE_TUNE-----------------------
def fine_enhanced_model(model, epochs, train_loader, valid_loader, test_loader, 
                        text_embeddings,  device):
    

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )
    evaluator = Evaluator(name='ogbn-arxiv')

    best_val_acc = 0
    best_model_state_few = None
    patience = 100
    patience_counter = 0
    epochs= 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_reasoning_loss = 0
        total_llm_loss = 0
        total_true_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            batch_text_emb = text_embeddings[batch.n_id]

            h, logits = model(batch.x, batch.edge_index, batch_text_emb)

            batch_size = batch.batch_size
            # print("batch_size=",batch_size)
      
          
            seed_logits=logits[:batch_size]
            # print("seed_logits=",seed_logits.shape)
            seed_labels = batch.y[:batch_size]
            # print("seed_labels",seed_labels.shape)
                            # ✅ بررسی سایز
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            # print("seed_labels",seed_labels.shape)
   
            
            # Use true labels for training if available
            # true_labels = batch.y if hasattr(batch, 'y') else None
            # loss, r_loss, l_loss = compute_enhanced_loss(
            #     h, logits, batch_reason_emb, batch_llm_labels
            # )
            # print("logits.shape=l",logits.shape)
            # print("batch.y=",batch.y.shape)
            # loss = F.cross_entropy(logits, batch.y.squeeze().long())
            loss = F.cross_entropy(seed_logits, seed_labels.view(-1).long())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # scheduler.step()
        

        # Validation
        model.eval()
        val_acc_total, val_count = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                # batch_llm_labels = llm_labels[batch.n_id]
                # print("batch.shape",len(batch))
                batch_text_emb = text_embeddings[batch.n_id]
                h, logits = model(batch.x, batch.edge_index, batch_text_emb)


                          
                seed_logits=logits[:batch_size]
                # print("seed_logits=",seed_logits.shape)
                seed_labels = batch.y[:batch_size]
                # print("seed_labels",seed_labels.shape)
                                # ✅ بررسی سایز
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]
                # print("seed_labels",seed_labels.shape)
    
               
                y_pred = torch.argmax(seed_logits, dim=1)

               
                y_pred = y_pred.cpu().long()
               
                y_true= seed_labels.cpu().long()
                # print("y_pred=",y_pred.shape)
                # print("y_true",y_true.shape)
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(1)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(1)

                acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})




                # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred.unsqueeze(1)})   # products
                # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
        
                val_acc_total += acc['acc'] * batch_size
                val_count += batch_size
            
            val_acc = val_acc_total / val_count
        scheduler.step(val_acc)
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_few = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        model.eval()
        test_acc_total, test_count = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # batch_llm_labels = llm_labels[batch.n_id]
                batch_text_emb = text_embeddings[batch.n_id]
                h, logits = model(batch.x, batch.edge_index, batch_text_emb)


                            
                seed_logits=logits[:batch_size]
                # print("seed_logits=",seed_logits.shape)
                seed_labels = batch.y[:batch_size]
                # print("seed_labels",seed_labels.shape)
                                # ✅ بررسی سایز
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]
                # print("seed_labels",seed_labels.shape)
                # acc = compute_accuracy(logits, batch.y)
                y_pred = torch.argmax(seed_logits, dim=1)

                # evaluator expects both tensors on CPU and long dtype
                y_pred = y_pred.cpu().long()
                # y_true = batch.y.cpu().long()
                y_true= seed_labels.cpu().long()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(1)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(1)

                acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})



                # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred.unsqueeze(1)})    #products
                # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
                # acc = evaluator.eval({'y_true': batch, 'y_pred': logits})
                test_acc_total += acc['acc'] * batch_size
                test_count += batch_size

            test_acc = test_acc_total / test_count


        print(f"[Epoch {epoch:03d}] Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Test with best model
    if best_model_state_few is not None:
        model.load_state_dict(best_model_state_few)
    
    model.eval()
    test_acc_total, test_count = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # batch_llm_labels = llm_labels[batch.n_id]
            batch_text_emb = text_embeddings[batch.n_id]
            h, logits = model(batch.x, batch.edge_index, batch_text_emb)


                           
            seed_logits=logits[:batch_size]
            # print("seed_logits=",seed_logits.shape)
            seed_labels = batch.y[:batch_size]
            # print("seed_labels",seed_labels.shape)
                            # ✅ بررسی سایز
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            # print("seed_labels",seed_labels.shape)
            # acc = compute_accuracy(logits, batch.y)
            y_pred = torch.argmax(seed_logits, dim=1)

            # evaluator expects both tensors on CPU and long dtype
            y_pred = y_pred.cpu().long()
            # y_true = batch.y.cpu().long()
            y_true= seed_labels.cpu().long()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)

            acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})



            # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred.unsqueeze(1)})    #products
            # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
            # acc = evaluator.eval({'y_true': batch, 'y_pred': logits})
            test_acc_total += acc['acc'] * batch_size
            test_count += batch_size

        test_acc = test_acc_total / test_count

    # torch.save(model.state_dict(), "enhanced_model_fine_tune.pt")
    return best_val_acc, test_acc
# #-------------------------------------------------FINE_TUNE-----------------------
#-------------------------------------------------ZERO-SHOT-----------------------
def zero_enhanced_model(model, epochs, train_loader, valid_loader, test_loader, 
                        text_embeddings,  device):
    
    
    evaluator = Evaluator(name='ogbn-arxiv')

    best_val_acc = 0
  
    
    # Test with best model
    
    model.eval()
    test_acc_total, test_count = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # batch_llm_labels = llm_labels[batch.n_id]
            batch_text_emb = text_embeddings[batch.n_id]
            h, logits = model(batch.x, batch.edge_index, batch_text_emb)
            # acc = compute_accuracy(logits, batch.y)
            y_pred = torch.argmax(logits, dim=1)

            # evaluator expects both tensors on CPU and long dtype
            y_pred = y_pred.cpu().long()
            # y_true = batch.y.cpu().long()
            y_true= batch.y.cpu().long()


           

            # همیشه هر دو را به 2D تبدیل کن اگر 1D هستند
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)

            acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})



            # acc = evaluator.eval({'y_true': y_true, 'y_pred': y_pred.unsqueeze(1)})   # fpr products
            # acc = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
            # acc = evaluator.eval({'y_true': batch, 'y_pred': logits})
            test_acc_total += acc['acc'] * batch.num_nodes
            test_count += batch.num_nodes

        test_acc = test_acc_total / test_count

    # torch.save(model.state_dict(), "enhanced_model_fine_tune.pt")
    return best_val_acc, test_acc
#-------------------------------------------------zero-shot-----------------------
