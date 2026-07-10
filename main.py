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
from trains_method2 import *
import os
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np



# Improved GNN Model
class EnhancedMultiModalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, GNN, num_gnn_layers, dropout=0.2):
        super().__init__()
        
        # Initial feature mapping
        self.struct_mapper = nn.Linear(in_channels, hidden_channels)    
        self.w = nn.Linear(hidden_channels, hidden_channels)  
      
        # GNN layers
        if GNN == 'GCN':
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_channels, hidden_channels) 
                for _ in range(num_gnn_layers)
            ])
        elif GNN == "SAGE":
            self.gnn_layers = nn.ModuleList([  
                SAGEConv(hidden_channels, hidden_channels)   
                for _ in range(num_gnn_layers)  
            ])
        elif GNN == 'GAT':
            num_heads = 8  
            self.gnn_layers = nn.ModuleList([  
                GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)  
                for _ in range(num_gnn_layers)  
            ])
        
        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced classifier 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels ),
            nn.LayerNorm(hidden_channels ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels , num_classes)
        )

    def forward(self, x, edge_index, text_emb):
        # Map features to same dimension
        struct_feat = self.struct_mapper(x)
        
        # text_feat = text_emb
        # Multi-modal fusion
        # gate_input = torch.cat([struct_feat, text_feat], dim=1)
        # h = self.fusion(gate_input)
        h= struct_feat
        # h=text_emb
        residual = h
        # GNN layers with residual connections
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Residual connection
            
            # if i > 0:  # Skip first layer for residual
            #     h = h + struct_feat
                # residual = h
            h = gnn_layer(h, edge_index)
            # text= self.w(text_emb)
            # h = h + text_emb
            # h = h + text
            h = self.layer_norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            
            
        # h = h + residual
        # Classification
        logits = self.classifier(h)
        return h, logits
class GNNMODEL(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, GNN, num_gnn_layers, dropout=0.2):
        super().__init__()
        self.num_gnn_layers=num_gnn_layers
        self.mapper = nn.Linear(in_channels, hidden_channels) 
        # GNN layers
        if GNN == 'GCN':
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_channels, hidden_channels) 
                for _ in range(num_gnn_layers)
            ])
        elif GNN == "SAGE":
            self.gnn_layers = nn.ModuleList([  
                SAGEConv(hidden_channels, hidden_channels)   
                for _ in range(num_gnn_layers)  
            ])
        elif GNN == 'GAT':
            num_heads = 8  
            self.gnn_layers = nn.ModuleList([  
                GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)  
                for _ in range(num_gnn_layers)  
            ])
        elif GNN =='MLP':
            self.num_gnn_layers=0

        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels ),
            nn.LayerNorm(hidden_channels ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels , num_classes)
        )

    def forward(self, x, edge_index, text_emb):
        
        h=self.mapper( x)
        if self.num_gnn_layers !=0:
            # GNN layers with residual connections
            for i, gnn_layer in enumerate(self.gnn_layers):
                
                h = gnn_layer(h, edge_index)
                h = self.layer_norms[i](h)
                h = F.gelu(h)
                h = self.dropout(h)
            
            

        logits = self.classifier(h)
        return h, logits


def create_loaders(data, batch_size):
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_idx,
        num_neighbors=[15, 10],  
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = NeighborLoader(
        data,
        input_nodes=data.valid_idx,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_idx,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader

def get_fewshot_train_idx(data, num_per_class):
    labels = data.y
    train_idx = data.train_idx

    selected_indices = []

    for c in torch.unique(labels[train_idx]):
        mask = (labels[train_idx] == c)
        class_train_idx = train_idx[mask.nonzero(as_tuple=True)[0]]

        if class_train_idx.ndim == 0:
            class_train_idx = class_train_idx.unsqueeze(0)

        if len(class_train_idx) >= num_per_class:
            selected = class_train_idx[:num_per_class]
        else:
            selected = class_train_idx

        selected = selected.to(torch.long)
        selected_indices.append(selected)

    fewshot_idx = torch.cat(selected_indices)
    return fewshot_idx

def label_similarity(y1, y2):
    """
    محاسبه درصد شباهت بین دو لیبل y1 و y2
    """
    y2 = torch.tensor(y2) if isinstance(y2, list) else y2
    # اطمینان از اینکه تنسورها هم‌شکل‌اند و روی CPU هستند
    y1 = y1.view(-1).cpu()
    y2 = y2.view(-1).cpu()

    assert y1.shape == y2.shape, "Label shapes must be equal"

    # محاسبه درصد برچسب‌های مساوی
    matches = (y1 == y2).sum().item()
    total = y1.numel()

    return 100.0 * matches / total



import torch
import faiss
import pandas as pd
from collections import defaultdict, Counter
def cluster_and_map_to_true_labels(
    reasoning_embeddings: torch.Tensor,
    true_labels: list,
    num_classes: int,
    dataset_name,
    node_ids=None,
    GNN="R"
):
    """
    Clusters reasoning embeddings using FAISS on GPU, maps cluster IDs to true labels using majority vote,
    and saves the mapped labels along with node IDs to a CSV.

    Args:
        reasoning_embeddings (torch.Tensor): (N, D) CUDA tensor of reasoning embeddings.
        true_labels (list[str] or list[int]): List of ground truth labels of each node (length N).
        num_classes (int): Number of clusters (usually equal to number of classes).
        node_ids (list[int], optional): Node IDs to use in output. Defaults to range(N).
        output_csv_path (str): Path to save the resulting CSV.

    Returns:
        mapped_cluster_labels (list): Mapped labels (same type as true_labels) for each node.
        cluster_to_true_label (dict): Mapping from cluster ID → true label.
    """
    output_csv_path="cluster_mapped_labels_"+dataset_name+"_qwen_"+GNN+".csv"
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        labels = [int(x) for x in df['mapped_label'].tolist()]

        return labels

    else:
        # assert reasoning_embeddings.is_cuda, "reasoning_embeddings must be on CUDA"
        N = reasoning_embeddings.size(0)
        assert len(true_labels) == N, "Mismatch in number of nodes and true_labels"

        # Step 1: Convert to float32 and move to CPU for FAISS
        embeddings_np = reasoning_embeddings.detach().cpu().numpy().astype('float32')

        # Step 2: Run KMeans clustering with FAISS on GPU
        d = embeddings_np.shape[1]
        kmeans = faiss.Kmeans(d, num_classes, gpu=False)
        kmeans.train(embeddings_np)
        _, cluster_labels = kmeans.index.search(embeddings_np, 1)
        cluster_labels = cluster_labels.reshape(-1)

        # Step 3: Majority vote mapping from cluster ID to true label
        cluster_groups = defaultdict(list)
        for i, cl_id in enumerate(cluster_labels):
            cluster_groups[cl_id].append(true_labels[i])

        cluster_to_true_label = {}
        for cl_id, labels in cluster_groups.items():
            most_common_label = Counter(labels).most_common(1)[0][0]
            cluster_to_true_label[cl_id] = most_common_label

        # Step 4: Generate mapped labels
        mapped_cluster_labels = [cluster_to_true_label[cl] for cl in cluster_labels]

        # Step 5: Save results
        if node_ids is None:
            node_ids = list(range(N))

        df = pd.DataFrame({
            'node_id': node_ids,
            'cluster_id': cluster_labels,
            'mapped_label': [int(label) for label in mapped_cluster_labels]
        })
        df.to_csv(output_csv_path, index=False)
        print(f"[✓] Saved clustered and mapped labels to: {output_csv_path}")

    return mapped_cluster_labels
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#                    DS
import torch
from typing import List, Optional, Tuple

class DawidSkeneTorch:
    """
    Dawid–Skene label aggregation (multiclass) in PyTorch.

    Notation:
      - N: number of items (nodes)
      - C: number of classes
      - M: number of annotators (label sources)

    Inputs:
      labels_list: list of LongTensor, each (N,), values in {0..C-1} or -1 for missing
      num_classes: C
      ignore_index: value treated as missing (-1 by default)
      max_iter, tol: EM stopping
      alpha: Dirichlet smoothing for confusion matrices (>= 0), e.g., 1.0
      init: 'uniform' | 'majority' initialization for posteriors
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        max_iter: int = 100,
        tol: float = 1e-5,
        alpha: float = 1.0,
        init: str = "majority",
        verbose: bool = False,
        annotator_weights: Optional[List[float]] = None,
    ):
        self.C = num_classes
        self.ignore_index = ignore_index
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.init = init
        self.verbose = verbose,
        self.annotator_weights=annotator_weights

        # learned params / outputs
        self.pi = None  # class prior, (C,)
        self.confusion_matrices = None  # list of (C, C)
        self.posteriors = None  # (N, C)
        self.hard_labels = None  # (N,)

    @staticmethod
    def _majority_init(labels_list: List[torch.Tensor], C: int, ignore_index: int) -> torch.Tensor:
        # returns (N, C) one-hot (or normalized if ties) from majority vote
        device = labels_list[0].device
        N = labels_list[0].shape[0]
        counts = torch.zeros((N, C), device=device, dtype=torch.float32)
        for lab in labels_list:
            mask = (lab != ignore_index)
            idx = lab[mask]
            if idx.numel() == 0: 
                continue
            rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
            counts[rows, idx] += 1.0
        # if all missing for an item, make uniform
        row_sums = counts.sum(dim=1, keepdim=True)
        missing_rows = (row_sums.squeeze(1) == 0)
        counts[missing_rows] = 1.0  # uniform tie
        # normalize to probabilities
        post = counts / counts.sum(dim=1, keepdim=True)
        return post

    def _uniform_init(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.full((N, self.C), 1.0 / self.C, device=device, dtype=torch.float32)

    def _init_confusions(self, labels_list: List[torch.Tensor], post: torch.Tensor) -> List[torch.Tensor]:
        """
        Initialize confusion matrices using expected counts under current posteriors.
        theta_m[k, l] ~ P(annotator m outputs l | true = k), shape (C, C)
        With Dirichlet(alpha) smoothing.
        """
        device = post.device
        N = post.shape[0]
        M = len(labels_list)
        thetas = []
        for m in range(M):
            lab = labels_list[m]
            theta = torch.full((self.C, self.C), self.alpha, device=device, dtype=torch.float32)  # prior counts
            for l in range(self.C):
                mask = (lab == l)
                if mask.any():
                    # expected true-class counts for items where annotator m said l
                    theta[:, l] += post[mask].sum(dim=0)  # sum over items gives expected counts per true class
            # normalize columns per true class (sum over l for each k)
            theta = theta / theta.sum(dim=1, keepdim=True)
            thetas.append(theta)
        return thetas

    def _m_step(
        self, labels_list: List[torch.Tensor], post: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Update class priors pi and confusion matrices theta_m using expected counts.
        """
        device = post.device
        N = post.shape[0]
        M = len(labels_list)

        # class prior
        pi = post.mean(dim=0)  # (C,)

        # confusion matrices
        thetas = []
        for m in range(M):
            lab = labels_list[m]
            theta_counts = torch.full((self.C, self.C), self.alpha, device=device, dtype=torch.float32)  # prior counts
            for l in range(self.C):
                mask = (lab == l)
                if mask.any():
                    theta_counts[:, l] += post[mask].sum(dim=0)
            theta = theta_counts / theta_counts.sum(dim=1, keepdim=True)  # normalize per true class
            thetas.append(theta)
        return pi, thetas
    def _e_step(self, labels_list: List[torch.Tensor], pi: torch.Tensor, thetas: List[torch.Tensor]) -> torch.Tensor:
        device = pi.device
        N = labels_list[0].shape[0]
        log_pi = (pi + 1e-12).log()
        log_post = log_pi.unsqueeze(0).repeat(N, 1)

        for m, lab in enumerate(labels_list):
            theta = thetas[m]
            mask = (lab != self.ignore_index)
            if mask.any():
                idx = lab[mask]
                log_theta_per_item = (theta[:, idx].T + 1e-12).log()
                weight = self.annotator_weights[m] if self.annotator_weights is not None else 1.0
                log_post[mask] += weight * log_theta_per_item
                

        log_post = log_post - torch.logsumexp(log_post, dim=1, keepdim=True)
        post = log_post.exp()
        return post

    def fit(
        self,
        labels_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Run EM and return:
          - posteriors: (N, C)
          - hard_labels: (N,)
          - confusion_matrices: list of (C, C)
        """
        assert len(labels_list) >= 1, "Need at least one annotator"
        device = labels_list[0].device
        N = labels_list[0].shape[0]
        # sanity shapes & devices
        for t in labels_list:
            assert t.shape == (N,), "All label tensors must have shape (N,)"
            assert t.device == device, "All label tensors must be on the same device"
            assert t.dtype == torch.long, "Labels must be LongTensor"

        # init posteriors
        if self.init == "majority":
            post = self._majority_init(labels_list, self.C, self.ignore_index)
        else:
            post = self._uniform_init(N, device)

        # init pi and thetas
        self.pi = post.mean(dim=0)
        self.confusion_matrices = self._init_confusions(labels_list, post)

        prev_ll = -float("inf")
        for it in range(self.max_iter):
            # E-step
            post = self._e_step(labels_list, self.pi, self.confusion_matrices)

            # M-step
            self.pi, self.confusion_matrices = self._m_step(labels_list, post)

            # monitor lower bound (sum of log-marginals); optional
            with torch.no_grad():
                log_pi = (self.pi + 1e-12).log()
                log_marg = log_pi.unsqueeze(0).repeat(N, 1)
                for m, lab in enumerate(labels_list):
                    theta = self.confusion_matrices[m]
                    mask = (lab != self.ignore_index)
                    if mask.any():
                        idx = lab[mask]
                        log_theta_per_item = (theta[:, idx].T + 1e-12).log()
                        log_marg[mask] += log_theta_per_item
                ll = torch.logsumexp(log_marg, dim=1).sum().item()

            if self.verbose:
                print(f"[DS] iter {it+1}: ll={ll:.4f}")

            if abs(ll - prev_ll) < self.tol * (1.0 + abs(prev_ll)):
                if self.verbose:
                    print(f"[DS] converged at iter {it+1}")
                break
            prev_ll = ll

        self.posteriors = post
        self.hard_labels = post.argmax(dim=1)

        return self.posteriors, self.hard_labels, self.confusion_matrices


# --------- Convenience wrapper for your 3 sources ---------

def ds_aggregate_three(
    labels1: torch.Tensor,
    labels2: torch.Tensor,
    labels3: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    max_iter: int = 100,
    tol: float = 1e-5,
    alpha: float = 1.0,
    init: str = "majority",
    verbose: bool = False,
):
    """
    Run Dawid–Skene on three sources of hard labels.
    Returns:
      posteriors (N, C), hard_labels (N,), confusion_matrices [3 x (C, C)]
    """
    ds = DawidSkeneTorch(
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_iter=max_iter,
        tol=tol,
        alpha=alpha,
        init=init,
        verbose=verbose,
    )
    ds.annotator_weights = [0.5,2.0,2.0]     # Cora_Llama
    # ds.annotator_weights = [2.0,0.5,0.5]     # Cora_Mistral
    # ds.annotator_weights = [0.5, 2.5,2.0]     # Cora_Qwen
    # ds.annotator_weights = [2.25,0.25,0.5]     # citeseer_Llama
    # ds.annotator_weights = [2.0,0.75,0.5]     # citeseer_Mistral
    # ds.annotator_weights = [2.0, 0.5,0.5]     # citeseer_Qwen
    # ds.annotator_weights = [0.75,1.5,1.5]     # pubmed_Llama
    # ds.annotator_weights = [0.25,2.25,0.5]     # pubmed_Mistral
    # ds.annotator_weights = [0.25, 0.5,2.5]     # pubmed_Qwen
    # ds.annotator_weights = [2.25,0.5,0.25]     # arxiv2023_Llama
    # ds.annotator_weights = [2.75,0.25,0.25]     # arxiv2023_Mistral
    # ds.annotator_weights = [0.25, 0.5,2.5]     # arxiv2023_Qwen
    # ds.annotator_weights = [2.0,0.75,0.25]     # products_Llama
    # ds.annotator_weights = [2.75,0.5,0.25]     # products_Mistral
    # ds.annotator_weights = [0.5, 0.25,2.25]     # products_Qwen
    # ds.annotator_weights = [1.5, 0.5,0.5]
    
    return ds.fit([labels1, labels2, labels3])

#                    DS
#---------------------------------------------------------------------


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train_mlp_on_embeddings(
    embeddings,     # torch.Tensor [N, D]
    data,
    device,
    hidden_dim=256,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-4
):
    """
    Train a 2-layer MLP on frozen node embeddings.
    """

    embeddings = embeddings.to(device)
    labels = data.y.squeeze().to(device)

    train_idx = data.train_idx.to(device)
    test_idx = data.test_idx.to(device)

    num_nodes, emb_dim = embeddings.shape
    num_classes = int(labels.max().item()) + 1
    
    model = MLPClassifier(
        in_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    print("🚀 Training MLP on node embeddings")
    print(f"Nodes: {num_nodes}, Emb dim: {emb_dim}, Classes: {num_classes}")

    for epoch in range(1, epochs + 1):
        # ===== Train =====
        model.train()
        optimizer.zero_grad()

        logits = model(embeddings[train_idx])
        loss = criterion(logits, labels[train_idx])

        loss.backward()
        optimizer.step()

        # Train accuracy
        train_pred = torch.argmax(logits, dim=1)
        train_acc = (train_pred == labels[train_idx]).float().mean().item()

        # ===== Test =====
        model.eval()
        with torch.no_grad():
            test_logits = model(embeddings[test_idx])
            test_pred = torch.argmax(test_logits, dim=1)
            test_acc = (test_pred == labels[test_idx]).float().mean().item()

        print(
            f"[Epoch {epoch:02d}] "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return model

#----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-modal GNN')
    parser.add_argument('dataset_name', default='cora', help='dataset name')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--Model', default='DiGTex', help='Models: DiGTex, GNN')
    parser.add_argument('--GNN', default='SAGE', help='type of GNN:  GCN, GAT, SAGE, MLP')
    parser.add_argument('--hidden_dim', default=768, type=int, help='hidden dimension')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--few_percent', default=1.0, type=float, help='few_percent')
    parser.add_argument('--LLM', default='all', help='all, llama, mistral, qwen')
    


    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset (same as original)
    dataset_name= args.dataset_name 
    model= args.Model
    #------------------------- LOAD DATASET
    if dataset_name=='products-subset':
        data, title, content=load_products_subset() 
        # _, _, _, labels = load_ogb_products()
        num_classes=46

    elif dataset_name=='products':
        data, title, content, labels = load_ogb_products()
        num_classes=len(torch.unique(data.y))

    elif dataset_name == 'arxiv':
        data, title, content=load_arxiv(dataset_name=dataset_name)
        num_classes=len(torch.unique(data.y))

    elif dataset_name=='cora':
        data, title, content=load_cora()
        num_classes=len(torch.unique(data.y))

    elif dataset_name=='citeseer':
        data, title, content=load_citeseer()
        num_classes=len(torch.unique(data.y))

    elif dataset_name=='pubmed':
        data, title, content=load_pubmed()
        num_classes=len(torch.unique(data.y))

    elif dataset_name=='arxiv_2023':
        data, title, content=load_arxiv(dataset_name=dataset_name)
        num_classes=len(torch.unique(data.y))
        data, title, content=load_arxiv_2023()

    # texts = ["Title: "+t + " Abstract: " + c for t, c in zip(title, content)]
    num_nodes = data.x.shape[0]

    # Load text embeddings
    csv_path = f'lm1_embeddings_{dataset_name}.csv'
    df = pd.read_csv(csv_path)
    text_embeddings = torch.zeros((num_nodes, 768), dtype=torch.float)

    for _, row in df.iterrows():
        node_id = int(row['id'])
        emb = torch.tensor(eval(row['embedding']), dtype=torch.float)
        text_embeddings[node_id] = emb
    
    text_embeddings = text_embeddings.to(device)
    
    # Enhanced model
    if model=='DiGTex':
        model = EnhancedMultiModalGNN(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            GNN=args.GNN,
            num_gnn_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    elif model=='GNN':
        model = GNNMODEL(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            GNN=args.GNN,
            num_gnn_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    # for name, param in model.named_parameters():
    #     print(name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



    num_nodes = data.x.shape[0]
#--------------------------------------------------------------------------------
    path="labels/"+args.LLM+"/"+dataset_name+"/"
#-------------------------------------------------------------------------------
    
    def load_all_label_files(num_nodes,path, device):
        # path = f"labels/all/{dataset_name}/"
        
        # تمام فایل‌های CSV که با predicted_labels شروع می‌شوند
        csv_files = sorted([f for f in os.listdir(path) if f.startswith("predicted_labels") and f.endswith(".csv")])

        all_labels = []   # لیست نهایی شامل هر labels_i

        print("Found CSV files:", csv_files)

        for csv_file in csv_files:
            csv_path = os.path.join(path, csv_file)
            print(f"\nReading file: {csv_path}")

            labels = torch.full((num_nodes,), 0, dtype=torch.long)

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # رد کردن header

                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        node_id = int(row[0])
                        label_id = int(row[2])
                        labels[node_id] = label_id
                    except Exception as e:
                        print(f"Error in row {row}: {e}")

            print("labels loaded.")
            labels = labels.to(device)

            # similarity check
            # print(f"similarity between labels ({csv_file}): ", label_similarity(data.y, labels))

            all_labels.append(labels)

        return all_labels
    all_labels = load_all_label_files(num_nodes,path, device)
      
#-------------------------------------------------------------------------------
    path ="labels/"+args.LLM+"/"+dataset_name+"/reasons/"

    reasoning_embeddings = torch.zeros((num_nodes, 768), dtype=torch.float)
    reasoning_texts = [""] * num_nodes  # جای هر نود یک متن می‌گذاریم
    csv_path = f"{path}reasoning_embeddings_{dataset_name}_qwen.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[0])
            reasoning_texts[node_id] = row[1]
           
            emb = torch.tensor(eval(row[2]), dtype=torch.float)
            reasoning_embeddings[node_id] = emb
    print("reasons loaded...")
    reasoning_embeddings = reasoning_embeddings.to(device)
    print("label_clustering loaded")
    # print("similarity between labels:= ",label_similarity(data.y, labels))
    labels6 = cluster_and_map_to_true_labels(                      #label2
    reasoning_embeddings=reasoning_embeddings,
    true_labels=data.y,
    num_classes=num_classes,
    dataset_name=dataset_name,
    node_ids=range(reasoning_embeddings.shape[0]),
)
    # print("similarity between labels reason:= ",label_similarity(data.y, torch.tensor(labels6)))

    labels6=torch.tensor(labels6).to(device)
    #----------------------------------------------------------------------
    #-------------------------------------majority voting
  

    # def majority_hard_labels(labels_list, num_classes, device='cpu'):
    #     """
    #     labels_list: list of tensors [num_nodes] از برچسب هر annotator
    #     num_classes: تعداد کل کلاس‌ها
    #     device: دستگاه (cpu یا cuda)

    #     خروجی: final_labels_hard [num_nodes] به صورت hard labels
    #     """
    #     num_nodes = labels_list[0].shape[0]
    #     num_annotators = len(labels_list)

    #     # [N, K]
    #     all_labels = torch.stack(labels_list, dim=1)

    #     final_labels_hard = torch.zeros(num_nodes, dtype=torch.long)

    #     for i in range(num_nodes):
    #         node_labels = all_labels[i]  # [K]

    #         counts = torch.bincount(node_labels, minlength=num_classes)

    #         max_count = counts.max()
    #         majority_classes = (counts == max_count).nonzero(as_tuple=True)[0]

    #         if majority_classes.numel() == 1:
    #             # اکثریت یکتا
    #             final_labels_hard[i] = majority_classes.item()
    #         else:
    #             # تساوی → برچسب annotator اول
    #             final_labels_hard[i] = labels_list[0][i]

    #     return final_labels_hard.to(device)
    # final_hard = majority_hard_labels([labels6, all_labels[0], all_labels[1]],num_classes=num_classes,device='cpu')

    #------------------------------------------------- soft with majority
    def majority_soft_labels(labels_list, num_classes, device='cpu'):
        """
        labels_list: list of tensors [num_nodes] از برچسب هر annotator
        num_classes: تعداد کل کلاس‌ها
        device: دستگاه (cpu یا cuda)
        
        خروجی: final_labels_soft [num_nodes, num_classes] به صورت soft labels
        """
        num_nodes = labels_list[0].shape[0]
        num_annotators = len(labels_list)
        
        # ترکیب تمام برچسب‌ها به یک tensor [num_nodes, num_annotators]
        all_labels = torch.stack(labels_list, dim=1)  # [N, K]

        # ماتریس نهایی soft
        final_labels_soft = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
        
        for i in range(num_nodes):
            # برچسب‌های نود i
            node_labels = all_labels[i]  # [K]
            # تعداد رای‌ها برای هر کلاس
            # print("i=",i)
            # print("node_labels=",node_labels)
            counts = torch.bincount(node_labels, minlength=num_classes).float()
            # وزن نهایی: تعداد رای هر کلاس تقسیم بر تعداد annotators
            final_labels_soft[i] = counts / num_annotators
        
        return final_labels_soft.to(device)

    # final_soft = majority_soft_labels([labels6, all_labels[0],all_labels[1]], num_classes=num_classes, device='cpu')
    # print(final_soft[0])
    #------------------------------------------------- soft with majority

    #------------------------------------------------ soft with DS
    posteriors, hard_labels, confs = ds_aggregate_three(
        labels6, all_labels[6], all_labels[10], num_classes=num_classes,
        ignore_index=-1,   # اگر missing ندارید، همین را بگذارید و مطمئن شوید هیچ -1ی وجود ندارد
        alpha=0.1,         # smoothing؛ می‌توانی 0.1 یا 0.5 هم تست کنی
        init="majority",   # یا "uniform"
        max_iter=100,
        tol=1e-5,
        verbose=True,
    )
        # بعد از اجرا، ببین کدوم annotator بهتره:
    print("\n📊 دقت تخمینی annotatorها (از confusion matrices):")
    for i, conf in enumerate(confs):
        accuracy = torch.diag(conf).mean().item()
        off_diag = (conf.sum() - torch.diag(conf).sum()) / (conf.numel() - conf.shape[0])
        print(f"  Annotator {i+1}: دقت={accuracy:.3f}, noise={off_diag:.3f}")

    # اگه می‌خوای ببینی posteriors چقدر مطمئنه:
    entropy = -(posteriors * (posteriors + 1e-12).log()).sum(1).mean()
    print(f"\n🎯 میانگین entropy: {entropy:.4f}")
    print(f"   (کمتر = مطمئن‌تر، بیشتر = نامطمئن‌تر)")
    # اگر خواستی در شیء data ذخیره کنی:
    data.soft_y = posteriors            # (N, C)
    data.y_ds = hard_labels             # (N,)
    # print("similarity between labels DS:= ",label_similarity(data.y, torch.tensor(data.y_ds)))
    # print("data.soft_y=",data.soft_y[0])
    # print("confs=",confs)
    #------------------------------------------------ soft with DS

    # Training
    data.edge_index = data.edge_index.contiguous()
    train_loader, valid_loader, test_loader = create_loaders(data, args.batch_size)
    # Enhanced model
    if args.Model=='DiGTex':
        data=data.to(device)
        best_val_acc, test_acc = train_enhanced_model(
            model,data, args.epochs, train_loader, valid_loader, test_loader,
            text_embeddings, reasoning_embeddings,data.soft_y ,  device, dataset_name=dataset_name
        )
    elif args.Model=='GNN':
        #---------------- for 10 % labeled
        # data.train_idx = torch.as_tensor(data.train_idx).to(device)
        data=data.to(device)
        # import math
        # num_per_class=round(num_nodes * args.few_percent/num_classes)
        # print("num_per_class",num_per_class)

        # fewshot_idx = get_fewshot_train_idx(data, num_per_class=num_per_class)
        # data.train_idx=fewshot_idx
        # train_loader, valid_loader, test_loader = create_loaders(data, args.batch_size)
        #-----------------------------------
        best_val_acc, test_acc = train_GNN(
            model, args.epochs, train_loader, valid_loader, test_loader,
            text_embeddings, data.y, device, dataset_name
        )


    print(f"\n✅ Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"🎯 Final Test Accuracy: {test_acc:.4f}")


    #-----------------------------------------------------------------
    logits_smoothed, pseudo_labels, confidence = correct_and_smooth(model, data, text_embeddings, device,confidence_threshold=0.85)

    pred = logits_smoothed.argmax(dim=1)
    acc = (pred[data.test_idx] == data.y[data.test_idx].to(pred.device)).float().mean().item()

    print(f"Zero-shot Accuracy after C&S1_0.85: {acc:.4f}")

    logits_smoothed, pseudo_labels, confidence = correct_and_smooth(model, data, text_embeddings, device,confidence_threshold=0.7)

    pred = logits_smoothed.argmax(dim=1)
    acc = (pred[data.test_idx] == data.y[data.test_idx].to(pred.device)).float().mean().item()

    print(f"Zero-shot Accuracy after C&S1_0.7: {acc:.4f}")
    #--------------------------------------------------------------------------
    num_nodes = data.x.shape[0]


#--------------------------- FEW-SHOT---------------------------
    print("____________________ few-Shot_________________________")
    model3 = EnhancedMultiModalGNN(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            GNN=args.GNN,
            num_gnn_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    model_name= dataset_name+"_model.pt"
    model3.load_state_dict(torch.load(model_name))        # لود وزن‌ها
    import copy
    # model3 = copy.deepcopy(model)
    model3 = model3.to(device)
    # model3 = model.to(device) 
    # for name, param in model1.named_parameters():
    #     print(name)
    for name, param in model3.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    total_params = sum(p.numel() for p in model3.parameters())
    trainable_params = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    data.train_idx = torch.as_tensor(data.train_idx).to(device)
    data=data.to(device)

    import math
    num_per_class=round(num_nodes * args.few_percent/num_classes)
    print("num_per_class",num_per_class)

    fewshot_idx = get_fewshot_train_idx(data, num_per_class=num_per_class)
    data.train_idx=fewshot_idx
    train_loader, valid_loader, test_loader = create_loaders(data, args.batch_size)
    best_val_acc, test_acc = fine_enhanced_model(
        model3, args.epochs, train_loader, valid_loader, test_loader,
        text_embeddings,  device)
    print(f"\n✅ Best Validation Accuracy(FEW_shot): {best_val_acc:.4f}")
    print(f"🎯 Final Test Accuracy (FFEW_shot): {test_acc:.4f}")
#-----------------------------------------------------------------------------------
#----------------------------------------------------------------------------
  




if __name__ == '__main__':
    main()