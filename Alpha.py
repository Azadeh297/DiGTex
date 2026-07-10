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
import itertools
from typing import Dict, Tuple, Callable, Optional, Union




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



#-------------------------------- Alpha


# -----------------------
# Krippendorff's Alpha (Nominal)
# -----------------------

def krippendorff_alpha(data: np.ndarray, distance_fn: Callable[[float, float], float]) -> float:
    """
    Calculate Krippendorff's alpha for nominal data.
    
    Args:
        data: numpy array with shape (num_annotators, num_items)
        distance_fn: function to compute distance between two values
    
    Returns:
        float: Krippendorff's alpha coefficient (-1 to 1)
    """
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    
    # Early return if insufficient data
    if np.sum(mask) < 2:
        return np.nan
    
    # Get all valid values for category counting
    valid_data = data[mask]
    categories = np.unique(valid_data)
    
    if len(categories) <= 1:
        return 1.0  # Perfect agreement if only one category
    
    # Compute observed disagreement (Do)
    Do = 0
    total_pairs = 0
    
    for i in range(data.shape[1]):  # each item
        item_labels = data[:, i][~np.isnan(data[:, i])]
        if len(item_labels) >= 2:
            for a, b in itertools.combinations(item_labels, 2):
                Do += distance_fn(a, b)
                total_pairs += 1
    
    if total_pairs == 0:
        return np.nan
    
    Do = Do / total_pairs
    
    # Compute expected disagreement (De)
    category_counts = {cat: np.sum(valid_data == cat) for cat in categories}
    total_labels = len(valid_data)
    
    De = 0
    for a, b in itertools.product(categories, repeat=2):
        if a != b:  # Only count disagreements for expected
            prob_a = category_counts[a] / total_labels
            prob_b = category_counts[b] / total_labels
            De += distance_fn(a, b) * prob_a * prob_b
    
    # Handle edge cases
    if De == 0:
        return 1.0 if Do == 0 else 0.0
    
    return 1 - (Do / De)


def nominal_distance(a: float, b: float) -> float:
    """Nominal distance function: 0 if equal, 1 if different."""
    return 0.0 if a == b else 1.0


def ordinal_distance(a: float, b: float) -> float:
    """Ordinal distance function: squared difference."""
    return (a - b) ** 2


def interval_distance(a: float, b: float) -> float:
    """Interval distance function: squared difference (same as ordinal for most purposes)."""
    return (a - b) ** 2


def ratio_distance(a: float, b: float) -> float:
    """Ratio distance function: squared relative difference."""
    if a == 0 and b == 0:
        return 0.0
    if a == 0 or b == 0:
        return 1.0
    return ((a - b) / (a + b)) ** 2


# -----------------------
# Main Function
# -----------------------

def compute_alpha_for_annotators(
    labels_matrix: Union[np.ndarray, torch.Tensor, list],
    distance_type: str = 'nominal'
) -> Tuple[float, Dict[int, float]]:
    """
    Compute Krippendorff's Alpha for full set and for each annotator removed.
    
    Args:
        labels_matrix: shape (num_samples, num_annotators) - labels for each sample by each annotator
        distance_type: 'nominal', 'ordinal', 'interval', or 'ratio'
    
    Returns:
        tuple: (full_alpha, alpha_per_removal)
            - full_alpha: float, alpha for all annotators
            - alpha_per_removal: dict {annotator_index: alpha_after_removal}
    """
    # Convert input to numpy array safely
    if isinstance(labels_matrix, torch.Tensor):
        labels_matrix = labels_matrix.detach().cpu().numpy()
    elif isinstance(labels_matrix, list):
        labels_matrix = np.array(labels_matrix)
    else:
        labels_matrix = np.array(labels_matrix)
    
    # Transpose to (num_annotators, num_samples) format expected by krippendorff_alpha
    data = labels_matrix.T
    
    # Select distance function
    distance_functions = {
        'nominal': nominal_distance,
        'ordinal': ordinal_distance, 
        'interval': interval_distance,
        'ratio': ratio_distance
    }
    
    if distance_type not in distance_functions:
        raise ValueError(f"Unknown distance_type: {distance_type}. Choose from {list(distance_functions.keys())}")
    
    distance_fn = distance_functions[distance_type]
    
    # Compute alpha for full dataset
    full_alpha = krippendorff_alpha(data, distance_fn=distance_fn)
    
    # Compute alpha with each annotator removed
    alpha_per_removal = {}
    for i in range(data.shape[0]):
        reduced_data = np.delete(data, i, axis=0)
        alpha_per_removal[i] = krippendorff_alpha(reduced_data, distance_fn=distance_fn)
    
    return full_alpha, alpha_per_removal


def analyze_annotator_reliability(labels_matrix: Union[np.ndarray, torch.Tensor, list], 
                                 distance_type: str = 'nominal',
                                 annotator_names: Optional[list] = None) -> Dict:
    """
    Comprehensive analysis of annotator reliability using Krippendorff's Alpha.
    
    Args:
        labels_matrix: shape (num_samples, num_annotators)
        distance_type: type of distance metric to use
        annotator_names: optional list of annotator names
    
    Returns:
        dict: comprehensive analysis results
    """
    full_alpha, alpha_per_removal = compute_alpha_for_annotators(labels_matrix, distance_type)
    
    # if annotator_names is None:
    #     annotator_names = [f"Annotator_{i}" for i in range(labels_matrix.shape[1])]
    # تعداد annotator ها را از shape ماتریس بگیرید
    num_annotators = labels_matrix.shape[1] if hasattr(labels_matrix, 'shape') else len(labels_matrix[0])
    if annotator_names is None:
        annotator_names = [f"Annotator_{i}" for i in range(num_annotators)]
    elif len(annotator_names) < num_annotators:
        # اگر تعداد نام‌ها کمتر از annotator هاست، باقی را اضافه کن
        annotator_names.extend([f"Annotator_{i}" for i in range(len(annotator_names), num_annotators)])
    
    # Calculate improvement when removing each annotator
    improvements = {}
    for i, alpha_without in alpha_per_removal.items():
        improvements[i] = alpha_without - full_alpha
    
    # Find most problematic annotator
    most_problematic_idx = max(improvements.keys(), key=lambda x: improvements[x])
    best_removal_improvement = improvements[most_problematic_idx]
    
    # Agreement interpretation
    def interpret_alpha(alpha):
        if np.isnan(alpha):
            return "Insufficient data"
        elif alpha < 0:
            return "Poor (worse than random)"
        elif alpha < 0.2:
            return "Slight agreement"
        elif alpha < 0.4:
            return "Fair agreement" 
        elif alpha < 0.6:
            return "Moderate agreement"
        elif alpha < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
    
    results = {
        'overall_alpha': full_alpha,
        'overall_interpretation': interpret_alpha(full_alpha),
        'alpha_per_removal': alpha_per_removal,
        'improvements_per_removal': improvements,
        'most_problematic_annotator': {
            'index': most_problematic_idx,
            'name': annotator_names[most_problematic_idx],
            'improvement_if_removed': best_removal_improvement
        },
        'annotator_rankings': sorted(
            [(i, annotator_names[i], improvements[i]) for i in improvements.keys()],
            key=lambda x: x[2], reverse=True
        ),
        'distance_type': distance_type
    }
    
    return results



#-                               Alpha
#---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-modal GNN')
    parser.add_argument('dataset_name', default='cora', help='dataset name')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--Model', default='DiGTex', help='Models: DiGTex, GNN')
    parser.add_argument('--GNN', default='SAGE', help='type of GNN:  GCN, GAT, SAGE, MLP')
    parser.add_argument('--hidden_dim', default=768, type=int, help='hidden dimension')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
    
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



    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



    def load_all_label_files(dataset_name, num_nodes, device):
        path = f"labels/all/{dataset_name}/"
        
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
    all_labels = load_all_label_files(dataset_name, num_nodes, device)


    #-----------------------------

 
    # -----------------------
# Example Usage and Testing
# -----------------------


    np.random.seed(42)
    
    labels_matrix = torch.stack([all_labels[6],all_labels[10]], dim=1)
    annotator_names = []
    
    print(f"شکل ماتریس برچسب‌ها: {labels_matrix.shape}")
    print(f"نمونه از داده‌ها:")
    print(labels_matrix[:5])  # نمایش 5 نمونه اول
    
    # if annotator_names is None:
    #     annotator_names = ['Annotator_1', 'Annotator_2', 'Annotator_3']
    
    # 2. تحلیل کامل قابلیت اعتماد annotator ها
    results = analyze_annotator_reliability(
        labels_matrix=labels_matrix,
        distance_type='nominal',  # برای داده‌های categorical
        annotator_names=annotator_names
    )
    print(f"Overall Krippendorff's Alpha: {results['overall_alpha']:.3f}")
    print(f"Interpretation: {results['overall_interpretation']}")
    print(f"\nMost problematic annotator: {results['most_problematic_annotator']['name']}")
    print(f"Alpha would improve by {results['most_problematic_annotator']['improvement_if_removed']:.3f} if removed")
    
    print(f"\nAnnotator rankings (by improvement if removed):")
    for idx, name, improvement in results['annotator_rankings']:
        print(f"  {name}: {improvement:+.3f}")

    





if __name__ == '__main__':
    main()