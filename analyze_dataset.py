"""
Analyze a generated PathStar dataset to verify its properties.
Usage: python analyze_dataset.py <dataset_dir>
Example: python analyze_dataset.py data/inweights_pathstar_v4_v4001_pet_elv2_plplain_d4000_l5_undirected_dt_tt
"""

import sys
import os
import pickle
import numpy as np
from collections import defaultdict

def analyze_dataset(dataset_dir):
    """Analyze a PathStar dataset and print statistics."""
    
    # Load metadata
    meta_path = os.path.join(dataset_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}")
        return
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Load data files
    paths_data = np.memmap(os.path.join(dataset_dir, 'paths.bin'), dtype=np.uint16, mode='r')
    edges_data = np.memmap(os.path.join(dataset_dir, 'edges.bin'), dtype=np.uint16, mode='r')
    
    # Handle empty validation file (when holdout_percentage=0)
    val_path = os.path.join(dataset_dir, 'val.bin')
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    else:
        val_data = np.array([], dtype=np.uint16)
    
    # Get parameters
    d = meta['d']
    l = meta['l']
    root_vertex = meta['root_vertex']
    GT_token = meta['special_tokens']['GT']
    LT_token = meta['special_tokens']['LT']
    EDGE_token = meta['special_tokens']['EDGE']
    PATH_token = meta['special_tokens']['PATH']
    predict_dir = meta.get('predict_direction_for_edge_task', False)
    use_directional_tokens = meta.get('use_directional_tokens', True)
    use_undirected = meta.get('use_undirected', False)

    
    PATHS_DATASET_SIZE = meta['PATHS_DATASET_SIZE']
    EDGES_DATASET_SIZE = meta['EDGES_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    # Calculate sequence lengths
    paths_seq_len = len(paths_data) // PATHS_DATASET_SIZE
    edges_seq_len = len(edges_data) // EDGES_DATASET_SIZE
    
    # Handle empty validation set (when holdout_percentage=0)
    if VAL_DATASET_SIZE > 0:
        val_seq_len = len(val_data) // VAL_DATASET_SIZE
    else:
        val_seq_len = paths_seq_len  # Use paths_seq_len as reference
    
    # Reshape data
    paths_data = paths_data.reshape(PATHS_DATASET_SIZE, paths_seq_len)
    edges_data = edges_data.reshape(EDGES_DATASET_SIZE, edges_seq_len)
    
    if VAL_DATASET_SIZE > 0:
        val_data = val_data.reshape(VAL_DATASET_SIZE, val_seq_len)
    else:
        val_data = val_data.reshape(0, val_seq_len)
    
    print("=" * 80)
    print(f"Dataset Analysis: {os.path.basename(dataset_dir)}")
    print("=" * 80)
    print()
    
    print("=== Metadata ===")
    print(f"Graph spokes (d): {d}")
    print(f"Path length (l): {l}")
    print(f"Root vertex: {root_vertex}")
    print(f"Holdout percentage: {meta.get('holdout_percentage', 0.0)}")
    print(f"Predict direction: {predict_dir}")
    print(f"Use directional tokens: {use_directional_tokens}")
    print(f"Use undirected: {use_undirected}")
    
    print("=== Dataset Sizes ===")
    print(f"Training paths: {PATHS_DATASET_SIZE}")
    print(f"Training edges: {EDGES_DATASET_SIZE}")
    print(f"Validation paths: {VAL_DATASET_SIZE}")
    print(f"Sequence lengths (without pause): paths={paths_seq_len}, edges={edges_seq_len}")
    print()
    
    # Analyze edge dataset
    print("=== Edge Dataset Analysis ===")
    
    # Print sample edges for inspection
    print("Sample edges (first 20):")
    for i in range(min(20, EDGES_DATASET_SIZE)):
        seq = edges_data[i]
        print(f"  Edge {i}: {seq}")
    print()
    
    # Count GT edges from root
    gt_edges_from_root = 0
    lt_edges_to_root = 0
    edge_start_nodes = defaultdict(int)
    edge_types = defaultdict(int)
    
    for i in range(EDGES_DATASET_SIZE):
        seq = edges_data[i]
        
        if seq[0] != EDGE_token:
            print(f"Warning: Edge sequence {i} doesn't start with EDGE token")
            continue
        
        if predict_dir:
            # Format: [EDGE, u, v, GT/LT, ...]
            u = seq[1]
            v = seq[2]
            direction = seq[3] if len(seq) > 3 else None
            
            edge_start_nodes[u] += 1
            if direction == GT_token:
                edge_types['GT'] += 1
                if u == root_vertex:
                    gt_edges_from_root += 1
            elif direction == GT_token:
                edge_types['LT'] += 1
                if v == root_vertex:
                    lt_edges_to_root += 1
        else:
            # Format: [EDGE, u, GT/LT, v, ...]
            u = seq[1]
            direction = seq[2] if len(seq) > 2 else None
            v = seq[3] if len(seq) > 3 else None
            
            edge_start_nodes[u] += 1
            if direction == GT_token:
                edge_types['GT'] += 1
                if u == root_vertex:
                    gt_edges_from_root += 1
            elif direction == LT_token:
                edge_types['LT'] += 1
                if v == root_vertex:
                    lt_edges_to_root += 1
    
    print(f"GT edges from root: {gt_edges_from_root} (expected: {d})")
    print(f"LT edges to root: {lt_edges_to_root} (expected: 0 for root)")
    print(f"Total GT edges: {edge_types['GT']}")
    print(f"Total LT edges: {edge_types['LT']}")
    print()
    
    if gt_edges_from_root != d:
        print(f"⚠️  WARNING: Expected {d} GT edges from root, found {gt_edges_from_root}")
        print(f"   This would affect the theoretical minimum calculation!")
    else:
        print(f"✓ Correct: Found exactly {d} GT edges from root")
    print()
    
    # Show top nodes by edge count
    print("Top 10 nodes by outgoing edge count:")
    sorted_nodes = sorted(edge_start_nodes.items(), key=lambda x: x[1], reverse=True)
    for node, count in sorted_nodes[:10]:
        if node == root_vertex:
            print(f"  Node {node} (ROOT): {count} edges")
        else:
            print(f"  Node {node}: {count} edges")
    print()
    
    # Analyze path dataset
    print("=== Path Dataset Analysis ===")
    path_leaf_nodes = set()
    for i in range(PATHS_DATASET_SIZE):
        seq = paths_data[i]
        if seq[0] == PATH_token:
            leaf = seq[1]
            path_leaf_nodes.add(leaf)
    
    print(f"Unique leaf nodes in paths: {len(path_leaf_nodes)}")
    print(f"Expected: {d - meta.get('holdout_percentage', 0) * d:.0f} (after holdout)")
    print()
    
    # Token counting for theoretical minimum
    print("=== Theoretical Minimum Calculation ===")
    path_target_len = meta.get('path_target_length', l)
    
    # For interleaved balanced dataset
    n_path_samples = PATHS_DATASET_SIZE
    n_edge_samples = EDGES_DATASET_SIZE
    
    # Check if paths would be upsampled
    if n_path_samples < n_edge_samples:
        print(f"Note: In training, paths would be upsampled from {n_path_samples} to {n_edge_samples}")
        n_path_samples_upsampled = n_edge_samples
    else:
        n_path_samples_upsampled = n_path_samples
    
    total_path_tokens = n_path_samples_upsampled * path_target_len
    total_edge_tokens = n_edge_samples * 1
    total_tokens = total_path_tokens + total_edge_tokens
    
    import math
    entropy_mass = gt_edges_from_root * math.log(gt_edges_from_root) if gt_edges_from_root > 0 else 0
    optimal_loss = entropy_mass / total_tokens if total_tokens > 0 else 0
    
    print(f"Path tokens (after upsampling): {total_path_tokens:,}")
    print(f"Edge tokens: {total_edge_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Entropy mass: {gt_edges_from_root} * log({gt_edges_from_root}) = {entropy_mass:.6f}")
    print(f"Theoretical minimum loss: {optimal_loss:.10f}")
    print()
    print(f"Training loss should NEVER go below: {optimal_loss:.10f}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py <dataset_dir>")
        print("\nExample:")
        print("  python analyze_dataset.py data/inweights_pathstar_v4_v4001_pet_elv2_plplain_d4000_l5_undirected_dt_tt")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    analyze_dataset(dataset_dir)

