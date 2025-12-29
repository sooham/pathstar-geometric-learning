"""
Visualize token embeddings from a trained GPT model checkpoint using UMAP.

This script loads a checkpoint (which now includes metadata) and creates multiple UMAP visualizations:

Standard visualizations:
1. All embeddings colored by token type (special vs node)
2. Node embeddings colored by token ID
3. 3D UMAP visualization
4. Node embeddings colored by similarity to root
5. Comprehensive summary figure

Path-based visualizations (if paths_by_leaf is in checkpoint metadata):
6. Sample of training paths with distinct colors
7. Path membership count heatmap
8. Train vs holdout path visualization
9. Depth in path structure visualization

Usage:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt
    
With custom save directory:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt --save_dir visualizations/
"""

import argparse
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

from model import GPTConfig, GPT
from umap_utils import apply_umap, plot_umap, umap_with_annotations


def load_checkpoint_and_model(checkpoint_path, device='cpu'):
    """
    Load a checkpoint and reconstruct the model.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on ('cpu', 'cuda', 'mps')
        
    Returns:
        model: Loaded GPT model
        checkpoint: Full checkpoint dict
        config: Training config from checkpoint
        meta: Metadata dict from checkpoint (contains paths_by_leaf, train_leaves, etc.)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments
    model_args = checkpoint['model_args']
    config = checkpoint.get('config', {})
    meta = checkpoint.get('meta', {})

    if meta is None:
        raise ValueError("No metadata found in checkpoint")
    
    print(f"Model args: {model_args}")
    
    # Create model config
    gptconf = GPTConfig(**model_args)
    
    # Initialize model
    model = GPT(gptconf)
    
    # Load state dict (handle potential _orig_mod. prefix from torch.compile)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Vocab size: {gptconf.vocab_size}")
    print(f"  - Embedding dim: {gptconf.n_embd}")
    print(f"  - Layers: {gptconf.n_layer}")
    print(f"  - Heads: {gptconf.n_head}")
    
    if meta:
        print(f"\nMetadata loaded from checkpoint:")
        print(f"  - d (spokes): {meta.get('d', 'N/A')}")
        print(f"  - l (length): {meta.get('l', 'N/A')}")
        print(f"  - vocab_size: {meta.get('vocab_size', 'N/A')}")
        print(f"  - root_vertex: {meta.get('root_vertex', 'N/A')}")
        if 'paths_by_leaf' in meta:
            print(f"  - paths_by_leaf: Available ({len(meta['paths_by_leaf'])} paths)")
        if 'train_leaves' in meta:
            print(f"  - train_leaves: {len(meta['train_leaves'])} leaves")
        if 'holdout_leaves' in meta:
            print(f"  - holdout_leaves: {len(meta['holdout_leaves'])} leaves")
    else:
        print("\nWarning: No metadata found in checkpoint")
    
    return model, checkpoint, config, meta


def load_metadata(data_dir):
    """
    Load metadata from the data directory.
    
    Args:
        data_dir: Path to the data directory containing meta.pkl
        
    Returns:
        meta: Metadata dictionary
    """
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        print(f"Loaded metadata from {meta_path}")
        return meta
    else:
        print(f"Warning: No metadata found at {meta_path}")
        return None


def extract_embeddings(model):
    """
    Extract token embeddings from the model.
    
    Args:
        model: GPT model
        
    Returns:
        embeddings: numpy array of shape (vocab_size, n_embd)
    """
    with torch.no_grad():
        embeddings = model.transformer.wte.weight.detach().cpu().numpy()
    
    print(f"Extracted embeddings: {embeddings.shape}")
    return embeddings


def create_token_labels(meta, vocab_size):
    """
    Create labels for each token based on metadata.
    
    Args:
        meta: Metadata dictionary
        vocab_size: Size of vocabulary
        
    Returns:
        labels: Dict with various labeling schemes
    """
    labels = {
        'token_type': np.zeros(vocab_size, dtype=int),  # 0=node, 1=special
        'token_names': [''] * vocab_size,
        'is_special': np.zeros(vocab_size, dtype=bool),
        'depth': np.full(vocab_size, -1, dtype=int),  # -1 for unknown/special
    }
    
    if meta is None:
        return labels
    
    # Mark special tokens
    special_tokens = meta.get('special_tokens', {})
    for name, token_id in special_tokens.items():
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            labels['token_type'][token_id] = 1
            labels['token_names'][token_id] = name
            labels['is_special'][token_id] = True
    
    # Use itos mapping for token names
    itos = meta.get('itos', {})
    for token_id, name in itos.items():
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            if labels['token_names'][token_id] == '':
                labels['token_names'][token_id] = str(name)
    
    # Try to extract depth information from path structure
    # PathStar has structure: root connects to d spokes of length l
    d = meta.get('d', 0)
    l = meta.get('l', 0)
    root_vertex = meta.get('root_vertex', None)
    
    if root_vertex is not None and 0 <= root_vertex < vocab_size:
        labels['depth'][root_vertex] = 0  # Root is at depth 0
        labels['token_names'][root_vertex] = f"ROOT({root_vertex})"
    
    # Mark node tokens that are not special
    num_special = len(special_tokens)
    for i in range(vocab_size):
        if not labels['is_special'][i]:
            labels['token_type'][i] = 0  # node token
    
    return labels


def visualize_all_embeddings(embeddings, labels, save_dir='out', prefix='embedding_umap'):
    """
    Create comprehensive UMAP visualizations of embeddings.
    
    Args:
        embeddings: numpy array of shape (vocab_size, n_embd)
        labels: Dict with labeling information
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vocab_size = embeddings.shape[0]
    print(f"\nCreating UMAP visualizations for {vocab_size} tokens...")
    
    # 1. UMAP of all embeddings, colored by token type (special vs node)
    print("  [1/4] Computing UMAP for all embeddings (colored by type)...")
    
    reduced_all = apply_umap(
        embeddings, 
        n_components=2, 
        n_neighbors=30, 
        min_dist=0.1,
        random_state=42
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot node tokens
    node_mask = ~labels['is_special']
    special_mask = labels['is_special']
    
    ax.scatter(
        reduced_all[node_mask, 0], 
        reduced_all[node_mask, 1],
        c='steelblue', alpha=0.5, s=20, label=f'Node tokens ({node_mask.sum()})'
    )
    
    # Plot special tokens with distinct markers
    ax.scatter(
        reduced_all[special_mask, 0], 
        reduced_all[special_mask, 1],
        c='red', alpha=1.0, s=100, marker='*', label=f'Special tokens ({special_mask.sum()})'
    )
    
    # Annotate special tokens
    for i in np.where(special_mask)[0]:
        name = labels['token_names'][i]
        if name:
            ax.annotate(
                name, 
                (reduced_all[i, 0], reduced_all[i, 1]),
                fontsize=9, fontweight='bold',
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Token Embeddings: Special vs Node Tokens', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path1 = os.path.join(save_dir, f'{prefix}_all_tokens.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path1}")
    
    # 2. UMAP of node tokens only, colored by token ID (as a proxy for structure)
    print("  [2/4] Computing UMAP for node tokens only (colored by token ID)...")
    
    node_embeddings = embeddings[node_mask]
    node_indices = np.where(node_mask)[0]
    
    reduced_nodes = apply_umap(
        node_embeddings,
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by token ID (normalized)
    colors = node_indices / node_indices.max()
    scatter = ax.scatter(
        reduced_nodes[:, 0], 
        reduced_nodes[:, 1],
        c=colors, cmap='viridis', alpha=0.7, s=15
    )
    plt.colorbar(scatter, ax=ax, label='Token ID (normalized)')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Node Token Embeddings (Colored by Token ID)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    path2 = os.path.join(save_dir, f'{prefix}_nodes_by_id.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path2}")
    
    # 3. 3D UMAP visualization
    print("  [3/4] Computing 3D UMAP...")
    
    reduced_3d = apply_umap(
        embeddings,
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42
    )
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot node tokens
    ax.scatter(
        reduced_3d[node_mask, 0],
        reduced_3d[node_mask, 1],
        reduced_3d[node_mask, 2],
        c='steelblue', alpha=0.4, s=10, label='Node tokens'
    )
    
    # Plot special tokens
    ax.scatter(
        reduced_3d[special_mask, 0],
        reduced_3d[special_mask, 1],
        reduced_3d[special_mask, 2],
        c='red', alpha=1.0, s=100, marker='*', label='Special tokens'
    )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title('Token Embeddings (3D UMAP)', fontsize=14)
    ax.legend()
    
    path3 = os.path.join(save_dir, f'{prefix}_3d.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path3}")
    
    # 4. Similarity-based coloring (cosine distance to root if available)
    print("  [4/4] Computing embedding similarities...")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms
    
    # Find root token
    root_idx = None
    for i, name in enumerate(labels['token_names']):
        if 'ROOT' in str(name).upper() or name == '0':
            root_idx = i
            break
    
    if root_idx is not None:
        # Compute cosine similarity to root
        root_emb = normalized[root_idx:root_idx+1]
        similarities = np.dot(normalized, root_emb.T).flatten()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            reduced_all[node_mask, 0],
            reduced_all[node_mask, 1],
            c=similarities[node_mask],
            cmap='RdYlGn',
            alpha=0.7, s=20
        )
        plt.colorbar(scatter, ax=ax, label='Cosine Similarity to Root')
        
        # Mark root
        ax.scatter(
            reduced_all[root_idx, 0],
            reduced_all[root_idx, 1],
            c='black', s=200, marker='X', label='Root', zorder=10
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('Node Embeddings Colored by Similarity to Root', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        path4 = os.path.join(save_dir, f'{prefix}_similarity_to_root.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path4}")
    else:
        print("    Skipping similarity plot (no root token found)")
    
    # 5. Create a comprehensive summary figure
    print("\n  Creating summary figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel 1: All tokens
    ax = axes[0, 0]
    ax.scatter(reduced_all[node_mask, 0], reduced_all[node_mask, 1],
               c='steelblue', alpha=0.5, s=15, label='Nodes')
    ax.scatter(reduced_all[special_mask, 0], reduced_all[special_mask, 1],
               c='red', s=100, marker='*', label='Special')
    ax.set_title('All Tokens', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Nodes colored by ID
    ax = axes[0, 1]
    scatter = ax.scatter(reduced_nodes[:, 0], reduced_nodes[:, 1],
                        c=colors, cmap='viridis', alpha=0.7, s=15)
    plt.colorbar(scatter, ax=ax, label='Token ID')
    ax.set_title('Nodes by Token ID', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Histogram of embedding norms
    ax = axes[1, 0]
    node_norms = np.linalg.norm(embeddings[node_mask], axis=1)
    special_norms = np.linalg.norm(embeddings[special_mask], axis=1)
    ax.hist(node_norms, bins=50, alpha=0.7, label='Nodes', color='steelblue')
    ax.hist(special_norms, bins=10, alpha=0.7, label='Special', color='red')
    ax.set_xlabel('Embedding Norm')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Embedding Norms', fontsize=12)
    ax.legend()
    
    # Panel 4: Cosine similarity distribution (random pairs)
    ax = axes[1, 1]
    n_samples = min(1000, len(node_indices))
    sample_idx = np.random.choice(len(node_indices), n_samples, replace=False)
    node_normed = normalized[node_mask]
    
    # Random pairs
    random_sims = []
    for _ in range(2000):
        i, j = np.random.choice(n_samples, 2, replace=False)
        sim = np.dot(node_normed[sample_idx[i]], node_normed[sample_idx[j]])
        random_sims.append(sim)
    
    ax.hist(random_sims, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero similarity')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Cosine Similarity Between Random Node Pairs', fontsize=12)
    ax.legend()
    
    plt.suptitle('Token Embedding Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path_summary = os.path.join(save_dir, f'{prefix}_summary.png')
    fig.savefig(path_summary, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary: {path_summary}")
    
    return reduced_all


def visualize_paths_in_umap(embeddings, labels, meta, save_dir='out', prefix='embedding_umap'):
    """
    Visualize embeddings with each path colored distinctly using path information from metadata.
    
    Args:
        embeddings: numpy array of shape (vocab_size, n_embd)
        labels: Dict with labeling information
        meta: Metadata dictionary containing path information
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    if meta is None or not meta:
        print("Skipping path-based visualization: No metadata available")
        return
    
    # Check if path information is available
    paths_by_leaf = meta.get('paths_by_leaf', None)
    train_leaves = meta.get('train_leaves', set())
    holdout_leaves = meta.get('holdout_leaves', set())
    root_vertex = meta.get('root_vertex', None)
    
    if paths_by_leaf is None:
        print("Skipping path-based visualization: 'paths_by_leaf' not available in metadata")
        return
    
    if root_vertex is None:
        print("Warning: 'root_vertex' not found in metadata")
    
    print("\n=== Path-Based Visualization ===")
    print(f"  Train leaves: {len(train_leaves)}")
    print(f"  Holdout leaves: {len(holdout_leaves)}")
    print(f"  Total paths: {len(paths_by_leaf)}")
    print(f"  Root vertex: {root_vertex}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    vocab_size = embeddings.shape[0]
    node_mask = ~labels['is_special']
    special_mask = labels['is_special']
    
    # Compute UMAP projection
    print("  Computing UMAP projection...")
    reduced_all = apply_umap(
        embeddings, 
        n_components=2, 
        n_neighbors=30, 
        min_dist=0.1,
        random_state=42
    )
    
    # Create a mapping from token to path indices it belongs to
    token_to_paths = {}
    for leaf_token, path_tokens in paths_by_leaf.items():
        for token in path_tokens:
            if token not in token_to_paths:
                token_to_paths[token] = []
            token_to_paths[token].append(leaf_token)
    
    # Separate train and holdout paths
    train_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in train_leaves]
    holdout_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in holdout_leaves]
    
    print(f"  Paths in training: {len(train_path_leaves)}")
    print(f"  Paths in holdout: {len(holdout_path_leaves)}")
    
    # ============================================================
    # Visualization 1: Sample of paths with distinct colors
    # ============================================================
    max_paths_to_show = 20  # Limit to avoid color confusion
    
    if len(train_path_leaves) > 0:
        print(f"\n  [1/4] Creating visualization: Sample of training paths with distinct colors...")
        
        # Sample a subset of paths to visualize
        num_paths_to_show = min(max_paths_to_show, len(train_path_leaves))
        sampled_leaves = np.random.choice(train_path_leaves, size=num_paths_to_show, replace=False)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Generate distinct colors for each path
        colors_list = cm.get_cmap('tab20' if num_paths_to_show <= 20 else 'hsv')(np.linspace(0, 1, num_paths_to_show))
        
        # Plot background: all node tokens in gray
        ax.scatter(
            reduced_all[node_mask, 0],
            reduced_all[node_mask, 1],
            c='lightgray', alpha=0.2, s=10, label='Other nodes', zorder=1
        )
        
        # Plot each path with a distinct color
        for path_idx, leaf_token in enumerate(sampled_leaves):
            path_tokens = paths_by_leaf[leaf_token]
            path_mask = np.array([t in path_tokens for t in range(vocab_size)])
            
            if path_mask.any():
                color = colors_list[path_idx]
                ax.scatter(
                    reduced_all[path_mask, 0],
                    reduced_all[path_mask, 1],
                    c=[color], alpha=0.8, s=50, 
                    label=f'Path to leaf {leaf_token}',
                    edgecolors='black', linewidths=0.5,
                    zorder=3
                )
        
        # Highlight root if available
        if root_vertex is not None and 0 <= root_vertex < vocab_size:
            ax.scatter(
                reduced_all[root_vertex, 0],
                reduced_all[root_vertex, 1],
                c='black', s=300, marker='X', 
                label=f'Root ({root_vertex})',
                edgecolors='white', linewidths=2,
                zorder=10
            )
        
        # Plot special tokens
        ax.scatter(
            reduced_all[special_mask, 0],
            reduced_all[special_mask, 1],
            c='red', alpha=0.6, s=100, marker='*',
            label='Special tokens',
            zorder=2
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'Sample of {num_paths_to_show} Training Paths (Distinct Colors)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        
        path1 = os.path.join(save_dir, f'{prefix}_paths_distinct.png')
        fig.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path1}")
    
    # ============================================================
    # Visualization 2: Heatmap showing path membership count
    # ============================================================
    print(f"  [2/4] Creating visualization: Path membership heatmap...")
    
    # Count how many paths each token belongs to
    path_membership_count = np.zeros(vocab_size, dtype=int)
    for token, path_list in token_to_paths.items():
        if 0 <= token < vocab_size:
            path_membership_count[token] = len(path_list)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot nodes colored by path membership count
    node_counts = path_membership_count[node_mask]
    scatter = ax.scatter(
        reduced_all[node_mask, 0],
        reduced_all[node_mask, 1],
        c=node_counts,
        cmap='YlOrRd',
        alpha=0.7, s=30,
        vmin=0, vmax=max(node_counts.max(), 1)
    )
    plt.colorbar(scatter, ax=ax, label='Number of Paths Containing Token')
    
    # Highlight root
    if root_vertex is not None and 0 <= root_vertex < vocab_size:
        ax.scatter(
            reduced_all[root_vertex, 0],
            reduced_all[root_vertex, 1],
            c='black', s=300, marker='X',
            label=f'Root (in {path_membership_count[root_vertex]} paths)',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Plot special tokens
    ax.scatter(
        reduced_all[special_mask, 0],
        reduced_all[special_mask, 1],
        c='blue', alpha=0.6, s=100, marker='*',
        label='Special tokens',
        zorder=5
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Token Embeddings: Path Membership Count', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path2 = os.path.join(save_dir, f'{prefix}_path_membership.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path2}")
    
    # ============================================================
    # Visualization 3: Train vs Holdout paths
    # ============================================================
    print(f"  [3/4] Creating visualization: Train vs Holdout paths...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create masks for train and holdout tokens
    train_path_tokens = set()
    for leaf in train_path_leaves:
        train_path_tokens.update(paths_by_leaf[leaf])
    
    holdout_path_tokens = set()
    for leaf in holdout_path_leaves:
        holdout_path_tokens.update(paths_by_leaf[leaf])
    
    # Tokens only in training paths
    train_only_mask = np.array([
        (t in train_path_tokens and t not in holdout_path_tokens and not labels['is_special'][t])
        for t in range(vocab_size)
    ])
    
    # Tokens only in holdout paths
    holdout_only_mask = np.array([
        (t in holdout_path_tokens and t not in train_path_tokens and not labels['is_special'][t])
        for t in range(vocab_size)
    ])
    
    # Tokens in both (shared nodes like root)
    shared_mask = np.array([
        (t in train_path_tokens and t in holdout_path_tokens and not labels['is_special'][t])
        for t in range(vocab_size)
    ])
    
    # Other nodes
    other_mask = np.array([
        (t not in train_path_tokens and t not in holdout_path_tokens and not labels['is_special'][t])
        for t in range(vocab_size)
    ])
    
    # Plot each category
    if other_mask.any():
        ax.scatter(
            reduced_all[other_mask, 0],
            reduced_all[other_mask, 1],
            c='lightgray', alpha=0.2, s=10,
            label=f'Other nodes ({other_mask.sum()})',
            zorder=1
        )
    
    if train_only_mask.any():
        ax.scatter(
            reduced_all[train_only_mask, 0],
            reduced_all[train_only_mask, 1],
            c='steelblue', alpha=0.6, s=40,
            label=f'Train-only paths ({train_only_mask.sum()})',
            zorder=3
        )
    
    if holdout_only_mask.any():
        ax.scatter(
            reduced_all[holdout_only_mask, 0],
            reduced_all[holdout_only_mask, 1],
            c='orange', alpha=0.6, s=40,
            label=f'Holdout-only paths ({holdout_only_mask.sum()})',
            zorder=3
        )
    
    if shared_mask.any():
        ax.scatter(
            reduced_all[shared_mask, 0],
            reduced_all[shared_mask, 1],
            c='green', alpha=0.8, s=60,
            label=f'Shared nodes ({shared_mask.sum()})',
            edgecolors='black', linewidths=0.5,
            zorder=5
        )
    
    # Highlight root
    if root_vertex is not None and 0 <= root_vertex < vocab_size:
        ax.scatter(
            reduced_all[root_vertex, 0],
            reduced_all[root_vertex, 1],
            c='black', s=300, marker='X',
            label=f'Root',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Plot special tokens
    ax.scatter(
        reduced_all[special_mask, 0],
        reduced_all[special_mask, 1],
        c='red', alpha=0.6, s=100, marker='*',
        label='Special tokens',
        zorder=2
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Token Embeddings: Train vs Holdout Paths', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path3 = os.path.join(save_dir, f'{prefix}_train_vs_holdout.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path3}")
    
    # ============================================================
    # Visualization 4: Depth in path structure
    # ============================================================
    print(f"  [4/4] Creating visualization: Depth in path structure...")
    
    # Calculate depth for each token (distance from root along path)
    token_depths = np.full(vocab_size, -1, dtype=int)
    
    if root_vertex is not None:
        token_depths[root_vertex] = 0
        
        # For each path, assign depths
        for leaf_token, path_tokens in paths_by_leaf.items():
            for depth, token in enumerate(path_tokens):
                if 0 <= token < vocab_size:
                    # If token already has a depth assigned, take the minimum
                    # (tokens can appear at different depths in different paths)
                    if token_depths[token] == -1:
                        token_depths[token] = depth
                    else:
                        token_depths[token] = min(token_depths[token], depth)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Filter to nodes with assigned depth
    has_depth_mask = (token_depths >= 0) & node_mask
    no_depth_mask = (token_depths < 0) & node_mask
    
    if no_depth_mask.any():
        ax.scatter(
            reduced_all[no_depth_mask, 0],
            reduced_all[no_depth_mask, 1],
            c='lightgray', alpha=0.2, s=10,
            label='Unknown depth',
            zorder=1
        )
    
    if has_depth_mask.any():
        depths = token_depths[has_depth_mask]
        scatter = ax.scatter(
            reduced_all[has_depth_mask, 0],
            reduced_all[has_depth_mask, 1],
            c=depths,
            cmap='viridis',
            alpha=0.7, s=40,
            vmin=0, vmax=depths.max()
        )
        plt.colorbar(scatter, ax=ax, label='Depth (Distance from Root)')
    
    # Highlight root
    if root_vertex is not None and 0 <= root_vertex < vocab_size:
        ax.scatter(
            reduced_all[root_vertex, 0],
            reduced_all[root_vertex, 1],
            c='black', s=300, marker='X',
            label='Root (depth=0)',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Plot special tokens
    ax.scatter(
        reduced_all[special_mask, 0],
        reduced_all[special_mask, 1],
        c='red', alpha=0.6, s=100, marker='*',
        label='Special tokens',
        zorder=5
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Token Embeddings: Depth in Path Structure', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path4 = os.path.join(save_dir, f'{prefix}_depth.png')
    fig.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path4}")
    
    # Print summary statistics
    print("\n  === Path Statistics ===")
    print(f"  Root appears in {path_membership_count[root_vertex] if root_vertex is not None else 0} paths")
    print(f"  Train-only tokens: {train_only_mask.sum()}")
    print(f"  Holdout-only tokens: {holdout_only_mask.sum()}")
    print(f"  Shared tokens: {shared_mask.sum()}")
    print(f"  Max path membership: {path_membership_count.max()}")
    print(f"  Max depth: {token_depths.max()}")
    print("  =======================\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GPT token embeddings using UMAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with checkpoint
  python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt
  
  # With custom output directory
  python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt --save_dir visualizations/
  
  # With custom prefix for output files
  python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt --prefix my_embeddings
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to load model on (cpu, cuda, mps)')
    parser.add_argument('--save_dir', type=str, default='out',
                        help='Directory to save visualizations')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Prefix for saved files (default: auto-generated from checkpoint name)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Token Embedding UMAP Visualization")
    print("=" * 60)
    
    # Auto-generate prefix from checkpoint filename if not provided
    if args.prefix is None:
        # Extract filename without extension
        checkpoint_basename = os.path.basename(args.checkpoint)
        if checkpoint_basename.endswith('.pt'):
            args.prefix = checkpoint_basename[:-3]  # Remove .pt extension
        else:
            args.prefix = checkpoint_basename
        print(f"\nAuto-generated prefix: {args.prefix}")
    
    # Load checkpoint and model (meta is now extracted from checkpoint)
    model, checkpoint, config, meta = load_checkpoint_and_model(args.checkpoint, args.device)
    
    # Extract embeddings
    embeddings = extract_embeddings(model)
    vocab_size = embeddings.shape[0]
    
    # Create labels
    labels = create_token_labels(meta, vocab_size)
    
    print(f"\nToken counts:")
    print(f"  - Special tokens: {labels['is_special'].sum()}")
    print(f"  - Node tokens: {(~labels['is_special']).sum()}")
    
    # Create standard visualizations
    visualize_all_embeddings(
        embeddings, 
        labels, 
        save_dir=args.save_dir,
        prefix=args.prefix
    )
    
    # Create path-based visualizations if metadata is available
    if meta and 'paths_by_leaf' in meta:
        visualize_paths_in_umap(
            embeddings,
            labels,
            meta,
            save_dir=args.save_dir,
            prefix=args.prefix
        )
    else:
        print("\nSkipping path-based visualizations: 'paths_by_leaf' not found in checkpoint metadata")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

