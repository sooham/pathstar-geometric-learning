"""
Visualize token embeddings from a trained GPT model checkpoint using UMAP.

This script loads a checkpoint and creates multiple UMAP visualizations:
1. All embeddings colored by token type (special vs node)
2. Node embeddings colored by depth in the path structure
3. Node embeddings with annotations for special tokens

Usage:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt
    
Or run with defaults:
    python visualize_embeddings_umap.py
"""

import argparse
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments
    model_args = checkpoint['model_args']
    config = checkpoint.get('config', {})
    
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
    
    return model, checkpoint, config


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


def main():
    parser = argparse.ArgumentParser(description='Visualize GPT token embeddings using UMAP')
    parser.add_argument('--checkpoint', type=str, 
                        default='out/ckpt_20251222T203000_fce0632_DSET_G1000L5P5PeUdirTtDt_L5E128H1MlpAgeluLnBiasD0WtEp15000Seed9004_2.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str,
                        default='data/inweights_pathstar_v4001_pet_elv2_plplain_d1000_l5_p5_undirected_dt_tt',
                        help='Path to data directory containing meta.pkl')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to load model on')
    parser.add_argument('--save_dir', type=str, default='out',
                        help='Directory to save visualizations')
    parser.add_argument('--prefix', type=str, default='embedding_umap',
                        help='Prefix for saved files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Token Embedding UMAP Visualization")
    print("=" * 60)
    
    # Load checkpoint and model
    model, checkpoint, config = load_checkpoint_and_model(args.checkpoint, args.device)
    
    # Load metadata
    meta = load_metadata(args.data_dir)
    if meta:
        print(f"\nDataset info from metadata:")
        print(f"  - d (spokes): {meta.get('d', 'N/A')}")
        print(f"  - l (length): {meta.get('l', 'N/A')}")
        print(f"  - vocab_size: {meta.get('vocab_size', 'N/A')}")
        print(f"  - Special tokens: {list(meta.get('special_tokens', {}).keys())}")
    
    # Extract embeddings
    embeddings = extract_embeddings(model)
    vocab_size = embeddings.shape[0]
    
    # Create labels
    labels = create_token_labels(meta, vocab_size)
    
    print(f"\nToken counts:")
    print(f"  - Special tokens: {labels['is_special'].sum()}")
    print(f"  - Node tokens: {(~labels['is_special']).sum()}")
    
    # Create visualizations
    visualize_all_embeddings(
        embeddings, 
        labels, 
        save_dir=args.save_dir,
        prefix=args.prefix
    )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

