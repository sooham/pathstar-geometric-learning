"""
Visualize token embeddings from a trained GPT model checkpoint using UMAP.

This script loads a checkpoint (which now includes metadata) and creates multiple UMAP visualizations
with two different n_neighbors values (15 and 100) for comparison:

Standard visualizations:
1. 3D UMAP visualization
2. Node embeddings colored by similarity to root
3. Comprehensive summary figure (with embedding norms, cosine similarities, similarity by distance, and path structure)

Path-based visualizations (if paths_by_leaf is in checkpoint metadata):
4. Sample of training paths with distinct colors and arrows
5. Train vs holdout path visualization
6. Depth in path structure visualization

Features:
- By default, root vertex and special tokens are excluded from UMAP
- Use --include-root to include the root vertex
- Use --include-special to include special tokens
- All visualizations are generated for both n_neighbors=15 and n_neighbors=100
- Files are saved with suffixes _n15 and _n100

Usage:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt
    
With custom save directory:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt --save_dir visualizations/
    
Include root and special tokens:
    python visualize_embeddings_umap.py --checkpoint out/ckpt_xxx.pt --include-root --include-special
"""

import argparse
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import GPTConfig, GPT
from umap_utils import apply_umap


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


def visualize_all_embeddings(embeddings, labels, meta, save_dir='out', prefix='embedding_umap', 
                           include_root=False, include_special=False, n_neighbors=15):
    """
    Create comprehensive UMAP visualizations of embeddings.
    
    Args:
        embeddings: numpy array of shape (vocab_size, n_embd)
        labels: Dict with labeling information
        meta: Metadata dictionary
        save_dir: Directory to save plots
        prefix: Prefix for saved files
        include_root: Whether to include root vertex in UMAP (default: False)
        include_special: Whether to include special tokens in UMAP (default: False)
        n_neighbors: Number of neighbors for UMAP (default: 15)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vocab_size = embeddings.shape[0]
    print(f"\nCreating UMAP visualizations for {vocab_size} tokens (n_neighbors={n_neighbors})...")
    print(f"  Include root: {include_root}")
    print(f"  Include special: {include_special}")
    
    # Add suffix for filename based on n_neighbors
    file_suffix = f"_n{n_neighbors}"
    
    # Identify root vertex
    root_vertex = meta.get('root_vertex', None) if meta else None
    
    # Create filtering mask for UMAP
    node_mask = ~labels['is_special']
    special_mask = labels['is_special']
    
    # Filter tokens for UMAP
    umap_mask = np.ones(vocab_size, dtype=bool)
    if not include_special:
        umap_mask &= ~special_mask
    if not include_root and root_vertex is not None:
        umap_mask[root_vertex] = False
    
    filtered_embeddings = embeddings[umap_mask]
    filtered_indices = np.where(umap_mask)[0]
    
    print(f"  UMAP will use {filtered_embeddings.shape[0]} tokens (filtered from {vocab_size})")
    
    if filtered_embeddings.shape[0] == 0:
        print("  Warning: No tokens to visualize after filtering!")
        return None
    
    # Compute UMAP on filtered embeddings
    print("  [1/2] Computing 3D UMAP...")
    
    reduced_3d = apply_umap(
        filtered_embeddings,
        n_components=3,
        n_neighbors=min(n_neighbors, filtered_embeddings.shape[0] - 1),
        min_dist=0.1,
        random_state=42
    )
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine which filtered tokens are nodes vs special
    filtered_is_special = labels['is_special'][umap_mask]
    filtered_is_node = ~filtered_is_special
    
    # Plot node tokens
    if filtered_is_node.any():
        ax.scatter(
            reduced_3d[filtered_is_node, 0],
            reduced_3d[filtered_is_node, 1],
            reduced_3d[filtered_is_node, 2],
            c='steelblue', alpha=0.4, s=10, label='Node tokens',
            edgecolors='gray', linewidths=0.3
        )
    
    # Plot special tokens if included
    if include_special and filtered_is_special.any():
        ax.scatter(
            reduced_3d[filtered_is_special, 0],
            reduced_3d[filtered_is_special, 1],
            reduced_3d[filtered_is_special, 2],
            c='red', alpha=1.0, s=100, marker='*', label='Special tokens',
            edgecolors='gray', linewidths=0.3
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    title = 'Token Embeddings (3D UMAP)'
    if not include_root:
        title += ' [Root Excluded]'
    if not include_special:
        title += ' [Special Excluded]'
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    path3 = os.path.join(save_dir, f'{prefix}_3d{file_suffix}.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path3}")
    
    # 2. Compute 2D UMAP for similarity plot
    print("  [2/2] Computing 2D UMAP for similarity visualization...")
    
    reduced_2d = apply_umap(
        filtered_embeddings,
        n_components=2,
        n_neighbors=min(n_neighbors, filtered_embeddings.shape[0] - 1),
        min_dist=0.1,
        random_state=42
    )
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    if root_vertex is not None:
        # Compute cosine similarity to root
        root_emb = normalized[root_vertex:root_vertex+1]
        similarities = np.dot(normalized, root_emb.T).flatten()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get similarities for filtered tokens
        filtered_sims = similarities[umap_mask]
        filtered_is_node_2d = ~labels['is_special'][umap_mask]
        
        if filtered_is_node_2d.any():
            scatter = ax.scatter(
                reduced_2d[filtered_is_node_2d, 0],
                reduced_2d[filtered_is_node_2d, 1],
                c=filtered_sims[filtered_is_node_2d],
                cmap='RdYlGn',
                alpha=0.7, s=20,
                edgecolors='gray', linewidths=0.3
            )
            plt.colorbar(scatter, ax=ax, label='Cosine Similarity to Root')
        
        # Mark root if included
        if include_root and root_vertex in filtered_indices:
            root_pos_in_filtered = np.where(filtered_indices == root_vertex)[0][0]
            ax.scatter(
                reduced_2d[root_pos_in_filtered, 0],
                reduced_2d[root_pos_in_filtered, 1],
                c='black', s=200, marker='X', label='Root', zorder=10
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        title = 'Node Embeddings Colored by Similarity to Root'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        path4 = os.path.join(save_dir, f'{prefix}_similarity_to_root{file_suffix}.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path4}")
    else:
        print("    Skipping similarity plot (no root token found)")
        similarities = None
    
    # 3. Create enhanced summary figure with new plots
    print("\n  Creating enhanced summary figure...")
    
    fig = plt.figure(figsize=(16, 14))
    
    # Panel 1: Histogram of embedding norms
    ax = plt.subplot(2, 2, 1)
    node_norms = np.linalg.norm(embeddings[node_mask], axis=1)
    special_norms = np.linalg.norm(embeddings[special_mask], axis=1)
    ax.hist(node_norms, bins=50, alpha=0.7, label='Nodes', color='steelblue')
    if len(special_norms) > 0:
        ax.hist(special_norms, bins=10, alpha=0.7, label='Special', color='red')
    ax.set_xlabel('Embedding Norm')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Embedding Norms', fontsize=12)
    ax.legend()
    
    # Panel 2: Cosine similarity distribution (random pairs)
    ax = plt.subplot(2, 2, 2)
    node_indices_all = np.where(node_mask)[0]
    n_samples = min(1000, len(node_indices_all))
    if n_samples >= 2:
        sample_idx = np.random.choice(len(node_indices_all), n_samples, replace=False)
        node_normed = normalized[node_mask]
        
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
    
    # Panel 3: Cosine similarity to root by distance
    ax = plt.subplot(2, 2, 3)
    if root_vertex is not None and meta and 'paths_by_leaf' in meta:
        paths_by_leaf = meta['paths_by_leaf']
        
        # Calculate distance from root for each token
        token_distances = {}
        for leaf_token, path_tokens in paths_by_leaf.items():
            for depth, token in enumerate(path_tokens):
                if token not in token_distances:
                    token_distances[token] = []
                token_distances[token].append(depth)
        
        # Group similarities by distance
        distance_to_sims = {}
        for token, distances in token_distances.items():
            if token < len(similarities) and not labels['is_special'][token]:
                min_dist = min(distances)
                if min_dist not in distance_to_sims:
                    distance_to_sims[min_dist] = []
                distance_to_sims[min_dist].append(similarities[token])
        
        if distance_to_sims:
            distances = sorted(distance_to_sims.keys())
            mins = [min(distance_to_sims[d]) for d in distances]
            maxs = [max(distance_to_sims[d]) for d in distances]
            avgs = [np.mean(distance_to_sims[d]) for d in distances]
            
            ax.plot(distances, avgs, 'o-', label='Average', linewidth=2, markersize=6)
            ax.fill_between(distances, mins, maxs, alpha=0.3, label='Min-Max Range')
            ax.set_xlabel('Distance from Root', fontsize=11)
            ax.set_ylabel('Cosine Similarity to Root', fontsize=11)
            ax.set_title('Similarity to Root by Distance', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Path data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Similarity to Root by Distance', fontsize=12)
    
    # Panel 4: 3D plot of sampled paths (cosine similarity to leaf)
    ax = plt.subplot(2, 2, 4, projection='3d')
    if root_vertex is not None and meta and 'paths_by_leaf' in meta:
        paths_by_leaf = meta['paths_by_leaf']
        train_leaves = meta.get('train_leaves', set())
        
        # Sample up to 20 training paths
        train_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in train_leaves]
        num_paths_to_sample = min(20, len(train_path_leaves))
        
        if num_paths_to_sample > 0:
            sampled_leaves = np.random.choice(train_path_leaves, size=num_paths_to_sample, replace=False)
            
            for path_idx, leaf_token in enumerate(sampled_leaves):
                path_tokens = paths_by_leaf[leaf_token]
                leaf_emb = normalized[leaf_token:leaf_token+1]
                
                # Calculate cosine similarity to leaf for each token in path
                path_sims = []
                for token in path_tokens:
                    if token < len(normalized):
                        sim = np.dot(normalized[token:token+1], leaf_emb.T).flatten()[0]
                        path_sims.append(sim)
                    else:
                        path_sims.append(0)
                
                # Plot path
                x_vals = list(range(len(path_tokens)))  # distance: 0=root, l-1=leaf
                y_vals = [path_idx] * len(path_tokens)  # path index
                z_vals = path_sims  # cosine similarity to leaf
                
                ax.plot(x_vals, y_vals, z_vals, 'o-', alpha=0.6, markersize=4)
            
            ax.set_xlabel('Distance in Path\n(0=Root, L-1=Leaf)', fontsize=10)
            ax.set_ylabel('Path Index', fontsize=10)
            ax.set_zlabel('Cosine Similarity to Leaf', fontsize=10)
            ax.set_title(f'Path Structure: Similarity to Leaf\n({num_paths_to_sample} sampled paths)', fontsize=12)
    else:
        ax.text(0.5, 0.5, 0.5, 'Path data not available', ha='center', va='center')
        ax.set_title('Path Structure', fontsize=12)
    
    plt.suptitle('Token Embedding Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path_summary = os.path.join(save_dir, f'{prefix}_summary{file_suffix}.png')
    fig.savefig(path_summary, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary: {path_summary}")
    
    return reduced_2d, filtered_indices, umap_mask


def visualize_paths_in_umap(embeddings, labels, meta, save_dir='out', prefix='embedding_umap',
                          include_root=False, include_special=False, n_neighbors=15):
    """
    Visualize embeddings with each path colored distinctly using path information from metadata.
    
    Args:
        embeddings: numpy array of shape (vocab_size, n_embd)
        labels: Dict with labeling information
        meta: Metadata dictionary containing path information
        save_dir: Directory to save plots
        prefix: Prefix for saved files
        include_root: Whether to include root vertex in UMAP (default: False)
        include_special: Whether to include special tokens in UMAP (default: False)
        n_neighbors: Number of neighbors for UMAP (default: 15)
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
    
    print(f"\n=== Path-Based Visualization (n_neighbors={n_neighbors}) ===")
    print(f"  Train leaves: {len(train_leaves)}")
    print(f"  Holdout leaves: {len(holdout_leaves)}")
    print(f"  Total paths: {len(paths_by_leaf)}")
    print(f"  Root vertex: {root_vertex}")
    print(f"  Include root: {include_root}")
    print(f"  Include special: {include_special}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Add suffix for filename based on n_neighbors
    file_suffix = f"_n{n_neighbors}"
    
    vocab_size = embeddings.shape[0]
    node_mask = ~labels['is_special']
    special_mask = labels['is_special']
    
    # Create filtering mask for UMAP
    umap_mask = np.ones(vocab_size, dtype=bool)
    if not include_special:
        umap_mask &= ~special_mask
    if not include_root and root_vertex is not None:
        umap_mask[root_vertex] = False
    
    filtered_embeddings = embeddings[umap_mask]
    filtered_indices = np.where(umap_mask)[0]
    
    print(f"  UMAP will use {filtered_embeddings.shape[0]} tokens (filtered from {vocab_size})")
    
    if filtered_embeddings.shape[0] == 0:
        print("  Warning: No tokens to visualize after filtering!")
        return
    
    # Compute UMAP projection on filtered embeddings
    print("  Computing UMAP projection...")
    reduced_filtered = apply_umap(
        filtered_embeddings, 
        n_components=2, 
        n_neighbors=min(n_neighbors, filtered_embeddings.shape[0] - 1), 
        min_dist=0.1,
        random_state=42
    )
    
    # Create a mapping from token ID to its position in the filtered UMAP
    token_to_umap_pos = {}
    for i, token_id in enumerate(filtered_indices):
        token_to_umap_pos[token_id] = i
    
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
    # Visualization 1: Sample of paths with distinct colors and arrows
    # ============================================================
    max_paths_to_show = 20
    
    if len(train_path_leaves) > 0:
        print(f"\n  [1/3] Creating visualization: Sample of training paths with arrows...")
        
        # Sample a subset of paths to visualize
        num_paths_to_show = min(max_paths_to_show, len(train_path_leaves))
        sampled_leaves = np.random.choice(train_path_leaves, size=num_paths_to_show, replace=False)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use only 5 bright distinguishable colors and cycle through them
        bright_colors = [
            '#FF6B6B',  # Bright Red
            '#4ECDC4',  # Bright Teal
            '#FFD93D',  # Bright Yellow
            '#6BCB77',  # Bright Green
            '#C77DFF',  # Bright Purple
        ]
        
        # Plot background: all filtered node tokens in gray
        background_mask = umap_mask & node_mask
        if root_vertex is not None and not include_root:
            background_mask[root_vertex] = False
        
        background_tokens = [t for t in filtered_indices if background_mask[t]]
        if background_tokens:
            background_positions = [token_to_umap_pos[t] for t in background_tokens]
            ax.scatter(
                reduced_filtered[background_positions, 0],
                reduced_filtered[background_positions, 1],
                c='lightgray', alpha=0.2, s=10, label='Other nodes', zorder=1,
                edgecolors='gray', linewidths=0.3
            )
        
        # Plot each path with a color (cycling through 5 colors)
        for path_idx, leaf_token in enumerate(sampled_leaves):
            path_tokens = paths_by_leaf[leaf_token]
            color = bright_colors[path_idx % len(bright_colors)]
            
            # Filter path tokens to only those in UMAP
            visible_path_tokens = [t for t in path_tokens if t in token_to_umap_pos]
            
            if len(visible_path_tokens) > 0:
                # Get UMAP positions for this path
                path_positions = [token_to_umap_pos[t] for t in visible_path_tokens]
                
                # Plot nodes
                ax.scatter(
                    reduced_filtered[path_positions, 0],
                    reduced_filtered[path_positions, 1],
                    c=color, alpha=0.8, s=50, 
                    label=f'Path {path_idx+1} (→ leaf {leaf_token})',
                    edgecolors='gray', linewidths=0.3,
                    zorder=3
                )
                
                # Draw arrows between consecutive nodes
                for i in range(len(visible_path_tokens) - 1):
                    start_token = visible_path_tokens[i]
                    end_token = visible_path_tokens[i + 1]
                    
                    start_pos = token_to_umap_pos[start_token]
                    end_pos = token_to_umap_pos[end_token]
                    
                    x_start, y_start = reduced_filtered[start_pos, 0], reduced_filtered[start_pos, 1]
                    x_end, y_end = reduced_filtered[end_pos, 0], reduced_filtered[end_pos, 1]
                    
                    # Draw arrow
                    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                              arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6),
                              zorder=2)
        
        # Highlight root if included
        if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
            root_pos = token_to_umap_pos[root_vertex]
            ax.scatter(
                reduced_filtered[root_pos, 0],
                reduced_filtered[root_pos, 1],
                c='black', s=300, marker='X', 
                label=f'Root ({root_vertex})',
                edgecolors='white', linewidths=2,
                zorder=10
            )
        
        # Plot special tokens if included
        if include_special:
            special_in_filtered = [t for t in filtered_indices if labels['is_special'][t]]
            if special_in_filtered:
                special_positions = [token_to_umap_pos[t] for t in special_in_filtered]
                ax.scatter(
                    reduced_filtered[special_positions, 0],
                    reduced_filtered[special_positions, 1],
                    c='red', alpha=0.6, s=100, marker='*',
                    label='Special tokens',
                    zorder=2,
                    edgecolors='gray', linewidths=0.3
                )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        title = f'Sample of {num_paths_to_show} Training Paths with Arrows'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        
        path1 = os.path.join(save_dir, f'{prefix}_paths_distinct{file_suffix}.png')
        fig.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path1}")
    
    # ============================================================
    # Visualization 2: Train vs Holdout paths
    # ============================================================
    print(f"  [2/3] Creating visualization: Train vs Holdout paths...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create sets of train and holdout tokens
    train_path_tokens = set()
    for leaf in train_path_leaves:
        train_path_tokens.update(paths_by_leaf[leaf])
    
    holdout_path_tokens = set()
    for leaf in holdout_path_leaves:
        holdout_path_tokens.update(paths_by_leaf[leaf])
    
    # Categorize tokens that are in the filtered view
    train_only_tokens = []
    holdout_only_tokens = []
    shared_tokens = []
    other_tokens = []
    
    for token_id in filtered_indices:
        if labels['is_special'][token_id]:
            continue
        
        in_train = token_id in train_path_tokens
        in_holdout = token_id in holdout_path_tokens
        
        if in_train and in_holdout:
            shared_tokens.append(token_id)
        elif in_train:
            train_only_tokens.append(token_id)
        elif in_holdout:
            holdout_only_tokens.append(token_id)
        else:
            other_tokens.append(token_id)
    
    # Plot each category
    if other_tokens:
        positions = [token_to_umap_pos[t] for t in other_tokens]
        ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c='lightgray', alpha=0.2, s=10,
            label=f'Other nodes ({len(other_tokens)})',
            zorder=1,
            edgecolors='gray', linewidths=0.3
        )
    
    if train_only_tokens:
        positions = [token_to_umap_pos[t] for t in train_only_tokens]
        ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c='steelblue', alpha=0.6, s=40,
            label=f'Train-only paths ({len(train_only_tokens)})',
            zorder=3,
            edgecolors='gray', linewidths=0.3
        )
    
    if holdout_only_tokens:
        positions = [token_to_umap_pos[t] for t in holdout_only_tokens]
        ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c='orange', alpha=0.6, s=40,
            label=f'Holdout-only paths ({len(holdout_only_tokens)})',
            zorder=3,
            edgecolors='gray', linewidths=0.3
        )
    
    if shared_tokens:
        positions = [token_to_umap_pos[t] for t in shared_tokens]
        ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c='green', alpha=0.8, s=60,
            label=f'Shared nodes ({len(shared_tokens)})',
            edgecolors='gray', linewidths=0.3,
            zorder=5
        )
    
    # Highlight root if included
    if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
        root_pos = token_to_umap_pos[root_vertex]
        ax.scatter(
            reduced_filtered[root_pos, 0],
            reduced_filtered[root_pos, 1],
            c='black', s=300, marker='X',
            label=f'Root',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Plot special tokens if included
    if include_special:
        special_in_filtered = [t for t in filtered_indices if labels['is_special'][t]]
        if special_in_filtered:
            positions = [token_to_umap_pos[t] for t in special_in_filtered]
            ax.scatter(
                reduced_filtered[positions, 0],
                reduced_filtered[positions, 1],
                c='red', alpha=0.6, s=100, marker='*',
                label='Special tokens',
                zorder=2,
                edgecolors='gray', linewidths=0.3
            )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    title = 'Token Embeddings: Train vs Holdout Paths'
    if not include_root:
        title += ' [Root Excluded]'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path2 = os.path.join(save_dir, f'{prefix}_train_vs_holdout{file_suffix}.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path2}")
    
    # ============================================================
    # Visualization 3: Depth in path structure
    # ============================================================
    print(f"  [3/3] Creating visualization: Depth in path structure...")
    
    # Calculate depth for each token (distance from root along path)
    token_depths = np.full(vocab_size, -1, dtype=int)
    
    if root_vertex is not None:
        token_depths[root_vertex] = 0
        
        # For each path, assign depths
        for leaf_token, path_tokens in paths_by_leaf.items():
            for depth, token in enumerate(path_tokens):
                if 0 <= token < vocab_size:
                    # If token already has a depth assigned, take the minimum
                    if token_depths[token] == -1:
                        token_depths[token] = depth
                    else:
                        token_depths[token] = min(token_depths[token], depth)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Categorize filtered tokens by depth
    has_depth_tokens = []
    no_depth_tokens = []
    
    for token_id in filtered_indices:
        if labels['is_special'][token_id]:
            continue
        
        if token_depths[token_id] >= 0:
            has_depth_tokens.append(token_id)
        else:
            no_depth_tokens.append(token_id)
    
    if no_depth_tokens:
        positions = [token_to_umap_pos[t] for t in no_depth_tokens]
        ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c='lightgray', alpha=0.2, s=10,
            label='Unknown depth',
            zorder=1,
            edgecolors='gray', linewidths=0.3
        )
    
    if has_depth_tokens:
        positions = [token_to_umap_pos[t] for t in has_depth_tokens]
        depths = [token_depths[t] for t in has_depth_tokens]
        scatter = ax.scatter(
            reduced_filtered[positions, 0],
            reduced_filtered[positions, 1],
            c=depths,
            cmap='viridis',
            alpha=0.7, s=40,
            vmin=0, vmax=max(depths),
            edgecolors='gray', linewidths=0.3
        )
        plt.colorbar(scatter, ax=ax, label='Depth (Distance from Root)')
    
    # Highlight root if included
    if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
        root_pos = token_to_umap_pos[root_vertex]
        ax.scatter(
            reduced_filtered[root_pos, 0],
            reduced_filtered[root_pos, 1],
            c='black', s=300, marker='X',
            label='Root (depth=0)',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Plot special tokens if included
    if include_special:
        special_in_filtered = [t for t in filtered_indices if labels['is_special'][t]]
        if special_in_filtered:
            positions = [token_to_umap_pos[t] for t in special_in_filtered]
            ax.scatter(
                reduced_filtered[positions, 0],
                reduced_filtered[positions, 1],
                c='red', alpha=0.6, s=100, marker='*',
                label='Special tokens',
                zorder=5,
                edgecolors='gray', linewidths=0.3
            )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    title = 'Token Embeddings: Depth in Path Structure'
    if not include_root:
        title += ' [Root Excluded]'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path3 = os.path.join(save_dir, f'{prefix}_depth{file_suffix}.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path3}")
    
    # Print summary statistics
    print("\n  === Path Statistics ===")
    if root_vertex is not None:
        root_path_count = len([leaf for leaf, path in paths_by_leaf.items() if root_vertex in path])
        print(f"  Root appears in {root_path_count} paths")
    print(f"  Train-only tokens: {len(train_only_tokens)}")
    print(f"  Holdout-only tokens: {len(holdout_only_tokens)}")
    print(f"  Shared tokens: {len(shared_tokens)}")
    if has_depth_tokens:
        print(f"  Max depth: {max(token_depths[t] for t in has_depth_tokens)}")
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
    parser.add_argument('--include-root', action='store_true',
                        help='Include root vertex in UMAP visualizations (excluded by default)')
    parser.add_argument('--include-special', action='store_true',
                        help='Include special tokens in UMAP visualizations (excluded by default)')
    
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
    
    # Create visualizations for both n_neighbors values
    for n_neighbors in [15, 100]:
        print("\n" + "=" * 60)
        print(f"Generating visualizations with n_neighbors={n_neighbors}")
        print("=" * 60)
        
        # Create standard visualizations
        visualize_all_embeddings(
            embeddings, 
            labels, 
            meta,
            save_dir=args.save_dir,
            prefix=args.prefix,
            include_root=args.include_root,
            include_special=args.include_special,
            n_neighbors=n_neighbors
        )
        
        # Create path-based visualizations if metadata is available
        if meta and 'paths_by_leaf' in meta:
            visualize_paths_in_umap(
                embeddings,
                labels,
                meta,
                save_dir=args.save_dir,
                prefix=args.prefix,
                include_root=args.include_root,
                include_special=args.include_special,
                n_neighbors=n_neighbors
            )
        else:
            print("\nSkipping path-based visualizations: 'paths_by_leaf' not found in checkpoint metadata")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

