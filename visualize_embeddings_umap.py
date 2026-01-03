"""
Visualize token embeddings from a trained GPT model checkpoint using UMAP.

This script loads a checkpoint (which now includes metadata) and creates multiple UMAP visualizations.
All UMAP projections use n_neighbors=15 and are generated in BOTH 2D and 3D.

Standard visualizations:
1. 3D UMAP visualization
2. Node embeddings colored by similarity to root (2D and 3D)
3. Comprehensive summary figure (4 panels)

Path-based visualizations (if paths_by_leaf is in checkpoint metadata):
4. Sample of training paths with distinct colors and arrows (2D and 3D)
5. Train vs holdout path visualization (2D and 3D)
6. Depth in path structure visualization (2D and 3D)
7. Leaf similarity comparison: within-path (5 paths) vs cross-path average
8. Position-distance similarity: similarity decay from each position (within-path vs cross-path)

Features:
- By default, root vertex and special tokens are EXCLUDED from UMAP
- Use --include-root to include the root vertex
- Use --include-special to include special tokens
- All UMAP visualizations generated with n_neighbors=15
- Both 2D and 3D versions created for all UMAP plots

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
from umap_utils import apply_umap, create_embedding_gif
import matplotlib.pyplot as plt


def plot_embeddings_2d_with_paths(embeddings, meta, save_path=None, epoch=None, iteration=None,
                                   include_root=True, include_special=False, num_paths=5,
                                   figsize=(12, 10), reference_reducer=None):
    """
    Create a simple 2D visualization of embeddings with sampled paths highlighted.
    
    This is a refactored utility for quick visualization during training.
    
    Supports "anchored UMAP" for smooth animations:
    - On first call (reference_reducer=None): fits UMAP and returns (fig, reducer)
    - On subsequent calls: uses reference_reducer for consistent coordinate system
    
    Args:
        embeddings: numpy array of shape (vocab_size, n_embd)
        meta: Metadata dictionary containing paths_by_leaf, special_tokens, etc.
        save_path: Path to save the plot (if None, returns figure without saving)
        epoch: Optional epoch number to display in title
        iteration: Optional iteration number to display in title
        include_root: Whether to include root vertex in UMAP (default: False)
        include_special: Whether to include special tokens in UMAP (default: False)
        num_paths: Number of paths to highlight (default: 5)
        figsize: Figure size as (width, height) in inches (default: (12, 10))
        reference_reducer: Optional pre-fitted UMAP reducer for anchored projections.
                          If provided, all embeddings will be projected into the same space.
        
    Returns:
        If reference_reducer is None: Tuple of (fig, fitted_reducer)
        If reference_reducer is provided: fig only
    """
    
    vocab_size, n_embd = embeddings.shape
    
    # Check if we can use raw embeddings (skip UMAP when dimension matches)
    use_raw_2d = (n_embd == 2)
    
    # Get metadata
    paths_by_leaf = meta.get('paths_by_leaf', {})
    train_leaves = meta.get('train_leaves', set())
    root_vertex = meta.get('root_vertex', None)
    special_tokens = meta.get('special_tokens', {})
    
    if not paths_by_leaf:
        raise ValueError("paths_by_leaf not found in metadata")
    
    # Create filtering mask
    is_special = np.zeros(vocab_size, dtype=bool)
    for token_id in special_tokens.values():
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            is_special[token_id] = True
    
    umap_mask = np.ones(vocab_size, dtype=bool)
    if not include_special:
        umap_mask &= ~is_special
    if not include_root and root_vertex is not None:
        umap_mask[root_vertex] = False
    
    filtered_embeddings = embeddings[umap_mask]
    filtered_indices = np.where(umap_mask)[0]
    
    # Compute 2D projection
    fitted_reducer = None
    if use_raw_2d:
        reduced_2d = filtered_embeddings
        viz_method = "Raw Embeddings"
    else:
        n_neighbors_val = min(15, filtered_embeddings.shape[0] - 1)
        
        if reference_reducer is not None:
            # Use anchored UMAP for consistent coordinate system
            reduced_2d = apply_umap(
                filtered_embeddings,
                n_components=2,
                n_neighbors=n_neighbors_val,
                min_dist=0.1,
                random_state=42,
                reference_reducer=reference_reducer
            )
        else:
            # Fit new UMAP and return reducer for future anchoring
            reduced_2d, fitted_reducer = apply_umap(
                filtered_embeddings,
                n_components=2,
                n_neighbors=n_neighbors_val,
                min_dist=0.1,
                random_state=42
            )
        viz_method = "UMAP"
    
    # Create token position mapping
    token_to_pos = {token_id: i for i, token_id in enumerate(filtered_indices)}
    
    # Sample paths consistently
    train_path_leaves = [leaf for leaf in paths_by_leaf.keys()]
    num_paths_to_show = min(num_paths, len(train_path_leaves))
    
    np.random.seed(42)
    sampled_leaves = []
    if num_paths_to_show > 0:
        sampled_leaves = np.random.choice(train_path_leaves, size=num_paths_to_show, replace=False).tolist()
    
    # Define colors
    BRIGHT_COLORS = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCB77', '#C77DFF']
    path_color_map = {leaf: BRIGHT_COLORS[i % len(BRIGHT_COLORS)] for i, leaf in enumerate(sampled_leaves)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background nodes
    background_tokens = [t for t in filtered_indices if not is_special[t]]
    if root_vertex is not None and not include_root and root_vertex in background_tokens:
        background_tokens.remove(root_vertex)
    
    if background_tokens:
        background_positions = [token_to_pos[t] for t in background_tokens if t in token_to_pos]
        if background_positions:
            ax.scatter(
                reduced_2d[background_positions, 0],
                reduced_2d[background_positions, 1],
                c='lightgray', alpha=0.2, s=10, label='Other nodes', zorder=1
            )
    
    # Plot sampled paths with arrows
    for path_idx, leaf_token in enumerate(sampled_leaves):
        path_tokens = paths_by_leaf[leaf_token]
        color = path_color_map[leaf_token]
        
        visible_path_tokens = [t for t in path_tokens if t in token_to_pos]
        
        if len(visible_path_tokens) > 0:
            path_positions = [token_to_pos[t] for t in visible_path_tokens]
            
            # Plot nodes
            ax.scatter(
                reduced_2d[path_positions, 0],
                reduced_2d[path_positions, 1],
                c=color, alpha=0.8, s=50,
                label=f'Path {path_idx+1}',
                edgecolors='black', linewidths=0.5,
                zorder=3
            )
            
            # Draw arrows
            for i in range(len(visible_path_tokens) - 1):
                start_pos = token_to_pos[visible_path_tokens[i]]
                end_pos = token_to_pos[visible_path_tokens[i + 1]]
                
                ax.annotate('', 
                           xy=(reduced_2d[end_pos, 0], reduced_2d[end_pos, 1]),
                           xytext=(reduced_2d[start_pos, 0], reduced_2d[start_pos, 1]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6),
                           zorder=2)
    
    # Highlight root if included
    if include_root and root_vertex is not None and root_vertex in token_to_pos:
        root_pos = token_to_pos[root_vertex]
        ax.scatter(
            reduced_2d[root_pos, 0],
            reduced_2d[root_pos, 1],
            c='black', s=300, marker='X',
            label='Root',
            edgecolors='white', linewidths=2,
            zorder=10
        )
    
    # Set labels and title
    ax.set_xlabel(f'{viz_method} 1', fontsize=12)
    ax.set_ylabel(f'{viz_method} 2', fontsize=12)
    
    # Build title with epoch/iteration info
    title = f'Token Embeddings (2D {viz_method})'
    if epoch is not None or iteration is not None:
        title += '\n'
        if epoch is not None:
            title += f'Epoch {epoch}'
        if iteration is not None:
            if epoch is not None:
                title += f', '
            title += f'Iter {iteration}'
    
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding plot: {save_path}")
    
    # Return fitted reducer if this was the first call (for anchored UMAP)
    if fitted_reducer is not None:
        return fig, fitted_reducer
    else:
        return fig


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
    
    # Remove neighborhood register tensors (these are computed/registered separately)
    neighborhood_keys = ['neighborhood_tensor', 'neighborhood_sizes_tensor', 'inv_neighborhood_sizes_tensor']
    for k in neighborhood_keys:
        if k in state_dict:
            state_dict.pop(k)
            print(f"  Removed '{k}' from state dict (not a model parameter)")
    
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
                           include_root=False, include_special=False):
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
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vocab_size, n_embd = embeddings.shape
    
    # Check if we can use raw embeddings (skip UMAP when dimension matches)
    use_raw_2d = (n_embd == 2)
    use_raw_3d = (n_embd == 3)
    
    if use_raw_2d:
        print(f"\nEmbeddings are 2D - using raw embeddings for 2D visualizations")
        print(f"Creating visualizations for {vocab_size} tokens...")
    elif use_raw_3d:
        print(f"\nEmbeddings are 3D - using raw embeddings for 3D visualizations")
        print(f"Creating visualizations for {vocab_size} tokens...")
    else:
        print(f"\nCreating UMAP visualizations for {vocab_size} tokens...")
    
    print(f"  Include root: {include_root}")
    print(f"  Include special: {include_special}")
    
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
    
    print(f"  Visualization will use {filtered_embeddings.shape[0]} tokens (filtered from {vocab_size})")
    
    if filtered_embeddings.shape[0] == 0:
        print("  Warning: No tokens to visualize after filtering!")
        return None
    
    # Compute embeddings for visualization
    n_neighbors_val = min(15, filtered_embeddings.shape[0] - 1)
    
    # 2D visualization
    if use_raw_2d:
        print("  [1/3] Using raw 2D embeddings...")
        reduced_2d = filtered_embeddings  # Already 2D
        viz_method_2d = "Raw Embeddings"
    else:
        print("  [1/3] Computing 2D UMAP (n_neighbors=15)...")
        reduced_2d, _ = apply_umap(
            filtered_embeddings,
            n_components=2,
            n_neighbors=n_neighbors_val,
            min_dist=0.1,
            random_state=42
        )
        viz_method_2d = "UMAP"
    
    # 3D visualization
    if use_raw_2d:
        # Skip 3D for 2D embeddings
        reduced_3d = None
        viz_method_3d = None
    elif use_raw_3d:
        print("  [2/3] Using raw 3D embeddings...")
        reduced_3d = filtered_embeddings  # Already 3D
        viz_method_3d = "Raw Embeddings"
    else:
        print("  [2/3] Computing 3D UMAP (n_neighbors=15)...")
        reduced_3d, _ = apply_umap(
            filtered_embeddings,
            n_components=3,
            n_neighbors=n_neighbors_val,
            min_dist=0.1,
            random_state=42
        )
        viz_method_3d = "UMAP"
    
    # Only create 3D plot if we have 3D data
    if reduced_3d is not None:
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
                c='steelblue', alpha=0.4, s=10, label='Node tokens'
            )
        
        # Plot special tokens if included
        if include_special and filtered_is_special.any():
            ax.scatter(
                reduced_3d[filtered_is_special, 0],
                reduced_3d[filtered_is_special, 1],
                reduced_3d[filtered_is_special, 2],
                c='red', alpha=1.0, s=100, marker='*', label='Special tokens'
            )
        
            ax.set_xlabel(f'{viz_method_3d} 1')
            ax.set_ylabel(f'{viz_method_3d} 2')
            ax.set_zlabel(f'{viz_method_3d} 3')
            title = f'Token Embeddings (3D {viz_method_3d})'
            if not include_root:
                title += ' [Root Excluded]'
            if not include_special:
                title += ' [Special Excluded]'
            ax.set_title(title, fontsize=14)
            ax.legend()
            
            path3 = os.path.join(save_dir, '3d.png')
            fig.savefig(path3, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {path3}")
    else:
        print(f"    Skipped 3D plot (embeddings are 2D)")
    
    # 3. Similarity to root visualizations (both 2D and 3D)
    print("  [3/3] Creating similarity to root visualizations...")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    if root_vertex is not None:
        # Compute cosine similarity to root
        root_emb = normalized[root_vertex:root_vertex+1]
        similarities = np.dot(normalized, root_emb.T).flatten()
        
        # Get similarities for filtered tokens
        filtered_sims = similarities[umap_mask]
        filtered_is_node_2d = ~labels['is_special'][umap_mask]
        
        # === 2D Similarity to Root ===
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if filtered_is_node_2d.any():
            scatter = ax.scatter(
                reduced_2d[filtered_is_node_2d, 0],
                reduced_2d[filtered_is_node_2d, 1],
                c=filtered_sims[filtered_is_node_2d],
                cmap='RdYlGn',
                alpha=0.7, s=20
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
        
        ax.set_xlabel(f'{viz_method_2d} 1', fontsize=12)
        ax.set_ylabel(f'{viz_method_2d} 2', fontsize=12)
        title = f'Node Embeddings Colored by Similarity to Root (2D {viz_method_2d})'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        path4_2d = os.path.join(save_dir, 'similarity_to_root_2d.png')
        fig.savefig(path4_2d, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path4_2d}")
        
        # === 3D Similarity to Root ===
        if reduced_3d is not None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if filtered_is_node_2d.any():
                scatter = ax.scatter(
                    reduced_3d[filtered_is_node_2d, 0],
                    reduced_3d[filtered_is_node_2d, 1],
                    reduced_3d[filtered_is_node_2d, 2],
                    c=filtered_sims[filtered_is_node_2d],
                    cmap='RdYlGn',
                    alpha=0.7, s=20
                )
                fig.colorbar(scatter, ax=ax, label='Cosine Similarity to Root', pad=0.1)
            
            # Mark root if included
            if include_root and root_vertex in filtered_indices:
                root_pos_in_filtered = np.where(filtered_indices == root_vertex)[0][0]
                ax.scatter(
                    reduced_3d[root_pos_in_filtered, 0],
                    reduced_3d[root_pos_in_filtered, 1],
                    reduced_3d[root_pos_in_filtered, 2],
                    c='black', s=200, marker='X', label='Root', zorder=10
                )
            
            ax.set_xlabel(f'{viz_method_3d} 1')
            ax.set_ylabel(f'{viz_method_3d} 2')
            ax.set_zlabel(f'{viz_method_3d} 3')
            title = f'Node Embeddings Colored by Similarity to Root (3D {viz_method_3d})'
            if not include_root:
                title += ' [Root Excluded]'
            ax.set_title(title, fontsize=14)
            ax.legend()
            
            path4_3d = os.path.join(save_dir, 'similarity_to_root_3d.png')
            fig.savefig(path4_3d, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {path4_3d}")
        else:
            print(f"    Skipped 3D similarity plot (embeddings are 2D)")
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
        
        # Sample up to 20 training paths with consistent seed
        train_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in train_leaves]
        num_paths_to_sample = min(20, len(train_path_leaves))
        
        if num_paths_to_sample > 0:
            # Use consistent sampling with seed
            np.random.seed(42)
            sampled_leaves = np.random.choice(train_path_leaves, size=num_paths_to_sample, replace=False)
            
            # Define consistent color palette
            BRIGHT_COLORS = [
                '#FF6B6B',  # Bright Red
                '#4ECDC4',  # Bright Teal
                '#FFD93D',  # Bright Yellow
                '#6BCB77',  # Bright Green
                '#C77DFF',  # Bright Purple
            ]
            
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
                
                # Plot path with consistent color
                x_vals = list(range(len(path_tokens)))  # distance: 0=root, l-1=leaf
                y_vals = [path_idx] * len(path_tokens)  # path index
                z_vals = path_sims  # cosine similarity to leaf
                color = BRIGHT_COLORS[path_idx % len(BRIGHT_COLORS)]
                
                ax.plot(x_vals, y_vals, z_vals, 'o-', alpha=0.6, markersize=4, color=color)
            
            ax.set_xlabel('Distance in Path\n(0=Root, L-1=Leaf)', fontsize=10)
            ax.set_ylabel('Path Index', fontsize=10)
            ax.set_zlabel('Cosine Similarity to Leaf', fontsize=10)
            ax.set_title(f'Path Structure: Similarity to Leaf\n({num_paths_to_sample} sampled paths)', fontsize=12)
    else:
        ax.text(0.5, 0.5, 0.5, 'Path data not available', ha='center', va='center')
        ax.set_title('Path Structure', fontsize=12)
    
    plt.suptitle('Token Embedding Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path_summary = os.path.join(save_dir, 'summary.png')
    fig.savefig(path_summary, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary: {path_summary}")
    
    return reduced_2d, filtered_indices, umap_mask


def visualize_paths_in_umap(embeddings, labels, meta, save_dir='out', prefix='embedding_umap',
                          include_root=False, include_special=False):
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
    
    vocab_size, n_embd = embeddings.shape
    
    # ============================================================
    # CONSISTENT COLOR MAPPING: Sample paths once and assign colors
    # ============================================================
    # Define consistent color palette (5 bright distinguishable colors)
    BRIGHT_COLORS = [
        '#FF6B6B',  # Bright Red
        '#4ECDC4',  # Bright Teal
        '#FFD93D',  # Bright Yellow
        '#6BCB77',  # Bright Green
        '#C77DFF',  # Bright Purple
    ]
    
    # Separate train and holdout paths
    train_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in train_leaves]
    holdout_path_leaves = [leaf for leaf in paths_by_leaf.keys() if leaf in holdout_leaves]
    all_leaves = train_path_leaves + holdout_path_leaves
    
    # Sample paths consistently with a fixed seed
    max_paths_to_show = 20
    num_paths_for_viz = min(5, len(all_leaves))  # For detailed visualizations
    num_paths_to_show = min(max_paths_to_show, len(all_leaves))  # For sampled paths viz
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Sample paths for detailed analysis (5 paths)
    sampled_leaves_5 = []
    if num_paths_for_viz > 0:
        sampled_leaves_5 = np.random.choice(all_leaves, size=num_paths_for_viz, replace=False).tolist()
    
    # Sample paths for the larger visualization (up to 20 paths)
    sampled_leaves_20 = []
    if num_paths_to_show > 0:
        # Reuse the same seed to ensure the first 5 are the same
        np.random.seed(42)
        sampled_leaves_20 = np.random.choice(all_leaves, size=num_paths_to_show, replace=False).tolist()
    
    # Create color mapping: leaf_id -> color
    # The first 5 paths get distinct colors, rest cycle through the same colors
    path_color_map = {}
    for idx, leaf in enumerate(sampled_leaves_20):
        path_color_map[leaf] = BRIGHT_COLORS[idx % len(BRIGHT_COLORS)]
    
    print(f"  Sampled {num_paths_for_viz} paths for detailed analysis: {sampled_leaves_5}")
    print(f"  Sampled {num_paths_to_show} paths for distinct path visualization")
    print(f"  Color mapping established for consistent visualization")
    # ============================================================
    
    # Check if we can use raw embeddings (skip UMAP when dimension matches)
    use_raw_2d = (n_embd == 2)
    use_raw_3d = (n_embd == 3)
    
    print("\n=== Path-Based Visualization ===")
    if use_raw_2d:
        print(f"  Using raw 2D embeddings (no UMAP needed)")
    elif use_raw_3d:
        print(f"  Using raw 3D embeddings (no UMAP needed)")
    print(f"  Train leaves: {len(train_leaves)}")
    print(f"  Holdout leaves: {len(holdout_leaves)}")
    print(f"  Total paths: {len(paths_by_leaf)}")
    print(f"  Root vertex: {root_vertex}")
    print(f"  Include root: {include_root}")
    print(f"  Include special: {include_special}")
    
    os.makedirs(save_dir, exist_ok=True)
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
    
    print(f"  Visualization will use {filtered_embeddings.shape[0]} tokens (filtered from {vocab_size})")
    
    if filtered_embeddings.shape[0] == 0:
        print("  Warning: No tokens to visualize after filtering!")
        return
    
    # Compute embeddings for visualization
    n_neighbors_val = min(15, filtered_embeddings.shape[0] - 1)
    
    # 2D visualization
    if use_raw_2d:
        print("  Using raw 2D embeddings...")
        reduced_2d = filtered_embeddings  # Already 2D
        viz_method_2d = "Raw Embeddings"
    else:
        print("  Computing 2D UMAP projection (n_neighbors=15)...")
        reduced_2d, _ = apply_umap(
            filtered_embeddings, 
            n_components=2, 
            n_neighbors=n_neighbors_val, 
            min_dist=0.1,
            random_state=42
        )
        viz_method_2d = "UMAP"
    
    # 3D visualization
    if use_raw_2d:
        # Skip 3D for 2D embeddings
        reduced_3d = None
        viz_method_3d = None
    elif use_raw_3d:
        print("  Using raw 3D embeddings...")
        reduced_3d = filtered_embeddings  # Already 3D
        viz_method_3d = "Raw Embeddings"
    else:
        print("  Computing 3D UMAP projection (n_neighbors=15)...")
        reduced_3d, _ = apply_umap(
            filtered_embeddings, 
            n_components=3, 
            n_neighbors=n_neighbors_val, 
            min_dist=0.1,
            random_state=42
        )
        viz_method_3d = "UMAP"
    
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
    
    print(f"  Paths in training: {len(train_path_leaves)}")
    print(f"  Paths in holdout: {len(holdout_path_leaves)}")
    
    # ============================================================
    # Visualization 1: Sample of paths with distinct colors and arrows
    # ============================================================
    if len(sampled_leaves_20) > 0:
        print(f"\n  [1/4] Creating visualization: Sample of training paths with arrows...")
        
        # === 2D Version ===
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot background: all filtered node tokens in gray
        background_mask = umap_mask & node_mask
        if root_vertex is not None and not include_root:
            background_mask[root_vertex] = False
        
        background_tokens = [t for t in filtered_indices if background_mask[t]]
        if background_tokens:
            background_positions = [token_to_umap_pos[t] for t in background_tokens]
            ax.scatter(
                reduced_2d[background_positions, 0],
                reduced_2d[background_positions, 1],
                c='lightgray', alpha=0.2, s=10, label='Other nodes', zorder=1
            )
        
        # Plot each path with a color (using consistent color mapping)
        for path_idx, leaf_token in enumerate(sampled_leaves_20):
            path_tokens = paths_by_leaf[leaf_token]
            color = path_color_map[leaf_token]
            
            # Filter path tokens to only those in UMAP
            visible_path_tokens = [t for t in path_tokens if t in token_to_umap_pos]
            
            if len(visible_path_tokens) > 0:
                # Get UMAP positions for this path
                path_positions = [token_to_umap_pos[t] for t in visible_path_tokens]
                
                # Plot nodes
                ax.scatter(
                    reduced_2d[path_positions, 0],
                    reduced_2d[path_positions, 1],
                    c=color, alpha=0.8, s=50, 
                    label=f'Path {path_idx+1} (leaf token {leaf_token})',
                    edgecolors='black', linewidths=0.5,
                    zorder=3
                )
                
                # Draw arrows between consecutive nodes
                for i in range(len(visible_path_tokens) - 1):
                    start_token = visible_path_tokens[i]
                    end_token = visible_path_tokens[i + 1]
                    
                    start_pos = token_to_umap_pos[start_token]
                    end_pos = token_to_umap_pos[end_token]
                    
                    x_start, y_start = reduced_2d[start_pos, 0], reduced_2d[start_pos, 1]
                    x_end, y_end = reduced_2d[end_pos, 0], reduced_2d[end_pos, 1]
                    
                    # Draw arrow
                    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                              arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6),
                              zorder=2)
        
        # Highlight root if included
        if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
            root_pos = token_to_umap_pos[root_vertex]
            ax.scatter(
                reduced_2d[root_pos, 0],
                reduced_2d[root_pos, 1],
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
                    reduced_2d[special_positions, 0],
                    reduced_2d[special_positions, 1],
                    c='red', alpha=0.6, s=100, marker='*',
                    label='Special tokens',
                    zorder=2
                )
        
        ax.set_xlabel(f'{viz_method_2d} 1', fontsize=12)
        ax.set_ylabel(f'{viz_method_2d} 2', fontsize=12)
        title = f'Sample of {num_paths_to_show} Training Paths with Arrows (2D {viz_method_2d})'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        
        path1_2d = os.path.join(save_dir, 'paths_distinct_2d.png')
        fig.savefig(path1_2d, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path1_2d}")
        
        # === 3D Version ===
        if reduced_3d is not None:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot background
            if background_tokens:
                background_positions = [token_to_umap_pos[t] for t in background_tokens]
                ax.scatter(
                    reduced_3d[background_positions, 0],
                    reduced_3d[background_positions, 1],
                    reduced_3d[background_positions, 2],
                    c='lightgray', alpha=0.2, s=10, label='Other nodes', zorder=1
                )
            
            # Plot each path
            for path_idx, leaf_token in enumerate(sampled_leaves_20):
                path_tokens = paths_by_leaf[leaf_token]
                color = path_color_map[leaf_token]
                
                visible_path_tokens = [t for t in path_tokens if t in token_to_umap_pos]
                
                if len(visible_path_tokens) > 0:
                    path_positions = [token_to_umap_pos[t] for t in visible_path_tokens]
                    
                    ax.scatter(
                        reduced_3d[path_positions, 0],
                        reduced_3d[path_positions, 1],
                        reduced_3d[path_positions, 2],
                        c=color, alpha=0.8, s=50, 
                        label=f'Path {path_idx+1} (leaf token {leaf_token})',
                        edgecolors='black', linewidths=0.5,
                        zorder=3
                    )
                    
                    # Draw 3D arrows/lines between consecutive nodes
                    for i in range(len(visible_path_tokens) - 1):
                        start_token = visible_path_tokens[i]
                        end_token = visible_path_tokens[i + 1]
                        
                        start_pos = token_to_umap_pos[start_token]
                        end_pos = token_to_umap_pos[end_token]
                        
                        xs = [reduced_3d[start_pos, 0], reduced_3d[end_pos, 0]]
                        ys = [reduced_3d[start_pos, 1], reduced_3d[end_pos, 1]]
                        zs = [reduced_3d[start_pos, 2], reduced_3d[end_pos, 2]]
                        
                        ax.plot(xs, ys, zs, color=color, lw=1.5, alpha=0.6, zorder=2)
            
            # Highlight root if included
            if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
                root_pos = token_to_umap_pos[root_vertex]
                ax.scatter(
                    reduced_3d[root_pos, 0],
                    reduced_3d[root_pos, 1],
                    reduced_3d[root_pos, 2],
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
                        reduced_3d[special_positions, 0],
                        reduced_3d[special_positions, 1],
                        reduced_3d[special_positions, 2],
                        c='red', alpha=0.6, s=100, marker='*',
                        label='Special tokens',
                        zorder=2
                    )
            
            ax.set_xlabel(f'{viz_method_3d} 1')
            ax.set_ylabel(f'{viz_method_3d} 2')
            ax.set_zlabel(f'{viz_method_3d} 3')
            title = f'Sample of {num_paths_to_show} Training Paths (3D {viz_method_3d})'
            if not include_root:
                title += ' [Root Excluded]'
            ax.set_title(title, fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            
            path1_3d = os.path.join(save_dir, 'paths_distinct_3d.png')
            fig.savefig(path1_3d, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {path1_3d}")
        else:
            print(f"    Skipped 3D paths plot (embeddings are 2D)")
    
    # ============================================================
    # Visualization 2: Train vs Holdout paths
    # ============================================================
    print(f"  [2/4] Creating visualization: Train vs Holdout paths...")
    
    # === 2D Version ===
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
    
    # Plot each category - 2D
    if other_tokens:
        positions = [token_to_umap_pos[t] for t in other_tokens]
        ax.scatter(
            reduced_2d[positions, 0],
            reduced_2d[positions, 1],
            c='lightgray', alpha=0.2, s=10,
            label=f'Other nodes ({len(other_tokens)})',
            zorder=1
        )
    
    if train_only_tokens:
        positions = [token_to_umap_pos[t] for t in train_only_tokens]
        ax.scatter(
            reduced_2d[positions, 0],
            reduced_2d[positions, 1],
            c='steelblue', alpha=0.6, s=40,
            label=f'Train-only paths ({len(train_only_tokens)})',
            zorder=3
        )
    
    if holdout_only_tokens:
        positions = [token_to_umap_pos[t] for t in holdout_only_tokens]
        ax.scatter(
            reduced_2d[positions, 0],
            reduced_2d[positions, 1],
            c='orange', alpha=0.6, s=40,
            label=f'Holdout-only paths ({len(holdout_only_tokens)})',
            zorder=3
        )
    
    if shared_tokens:
        positions = [token_to_umap_pos[t] for t in shared_tokens]
        ax.scatter(
            reduced_2d[positions, 0],
            reduced_2d[positions, 1],
            c='green', alpha=0.8, s=60,
            label=f'Shared nodes ({len(shared_tokens)})',
            edgecolors='black', linewidths=0.5,
            zorder=5
        )
    
    # Highlight root if included
    if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
        root_pos = token_to_umap_pos[root_vertex]
        ax.scatter(
            reduced_2d[root_pos, 0],
            reduced_2d[root_pos, 1],
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
                reduced_2d[positions, 0],
                reduced_2d[positions, 1],
                c='red', alpha=0.6, s=100, marker='*',
                label='Special tokens',
                zorder=2
            )
    
    ax.set_xlabel(f'{viz_method_2d} 1', fontsize=12)
    ax.set_ylabel(f'{viz_method_2d} 2', fontsize=12)
    title = f'Token Embeddings: Train vs Holdout Paths (2D {viz_method_2d})'
    if not include_root:
        title += ' [Root Excluded]'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path2_2d = os.path.join(save_dir, 'train_vs_holdout_2d.png')
    fig.savefig(path2_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path2_2d}")
    
    # === 3D Version ===
    if reduced_3d is not None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if other_tokens:
            positions = [token_to_umap_pos[t] for t in other_tokens]
            ax.scatter(
                reduced_3d[positions, 0],
                reduced_3d[positions, 1],
                reduced_3d[positions, 2],
                c='lightgray', alpha=0.2, s=10,
                label=f'Other nodes ({len(other_tokens)})',
                zorder=1
            )
        
        if train_only_tokens:
            positions = [token_to_umap_pos[t] for t in train_only_tokens]
            ax.scatter(
                reduced_3d[positions, 0],
                reduced_3d[positions, 1],
                reduced_3d[positions, 2],
                c='steelblue', alpha=0.6, s=40,
                label=f'Train-only paths ({len(train_only_tokens)})',
                zorder=3
            )
        
        if holdout_only_tokens:
            positions = [token_to_umap_pos[t] for t in holdout_only_tokens]
            ax.scatter(
                reduced_3d[positions, 0],
                reduced_3d[positions, 1],
                reduced_3d[positions, 2],
                c='orange', alpha=0.6, s=40,
                label=f'Holdout-only paths ({len(holdout_only_tokens)})',
                zorder=3
            )
        
        if shared_tokens:
            positions = [token_to_umap_pos[t] for t in shared_tokens]
            ax.scatter(
                reduced_3d[positions, 0],
                reduced_3d[positions, 1],
                reduced_3d[positions, 2],
                c='green', alpha=0.8, s=60,
                label=f'Shared nodes ({len(shared_tokens)})',
                edgecolors='black', linewidths=0.5,
                zorder=5
            )
        
        # Highlight root if included
        if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
            root_pos = token_to_umap_pos[root_vertex]
            ax.scatter(
                reduced_3d[root_pos, 0],
                reduced_3d[root_pos, 1],
                reduced_3d[root_pos, 2],
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
                    reduced_3d[positions, 0],
                    reduced_3d[positions, 1],
                    reduced_3d[positions, 2],
                    c='red', alpha=0.6, s=100, marker='*',
                    label='Special tokens',
                    zorder=2
                )
        
        ax.set_xlabel(f'{viz_method_3d} 1')
        ax.set_ylabel(f'{viz_method_3d} 2')
        ax.set_zlabel(f'{viz_method_3d} 3')
        title = f'Token Embeddings: Train vs Holdout Paths (3D {viz_method_3d})'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        
        path2_3d = os.path.join(save_dir, 'train_vs_holdout_3d.png')
        fig.savefig(path2_3d, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path2_3d}")
    else:
        print(f"    Skipped 3D train vs holdout plot (embeddings are 2D)")
    
    # ============================================================
    # Visualization 3: Depth in path structure
    # ============================================================
    print(f"  [3/4] Creating visualization: Depth in path structure...")
    
    # Define discrete colors for depth levels (up to 10 distinct colors)
    DEPTH_COLORS = [
        '#E63946',  # Red (depth 0 - root)
        '#F77F00',  # Orange
        '#FCBF49',  # Yellow
        '#06D6A0',  # Teal
        '#118AB2',  # Blue
        '#073B4C',  # Dark Blue
        '#9D4EDD',  # Purple
        '#FF006E',  # Hot Pink
        '#8338EC',  # Violet
        '#3A86FF',  # Light Blue
    ]
    
    # Define distinct marker shapes for depth levels (up to 10 distinct shapes)
    DEPTH_MARKERS = [
        'o',   # Circle (depth 0 - root)
        's',   # Square 4 sides
        '^',   # Triangle up 3 sides
        'D',   # Diamond  4 sides
        'p',   # Pentagon 5 sides
        'h',   # Hexagon 6 sides
        '*',   # Star
        'v',   # Triangle down
        'P',   # Plus (filled)
        'X',   # X (filled)
    ]
    
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
    
    # === 2D Version ===
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
            reduced_2d[positions, 0],
            reduced_2d[positions, 1],
            c='lightgray', alpha=0.2, s=10,
            label='Unknown depth',
            zorder=1
        )
    
    if has_depth_tokens:
        positions = [token_to_umap_pos[t] for t in has_depth_tokens]
        depths = [token_depths[t] for t in has_depth_tokens]
        max_depth = max(depths)
        
        # Check if depth exceeds our discrete color/marker palette
        if max_depth >= len(DEPTH_COLORS):
            # Fallback to continuous colormap with circles for depths > 10
            print(f"    Note: Max depth ({max_depth}) >= {len(DEPTH_COLORS)}, using continuous colormap with circles")
            scatter = ax.scatter(
                reduced_2d[positions, 0],
                reduced_2d[positions, 1],
                c=depths,
                cmap='viridis',
                marker='o',
                alpha=0.7, s=40,
                vmin=0, vmax=max_depth
            )
            plt.colorbar(scatter, ax=ax, label='Depth (Distance from Root)')
        else:
            # Use discrete colors and shapes for depths 0-9
            # Plot each depth level with its own distinct color and shape
            for depth_level in range(max_depth + 1):
                depth_mask = np.array([d == depth_level for d in depths])
                if depth_mask.any():
                    color = DEPTH_COLORS[depth_level]
                    marker = DEPTH_MARKERS[depth_level]
                    ax.scatter(
                        reduced_2d[np.array(positions)[depth_mask], 0],
                        reduced_2d[np.array(positions)[depth_mask], 1],
                        c=color,
                        marker=marker,
                        alpha=0.7, s=60,
                        label=f'Depth {depth_level}' + (' (root)' if depth_level == 0 else ''),
                        edgecolors='black', linewidths=0.3
                    )
    
    # Highlight root if included
    if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
        root_pos = token_to_umap_pos[root_vertex]
        ax.scatter(
            reduced_2d[root_pos, 0],
            reduced_2d[root_pos, 1],
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
                reduced_2d[positions, 0],
                reduced_2d[positions, 1],
                c='red', alpha=0.6, s=100, marker='*',
                label='Special tokens',
                zorder=5
            )
    
    ax.set_xlabel(f'{viz_method_2d} 1', fontsize=12)
    ax.set_ylabel(f'{viz_method_2d} 2', fontsize=12)
    title = f'Token Embeddings: Depth in Path Structure (2D {viz_method_2d})'
    if not include_root:
        title += ' [Root Excluded]'
    ax.set_title(title, fontsize=14)
    # Only show legend if using discrete colors/shapes (max_depth < 10)
    if has_depth_tokens and max(token_depths[t] for t in has_depth_tokens) < len(DEPTH_COLORS):
        ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    path3_2d = os.path.join(save_dir, 'depth_2d.png')
    fig.savefig(path3_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path3_2d}")
    
    # === 3D Version ===
    if reduced_3d is not None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if no_depth_tokens:
            positions = [token_to_umap_pos[t] for t in no_depth_tokens]
            ax.scatter(
                reduced_3d[positions, 0],
                reduced_3d[positions, 1],
                reduced_3d[positions, 2],
                c='lightgray', alpha=0.2, s=10,
                label='Unknown depth',
                zorder=1
            )
        
        if has_depth_tokens:
            positions = [token_to_umap_pos[t] for t in has_depth_tokens]
            depths = [token_depths[t] for t in has_depth_tokens]
            max_depth = max(depths)
            
            # Check if depth exceeds our discrete color/marker palette
            if max_depth >= len(DEPTH_COLORS):
                # Fallback to continuous colormap with circles for depths > 10
                print(f"    Note: Max depth ({max_depth}) >= {len(DEPTH_COLORS)}, using continuous colormap with circles")
                scatter = ax.scatter(
                    reduced_3d[positions, 0],
                    reduced_3d[positions, 1],
                    reduced_3d[positions, 2],
                    c=depths,
                    cmap='viridis',
                    marker='o',
                    alpha=0.7, s=40,
                    vmin=0, vmax=max_depth
                )
                fig.colorbar(scatter, ax=ax, label='Depth (Distance from Root)', pad=0.1)
            else:
                # Use discrete colors and shapes for depths 0-9
                # Plot each depth level with its own distinct color and shape
                for depth_level in range(max_depth + 1):
                    depth_mask = np.array([d == depth_level for d in depths])
                    if depth_mask.any():
                        color = DEPTH_COLORS[depth_level]
                        marker = DEPTH_MARKERS[depth_level]
                        ax.scatter(
                            reduced_3d[np.array(positions)[depth_mask], 0],
                            reduced_3d[np.array(positions)[depth_mask], 1],
                            reduced_3d[np.array(positions)[depth_mask], 2],
                            c=color,
                            marker=marker,
                            alpha=0.7, s=60,
                            label=f'Depth {depth_level}' + (' (root)' if depth_level == 0 else ''),
                            edgecolors='black', linewidths=0.3
                        )
        
        # Highlight root if included
        if include_root and root_vertex is not None and root_vertex in token_to_umap_pos:
            root_pos = token_to_umap_pos[root_vertex]
            ax.scatter(
                reduced_3d[root_pos, 0],
                reduced_3d[root_pos, 1],
                reduced_3d[root_pos, 2],
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
                    reduced_3d[positions, 0],
                    reduced_3d[positions, 1],
                    reduced_3d[positions, 2],
                    c='red', alpha=0.6, s=100, marker='*',
                    label='Special tokens',
                    zorder=5
                )
        
        ax.set_xlabel(f'{viz_method_3d} 1')
        ax.set_ylabel(f'{viz_method_3d} 2')
        ax.set_zlabel(f'{viz_method_3d} 3')
        title = f'Token Embeddings: Depth in Path Structure (3D {viz_method_3d})'
        if not include_root:
            title += ' [Root Excluded]'
        ax.set_title(title, fontsize=14)
        # Only show legend if using discrete colors/shapes (max_depth < 10)
        if has_depth_tokens and max(token_depths[t] for t in has_depth_tokens) < len(DEPTH_COLORS):
            ax.legend(fontsize=9, loc='best', ncol=2)
        
        path3_3d = os.path.join(save_dir, 'depth_3d.png')
        fig.savefig(path3_3d, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path3_3d}")
    else:
        print(f"    Skipped 3D depth plot (embeddings are 2D)")
    
    # ============================================================
    # Visualization 4: Leaf similarity within path vs across paths
    # ============================================================
    print(f"  [4/4] Creating visualization: Leaf similarity within vs across paths...")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    # Use the consistently sampled 5 training paths
    if len(sampled_leaves_5) > 0:
        
        # Store data for all selected paths
        all_selected_data = []
        all_cross_path_data = []
        
        # Get other paths (excluding all 5 selected)
        other_paths = [leaf for leaf in train_path_leaves if leaf not in sampled_leaves_5]
        
        # Group nodes by distance from root across all other paths
        distance_to_nodes = {}
        for leaf in other_paths:
            path = paths_by_leaf[leaf]
            for dist, token in enumerate(path):
                if dist not in distance_to_nodes:
                    distance_to_nodes[dist] = []
                distance_to_nodes[dist].append(token)
        
        for idx, selected_leaf in enumerate(sampled_leaves_5):
            selected_path = paths_by_leaf[selected_leaf]
            
            # Calculate distances for selected path
            selected_distances = list(range(len(selected_path)))
            
            # WITHIN-PATH: Calculate cosine similarities for selected path (leaf vs each predecessor)
            leaf_emb = normalized[selected_leaf:selected_leaf+1]
            selected_sims = []
            for token in selected_path:
                if token < len(normalized):
                    sim = np.dot(normalized[token:token+1], leaf_emb.T).flatten()[0]
                    selected_sims.append(sim)
                else:
                    selected_sims.append(0)
            
            all_selected_data.append({
                'leaf': selected_leaf,
                'distances': selected_distances,
                'similarities': selected_sims,
                'color': path_color_map[selected_leaf]
            })
            
            # CROSS-PATH: Calculate average similarity of THIS leaf to nodes from OTHER paths
            cross_distances = []
            cross_sims = []
            
            for dist in sorted(distance_to_nodes.keys()):
                # For this distance, compute average similarity between nodes at this distance
                # (from OTHER paths) and THIS selected leaf
                sims_at_dist = []
                for token in distance_to_nodes[dist]:
                    if token < len(normalized):
                        # Similarity of this token (from another path) to the selected leaf
                        sim = np.dot(normalized[token:token+1], leaf_emb.T).flatten()[0]
                        sims_at_dist.append(sim)
                
                if sims_at_dist:
                    cross_distances.append(dist)
                    cross_sims.append(np.mean(sims_at_dist))
            
            all_cross_path_data.append({
                'leaf': selected_leaf,
                'distances': cross_distances,
                'similarities': cross_sims,
                'color': path_color_map[selected_leaf]
            })
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: All 5 selected paths - WITHIN-PATH (leaf vs its own predecessors)
        for path_data in all_selected_data:
            ax1.plot(path_data['distances'], path_data['similarities'], 'o-', 
                    linewidth=2, markersize=6, alpha=0.7,
                    color=path_data['color'], 
                    label=f"Leaf {path_data['leaf']}")
        
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, 
                   label='Perfect similarity')
        ax1.set_xlabel('Distance from Root (within path)', fontsize=12)
        ax1.set_ylabel('Cosine Similarity to Own Leaf', fontsize=12)
        ax1.set_title(f'Within-Path Similarity\n(Each leaf to its own predecessors)', fontsize=13)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Calculate y-axis limits based on all selected paths
        all_sims = [sim for path_data in all_selected_data for sim in path_data['similarities']]
        ax1.set_ylim([min(0, min(all_sims) - 0.1), 1.05])
        
        # Annotate first path only to avoid clutter
        first_path_data = all_selected_data[0]
        if len(first_path_data['distances']) > 0:
            # Root annotation (at distance 0)
            ax1.text(0, -0.08, 'Root', ha='center', va='top',
                    fontsize=9, color='darkgreen', fontweight='bold',
                    transform=ax1.get_xaxis_transform())
            # Leaf annotation (at max distance)
            max_dist = max(first_path_data['distances'])
            ax1.text(max_dist, -0.08, 'Leaf\n(self)', ha='center', va='top',
                    fontsize=9, color='darkred', fontweight='bold',
                    transform=ax1.get_xaxis_transform())
        
        # Plot 2: All 5 selected paths - CROSS-PATH (each leaf to nodes from OTHER paths)
        for cross_data in all_cross_path_data:
            ax2.plot(cross_data['distances'], cross_data['similarities'], 'o-', 
                    linewidth=2, markersize=6, alpha=0.7,
                    color=cross_data['color'], 
                    label=f"Leaf {cross_data['leaf']} → other paths")
        
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, 
                   label='Perfect similarity')
        ax2.set_xlabel('Distance from Root (in other paths)', fontsize=12)
        ax2.set_ylabel('Avg Cosine Similarity to Selected Leaf', fontsize=12)
        ax2.set_title(f'Cross-Path Average Similarity\n(Each leaf to nodes from {len(other_paths)} other paths)', 
                     fontsize=13)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Calculate y-axis limits for cross-path
        all_cross_sims = [sim for cross_data in all_cross_path_data for sim in cross_data['similarities']]
        if all_cross_sims:
            ax2.set_ylim([min(0, min(all_cross_sims) - 0.1), 1.05])
        
        # Annotate special points
        if all_cross_path_data and len(all_cross_path_data[0]['distances']) > 0:
            ax2.text(0, -0.08, 'Root', ha='center', va='top',
                    fontsize=9, color='darkgreen', fontweight='bold',
                    transform=ax2.get_xaxis_transform())
            if len(all_cross_path_data[0]['distances']) > 1:
                max_dist_cross = max(all_cross_path_data[0]['distances'])
                ax2.text(max_dist_cross, -0.08, 'Leaves\n(other)', ha='center', va='top',
                        fontsize=9, color='darkred', fontweight='bold',
                        transform=ax2.get_xaxis_transform())
        
        plt.suptitle(f'Leaf-to-Node Similarity: Within-Path vs Cross-Path ({len(sampled_leaves_5)} selected paths)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path4 = os.path.join(save_dir, 'leaf_similarity_comparison.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path4}")
        
        # ============================================================
        # Additional visualization: Distance-based similarity from each position
        # ============================================================
        print(f"    [4b/4] Creating visualization: Position-based distance similarity...")
        
        # Select the first path from the 5 sampled paths for this analysis
        selected_leaf_A = sampled_leaves_5[0]
        selected_path_A = paths_by_leaf[selected_leaf_A]
        path_length = len(selected_path_A)
        
        # Create 2x2 subplot layout for plots 3 and 4
        fig, ((ax3, ax4)) = plt.subplots(1, 2, figsize=(18, 6))
        
        # PLOT 3: Within-path similarity (node at position i to nodes at distance x within same path)
        # For each position i in path A
        for i in range(path_length):
            node_i = selected_path_A[i]
            if node_i >= len(normalized):
                continue
            
            distances_from_i = []
            sims_from_i = []
            
            # For each possible distance x
            max_distance = path_length - 1
            for x in range(max_distance + 1):
                # Find all nodes at distance x from position i
                nodes_at_dist_x = []
                for j in range(path_length):
                    if abs(j - i) == x:
                        nodes_at_dist_x.append(selected_path_A[j])
                
                if nodes_at_dist_x:
                    # Calculate average similarity to these nodes
                    sims = []
                    node_i_emb = normalized[node_i:node_i+1]
                    for node_j in nodes_at_dist_x:
                        if node_j < len(normalized):
                            sim = np.dot(normalized[node_j:node_j+1], node_i_emb.T).flatten()[0]
                            sims.append(sim)
                    
                    if sims:
                        distances_from_i.append(x)
                        sims_from_i.append(np.mean(sims))
            
            # Plot line for this position i
            if distances_from_i:
                ax3.plot(distances_from_i, sims_from_i, 'o-', 
                        linewidth=1.5, markersize=4, alpha=0.7,
                        label=f'Pos {i}' + (' (root)' if i == 0 else ' (leaf)' if i == path_length-1 else ''))
        
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=0.0, color='gray', linestyle='-', alpha=0.2, linewidth=1)
        ax3.set_xlabel('Distance from Node', fontsize=12)
        ax3.set_ylabel('Avg Cosine Similarity', fontsize=12)
        ax3.set_title(f'Within-Path Similarity by Distance\n(Path to leaf {selected_leaf_A}, from each position)', fontsize=13)
        ax3.legend(fontsize=8, loc='best', ncol=2)
        ax3.grid(True, alpha=0.3)
        
        # PLOT 4: Cross-path similarity (node at position i to nodes at distance x in OTHER paths)
        # For each position i in path A
        for i in range(path_length):
            node_i = selected_path_A[i]
            if node_i >= len(normalized):
                continue
            
            distances_from_i = []
            sims_from_i = []
            
            # For each possible distance x
            max_distance = path_length - 1
            for x in range(max_distance + 1):
                # Find all nodes at positions (i-x) or (i+x) in OTHER paths
                nodes_at_dist_x = []
                for other_leaf in other_paths:
                    other_path = paths_by_leaf[other_leaf]
                    # Look at positions that are distance x away from i
                    for j in range(len(other_path)):
                        if abs(j - i) == x and j < len(other_path):
                            nodes_at_dist_x.append(other_path[j])
                
                if nodes_at_dist_x:
                    # Calculate average similarity to these nodes
                    sims = []
                    node_i_emb = normalized[node_i:node_i+1]
                    for node_j in nodes_at_dist_x:
                        if node_j < len(normalized):
                            sim = np.dot(normalized[node_j:node_j+1], node_i_emb.T).flatten()[0]
                            sims.append(sim)
                    
                    if sims:
                        distances_from_i.append(x)
                        sims_from_i.append(np.mean(sims))
            
            # Plot line for this position i
            if distances_from_i:
                ax4.plot(distances_from_i, sims_from_i, 'o-', 
                        linewidth=1.5, markersize=4, alpha=0.7,
                        label=f'Pos {i}' + (' (root)' if i == 0 else ' (leaf)' if i == path_length-1 else ''))
        
        ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax4.axhline(y=0.0, color='gray', linestyle='-', alpha=0.2, linewidth=1)
        ax4.set_xlabel('Distance from Node Position', fontsize=12)
        ax4.set_ylabel('Avg Cosine Similarity', fontsize=12)
        ax4.set_title(f'Cross-Path Similarity by Distance\n(Path to leaf {selected_leaf_A} nodes → other paths)', fontsize=13)
        ax4.legend(fontsize=8, loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Position-Distance Similarity Analysis (Path to leaf token {selected_leaf_A})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path4b = os.path.join(save_dir, 'position_distance_similarity.png')
        fig.savefig(path4b, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path4b}")
        
    else:
        print("    Skipping (no training paths available)")
    
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
    
    # Create standard visualizations
    visualize_all_embeddings(
        embeddings, 
        labels, 
        meta,
        save_dir=args.save_dir,
        prefix=args.prefix,
        include_root=args.include_root,
        include_special=args.include_special
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
            include_special=args.include_special
        )
    else:
        print("\nSkipping path-based visualizations: 'paths_by_leaf' not found in checkpoint metadata")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

