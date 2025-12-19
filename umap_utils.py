"""
UMAP utility functions for dimensionality reduction and visualization.
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple


def apply_umap(
    data: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: Optional[int] = 42,
    columns_are_samples: bool = False,
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction to a matrix of vectors.
    
    Args:
        data: Input matrix. By default, rows are samples and columns are features.
              If columns_are_samples=True, will transpose so columns become samples.
        n_components: Number of output dimensions (default: 2 for visualization).
        n_neighbors: Number of neighbors for UMAP (controls local vs global structure).
                     Larger values = more global structure preserved.
        min_dist: Minimum distance between points in low-dim space.
                  Smaller values = tighter clusters.
        metric: Distance metric ('euclidean', 'cosine', 'manhattan', etc.).
        random_state: Random seed for reproducibility.
        columns_are_samples: If True, treat columns as samples (will transpose input).
    
    Returns:
        np.ndarray: Reduced data with shape (n_samples, n_components).
    
    Example:
        >>> embeddings = np.random.randn(100, 64)  # 100 samples, 64 features
        >>> reduced = apply_umap(embeddings, n_components=2)
        >>> reduced.shape
        (100, 2)
        
        >>> # If your samples are columns:
        >>> data = np.random.randn(64, 100)  # 64 features, 100 samples as columns
        >>> reduced = apply_umap(data, columns_are_samples=True)
        >>> reduced.shape
        (100, 2)
    """
    if columns_are_samples:
        data = data.T  # Transpose so rows become samples
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    
    return reducer.fit_transform(data)


def plot_umap(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Projection",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    s: float = 10,
    **umap_kwargs,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Apply UMAP and create a scatter plot visualization.
    
    Args:
        data: Input matrix (rows are samples, columns are features).
        labels: Optional labels for coloring points. Can be numeric or categorical.
        title: Plot title.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.
        cmap: Colormap for points (used when labels are provided).
        alpha: Point transparency.
        s: Point size.
        **umap_kwargs: Additional arguments passed to apply_umap().
    
    Returns:
        Tuple of (figure, axes, reduced_data).
    
    Example:
        >>> embeddings = np.random.randn(100, 64)
        >>> labels = np.random.randint(0, 5, 100)  # 5 classes
        >>> fig, ax, coords = plot_umap(embeddings, labels=labels, save_path="umap.png")
    """
    # Apply UMAP
    reduced = apply_umap(data, n_components=2, **umap_kwargs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, cmap=cmap, alpha=alpha, s=s
        )
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=alpha, s=s)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved UMAP plot to: {save_path}")
    
    return fig, ax, reduced


def umap_with_annotations(
    data: np.ndarray,
    annotations: list,
    title: str = "UMAP with Annotations",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    annotate_all: bool = False,
    annotate_indices: Optional[list] = None,
    fontsize: int = 8,
    **umap_kwargs,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Apply UMAP and annotate specific points with text labels.
    
    Args:
        data: Input matrix (rows are samples, columns are features).
        annotations: List of text labels for each point.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.
        annotate_all: If True, annotate all points (can be cluttered).
        annotate_indices: List of specific indices to annotate. If None and 
                          annotate_all=False, no annotations are shown.
        fontsize: Font size for annotations.
        **umap_kwargs: Additional arguments passed to apply_umap().
    
    Returns:
        Tuple of (figure, axes, reduced_data).
    """
    reduced = apply_umap(data, n_components=2, **umap_kwargs)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=20)
    
    # Add annotations
    indices_to_annotate = []
    if annotate_all:
        indices_to_annotate = list(range(len(annotations)))
    elif annotate_indices is not None:
        indices_to_annotate = annotate_indices
    
    for i in indices_to_annotate:
        ax.annotate(
            str(annotations[i]),
            (reduced[i, 0], reduced[i, 1]),
            fontsize=fontsize,
            alpha=0.8,
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated UMAP plot to: {save_path}")
    
    return fig, ax, reduced


if __name__ == "__main__":
    # Demo usage
    print("UMAP Utils Demo")
    print("=" * 40)
    
    # Generate sample data: 3 clusters
    np.random.seed(42)
    n_per_cluster = 50
    dim = 64
    
    cluster1 = np.random.randn(n_per_cluster, dim) + np.array([5] * dim)
    cluster2 = np.random.randn(n_per_cluster, dim) + np.array([-5] * dim)
    cluster3 = np.random.randn(n_per_cluster, dim) + np.array([0] * dim)
    
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    
    print(f"Input shape: {data.shape}")
    
    # Basic UMAP
    reduced = apply_umap(data)
    print(f"Reduced shape: {reduced.shape}")
    
    # Plot with labels
    fig, ax, coords = plot_umap(
        data, 
        labels=labels,
        title="Demo: 3 Clusters in UMAP Space",
        save_path="out/umap_demo.png"
    )
    
    print("\nDone! Check out/umap_demo.png")
