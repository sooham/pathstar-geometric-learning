"""
Debug script to calculate and verify theoretical minimum loss.
Usage: python debug_optimal_loss.py
"""
import math

def calculate_optimal_loss(d, l, holdout_percentage, use_undirected=True, 
                          interleave_dataset=True, balance_interleaved_datasets=True,
                          use_directional_tokens_in_path=False):
    """
    Calculate theoretical minimum loss for PathStar training.
    
    Args:
        d: number of spokes
        l: path length (number of nodes from root to leaf inclusive)
        holdout_percentage: fraction of paths held out for validation
        use_undirected: whether edges are undirected
        interleave_dataset: whether paths and edges are interleaved
        balance_interleaved_datasets: whether paths are upsampled to match edges
        use_directional_tokens_in_path: whether GT tokens are interleaved in path sequences
    """
    
    # Calculate dataset sizes
    num_graph_edges = d * (l - 1)  # Each spoke has (l-1) edges
    edges_size = (2 if use_undirected else 1) * num_graph_edges
    
    # Calculate paths after holdout
    num_holdout = math.ceil(d * holdout_percentage)
    paths_size = d - num_holdout
    
    print(f"=== Dataset Configuration ===")
    print(f"Graph spokes (d): {d}")
    print(f"Path length (l): {l}")
    print(f"Holdout percentage: {holdout_percentage}")
    print(f"Holdout paths: {num_holdout}")
    print(f"Training paths: {paths_size}")
    print(f"Total graph edges: {num_graph_edges}")
    print(f"Edge dataset size: {edges_size} ({'undirected' if use_undirected else 'directed'})")
    print()
    
    # Calculate effective samples
    if interleave_dataset and balance_interleaved_datasets and paths_size < edges_size:
        n_path_samples = edges_size  # Upsampled
        print(f"[Balancing] Paths upsampled from {paths_size} to {edges_size}")
    else:
        n_path_samples = paths_size
    
    n_edge_samples = edges_size
    
    # Calculate tokens per sample
    path_target_len = (2 * l - 1) if use_directional_tokens_in_path else l
    edge_tokens_per_sample = 1  # Only 1 token contributes to loss per edge
    
    # Calculate total tokens
    total_path_tokens = n_path_samples * path_target_len
    total_edge_tokens = n_edge_samples * edge_tokens_per_sample
    total_tokens = total_path_tokens + total_edge_tokens
    
    print(f"=== Token Counting ===")
    print(f"Path samples (after upsampling): {n_path_samples}")
    print(f"Path tokens per sample: {path_target_len}")
    print(f"Total path tokens: {total_path_tokens:,}")
    print(f"Edge samples: {n_edge_samples}")
    print(f"Edge tokens per sample: {edge_tokens_per_sample}")
    print(f"Total edge tokens: {total_edge_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print()
    
    # Calculate entropy mass
    # Only GT edges from root have entropy
    # For PathStar, root has exactly d children
    # When undirected, there are d edges with GT direction from root
    # Each has log(d) entropy (uniformly choosing among d children)
    num_gt_edges_from_root = d
    entropy_per_edge = math.log(d)  # Natural log
    entropy_mass = num_gt_edges_from_root * entropy_per_edge
    
    print(f"=== Entropy Calculation ===")
    print(f"GT edges from root: {num_gt_edges_from_root}")
    print(f"Entropy per GT edge from root: log({d}) = {entropy_per_edge:.6f}")
    print(f"Total entropy mass: {num_gt_edges_from_root} * log({d}) = {entropy_mass:.6f}")
    print()
    
    # Calculate optimal loss
    optimal_loss = entropy_mass / total_tokens
    
    print(f"=== Theoretical Minimum Loss ===")
    print(f"Optimal loss = {entropy_mass:.6f} / {total_tokens:,}")
    print(f"            = {optimal_loss:.10f}")
    print()
    
    # Sanity checks
    print(f"=== Sanity Checks ===")
    print(f"If all path tokens have 0 loss: path contribution = 0")
    print(f"If {num_gt_edges_from_root} GT edges each have log({d}) loss: edge contribution = {entropy_mass:.6f}")
    print(f"Average loss over {total_tokens:,} tokens = {optimal_loss:.10f}")
    print()
    
    # What training loss would we expect to see?
    print(f"=== Expected Training Loss ===")
    print(f"Training loss should approach but NEVER go below: {optimal_loss:.10f}")
    print(f"If training loss goes below this, there's a bug in either:")
    print(f"  1. The theoretical minimum calculation")
    print(f"  2. The training loss calculation")
    print(f"  3. The masking logic")
    print()
    
    return optimal_loss


if __name__ == "__main__":
    # Example 1: User's scenario (d=5, l=100)
    print("=" * 80)
    print("EXAMPLE 1: d=5, l=100 (user's scenario)")
    print("=" * 80)
    optimal_loss_1 = calculate_optimal_loss(
        d=5, 
        l=100, 
        holdout_percentage=0.2,
        use_undirected=True,
        interleave_dataset=True,
        balance_interleaved_datasets=True,
        use_directional_tokens_in_path=False
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: d=100, l=5 (mentioned by user)")
    print("=" * 80)
    optimal_loss_2 = calculate_optimal_loss(
        d=100,
        l=5,
        holdout_percentage=0.2,
        use_undirected=True,
        interleave_dataset=True,
        balance_interleaved_datasets=True,
        use_directional_tokens_in_path=False
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: d=8000, l=3 (from test_final.yaml)")
    print("=" * 80)
    optimal_loss_3 = calculate_optimal_loss(
        d=8000,
        l=3,
        holdout_percentage=0.2,
        use_undirected=True,
        interleave_dataset=True,
        balance_interleaved_datasets=True,
        use_directional_tokens_in_path=False
    )

