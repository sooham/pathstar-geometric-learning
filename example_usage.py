#!/usr/bin/env python3
"""
Example usage of InContextPathStar and InWeightsPathStar classes
for generating path-star graph learning tasks.
"""
import torch
from generate_pathstar import InContextPathStar, InWeightsPathStar


def example_incontext_pathstar():
    """
    Example: In-Context Path-Star Task (B&N'24 style)
    
    Each training example has a fresh, randomly-labeled graph.
    Input: [adjacency_list, PAUSE, root, goal]
    Output: [path from root to goal]
    """
    print("=" * 80)
    print("IN-CONTEXT PATH-STAR TASK")
    print("=" * 80)
    
    # Create generator: 5 paths, each of length 5
    generator = InContextPathStar(d=5, l=5, vocab_size=200)
    
    print("\nGenerator Configuration:")
    print(f"  Number of paths (d): {generator.d}")
    print(f"  Path length (l): {generator.l}")
    print(f"  Nodes per graph: {generator.total_nodes}")
    print(f"  Edges per graph: {generator.total_edges}")
    print(f"  Vocabulary size: {generator.vocab_size}")
    print(f"  Effective vocab size: {generator.effective_vocab_size}")
    
    # Generate a single example
    print("\n--- Single Training Example ---")
    example = generator.generate_training_example(
        use_directional_tokens=False,
        num_pause_tokens=3
    )
    
    print(f"Root (start): {example['root']}")
    print(f"Goal (leaf): {example['goal']}")
    print(f"True path: {example['target']}")
    print(f"Prefix length: {len(example['prefix'])}")
    print(f"Prefix (first 20 tokens): {example['prefix'][:20]}")
    print(f"Full sequence length: {len(example['full_sequence'])}")
    
    # Generate a batch
    print("\n--- Training Batch ---")
    batch_size = 100
    training_data = generator.generate_training_set(
        size=batch_size,
        use_directional_tokens=False,
        num_pause_tokens=3,
        return_tensors=True
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Prefixes shape: {training_data['prefixes'].shape}")
    print(f"Targets shape: {training_data['targets'].shape}")
    print(f"Full sequences shape: {training_data['full_sequences'].shape}")
    
    # Show statistics
    print(f"\nSequence length statistics:")
    print(f"  Prefix length (all same): {training_data['prefix_lengths'][0].item()}")
    print(f"  Target length (all same): {training_data['target_lengths'][0].item()}")
    

def example_inweights_pathstar():
    """
    Example: In-Weights Path-Star Task
    
    Single fixed graph, model learns paths in weights.
    Input: [leaf, PAUSE, root, n_2, ..., n_ℓ]
    Output: predict next token (path memorization)
    """
    print("\n" + "=" * 80)
    print("IN-WEIGHTS PATH-STAR TASK")
    print("=" * 80)
    
    # Create generator: 5 paths, each of length 5
    generator = InWeightsPathStar(d=5, l=5)
    
    print("\nGenerator Configuration:")
    print(generator)
    
    # Generate adjacency list for the fixed graph
    print("\n--- Adjacency List (shuffled) ---")
    adj_list = generator.generate_adjacency_list()
    print(f"Edges (first 10): {adj_list[:10]}")
    print(f"Total edges: {len(adj_list)}")
    
    # Generate path prediction training set
    print("\n--- Path Prediction Training Set ---")
    batch_size = 100
    pause_token = 999  # Special token for pause
    sequences = generator.generate_path_prediction_training_set(
        size=batch_size,
        pause_token=pause_token,
        num_pause_tokens=3
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"First sequence: {sequences[0].tolist()}")
    print(f"  - First token (leaf): {sequences[0][0].item()}")
    print(f"  - Pause tokens: {sequences[0][1:4].tolist()}")
    print(f"  - Path (root to leaf): {sequences[0][4:].tolist()}")
    
    # Generate edge memorization training set
    print("\n--- Edge Memorization Training Set ---")
    x, y = generator.generate_edge_memorization_training_set(
        size=batch_size,
        undirected=True
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Predecessors (x) shape: {x.shape}")
    print(f"Successors (y) shape: {y.shape}")
    print(f"First 10 edges: {list(zip(x[:10].tolist(), y[:10].tolist()))}")


def example_with_custom_mapping():
    """
    Example: Using InWeightsPathStar with custom node mapping
    """
    print("\n" + "=" * 80)
    print("IN-WEIGHTS PATH-STAR WITH CUSTOM MAPPING")
    print("=" * 80)
    
    # Create a custom mapping (e.g., random permutation)
    import random
    d, l = 3, 4
    total_nodes = d * (l - 1) + 1
    
    # Create mapping: canonical_id -> custom_token
    tokens = list(range(100, 100 + total_nodes))
    random.shuffle(tokens)
    mapping = {i: tokens[i] for i in range(total_nodes)}
    
    print(f"\nCustom mapping: {mapping}")
    
    # Create generator with custom mapping
    generator = InWeightsPathStar(d=d, l=l, mapping=mapping)
    
    print("\nMapped generator:")
    print(generator)
    
    # Generate training data with mapped nodes
    sequences = generator.generate_path_prediction_training_set(
        size=5,
        pause_token=999,
        num_pause_tokens=1
    )
    
    print(f"\nTraining sequences (with mapping):")
    for i, seq in enumerate(sequences[:3]):
        print(f"  Example {i+1}: {seq.tolist()}")


def example_comparison():
    """
    Compare the two approaches
    """
    print("\n" + "=" * 80)
    print("COMPARISON: In-Context vs In-Weights")
    print("=" * 80)
    
    d, l = 5, 5
    
    print("\nKey Differences:")
    print("\n1. IN-CONTEXT (B&N'24):")
    print("   - Each example has a FRESH, randomly-labeled graph")
    print("   - Model must learn to parse adjacency list in-context")
    print("   - Tests generalization to new graphs")
    print("   - Prefix includes full adjacency list")
    
    incontext = InContextPathStar(d=d, l=l, vocab_size=200)
    example_ic = incontext.generate_training_example(num_pause_tokens=3)
    print(f"   - Example prefix length: {len(example_ic['prefix'])}")
    print(f"   - Example target length: {len(example_ic['target'])}")
    
    print("\n2. IN-WEIGHTS:")
    print("   - Single FIXED graph throughout training")
    print("   - Model memorizes graph structure in weights")
    print("   - Tests path recall from memory")
    print("   - Input is just (leaf, pause, root, ...path)")
    
    inweights = InWeightsPathStar(d=d, l=l)
    sequences = inweights.generate_path_prediction_training_set(
        size=1, pause_token=999, num_pause_tokens=3
    )
    print(f"   - Sequence length: {sequences.shape[1]}")
    print(f"   - No adjacency list in input!")


if __name__ == "__main__":
    # Run all examples
    example_incontext_pathstar()
    example_inweights_pathstar()
    example_with_custom_mapping()
    example_comparison()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)

