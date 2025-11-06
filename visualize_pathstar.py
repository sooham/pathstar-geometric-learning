#!/usr/bin/env python3
"""
Visualize path-star graph structure in ASCII art
"""
from generate_pathstar import InContextPathStar, InWeightsPathStar


def visualize_pathstar_structure(d, l):
    """
    Create ASCII art visualization of path-star structure
    """
    print(f"\nPath-Star Graph: d={d} paths, l={l} length")
    print(f"Total nodes: {d * (l-1) + 1}")
    print(f"Total edges: {d * (l-1)}")
    print()
    
    # Create generator to get actual structure
    gen = InWeightsPathStar(d=d, l=l)
    
    print("Structure:")
    print()
    
    # Draw each path
    for i, leaf in enumerate(gen.v_leaf):
        path = gen.paths_by_leaf[leaf]
        
        # Create visual representation
        if i == 0:
            # First path - show root
            spaces = " " * 4
            print(f"{spaces}[{path[0]}] (root)")
            print(f"{spaces} |")
        
        # Draw the path
        for j in range(1, len(path)):
            node = path[j]
            if j == 1:
                # First node after root
                if i == 0:
                    print(f"{spaces} +-- [{node}]", end="")
                else:
                    print(f"{spaces} |")
                    print(f"{spaces} +-- [{node}]", end="")
            elif j == len(path) - 1:
                # Leaf node
                print(f" -- [{node}] (leaf)")
            else:
                # Intermediate node
                print(f" -- [{node}]", end="")
    
    print()
    print("Adjacency List:")
    for node in sorted(gen.adj_list.keys()):
        if gen.adj_list[node]:
            print(f"  {node:3d} -> {gen.adj_list[node]}")
    
    print()
    print("Paths by Leaf:")
    for leaf in sorted(gen.paths_by_leaf.keys()):
        path = gen.paths_by_leaf[leaf]
        print(f"  Leaf {leaf:2d}: {' -> '.join(map(str, path))}")


def visualize_incontext_example(d, l):
    """
    Show an in-context learning example
    """
    print("\n" + "=" * 80)
    print("IN-CONTEXT EXAMPLE")
    print("=" * 80)
    
    gen = InContextPathStar(d=d, l=l, vocab_size=100)
    example = gen.generate_training_example(
        use_directional_tokens=False,
        num_pause_tokens=2
    )
    
    print(f"\nGraph Configuration: d={d}, l={l}")
    print(f"Root node (mapped): {example['root']}")
    print(f"Goal node (mapped): {example['goal']}")
    
    # Parse adjacency sequence
    adj_seq = example['adjacency_sequence']
    edges = [(adj_seq[i], adj_seq[i+1]) for i in range(0, len(adj_seq), 2)]
    
    print(f"\nAdjacency List (shuffled edge pairs):")
    for i, (u, v) in enumerate(edges[:10]):
        print(f"  Edge {i+1:2d}: {u} -> {v}")
    if len(edges) > 10:
        print(f"  ... and {len(edges) - 10} more edges")
    
    print(f"\nPrefix (input to model):")
    print(f"  Length: {len(example['prefix'])} tokens")
    print(f"  Format: [adjacency_list, PAUSE, PAUSE, root, goal]")
    print(f"  First 15 tokens: {example['prefix'][:15]}")
    print(f"  Last 5 tokens: {example['prefix'][-5:]}")
    
    print(f"\nTarget (expected output):")
    print(f"  Path from {example['root']} to {example['goal']}")
    print(f"  Path: {example['target']}")
    print(f"  Length: {len(example['target'])} nodes")
    
    # Verify path correctness
    print(f"\nVerification:")
    print(f"  ✓ Path starts at root: {example['target'][0] == example['root']}")
    print(f"  ✓ Path ends at goal: {example['target'][-1] == example['goal']}")
    print(f"  ✓ Path length matches l: {len(example['target']) == l}")


def visualize_inweights_example(d, l):
    """
    Show an in-weights learning example
    """
    print("\n" + "=" * 80)
    print("IN-WEIGHTS EXAMPLE")
    print("=" * 80)
    
    gen = InWeightsPathStar(d=d, l=l)
    
    print(f"\nGraph Configuration: d={d}, l={l}")
    print(f"Root node: {gen.v_root}")
    print(f"Leaf nodes: {gen.v_leaf}")
    
    # Generate some training sequences
    sequences = gen.generate_path_prediction_training_set(
        size=5,
        pause_token=999,
        num_pause_tokens=2
    )
    
    print(f"\nTraining Sequences:")
    print(f"  Format: [leaf, PAUSE, PAUSE, root, n₂, ..., leaf]")
    print()
    for i, seq in enumerate(sequences):
        seq_list = seq.tolist()
        leaf = seq_list[0]
        pause = seq_list[1:3]
        path = seq_list[3:]
        
        print(f"  Sequence {i+1}:")
        print(f"    Leaf: {leaf}")
        print(f"    Pause: {pause}")
        print(f"    Path: {path}")
        print(f"    Full: {seq_list}")
        print()


def compare_sequence_lengths():
    """
    Compare sequence lengths for different configurations
    """
    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH COMPARISON")
    print("=" * 80)
    
    configs = [
        (2, 3),
        (3, 4),
        (5, 5),
        (10, 5),
        (5, 10),
        (10, 10),
    ]
    
    print("\n{:>4s} {:>4s} {:>6s} {:>6s} {:>15s} {:>15s}".format(
        "d", "l", "nodes", "edges", "InContext len", "InWeights len"
    ))
    print("-" * 70)
    
    for d, l in configs:
        nodes = d * (l - 1) + 1
        edges = d * (l - 1)
        
        # InContext: adjacency_list (2 * edges) + pause (2) + query (2) + target (l)
        incontext_len = 2 * edges + 2 + 2 + l
        
        # InWeights: leaf (1) + pause (2) + path (l)
        inweights_len = 1 + 2 + l
        
        print("{:>4d} {:>4d} {:>6d} {:>6d} {:>15d} {:>15d}".format(
            d, l, nodes, edges, incontext_len, inweights_len
        ))


if __name__ == "__main__":
    # Visualize small path-star structures
    print("=" * 80)
    print("PATH-STAR STRUCTURE VISUALIZATION")
    print("=" * 80)
    
    visualize_pathstar_structure(d=3, l=4)
    
    print("\n" + "=" * 80)
    
    visualize_pathstar_structure(d=5, l=3)
    
    # Show example tasks
    visualize_incontext_example(d=3, l=4)
    visualize_inweights_example(d=3, l=4)
    
    # Compare sequence lengths
    compare_sequence_lengths()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)

