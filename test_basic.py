#!/usr/bin/env python3
"""
Basic test suite to verify implementations are correct
"""
import torch
from generate_pathstar import InContextPathStar, InWeightsPathStar


def test_incontext_basic():
    """Test basic InContextPathStar functionality"""
    print("Testing InContextPathStar...")
    
    gen = InContextPathStar(d=3, l=4, vocab_size=100)
    
    # Test single example
    example = gen.generate_training_example()
    assert 'prefix' in example
    assert 'target' in example
    assert 'root' in example
    assert 'goal' in example
    
    # Verify path correctness
    assert example['target'][0] == example['root'], "Path should start at root"
    assert example['target'][-1] == example['goal'], "Path should end at goal"
    assert len(example['target']) == gen.l, f"Path length should be {gen.l}"
    
    # Test batch generation
    batch = gen.generate_training_set(size=10)
    assert batch['prefixes'].shape[0] == 10
    assert batch['targets'].shape[0] == 10
    
    print("  ✓ Basic functionality")
    print("  ✓ Single example generation")
    print("  ✓ Batch generation")
    print("  ✓ Path correctness")


def test_incontext_randomness():
    """Test that InContextPathStar generates different graphs"""
    print("\nTesting InContextPathStar randomness...")
    
    gen = InContextPathStar(d=3, l=4, vocab_size=100)
    
    # Generate multiple examples
    examples = [gen.generate_training_example() for _ in range(5)]
    
    # Check that root nodes vary (with high probability)
    roots = [ex['root'] for ex in examples]
    unique_roots = len(set(roots))
    
    # With vocab_size=100 and 10 nodes, we expect high diversity
    assert unique_roots >= 3, "Should have at least 3 different root nodes in 5 examples"
    
    print(f"  ✓ Generated {unique_roots} unique root nodes in 5 examples")


def test_incontext_directional():
    """Test directional token functionality"""
    print("\nTesting InContextPathStar with directional tokens...")
    
    gen = InContextPathStar(d=3, l=4, vocab_size=100)
    
    # Without directional tokens
    ex1 = gen.generate_training_example(use_directional_tokens=False)
    adj_seq_len1 = len(ex1['adjacency_sequence'])
    
    # With directional tokens
    ex2 = gen.generate_training_example(use_directional_tokens=True)
    adj_seq_len2 = len(ex2['adjacency_sequence'])
    
    # With directional tokens, length should be 1.5x (each edge: u, >, v instead of u, v)
    expected_ratio = 1.5
    actual_ratio = adj_seq_len2 / adj_seq_len1
    
    assert abs(actual_ratio - expected_ratio) < 0.01, \
        f"Directional tokens should increase length by 1.5x, got {actual_ratio}"
    
    # Check that GT token is present
    gt_token = gen.SPECIAL_TOKENS['GT']
    assert gt_token in ex2['adjacency_sequence'], "GT token should be in adjacency sequence"
    
    print("  ✓ Directional token format")
    print(f"  ✓ Length ratio: {actual_ratio:.2f} (expected: 1.5)")


def test_inweights_basic():
    """Test basic InWeightsPathStar functionality"""
    print("\nTesting InWeightsPathStar...")
    
    gen = InWeightsPathStar(d=3, l=4)
    
    # Test adjacency list
    adj_list = gen.generate_adjacency_list()
    assert len(adj_list) == gen.d * (gen.l - 1), "Should have d*(l-1) edges"
    
    # Test paths by leaf
    paths = gen.generate_paths_by_leaf()
    assert len(paths) == gen.d, f"Should have {gen.d} paths"
    
    # Verify each path
    for leaf, path in paths.items():
        assert path[0] == gen.v_root, "Path should start at root"
        assert path[-1] == leaf, "Path should end at leaf"
        assert len(path) == gen.l, f"Path should have length {gen.l}"
    
    print("  ✓ Adjacency list generation")
    print("  ✓ Path structure")
    print("  ✓ Path correctness")


def test_inweights_training_sets():
    """Test InWeightsPathStar training set generation"""
    print("\nTesting InWeightsPathStar training sets...")
    
    gen = InWeightsPathStar(d=5, l=5)
    
    # Test edge memorization
    x, y = gen.generate_edge_memorization_training_set(size=100, undirected=True)
    assert x.shape[0] == 100, "Should have 100 samples"
    assert y.shape[0] == 100, "Should have 100 samples"
    
    # Test path prediction
    sequences = gen.generate_path_prediction_training_set(
        size=50, 
        pause_token=999, 
        num_pause_tokens=3
    )
    assert sequences.shape[0] == 50, "Should have 50 samples"
    assert sequences.shape[1] == gen.l + 1 + 3, "Sequence length should be l+1+num_pause_tokens"
    
    # Verify sequence structure
    for seq in sequences[:5]:
        leaf = seq[0].item()
        assert leaf in gen.v_leaf, "First token should be a leaf"
        
        pause_tokens = seq[1:4]
        assert all(p == 999 for p in pause_tokens), "Should have pause tokens"
        
        path = seq[4:].tolist()
        assert path[0] == gen.v_root, "Path should start at root"
    
    print("  ✓ Edge memorization set")
    print("  ✓ Path prediction set")
    print("  ✓ Sequence structure")


def test_inweights_mapping():
    """Test InWeightsPathStar with custom mapping"""
    print("\nTesting InWeightsPathStar with custom mapping...")
    
    d, l = 3, 4
    total_nodes = d * (l - 1) + 1
    
    # Create custom mapping
    mapping = {i: i + 100 for i in range(total_nodes)}
    
    gen = InWeightsPathStar(d=d, l=l, mapping=mapping)
    
    # Verify mapping is applied
    assert gen.v_root == 100, "Root should be mapped to 100"
    
    # Verify all vertices are mapped
    for v in gen.vertices:
        assert v >= 100, "All vertices should be mapped"
    
    # Generate training data with mapped nodes
    sequences = gen.generate_path_prediction_training_set(
        size=10,
        pause_token=999,
        num_pause_tokens=1
    )
    
    # Verify sequences use mapped nodes
    for seq in sequences:
        path = seq[2:].tolist()  # Skip leaf and pause
        assert path[0] == gen.v_root, "Should use mapped root"
        assert all(100 <= p < 200 for p in path), "All path nodes should be mapped"
    
    print("  ✓ Custom mapping application")
    print("  ✓ Mapped training data")


def test_graph_structure():
    """Test that graph structure is correct"""
    print("\nTesting graph structure...")
    
    configs = [(2, 3), (3, 4), (5, 5)]
    
    for d, l in configs:
        gen = InWeightsPathStar(d=d, l=l)
        
        # Test node count
        expected_nodes = d * (l - 1) + 1
        assert gen.total_vert == expected_nodes, \
            f"d={d}, l={l}: Expected {expected_nodes} nodes, got {gen.total_vert}"
        
        # Test edge count
        edge_count = sum(len(neighbors) for neighbors in gen.adj_list.values())
        expected_edges = d * (l - 1)
        assert edge_count == expected_edges, \
            f"d={d}, l={l}: Expected {expected_edges} edges, got {edge_count}"
        
        # Test leaf count
        assert len(gen.v_leaf) == d, \
            f"d={d}, l={l}: Expected {d} leaves, got {len(gen.v_leaf)}"
        
        # Test root degree
        assert len(gen.adj_list[gen.v_root]) == d, \
            f"d={d}, l={l}: Root should have degree {d}, got {len(gen.adj_list[gen.v_root])}"
    
    print(f"  ✓ Tested {len(configs)} configurations")
    print("  ✓ Node counts correct")
    print("  ✓ Edge counts correct")
    print("  ✓ Graph properties correct")


def test_special_tokens():
    """Test special token definitions"""
    print("\nTesting special tokens...")
    
    gen = InContextPathStar(d=3, l=4, vocab_size=50)
    
    # Check all special tokens are defined
    expected_tokens = ['PAUSE', 'PAD', 'GT', 'LT', 'SEP', 'START', 'GOAL', 'PATH_START', 'EOS']
    for token_name in expected_tokens:
        assert token_name in gen.SPECIAL_TOKENS, f"Missing token: {token_name}"
    
    # Check tokens don't overlap with vocabulary
    for token_name, token_id in gen.SPECIAL_TOKENS.items():
        assert token_id >= gen.vocab_size, \
            f"Token {token_name} ({token_id}) overlaps with vocabulary (size {gen.vocab_size})"
    
    # Check effective vocab size
    assert gen.effective_vocab_size == gen.vocab_size + 9, \
        "Effective vocab size should be vocab_size + 9"
    
    print("  ✓ All special tokens defined")
    print("  ✓ No overlap with vocabulary")
    print("  ✓ Effective vocab size correct")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING BASIC TESTS")
    print("=" * 80)
    
    try:
        test_incontext_basic()
        test_incontext_randomness()
        test_incontext_directional()
        test_inweights_basic()
        test_inweights_training_sets()
        test_inweights_mapping()
        test_graph_structure()
        test_special_tokens()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise

