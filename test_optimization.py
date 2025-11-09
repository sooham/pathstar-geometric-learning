#!/usr/bin/env python3
"""
Test script to verify the optimized InWeightsPathStar implementation.
Tests correctness with small graphs and benchmarks performance with larger graphs.
"""

import time
import numpy as np
from pathstar import InWeightsPathStar
import os
import shutil

def test_small_graph_correctness():
    """Test with small graph (d=10, l=10) to verify output correctness"""
    print("="*80)
    print("TEST 1: Small Graph Correctness (d=10, l=10)")
    print("="*80)
    
    # Create a small graph
    d, l = 10, 10
    vocab_size = 1000
    holdout_percentage = 0.2
    
    print(f"\nCreating InWeightsPathStar(d={d}, l={l}, vocab_size={vocab_size}, holdout_percentage={holdout_percentage})")
    start_time = time.time()
    generator = InWeightsPathStar(d=d, l=l, vocab_size=vocab_size, holdout_percentage=holdout_percentage)
    init_time = time.time() - start_time
    
    print(f"\n✓ Graph initialized in {init_time:.3f}s")
    print(f"  Total vertices: {generator.total_vert}")
    print(f"  Root: {generator.v_root}")
    print(f"  Leaves: {generator.v_leaf}")
    print(f"  Train leaves: {len(generator.train_leaves)}")
    print(f"  Holdout leaves: {len(generator.holdout_leaves)}")
    
    # Test path generation
    print("\n--- Testing Path Generation ---")
    num_paths = len(generator.train_leaves)
    start_time = time.time()
    path_sequences = generator.generate_path_prediction_training_set(
        size=num_paths,
        num_pause_tokens=2,
        obey_holdout=True
    )
    path_gen_time = time.time() - start_time
    
    print(f"✓ Generated {num_paths} path sequences in {path_gen_time:.3f}s")
    print(f"  Shape: {path_sequences.shape}")
    print(f"  First sequence: {path_sequences[0].tolist()}")
    
    # Verify path structure
    first_seq = path_sequences[0].tolist()
    assert first_seq[0] == generator.TASK_TOKENS['PATH'], "First token should be PATH"
    assert first_seq[2] == generator.pause_token, "Third token should be PAUSE"
    assert first_seq[3] == generator.pause_token, "Fourth token should be PAUSE"
    print("✓ Path sequence structure is correct")
    
    # Test edge generation
    print("\n--- Testing Edge Generation ---")
    num_edges = d * (l - 1) * 2  # undirected
    start_time = time.time()
    edge_sequences = generator.generate_edge_memorization_training_set(
        size=num_edges,
        undirected=True,
        use_directional_tokens=True
    )
    edge_gen_time = time.time() - start_time
    
    print(f"✓ Generated {num_edges} edge sequences in {edge_gen_time:.3f}s")
    print(f"  Shape: {edge_sequences.shape}")
    print(f"  First 5 edges: {edge_sequences[:5].tolist()}")
    
    # Verify edge structure
    first_edge = edge_sequences[0].tolist()
    assert first_edge[0] in [generator.SPECIAL_TOKENS['GT'], generator.SPECIAL_TOKENS['LT']], "First token should be GT or LT"
    print("✓ Edge sequence structure is correct")
    
    # Test dataset preparation
    print("\n--- Testing Dataset Preparation ---")
    output_dir = './data/test_small'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    start_time = time.time()
    result_dir = generator.prepare(
        num_pause_tokens=2,
        output_dir='./data/test_small',
        use_undirected=True,
        use_directional_tokens=True
    )
    prepare_time = time.time() - start_time
    
    print(f"\n✓ Dataset prepared in {prepare_time:.3f}s")
    print(f"  Output directory: {result_dir}")
    
    # Verify files exist
    train_file = os.path.join(result_dir, 'train.bin')
    val_file = os.path.join(result_dir, 'val.bin')
    meta_file = os.path.join(result_dir, 'meta.pkl')
    
    assert os.path.exists(train_file), "train.bin should exist"
    assert os.path.exists(val_file), "val.bin should exist"
    assert os.path.exists(meta_file), "meta.pkl should exist"
    print("✓ All dataset files created")
    
    # Verify file sizes
    train_size = os.path.getsize(train_file) / (1024**2)
    val_size = os.path.getsize(val_file) / (1024**2)
    print(f"  train.bin size: {train_size:.2f} MB")
    print(f"  val.bin size: {val_size:.2f} MB")
    
    # Load and verify a few samples
    import pickle
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    seq_length = l + 2 + 2  # l + PATH token + leaf + 2 PAUSE tokens
    train_data = np.memmap(train_file, dtype=np.uint16, mode='r', shape=(meta['train_size'], seq_length))
    print(f"\n✓ Loaded train data: shape {train_data.shape}")
    print(f"  First sequence: {train_data[0]}")
    print(f"  Last sequence: {train_data[-1]}")
    
    print("\n" + "="*80)
    print("✓ ALL SMALL GRAPH TESTS PASSED")
    print("="*80)
    
    return {
        'init_time': init_time,
        'path_gen_time': path_gen_time,
        'edge_gen_time': edge_gen_time,
        'prepare_time': prepare_time
    }


def benchmark_medium_graph():
    """Benchmark with medium graph (d=10000, l=50)"""
    print("\n" + "="*80)
    print("TEST 2: Medium Graph Performance (d=10000, l=50)")
    print("="*80)
    
    d, l = 10000, 50
    vocab_size = 500000  # d * (l-1) + 1 = 490,001
    holdout_percentage = 0.1
    
    print(f"\nCreating InWeightsPathStar(d={d}, l={l}, vocab_size={vocab_size})")
    start_time = time.time()
    generator = InWeightsPathStar(d=d, l=l, vocab_size=vocab_size, holdout_percentage=holdout_percentage)
    init_time = time.time() - start_time
    
    print(f"✓ Graph initialized in {init_time:.3f}s")
    print(f"  Total vertices: {generator.total_vert}")
    print(f"  Estimated edges: {d * (l-1)}")
    
    # Test preparation (the most intensive operation)
    print("\n--- Testing Dataset Preparation ---")
    output_dir = './data/test_medium'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    start_time = time.time()
    result_dir = generator.prepare(
        num_pause_tokens=1,
        output_dir='./data/test_medium',
        use_undirected=True,
        use_directional_tokens=True
    )
    prepare_time = time.time() - start_time
    
    print(f"\n✓ Dataset prepared in {prepare_time:.2f}s ({prepare_time/60:.2f} minutes)")
    
    # Verify files
    train_file = os.path.join(result_dir, 'train.bin')
    val_file = os.path.join(result_dir, 'val.bin')
    
    train_size = os.path.getsize(train_file) / (1024**3)
    val_size = os.path.getsize(val_file) / (1024**3)
    print(f"  train.bin size: {train_size:.2f} GB")
    print(f"  val.bin size: {val_size:.2f} GB")
    
    print("\n" + "="*80)
    print("✓ MEDIUM GRAPH BENCHMARK COMPLETE")
    print("="*80)
    
    return {
        'init_time': init_time,
        'prepare_time': prepare_time,
        'train_size_gb': train_size,
        'val_size_gb': val_size
    }


def benchmark_large_graph():
    """Benchmark with large graph (d=1000000, l=50)"""
    print("\n" + "="*80)
    print("TEST 3: Large Graph Performance (d=1,000,000, l=50)")
    print("="*80)
    
    d, l = 1000000, 50
    vocab_size = 50000000  # d * (l-1) + 1 = 49,000,001
    holdout_percentage = 0.01  # Only 1% holdout for large graph
    
    print(f"\nCreating InWeightsPathStar(d={d}, l={l}, vocab_size={vocab_size})")
    print(f"Expected vertices: {d * (l-1) + 1}")
    print(f"Expected edges: {d * (l-1)}")
    print(f"Expected dataset size: ~{d * (l-1) * 2 * (l+3) * 2 / (1024**3):.1f} GB")
    
    start_time = time.time()
    generator = InWeightsPathStar(d=d, l=l, vocab_size=vocab_size, holdout_percentage=holdout_percentage)
    init_time = time.time() - start_time
    
    print(f"\n✓ Graph initialized in {init_time:.2f}s ({init_time/60:.2f} minutes)")
    print(f"  Total vertices: {generator.total_vert}")
    print(f"  Train leaves: {len(generator.train_leaves)}")
    print(f"  Holdout leaves: {len(generator.holdout_leaves)}")
    
    # Test preparation
    print("\n--- Testing Dataset Preparation ---")
    output_dir = './data/test_large'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    start_time = time.time()
    result_dir = generator.prepare(
        num_pause_tokens=1,
        output_dir='./data/test_large',
        use_undirected=True,
        use_directional_tokens=True
    )
    prepare_time = time.time() - start_time
    
    print(f"\n✓ Dataset prepared in {prepare_time:.2f}s ({prepare_time/60:.2f} minutes)")
    
    # Verify files
    train_file = os.path.join(result_dir, 'train.bin')
    val_file = os.path.join(result_dir, 'val.bin')
    
    train_size = os.path.getsize(train_file) / (1024**3)
    val_size = os.path.getsize(val_file) / (1024**3)
    print(f"  train.bin size: {train_size:.2f} GB")
    print(f"  val.bin size: {val_size:.2f} GB")
    print(f"  Total size: {train_size + val_size:.2f} GB")
    
    print("\n" + "="*80)
    print("✓ LARGE GRAPH BENCHMARK COMPLETE")
    print("="*80)
    
    return {
        'init_time': init_time,
        'prepare_time': prepare_time,
        'train_size_gb': train_size,
        'val_size_gb': val_size,
        'total_size_gb': train_size + val_size
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and benchmark InWeightsPathStar optimization')
    parser.add_argument('--skip-small', action='store_true', help='Skip small graph test')
    parser.add_argument('--skip-medium', action='store_true', help='Skip medium graph test')
    parser.add_argument('--skip-large', action='store_true', help='Skip large graph test')
    args = parser.parse_args()
    
    results = {}
    
    if not args.skip_small:
        try:
            results['small'] = test_small_graph_correctness()
        except Exception as e:
            print(f"\n❌ Small graph test FAILED: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    if not args.skip_medium:
        try:
            results['medium'] = benchmark_medium_graph()
        except Exception as e:
            print(f"\n❌ Medium graph test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    if not args.skip_large:
        try:
            results['large'] = benchmark_large_graph()
        except Exception as e:
            print(f"\n❌ Large graph test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if 'small' in results:
        print(f"\nSmall Graph (d=10, l=10):")
        print(f"  Init: {results['small']['init_time']:.3f}s")
        print(f"  Path gen: {results['small']['path_gen_time']:.3f}s")
        print(f"  Edge gen: {results['small']['edge_gen_time']:.3f}s")
        print(f"  Prepare: {results['small']['prepare_time']:.3f}s")
    
    if 'medium' in results:
        print(f"\nMedium Graph (d=10000, l=50):")
        print(f"  Init: {results['medium']['init_time']:.2f}s")
        print(f"  Prepare: {results['medium']['prepare_time']:.2f}s ({results['medium']['prepare_time']/60:.2f} min)")
        print(f"  Dataset size: {results['medium']['train_size_gb']:.2f} GB")
    
    if 'large' in results:
        print(f"\nLarge Graph (d=1000000, l=50):")
        print(f"  Init: {results['large']['init_time']:.2f}s ({results['large']['init_time']/60:.2f} min)")
        print(f"  Prepare: {results['large']['prepare_time']:.2f}s ({results['large']['prepare_time']/60:.2f} min)")
        print(f"  Dataset size: {results['large']['total_size_gb']:.2f} GB")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETE")
    print("="*80)

