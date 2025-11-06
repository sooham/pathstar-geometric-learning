"""
Demonstration of PAUSE and PAD token masking in training.

This script shows:
1. How tokens are loaded from a dataset
2. How PAUSE and PAD tokens are identified
3. How masking is applied to targets
4. The effect on loss calculation
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

def demo_masking(data_dir='data/inweights_pathstar_d10_l5'):
    """
    Demonstrate the masking behavior with a real dataset.
    """
    print("="*70)
    print("PAUSE and PAD Token Masking Demonstration")
    print("="*70)
    
    # Load metadata
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        print(f"Error: Dataset not found at {data_dir}")
        print("Please generate a dataset first using pathstar.py")
        return
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    print(f"\n1. DATASET INFORMATION")
    print(f"   Dataset: {data_dir}")
    print(f"   Vocab size: {meta['vocab_size']}")
    print(f"   d={meta['d']}, l={meta['l']}")
    
    # Load special tokens
    pause_token_id = meta.get('pause_token')
    pad_token_id = meta.get('pad_token')
    
    if pause_token_id is None and pad_token_id is None:
        print("\n   ⚠️  No special tokens found in this dataset")
        print("   This dataset was created with an older version")
        return
    
    print(f"\n2. SPECIAL TOKENS")
    print(f"   PAUSE token: {pause_token_id}")
    print(f"   PAD token: {pad_token_id}")
    if 'task_tokens' in meta:
        print(f"   Task tokens: {meta['task_tokens']}")
    
    # Load training data
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    print(f"\n3. SAMPLE SEQUENCE (before masking)")
    # Get a sample sequence
    block_size = 32
    start_idx = 0
    sample_x = train_data[start_idx:start_idx+block_size]
    sample_y = train_data[start_idx+1:start_idx+1+block_size]
    
    print(f"   Input (x):  {sample_x[:20]}...")
    print(f"   Target (y): {sample_y[:20]}...")
    
    # Count special tokens
    pause_count = np.sum(sample_y == pause_token_id) if pause_token_id is not None else 0
    pad_count = np.sum(sample_y == pad_token_id) if pad_token_id is not None else 0
    
    print(f"\n   Special token counts in target:")
    print(f"   - PAUSE: {pause_count}")
    print(f"   - PAD: {pad_count}")
    print(f"   - Regular tokens: {len(sample_y) - pause_count - pad_count}")
    
    # Apply masking
    print(f"\n4. APPLYING MASKING")
    y_tensor = torch.from_numpy(sample_y.astype(np.int64))
    y_masked = y_tensor.clone()
    
    if pause_token_id is not None:
        y_masked[y_masked == pause_token_id] = -1
        print(f"   ✓ Replaced {pause_count} PAUSE tokens with -1")
    
    if pad_token_id is not None:
        y_masked[y_masked == pad_token_id] = -1
        print(f"   ✓ Replaced {pad_count} PAD tokens with -1")
    
    print(f"\n5. SAMPLE SEQUENCE (after masking)")
    print(f"   Target (y): {y_masked[:20]}...")
    
    # Show the effect on loss calculation
    print(f"\n6. LOSS CALCULATION COMPARISON")
    
    # Create dummy logits (uniform distribution)
    vocab_size = meta['vocab_size']
    logits = torch.randn(len(sample_y), vocab_size)
    
    # Loss without masking (incorrect)
    loss_without_masking = F.cross_entropy(logits, y_tensor)
    
    # Loss with masking (correct)
    loss_with_masking = F.cross_entropy(logits, y_masked, ignore_index=-1)
    
    print(f"   Loss WITHOUT masking: {loss_without_masking:.4f}")
    print(f"   Loss WITH masking:    {loss_with_masking:.4f}")
    print(f"   Difference:           {abs(loss_without_masking - loss_with_masking):.4f}")
    
    # Calculate effective batch size
    effective_tokens = (y_masked != -1).sum().item()
    total_tokens = len(y_masked)
    
    print(f"\n7. EFFECTIVE TRAINING")
    print(f"   Total tokens:     {total_tokens}")
    print(f"   Masked tokens:    {total_tokens - effective_tokens}")
    print(f"   Effective tokens: {effective_tokens}")
    print(f"   Masking rate:     {100 * (total_tokens - effective_tokens) / total_tokens:.1f}%")
    
    print(f"\n8. INTERPRETATION")
    print(f"   ✓ Model is trained on {effective_tokens}/{total_tokens} tokens")
    print(f"   ✓ PAUSE tokens allow 'thinking' without penalty")
    print(f"   ✓ PAD tokens don't bias the model")
    print(f"   ✓ Gradients computed only for meaningful predictions")
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    
    # Show a visual representation
    print(f"\nVISUAL REPRESENTATION (first 20 tokens):")
    print(f"Position: ", end="")
    for i in range(min(20, len(sample_y))):
        print(f"{i:4d} ", end="")
    print()
    
    print(f"Original: ", end="")
    for i in range(min(20, len(sample_y))):
        token = sample_y[i]
        if token == pause_token_id:
            print(" PSE ", end="")
        elif token == pad_token_id:
            print(" PAD ", end="")
        else:
            print(f"{token:4d} ", end="")
    print()
    
    print(f"Masked:   ", end="")
    for i in range(min(20, len(y_masked))):
        token = y_masked[i].item()
        if token == -1:
            print("  -1 ", end="")
        else:
            print(f"{token:4d} ", end="")
    print()
    
    print(f"\nLegend: PSE=PAUSE, PAD=PAD, -1=masked (ignored in loss)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data/inweights_pathstar_d10_l5'
    
    demo_masking(data_dir)

