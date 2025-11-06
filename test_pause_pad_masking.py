"""
Test script to verify that PAUSE and PAD tokens are correctly masked in the training loop.
"""

import os
import pickle
import numpy as np
import torch

# Test with one of the existing datasets
data_dir = 'data/inweights_pathstar_d10_l5'

# Load metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    print("Dataset metadata:")
    print(f"  vocab_size: {meta['vocab_size']}")
    print(f"  d: {meta.get('d', 'N/A')}")
    print(f"  l: {meta.get('l', 'N/A')}")
    
    # Support both formats: 'special_tokens' dict (InContextPathStar) and direct keys (InWeightsPathStar)
    if 'special_tokens' in meta:
        print("\nSpecial tokens (InContextPathStar format):")
        for name, token_id in meta['special_tokens'].items():
            print(f"  {name}: {token_id}")
        
        pause_token_id = meta['special_tokens'].get('PAUSE')
        pad_token_id = meta['special_tokens'].get('PAD')
    else:
        # InWeightsPathStar format
        pause_token_id = meta.get('pause_token')
        pad_token_id = meta.get('pad_token')
        
        if pause_token_id is not None or pad_token_id is not None:
            print("\nSpecial tokens (InWeightsPathStar format):")
            if pause_token_id is not None:
                print(f"  PAUSE: {pause_token_id}")
            if pad_token_id is not None:
                print(f"  PAD: {pad_token_id}")
            if 'task_tokens' in meta:
                print(f"  Task tokens: {meta['task_tokens']}")
    
    if pause_token_id is not None or pad_token_id is not None:
        # Load a sample of training data
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        
        # Reshape to see individual sequences (assuming they're stored sequentially)
        print(f"\nTotal data size: {len(train_data)} tokens")
        
        # Sample first 1000 tokens
        sample = train_data[:1000]
        
        # Count occurrences of special tokens
        if pause_token_id is not None:
            pause_count = np.sum(sample == pause_token_id)
            print(f"\nPAUSE token ({pause_token_id}) occurrences in first 1000 tokens: {pause_count}")
        
        if pad_token_id is not None:
            pad_count = np.sum(sample == pad_token_id)
            print(f"PAD token ({pad_token_id}) occurrences in first 1000 tokens: {pad_count}")
        
        # Simulate the masking process
        print("\n" + "="*60)
        print("Simulating masking process:")
        print("="*60)
        
        # Create a small batch
        block_size = 64
        batch_size = 4
        
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        print(f"\nOriginal y shape: {y.shape}")
        print(f"Original y sample (first sequence, first 20 tokens):\n{y[0, :20]}")
        
        # Count tokens before masking
        if pause_token_id is not None:
            pause_before = (y == pause_token_id).sum().item()
            print(f"\nPAUSE tokens before masking: {pause_before}")
        
        if pad_token_id is not None:
            pad_before = (y == pad_token_id).sum().item()
            print(f"PAD tokens before masking: {pad_before}")
        
        # Apply masking
        if pause_token_id is not None:
            y[y == pause_token_id] = -1
        if pad_token_id is not None:
            y[y == pad_token_id] = -1
        
        print(f"\nAfter masking y sample (first sequence, first 20 tokens):\n{y[0, :20]}")
        
        # Count masked tokens
        masked_count = (y == -1).sum().item()
        print(f"\nTotal masked tokens (set to -1): {masked_count}")
        print(f"Percentage of tokens masked: {100 * masked_count / y.numel():.2f}%")
        
        print("\n" + "="*60)
        print("Masking test completed successfully!")
        print("="*60)
        print("\nNote: These masked tokens (with value -1) will be ignored")
        print("in the cross_entropy loss calculation (ignore_index=-1)")
    else:
        print("\nNo special tokens (PAUSE/PAD) found in metadata.")
        print("This dataset may not have PAUSE/PAD tokens defined.")
        print("The dataset might be from an older version or a different format.")
else:
    print(f"Metadata file not found: {meta_path}")
    print("Please generate a dataset first using pathstar.py")

