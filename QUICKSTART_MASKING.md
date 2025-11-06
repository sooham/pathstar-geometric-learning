# Quick Start: PAUSE and PAD Token Masking

## TL;DR
PAUSE and PAD tokens are now automatically masked during training. No action required - just train as usual!

## What Changed?
- ✅ PAUSE tokens (for model "thinking") are excluded from loss
- ✅ PAD tokens (for sequence alignment) are excluded from loss
- ✅ Automatic detection from dataset metadata
- ✅ Works with both InContext and InWeights datasets

## Quick Test
```bash
# Test the masking on a dataset
python test_pause_pad_masking.py

# Expected output:
# Loaded special tokens: PAUSE=41, PAD=42
# Total masked tokens: 156 (60.94%)
# Masking test completed successfully!
```

## Training
```bash
# Train as usual - masking is automatic
python train.py --dataset=inweights_pathstar_d10_l5 --batch_size=32

# You'll see:
# Loaded special tokens: PAUSE=41, PAD=42
# Note: PAUSE and PAD tokens will be masked in loss calculation
```

## Generate New Dataset
```bash
# New datasets will include special tokens
python pathstar.py --mode=inweights --d=10 --l=5
```

## Why This Matters
**Without masking**: Model learns to predict PAUSE/PAD tokens (wrong!)  
**With masking**: Model focuses only on actual graph nodes and paths (correct!)

## Check Your Dataset
```python
import pickle

with open('data/your_dataset/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
    
# InWeightsPathStar format:
print(f"PAUSE: {meta.get('pause_token')}")
print(f"PAD: {meta.get('pad_token')}")

# InContextPathStar format:
print(f"Special tokens: {meta.get('special_tokens')}")
```

## Troubleshooting

### "No special tokens found"
- Your dataset was created with an older version
- Regenerate the dataset with current code
- Or continue training (masking will be disabled, but training works)

### Dataset too small error
- This is unrelated to masking
- The InWeights dataset format uses fixed sequences
- Use appropriate block_size for your dataset

## Documentation
- **Full details**: See `PAUSE_PAD_MASKING.md`
- **Change summary**: See `CHANGES_PAUSE_PAD.md`
- **Test script**: `test_pause_pad_masking.py`

## Questions?
The masking is implemented in `train.py` lines 87-129. It's simple:
1. Load special token IDs from metadata
2. Replace them with `-1` in target tensors
3. PyTorch's cross_entropy ignores `-1` (via `ignore_index=-1`)

That's it! Happy training! 🚀

