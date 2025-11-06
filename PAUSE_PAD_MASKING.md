# PAUSE and PAD Token Masking in Training

## Overview

The training script has been modified to properly handle special tokens (`PAUSE` and `PAD`) during loss calculation. These tokens serve specific purposes and should not contribute to the training loss.

## Special Tokens

### PAUSE Token
- **Purpose**: Allows the model to "think" or process information without being penalized for predictions
- **Usage**: Inserted between the graph structure and the query to give the model computational steps
- **Location**: Between adjacency list and the start/goal nodes in sequences
- **Loss Behavior**: **NOT included in loss calculation** - the model is not trained to predict PAUSE tokens

### PAD Token
- **Purpose**: Padding sequences to uniform length for batching
- **Usage**: Added at the end of shorter sequences to match the longest sequence in a batch
- **Loss Behavior**: **NOT included in loss calculation** - padding should not affect training

## Implementation Details

### Changes to `train.py`

1. **Metadata Loading** (lines 87-113):
   ```python
   # Load special token IDs from metadata
   if 'special_tokens' in meta:
       # InContextPathStar format
       pause_token_id = meta['special_tokens'].get('PAUSE')
       pad_token_id = meta['special_tokens'].get('PAD')
   else:
       # InWeightsPathStar format
       pause_token_id = meta.get('pause_token')
       pad_token_id = meta.get('pad_token')
   ```

2. **Token Masking in `get_batch()`** (lines 115-129):
   ```python
   # Mask out PAUSE and PAD tokens in targets
   if pause_token_id is not None:
       y[y == pause_token_id] = -1
   if pad_token_id is not None:
       y[y == pad_token_id] = -1
   ```

3. **Loss Calculation** (in `model.py`, line 207):
   ```python
   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                         targets.view(-1), 
                         ignore_index=-1)
   ```
   
   The `ignore_index=-1` parameter tells PyTorch's cross-entropy loss to ignore any target tokens with value `-1`.

### Supported Dataset Formats

The implementation supports both dataset formats:

1. **InContextPathStar**: Special tokens stored in `meta['special_tokens']` dictionary
2. **InWeightsPathStar**: Special tokens stored directly as `meta['pause_token']` and `meta['pad_token']`

### Backward Compatibility

For older datasets that don't have special tokens defined:
- The script will print a warning message
- Training will proceed normally without masking
- No errors will be raised

## Example Usage

### Training with a Dataset

```bash
# Train with default settings
python train.py --dataset=inweights_pathstar_d5_l3

# The script will automatically:
# 1. Load the dataset metadata
# 2. Detect PAUSE and PAD token IDs
# 3. Mask these tokens in the loss calculation
```

### Expected Output

```
found vocab_size = 15 (inside data/inweights_pathstar_d5_l3/meta.pkl)
Loaded special tokens: PAUSE=11, PAD=12
Note: PAUSE and PAD tokens will be masked in loss calculation (ignore_index=-1)
```

## Testing

A test script `test_pause_pad_masking.py` is provided to verify the masking behavior:

```bash
python test_pause_pad_masking.py
```

This will:
1. Load a dataset and its metadata
2. Display special token information
3. Simulate the masking process on a sample batch
4. Show statistics about masked tokens

### Example Test Output

```
Special tokens (InWeightsPathStar format):
  PAUSE: 11
  PAD: 12
  Task tokens: {'PATH': 13, 'EDGE': 14}

PAUSE token (11) occurrences in first 1000 tokens: 8
PAD token (12) occurrences in first 1000 tokens: 80

============================================================
Simulating masking process:
============================================================

Original y shape: torch.Size([4, 64])
PAUSE tokens before masking: 14
PAD tokens before masking: 117

After masking: 131 tokens set to -1
Percentage of tokens masked: 51.17%
```

## Why This Matters

### Without Masking
- The model would be trained to predict PAUSE tokens, which doesn't make sense conceptually
- Padding tokens would contribute to the loss, potentially biasing the model
- The model might learn to output PAUSE/PAD tokens inappropriately

### With Masking
- The model focuses only on learning meaningful predictions (graph nodes, paths)
- PAUSE tokens serve their intended purpose: giving the model "thinking time"
- Training is more efficient and focused on the actual task

## Technical Details

### Loss Calculation Flow

1. **Data Loading**: Sequences are loaded from `.bin` files as `uint16` arrays
2. **Batch Creation**: Random sequences are sampled and converted to tensors
3. **Target Masking**: PAUSE and PAD tokens in targets are replaced with `-1`
4. **Forward Pass**: Model processes input sequences
5. **Loss Computation**: Cross-entropy loss ignores targets with value `-1`
6. **Backward Pass**: Gradients are computed only for non-masked tokens

### Memory and Performance

- Masking is done in-place on the target tensor: `y[y == pause_token_id] = -1`
- This is a fast O(n) operation with minimal memory overhead
- The masking happens on the CPU before data is transferred to GPU
- No impact on training speed or memory usage

## Configuration File

A new `configurator.py` file has been added to support command-line argument parsing:

```bash
# Override default config values
python train.py --batch_size=32 --learning_rate=1e-4 --compile=False
```

## Files Modified

1. **train.py**: Added special token loading and masking logic
2. **configurator.py**: Created for command-line argument parsing
3. **test_pause_pad_masking.py**: Created for testing the masking behavior

## Files Unchanged

- **model.py**: Already had `ignore_index=-1` in the loss calculation
- **pathstar.py**: Already generates datasets with PAUSE and PAD tokens

## Future Considerations

### Additional Special Tokens

If more special tokens need to be masked in the future:

```python
# In train.py, add to the masking section:
task_token_id = meta.get('task_tokens', {}).get('EDGE')
if task_token_id is not None:
    y[y == task_token_id] = -1
```

### Per-Token Masking Control

For more fine-grained control, you could add a configuration option:

```python
# In train.py config section:
mask_pause = True  # Whether to mask PAUSE tokens
mask_pad = True    # Whether to mask PAD tokens

# In get_batch():
if mask_pause and pause_token_id is not None:
    y[y == pause_token_id] = -1
if mask_pad and pad_token_id is not None:
    y[y == pad_token_id] = -1
```

## References

- PyTorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- The `ignore_index` parameter is standard in sequence modeling tasks (e.g., machine translation, language modeling)

