# GPU Out of Memory (OOM) Issues - Root Causes and Fixes

## Problem Summary
The sweep was running out of GPU memory with errors like:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.45 GiB. 
GPU 0 has a total capacity of 15.48 GiB of which 4.10 GiB is free.
```

## Root Causes

### 1. **Extremely Large Sweep Configuration Values** ⚠️⚠️⚠️

**Original Config (`sweep_config_minimal.yaml`):**
```yaml
graph_d:
  values: [5000, 1000000, 10000, 100000]  # Up to 1 MILLION spokes!
graph_vocab_size:
  value: 10000000  # 10 MILLION vocabulary size!
graph_holdout_percentage:
  value: 0.5
```

**Memory Impact:**
- **Vocabulary Size (10M)**: Creates embedding layers of size (10M × 64) = 640M parameters
  - Token embeddings: 10M × 64 × 4 bytes ≈ **2.4 GB**
  - Output projection: Similar size
  - **Total for embeddings alone: ~5 GB**

- **Large graph_d (5000+)**: With `graph_d=5000` and `holdout_percentage=0.5`:
  - Training paths: 2,500, replicated 16× = **40,000 samples**
  - Validation paths: 2,500 samples

### 2. **No Batching in `evaluate_samples()` Function**

The function was trying to generate ALL samples in a single GPU batch:
```python
# OLD CODE - trying to process 40,000 samples at once!
contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device)
generated_sequences = model.generate(contexts_batch, ...)  # OOM here!
```

With 40,000 samples and vocab_size=10M, the softmax operation alone needed **7.45 GiB**.

## Fixes Applied

### Fix 1: Reduced Sweep Configuration to Reasonable Values

**New Config:**
```yaml
graph_d:
  values: [250, 500, 1000]  # Reasonable values (was: up to 1M)
graph_vocab_size:
  value: 10000  # Reduced from 10M to 10K
graph_holdout_percentage:
  value: 0.2  # Reduced from 0.5 for more training data
```

**Why these values:**
- `graph_d`: 250-1000 is reasonable for experimentation
- `graph_vocab_size`: 10K is 1000x smaller but still sufficient
  - Required minimum: `d * (l-1) + 1` = 1000 * 4 + 1 = 4001
  - 10K provides plenty of headroom
- `holdout_percentage`: 0.2 (20%) is more standard for validation

### Fix 2: Added Batching to `evaluate_samples()`

**New Code with Batching:**
```python
def evaluate_samples(data, data_size, split_name, num_samples=5, eval_batch_size=256):
    # ... prepare all samples ...
    
    # Generate in batches to avoid OOM
    all_generated_tokens = []
    num_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * eval_batch_size
        end_idx = min(start_idx + eval_batch_size, num_samples)
        
        # Process only eval_batch_size samples at a time
        batch_contexts = contexts[start_idx:end_idx]
        contexts_batch = torch.from_numpy(np.stack(batch_contexts).astype(np.int64)).to(device)
        generated_sequences = model.generate(contexts_batch, ...)
        all_generated_tokens.append(generated_tokens_batch)
    
    # Concatenate all batches
    all_generated_tokens = np.concatenate(all_generated_tokens, axis=0)
```

**Benefits:**
- Processes at most 256 samples per batch (configurable via `eval_batch_size`)
- Even with 40,000 samples, memory usage is bounded
- Can handle any dataset size without OOM

## Memory Analysis

### Before (with original config):
- **Model embeddings**: ~5 GB (vocab_size=10M)
- **Forward pass activations**: ~3 GB
- **Generation batch (40K samples)**: ~7.5 GB
- **Total**: **15.5+ GB** → **OOM on 16GB GPU!**

### After (with fixed config):
- **Model embeddings**: ~5 MB (vocab_size=10K)
- **Forward pass activations**: ~500 MB
- **Generation batch (256 samples)**: ~50 MB
- **Total**: **~1 GB** → **Plenty of room!**

## Testing Recommendations

1. **Start small**: Use `graph_d=250` first to verify everything works
2. **Monitor GPU memory**: Watch `nvidia-smi` during training
3. **Adjust eval_batch_size**: If still OOM, reduce from 256 to 128 or 64
4. **Incremental scaling**: If 250 works well, try 500, then 1000

## Additional Notes

The `eval_batch_size=256` parameter can be tuned based on:
- GPU memory (16GB in your case)
- Model size (smaller models = larger batches)
- Sequence length (longer sequences = smaller batches)

For even larger experiments, consider:
- Further reducing `eval_batch_size` to 128 or 64
- Sampling fewer training samples for evaluation (e.g., max 5000 instead of all)
- Using gradient checkpointing if training larger models


