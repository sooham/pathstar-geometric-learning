# Wandb Sweep Guide for PathStar Training

This guide explains how to run hyperparameter sweeps for the PathStar geometric learning experiments.

## Overview

The refactored `train.py` now supports:
- **Standalone training** with command-line arguments
- **Wandb sweeps** for hyperparameter optimization
- **Multi-GPU parallel sweeps** for faster experimentation

## Quick Start

### 1. Single GPU Sweep

```bash
# Create and run sweep on one GPU
python3 run_sweep.py \
    --sweep_config sweep_config_minimal.yaml \
    --project pathstar_sweep \
    --count 10
```

### 2. Multi-GPU Sweep (Recommended)

```bash
# Easy way: Use the convenience script
./run_multi_gpu_sweep.sh sweep_config_minimal.yaml 5 pathstar_sweep

# Manual way:
# Step 1: Create the sweep
python3 run_sweep.py \
    --sweep_config sweep_config_minimal.yaml \
    --project pathstar_sweep \
    --create_only

# Step 2: Run agents on each GPU (in separate terminals)
CUDA_VISIBLE_DEVICES=0 python3 run_sweep.py --sweep_id <SWEEP_ID> --project pathstar_sweep --count 5 &
CUDA_VISIBLE_DEVICES=1 python3 run_sweep.py --sweep_id <SWEEP_ID> --project pathstar_sweep --count 5 &
```

## Configuration Files

### sweep_config.yaml (Full Configuration)

This file contains extensive hyperparameter ranges for thorough exploration:
- **Dataset parameters**: `graph_d`, `graph_l`, `graph_vocab_size`, `graph_holdout_percentage`, `num_pause_tokens`
- **Model architecture**: `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- **Optimization**: `learning_rate`, `label_smoothing`, `weight_decay`, `beta1`, `beta2`, `grad_clip`
- **Learning rate schedule**: `warmup_frac`, `lr_decay_frac`, `min_lr`

### sweep_config_minimal.yaml (Quick Testing)

Minimal configuration for rapid iteration with fewer parameter combinations.

## Creating Custom Sweep Configurations

Example YAML structure:

```yaml
program: train.py
method: bayes  # or 'grid', 'random'
metric:
  name: val/loss/overall
  goal: minimize

parameters:
  # Single value (fixed)
  graph_l:
    value: 5
  
  # Multiple values (sweep)
  graph_d:
    values: [50, 100, 250, 500]
  
  # Distribution sampling
  learning_rate:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
```

### Sweep Methods

1. **Grid Search** (`method: grid`)
   - Tries all combinations
   - Best for small parameter spaces
   - Example: 4 values × 3 values = 12 runs

2. **Random Search** (`method: random`)
   - Samples random combinations
   - Good for large parameter spaces
   - Specify number of runs

3. **Bayesian Optimization** (`method: bayes`)
   - Uses previous results to guide search
   - Best for expensive experiments
   - Requires fewer runs to find good parameters

## Sweepable Parameters

### Dataset Parameters

- `graph_d`: Number of spokes/paths (e.g., 5, 10, 50, 100, 500, 1000)
- `graph_l`: Path length (e.g., 5, 7, 9, 11)
- `graph_vocab_size`: Vocabulary size (must be ≥ d × (l-1) + 1)
- `graph_holdout_percentage`: Validation split (0.0 to 1.0)
- `num_pause_tokens`: Number of pause tokens (e.g., 1, 3, 5)
- `use_undirected`: true or false
- `use_directional_tokens`: true or false

### Model Architecture

- `n_layer`: Number of transformer layers (e.g., 2, 3, 4, 6, 12)
- `n_head`: Number of attention heads (e.g., 4, 8, 12)
- `n_embd`: Embedding dimension (e.g., 64, 96, 128, 192, 384)
- `dropout`: Dropout rate (0.0 to 0.3)
- `bias`: Use bias in layers (true/false)

### Training Hyperparameters

- `learning_rate`: Peak learning rate (1e-5 to 1e-2)
- `label_smoothing`: Label smoothing epsilon (0.0 to 0.2)
- `weight_decay`: AdamW weight decay (0.0 to 0.1)
- `beta1`: AdamW beta1 (typically 0.9)
- `beta2`: AdamW beta2 (0.95, 0.98, 0.99)
- `grad_clip`: Gradient clipping (0.0 = disabled, or 0.5 to 2.0)

### Learning Rate Schedule

- `decay_lr`: Enable LR decay (true/false)
- `warmup_frac`: Warmup fraction of total steps (0.05 to 0.15)
- `lr_decay_frac`: Decay phase fraction (0.95 to 1.0)
- `min_lr`: Minimum learning rate (1e-6 to 1e-4)

### Training Control

- `epochs`: Number of epochs (fixed, e.g., 5000, 10000, 50000)
- `gradient_accumulation_steps`: Gradient accumulation (typically 1)
- `eval_interval`: Evaluation frequency (e.g., 100, 200)

## Monitoring Sweeps

### Wandb Dashboard

View your sweeps at:
```
https://wandb.ai/<your-entity>/<project-name>/sweeps/<sweep-id>
```

Key metrics tracked:
- `val/loss/overall`: Overall validation loss
- `val/loss/token_{i}`: Per-token validation loss
- `val/accuracy/token_{i}`: Per-token accuracy (autoregressive)
- `train/loss/overall`: Training loss
- `lr`: Current learning rate
- `mfu`: Model FLOPs Utilization

### Local Logs

When using multi-GPU sweeps:
```bash
# Monitor GPU 0
tail -f gpu_0_sweep.log

# Monitor GPU 1
tail -f gpu_1_sweep.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Advanced Usage

### Resume Interrupted Sweep

```bash
# If agents crashed, restart them with the same sweep_id
CUDA_VISIBLE_DEVICES=0 python3 run_sweep.py \
    --sweep_id <SWEEP_ID> \
    --project pathstar_sweep \
    --count 10
```

### Run Specific Number of Experiments

```bash
# Run exactly 50 experiments total (25 per GPU)
./run_multi_gpu_sweep.sh sweep_config.yaml 25 pathstar_sweep
```

### Custom GPU Allocation

```bash
# Use only specific GPUs (e.g., GPUs 2 and 3)
CUDA_VISIBLE_DEVICES=2 python3 run_sweep.py --sweep_id <ID> --project <PROJECT> --count 5 &
CUDA_VISIBLE_DEVICES=3 python3 run_sweep.py --sweep_id <ID> --project <PROJECT> --count 5 &
```

## Standalone Training (No Sweep)

You can still run regular training without sweeps:

```bash
# Default parameters
python3 train.py

# Override parameters via command line
python3 train.py \
    --graph_d=100 \
    --graph_l=7 \
    --n_layer=4 \
    --n_embd=128 \
    --learning_rate=5e-4
```

## Troubleshooting

### Issue: "vocab_size must be >= d * (l-1) + 1"

**Solution**: Increase `graph_vocab_size` in your sweep config:
```yaml
graph_vocab_size:
  value: 50000  # Large enough for all swept graph_d and graph_l combinations
```

### Issue: Dataset regeneration on every run

**Cause**: Dataset parameters are being swept, creating new datasets each time.

**Solution**: For architecture/optimization sweeps, fix dataset parameters:
```yaml
graph_d:
  value: 100  # Fixed
graph_l:
  value: 5    # Fixed
```

### Issue: Agents not using different GPUs

**Solution**: Ensure `CUDA_VISIBLE_DEVICES` is set correctly:
```bash
# Check current setting
echo $CUDA_VISIBLE_DEVICES

# Verify in Python
python3 -c "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))"
```

### Issue: Out of memory errors

**Solutions**:
1. Reduce batch size (note: batch size is auto-calculated from dataset size in current implementation)
2. Reduce model size (`n_layer`, `n_embd`)
3. Reduce sequence length (`graph_l`)
4. Enable gradient checkpointing (requires code modification)

## Best Practices

1. **Start Small**: Begin with `sweep_config_minimal.yaml` to test your setup
2. **Fix Datasets**: Keep dataset parameters fixed for faster sweeps
3. **Use Bayesian**: For expensive experiments, `method: bayes` is more efficient than grid search
4. **Monitor Early**: Check first few runs to catch configuration errors
5. **Log Files**: Keep GPU log files for debugging crashed runs
6. **Sweep Incrementally**: Run small sweeps (10-20 runs) before large ones

## Example Workflows

### Workflow 1: Dataset Size Scaling Study

```yaml
method: grid
parameters:
  # Sweep dataset size
  graph_d:
    values: [10, 50, 100, 500, 1000, 5000]
  graph_l:
    value: 5
  
  # Fix model architecture
  n_layer:
    value: 3
  n_embd:
    value: 96
  
  # Fix optimization
  learning_rate:
    value: 1e-3
```

### Workflow 2: Architecture Search

```yaml
method: bayes
parameters:
  # Fix dataset
  graph_d:
    value: 250
  graph_l:
    value: 5
  
  # Sweep architecture
  n_layer:
    values: [2, 3, 4, 6]
  n_head:
    values: [4, 8, 12]
  n_embd:
    values: [64, 96, 128, 192]
  dropout:
    values: [0.0, 0.1, 0.2]
```

### Workflow 3: Learning Rate Optimization

```yaml
method: random
parameters:
  # Fix dataset and model
  graph_d:
    value: 250
  n_layer:
    value: 3
  n_embd:
    value: 96
  
  # Sweep optimization
  learning_rate:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  label_smoothing:
    values: [0.0, 0.05, 0.1, 0.15]
  weight_decay:
    values: [0.0, 0.001, 0.01, 0.1]
```

## Additional Resources

- [Wandb Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Bayesian Optimization Guide](https://docs.wandb.ai/guides/sweeps/sweep-config-keys#bayesian-search)
- [PathStar Paper/Repository](https://github.com/your-repo) (if applicable)

