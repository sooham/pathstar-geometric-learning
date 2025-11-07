# Running Parallel Experiments on Multiple GPUs

## Overview

The training script has been modified to support running multiple experiments in parallel on different GPUs without checkpoint conflicts. Each experiment now creates a unique checkpoint file based on the experiment name and GPU ID.

## Checkpoint Naming

Checkpoints are now saved with the following format:
```
ckpt_exp_{experiment_name}_gpu_{gpu_id}.pt
```

For example:
- `ckpt_exp_exp1_lr1e3_gpu_0.pt`
- `ckpt_exp_exp2_lr1e4_gpu_1.pt`
- `ckpt_exp_exp3_larger_model_gpu_2.pt`

## New Command-Line Arguments

Two new configuration parameters have been added:

### `--experiment_name`
- **Description**: A unique identifier for your experiment
- **Default**: If not provided, uses the `wandb_run_name` value
- **Example**: `--experiment_name="exp1_lr1e3"`

### `--gpu_id`
- **Description**: GPU identifier for this experiment
- **Default**: Automatically determined from:
  1. `CUDA_VISIBLE_DEVICES` environment variable (first GPU)
  2. Current CUDA device via `torch.cuda.current_device()`
  3. Falls back to 'cpu' if no GPU is available
- **Example**: `--gpu_id=0`

## Usage Examples

### Single Experiment
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --experiment_name="my_experiment" \
    --gpu_id=0 \
    --learning_rate=1e-3
```

### Parallel Experiments on Different GPUs
```bash
# GPU 0: Low learning rate
CUDA_VISIBLE_DEVICES=0 python train.py \
    --experiment_name="exp_lr1e4" \
    --gpu_id=0 \
    --learning_rate=1e-4 &

# GPU 1: High learning rate
CUDA_VISIBLE_DEVICES=1 python train.py \
    --experiment_name="exp_lr1e3" \
    --gpu_id=1 \
    --learning_rate=1e-3 &

# GPU 2: Different architecture
CUDA_VISIBLE_DEVICES=2 python train.py \
    --experiment_name="exp_large_model" \
    --gpu_id=2 \
    --n_layer=2 \
    --n_head=16 &

wait  # Wait for all experiments to complete
```

### Resuming a Specific Experiment
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --init_from=resume \
    --experiment_name="exp_lr1e3" \
    --gpu_id=0
```

**Important**: When resuming, you must provide the same `experiment_name` and `gpu_id` that were used during initial training, or the checkpoint won't be found.

## Automatic GPU Detection

If you don't specify `--gpu_id`, the script will automatically determine it:

1. **From CUDA_VISIBLE_DEVICES**: 
   ```bash
   CUDA_VISIBLE_DEVICES=2 python train.py --experiment_name="my_exp"
   # Will use gpu_id=2
   ```

2. **From PyTorch's current device**:
   ```bash
   python train.py --experiment_name="my_exp"
   # Will use torch.cuda.current_device()
   ```

## Best Practices

1. **Always use unique experiment names** when running parallel experiments
2. **Be explicit with GPU IDs** to avoid confusion, especially when using CUDA_VISIBLE_DEVICES
3. **Use the same experiment_name and gpu_id** when resuming training
4. **Keep track of your experiments** - consider using descriptive names that include key hyperparameters
5. **Use wandb logging** to monitor all experiments simultaneously

## Example Script

See `run_parallel_experiments.sh` for a complete example of running multiple experiments in parallel.

## Troubleshooting

### Checkpoint not found when resuming
- Verify that `experiment_name` and `gpu_id` match the original training run
- Check the `out_dir` for available checkpoints: `ls out/ckpt_*.pt`

### Checkpoints still conflicting
- Ensure each parallel experiment uses a unique `experiment_name`
- Verify that `gpu_id` is being set correctly (check the script output for "Checkpoint will be saved as...")

### Multiple experiments writing to same checkpoint
- Double-check that you're passing different `--experiment_name` values to each training command
- Consider using different `--out_dir` values for completely separate experiment groups

