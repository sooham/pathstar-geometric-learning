# Training Script Refactoring Summary

## Overview

The `train.py` script has been refactored to support **wandb sweeps** for hyperparameter optimization and **multi-GPU parallel execution**. This enables systematic exploration of dataset parameters, model architectures, and training hyperparameters.

## What Changed

### 1. Core Refactoring: train.py

**Before**: Monolithic script with hard-coded parameters
**After**: Modular design with function-based architecture

#### Key Changes:

1. **Wrapped training in `train()` function**
   - Takes optional `config` dict parameter
   - Supports both standalone and sweep modes
   - Merges config with defaults properly

2. **Added `get_default_config()` function**
   - Centralizes all default parameters
   - Returns comprehensive config dict
   - Makes defaults explicit and documented

3. **Added `sweep_train()` wrapper**
   - Entry point for wandb sweep agents
   - Automatically receives sweep hyperparameters
   - Calls `train()` with wandb.config

4. **Reorganized parameter initialization**
   - Dataset generation happens after config is finalized
   - Proper handling of command-line overrides
   - All dependent calculations moved to correct location

### 2. New Files Created

#### sweep_config.yaml
Comprehensive sweep configuration with extensive parameter ranges:
- Dataset: `graph_d`, `graph_l`, `graph_vocab_size`, `graph_holdout_percentage`
- Architecture: `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- Optimization: `learning_rate`, `label_smoothing`, `weight_decay`, etc.
- Methods: Supports `bayes`, `grid`, and `random` search

#### sweep_config_minimal.yaml
Lightweight config for quick testing:
- Sweeps only `graph_d` (6 values)
- Fixes other parameters for fast iteration
- Grid search method (6 total runs)
- Useful for testing setup before large sweeps

#### run_sweep.py
Python script for managing sweeps:
- Create new sweeps from YAML config
- Join existing sweeps (multi-GPU support)
- Set GPU via `--gpu_id` argument
- Supports `--count` for limited runs
- `--create_only` for sweep creation without running

#### run_multi_gpu_sweep.sh
Bash convenience script:
- Automatically creates sweep
- Launches agents on all available GPUs
- Logs each GPU to separate file
- Handles background processes
- Simple usage: `./run_multi_gpu_sweep.sh config.yaml 10`

#### SWEEP_GUIDE.md
Comprehensive documentation:
- Quick start examples
- Parameter explanations
- Best practices
- Troubleshooting guide
- Example workflows for common experiments

### 3. Fixed Issues

#### Syntax Errors in sweep_config_minimal.yaml
- **Problem**: Mixed `value:` with lists `value: [...]`
- **Fix**: Use `values: [...]` for lists, `value: x` for single values
- **Impact**: Sweeps now parse correctly

#### Wandb Integration Issues
- **Problem**: Incorrect detection of sweep mode
- **Fix**: Created proper `sweep_train()` wrapper function
- **Impact**: Wandb agents can now call training correctly

#### Multi-GPU Support
- **Problem**: No mechanism to distribute runs across GPUs
- **Fix**: Added sweep ID sharing and CUDA_VISIBLE_DEVICES handling
- **Impact**: Multiple agents can work on same sweep in parallel

## How to Use

### Standalone Training (No Changes Required)

```bash
# Still works exactly as before
python3 train.py

# With command-line overrides
python3 train.py --graph_d=100 --learning_rate=5e-4
```

### Single GPU Sweep

```bash
python3 run_sweep.py \
    --sweep_config sweep_config_minimal.yaml \
    --project pathstar_sweep_dataset \
    --count 10
```

### Multi-GPU Sweep (Recommended)

```bash
# Easy way
./run_multi_gpu_sweep.sh sweep_config_minimal.yaml 5 pathstar_sweep_dataset

# Manual way
# Step 1: Create sweep
python3 run_sweep.py \
    --sweep_config sweep_config_minimal.yaml \
    --project pathstar_sweep_dataset \
    --create_only

# Step 2: Launch agents on each GPU
CUDA_VISIBLE_DEVICES=0 python3 run_sweep.py --sweep_id <ID> --project pathstar_sweep_dataset --count 5 &
CUDA_VISIBLE_DEVICES=1 python3 run_sweep.py --sweep_id <ID> --project pathstar_sweep_dataset --count 5 &
```

## Architecture Details

### Parameter Flow

```
1. get_default_config() → Default values
2. wandb.config OR configurator.py → Overrides
3. generate_dataset_if_needed() → Create/load dataset
4. Calculate dependent parameters → batch_size, block_size, etc.
5. Initialize model and optimizer
6. Training loop
```

### Sweep Execution Flow

```
1. run_sweep.py creates sweep → wandb server stores config
2. wandb agent pulls next hyperparameter combination
3. agent calls sweep_train() with hyperparameters
4. sweep_train() calls train(config=dict(wandb.config))
5. train() executes with sweep hyperparameters
6. Results logged to wandb
7. Repeat steps 2-6 until count reached or sweep complete
```

### Multi-GPU Parallelism

```
GPU 0: Agent 1 → Run 1 → Run 3 → Run 5 → ...
GPU 1: Agent 2 → Run 2 → Run 4 → Run 6 → ...

Both agents:
- Share same sweep_id
- Pull next hyperparameters from wandb server
- Run independently
- Log to shared wandb project
- No coordination required (wandb handles it)
```

## Backward Compatibility

✅ **Fully backward compatible**
- Old command-line usage still works
- configurator.py still works
- No changes to model.py or pathstar.py required
- Checkpoint format unchanged

## Testing Checklist

- [x] Standalone training works
- [x] Command-line arguments work
- [x] Sweep YAML parses correctly
- [x] train.py imports successfully
- [ ] Single-GPU sweep runs (requires wandb login)
- [ ] Multi-GPU sweep runs (requires 2+ GPUs)
- [ ] Dataset generation works in sweep mode
- [ ] Metrics log correctly to wandb

## Performance Improvements

### Before:
- Manual hyperparameter tuning
- One experiment at a time
- No systematic parameter exploration
- Manual result tracking

### After:
- Automated hyperparameter search
- Parallel experiments on multiple GPUs
- Systematic sweep with Bayesian/grid/random search
- Automatic wandb logging and tracking
- Easy comparison of results

### Example Speed-up:
- **Before**: 100 experiments × 2 hours each = 200 hours (8.3 days)
- **After**: 100 experiments ÷ 2 GPUs × 2 hours each = 100 hours (4.2 days)
- **Speedup**: 2x (scales linearly with GPU count)

## Common Use Cases

### 1. Dataset Scaling Study
Sweep `graph_d` and `graph_l` to understand how performance scales with graph size.

### 2. Architecture Search
Find optimal `n_layer`, `n_head`, `n_embd` for your dataset.

### 3. Learning Rate Tuning
Use Bayesian search to find best `learning_rate`, `warmup_frac`, `min_lr`.

### 4. Regularization Study
Sweep `dropout`, `label_smoothing`, `weight_decay` to prevent overfitting.

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic batch sizing**: Adjust batch size based on model size to maximize GPU utilization
2. **Early stopping**: Terminate poor-performing runs early to save compute
3. **Resume from checkpoint**: Continue sweeps from interrupted runs
4. **Curriculum learning**: Sweep training curriculum parameters
5. **Multi-node sweeps**: Distribute across multiple machines
6. **Hyperband scheduling**: Adaptive resource allocation

## Files Modified

- ✏️ `train.py` - Refactored for sweeps
- 📄 `sweep_config.yaml` - New sweep configuration
- 📄 `sweep_config_minimal.yaml` - New minimal sweep config
- 📄 `run_sweep.py` - New sweep runner
- 📄 `run_multi_gpu_sweep.sh` - New multi-GPU script
- 📄 `SWEEP_GUIDE.md` - New documentation
- 📄 `REFACTORING_SUMMARY.md` - This file
- 💾 `train_old.py` - Backup of original (can be deleted)

## Migration Guide

If you have existing training scripts or workflows:

### Option 1: No Changes
Continue using `python3 train.py` as before. Everything still works.

### Option 2: Adopt Sweeps Gradually
1. Start with `sweep_config_minimal.yaml` for testing
2. Run single-GPU sweeps first
3. Scale to multi-GPU when comfortable
4. Create custom sweep configs for your experiments

### Option 3: Full Migration
1. Convert your hyperparameter lists to YAML configs
2. Use `run_multi_gpu_sweep.sh` for all experiments
3. Monitor all runs in wandb dashboard
4. Analyze results using wandb's tools

## Support

For issues or questions:
1. Check `SWEEP_GUIDE.md` for common problems
2. Verify YAML syntax with: `python3 -c "import yaml; yaml.safe_load(open('your_config.yaml'))"`
3. Test imports with: `python3 -c "from train import sweep_train; print('OK')"`
4. Check wandb setup: `wandb login` and `wandb status`




