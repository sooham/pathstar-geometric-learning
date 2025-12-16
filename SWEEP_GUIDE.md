# PathStar WandB Sweep Guide

This guide explains how to run hyperparameter sweeps for the PathStar geometric learning experiments. The system supports single-GPU execution and automated multi-GPU distribution.

## 1. Quick Start (Multi-GPU)

The recommended way to run sweeps is using the `run_multi_gpu_sweep.sh` script.

### Grid Search (Auto-Count)
For grid searches, the system automatically calculates the total number of combinations and distributes them evenly across all available GPUs.

```bash
# Basic usage (defaults to project 'pathstar_sweep_dataset')
./run_multi_gpu_sweep.sh sweep_config.yaml

# With custom project name
./run_multi_gpu_sweep.sh sweep_config.yaml my_project

# With custom project and entity (for teams)
./run_multi_gpu_sweep.sh sweep_config.yaml my_project my_entity
```

### Random/Bayes Search
For non-grid methods, agents will run indefinitely until stopped. You must manually monitor and kill them when sufficient data is collected.

```bash
./run_multi_gpu_sweep.sh sweep_config_bayes.yaml my_project
```

## 2. Manual & Single-GPU Usage

You can use the Python script directly for more control (e.g., running on a specific GPU or setting a specific run count).

### Single, Specific GPU
```bash
# Run 10 experiments on GPU 0
CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
    --sweep_id <sweep_id> \
    --project my_project \
    --count 10
```

### Creating a Sweep Manually
```bash
# Create sweep without running agents
python run_sweep.py --sweep_config sweep_config.yaml --create_only
```

### Resume/Join Existing Sweep
```bash
# Join a sweep by ID
python run_sweep.py --sweep_id <existing_sweep_id> --project my_project
```

## 3. How "Auto-Count" Works

When using `method: grid` in your YAML config:

1.  **Calculation**: The script parses the YAML to count all parameter combinations (e.g., `3 layers * 2 heads = 6 total runs`).
2.  **Detection**: It detects the number of available GPUs (e.g., 2 GPUs).
3.  **Distribution**: It assigns runs to each GPU.
    *   *Example*: 6 runs / 2 GPUs = 3 runs per GPU.
    *   *Example*: 5 runs / 2 GPUs = GPU 0 gets 3 runs, GPU 1 gets 2 runs.

**Note**: This feature only works for `method: grid`. For `bayes` or `random`, you must use `python run_sweep.py --count N` if you want a fixed number of runs, otherwise they run forever.

## 4. Sweep Configuration

### Example Structure (`sweep_config.yaml`)

```yaml
program: train.py
method: grid  # or 'bayes', 'random'
metric:
  name: val/loss/overall
  goal: minimize
parameters:
  # Fixed values (not counted in grid combinations)
  graph_l:
    value: 5
  
  # Swept values (counted)
  graph_d:
    values: [50, 100, 250]
  learning_rate:
    values: [1e-3, 1e-4]
```

### Key Parameters
*   **Dataset**: `graph_d`, `graph_l`, `graph_holdout_percentage`
*   **Model**: `n_layer`, `n_head`, `n_embd`, `dropout`
*   **Training**: `learning_rate`, `weight_decay`, `label_smoothing`

## 5. Troubleshooting & Fixes

### Common Issues
*   **"wandb agent cannot be found"**: Ensure you have run `wandb login`. Check that the project name matches exactly.
*   **grep errors on macOS**: The script has been patched to use POSIX-compliant headers compatible with both macOS and Linux.
*   **OOM Errors**: Reduce `max_batch_size` implicitly by reducing model size or `graph_l` in your config.

### Cleaning Up
To stop all sweep agents:
```bash
# Kill all sweep processes
pkill -f run_sweep.py
```

## 6. Architecture

*   **`run_multi_gpu_sweep.sh`**: Orchestrator. Creates the sweep, detects GPUs, and launches background Python agents.
*   **`run_sweep.py`**: The agent. Connects to WandB, executes `train.py`, and handles signal interrupts (SIGINT/SIGTERM) for graceful shutdown.
*   **`train.py`**: The training application.

---
*Consolidated from SWEEP_GUIDE.md, SWEEP_USAGE.md, SWEEP_FIXES.md, and AUTO_COUNT_FEATURE.md*
