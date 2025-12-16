# Agents Guide to PathStar Codebase

This document provides high-level context and technical details for AI Agents working on the `pathstar-geometric-learning` codebase.

## 1. Architecture Overview

 The project experiments with Transformer models (GPT) learning graph structures, specifically "PathStar" graphs (stars with long spokes).

### Data Pipeline (`pathstar.py`)
*   **Graph Class**: `InWeightsPathStar`
    *   Structure: `d` spokes (paths) of length `l` connected to a central root.
    *   **Vocabulary**: Nodes are mapped to integer tokens. Support for `randomize_vocab_size`.
    *   **Tokens**: Includes special tokens for `PAD`, `PAUSE`, `GT` (>), `LT` (<), `PATH`, `EDGE`.
*   **Tasks**:
    1.  **Path Prediction**: Given a leaf node, predict the path to the root.
    2.  **Edge Memorization**: Memorize direct edges `(u, v)`.
*   **Output**: Generates `.bin` files (numpy arrays) for training (`paths.bin`, `edges.bin`) and validation.

### Model (`model.py`)
*   **Type**: Decoder-only Transformer (GPT-2 style).
*   **Key Classes**:
    *   `GPT`: Main model class.
    *   `CausalSelfAttention`: Standard attention mechanism (supports `torch.nn.functional.scaled_dot_product_attention` for Flash Attention).
    *   `GPTConfig`: Dataclass for configuration (layers, heads, embeddings, etc.).

### Training (`train.py`)
*   **Workflow**:
    1.  Initializes `InWeightsPathStar` to generate/load data.
    2.  Calculates optimal batch size based on GPU memory.
    3.  Runs training loop with interleaved edge and path path training if configured.
    4.  Logs metrics to Weights & Biases (WandB).
*   **Key Functions**:
    *   `train()`: Main entry point.
    *   `calculate_optimal_batch_size_for_training()`: Heuristic for memory management.
    *   `evaluate()`: Performs autoregressive sampling to check accuracy.

## 2. Automation & Sweeps

The codebase has a robust system for hyperparameter sweeps, managed by `run_multi_gpu_sweep.sh` and `run_sweep.py` (implied existence from script usage, though `run_sweep.py` logic is likely separate or imported).

*   **`run_multi_gpu_sweep.sh`**:
    *   **Feature**: Auto-detects GPUs and launches parallel agents.
    *   **Feature**: "Auto Count" for grid search - it parses the YAML, calculates total combinations, and divides them among available GPUs.
*   **Documentation Files**:
    *   `SWEEP_GUIDE.md`, `SWEEP_USAGE.md`, `AUTO_COUNT_FEATURE.md`: These contain specific logic details about how the sweep automation works. **Trust these files** when understanding the sweep system's capabilities.

## 3. Important Notes for Editing

*   **Data Generation**: When modifying `pathstar.py`, ensure that `_generate_dataset_name` is updated if new parameters are added, to avoid loading stale datasets from disk.
*   **Model Config**: `GPTConfig` in `model.py` is the source of truth for model hyperparameters.
*   **Memory**: `train.py` attempts to be smart about memory. If adding large overheads, check `calculate_optimal_batch_size_for_training`.

## 4. Files to Ignore
*   Tests: `test_*.py`
*   Infrastructure: `setup_vastai.py`
*   Visualization: `visualize_pathstar.py`
