# PathStar Geometric Learning

This project implements geometric learning experiments on "PathStar" graphs using a GPT-based model, replicating concepts from the paper `pathstar_paper.pdf`.

## Project Overview

The core objective is to learn structural properties of a star-like graph structure where "spokes" (paths) radiate from a central root. The model is trained to perform tasks such as path completion and edge memorization.

## Key Files

### Core Implementation
*   **`pathstar.py`**: A dataset generator for the PathStar graph. It creates:
    *   Graph structure with `d` spokes of length `l`.
    *   Adjacency lists and path traversals.
    *   Training datasets for:
        *   **Edge Memorization**: `(u, v)` pairs.
        *   **Path Prediction**: Predicting sequences of nodes from leaf to root (or vice versa).
    *   Supports special tokens (PAD, PAUSE, DIRECTIONAL).
*   **`model.py`**: A full PyTorch implementation of a GPT Language Model (Transformer Decoder).
    *   Includes `CausalSelfAttention` (with optional Flash Attention).
    *   Configurable via `GPTConfig`.
*   **`train.py`**: The main training script.
    *   Handles dataset loading/generation.
    *   Implements the training loop with WandB logging.
    *   Supports teacher forcing and autoregressive evaluation.
    *   Includes memory optimization (batch size calculation).

### Sweep & Automation
*   **`run_multi_gpu_sweep.sh`**: A bash script to orchestrate WandB hyperparameter sweeps across multiple GPUs.
    *   Automatically distributes run counts for grid searches.
    *   Handles process management and cleanup.

## Paper
*   **`pathstar_paper.pdf`**: The reference paper for this implementation.

## Usage

### Training
To train the model with default settings:
```bash
python train.py
```

### Running Sweeps
To run a multi-GPU sweep:
```bash
./run_multi_gpu_sweep.sh sweep_config.yaml [project_name]
```
