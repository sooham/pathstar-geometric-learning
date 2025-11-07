#!/bin/bash
# Example script to run parallel experiments on different GPUs
# Each experiment will have its own checkpoint file

# Experiment 1 on GPU 0 with specific hyperparameters
CUDA_VISIBLE_DEVICES=0 python train.py \
    --experiment_name="exp1_lr1e3" \
    --gpu_id=0 \
    --learning_rate=1e-3 \
    --n_layer=1 \
    --n_head=8 \
    --n_embd=96 &

# Experiment 2 on GPU 1 with different hyperparameters
CUDA_VISIBLE_DEVICES=1 python train.py \
    --experiment_name="exp2_lr1e4" \
    --gpu_id=1 \
    --learning_rate=1e-4 \
    --n_layer=2 \
    --n_head=8 \
    --n_embd=96 &

# Experiment 3 on GPU 2
CUDA_VISIBLE_DEVICES=2 python train.py \
    --experiment_name="exp3_larger_model" \
    --gpu_id=2 \
    --learning_rate=1e-3 \
    --n_layer=2 \
    --n_head=16 \
    --n_embd=128 &

# Wait for all background processes to complete
wait

echo "All experiments completed!"

