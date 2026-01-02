"""
This training script runs on a single GPU and supports wandb sweeps.
Trains on a combined dataset of edges and paths with interleaved sampling.

To run standalone:
$ python train.py --batch_size=32 --compile=False

To run with wandb sweep:
$ wandb sweep sweep_config.yaml
$ wandb agent <sweep_id>
"""

from datetime import datetime
from collections import defaultdict
import random
import wandb
import os
import time
import math
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from model import GPTConfig, GPT
from pathstar import InWeightsPathStar, add_pause_tokens_to_batch, add_pause_tokens_to_edges
from learning_rate_scheduler import get_lr, initialize_lr_scheduler

from live_display import LiveTrainingPanel, get_rich_token_str
from utils import clear_gpu_memory, get_git_commit_id, detect_device, set_dtype, compute_token_colors


# GOOD
def get_default_config():
    """
    Returns default configuration for training.
    This can be overridden by command-line args, config files, or wandb sweep.
    """
    return {
        # I/O
        'init_from': 'scratch',  # 'scratch' or 'resume'
        'out_dir': 'out',
        'eval_interval': 10,
        'print_eval_interval': 100, # Interval to print generated samples
        'log_interval': 1,
        'eval_only': False,
        'always_save_checkpoint': True,
        
        # wandb logging
        'wandb_log': True,
        'wandb_project': 'pathstar_interleave',
        'wandb_run_name': None,  # Will be auto-generated
        
        # Visualization
        'live_display': True,  # If True, show Rich Live display with training slices, metrics, etc.
        'vis_interval': 100, # Interval to update training slice visualization
        'show_training_slices': False,  # If True, show training batch slices in live display
        'log_attention_maps': False,  # If True, log attention map heatmaps to wandb
        'attention_map_interval': 500,  # How often to log attention maps (iterations)
        'attention_map_samples': 3,  # Number of samples to visualize
        'log_activation_stats': True,  # If True, log activation mean/variance per layer to wandb
        'analyze_embedding_geometry': False,  # If True, compute and log embedding geometry metrics during eval
        'show_edge_memorization_metrics': False, # If True, show and log % of edges memorized by model (works in both normal and edge_only modes)
        # Debugging
        'debug_masking': False,          # If True, show target masks applied to Y
        'debug_masking_samples': 2,      # How many batch rows to show
        'debug_masking_max_len': 32,     # Max tokens to show per row
        'detect_anomaly': True,         # If True, enable PyTorch anomaly detection (slow but thorough)
        'check_nan_interval': 50,        # Check for NaNs every N iterations (0 = disabled)
        
        # Dataset generation parameters
        'graph_d': 1000,
        'graph_l': 5,
        'randomize_vocab_size': 'auto',
        'graph_holdout_percentage': 0.2,
        'num_pause_tokens': 5,
        'use_undirected': True,
        'use_directional_tokens': False,
        # If True, PATH task sequences interleave GT tokens between edges:
        #   [PATH] leaf (PAUSE)xN root GT n2 GT n3 ... GT leaf
        'use_task_tokens_in_path': False,
        
        # Scheduled sampling / autoregressive substitution for PATH tasks:
        # During training, with probability p_autoregressive_substitution, substitute
        # teacher-forced tokens with the model's own predictions. This helps bridge
        # the gap between teacher forcing and autoregressive inference.
        # 0.0 = pure teacher forcing, 1.0 = pure autoregressive (no teacher forcing)
        # Only applies to PATH tasks, not EDGE tasks.
        'p_autoregressive_substitution': 0.0,
        
        # Training parameters
        'gradient_accumulation_steps': 1,
        # If set, this caps the memory-based auto batch size.
        # This is the per-step microbatch size; effective batch size is
        #   batch_size * gradient_accumulation_steps.
        # Keep this <= the auto-computed value to avoid OOM.
        'batch_size': None,
        'epochs': 1000,
        # Early termination when val loss falls below this threshold (None = disabled)
        'target_val_loss': None,
        # Evaluation batch sizes (kept separate from training batch_size)
        'eval_batch_size': 5000,
        'edge_eval_batch_size': 5000,
        
        # Model architecture
        'n_layer': 3,
        'n_head': 8,
        'n_embd': 96,
        'dropout': 0.0,  # Dropout for attention, MLP, and residual connections
        'use_layernorm': True,
        'use_mlp': True,
        'use_pos_embeddings': True,  # If True, use positional embeddings. If False, no positional information
        'activation': 'GELU',
        'embd_dropout': 0.0,
        'holdout_percentage': 0.0, # Percentage of paths to hold out for validation
        'balance_interleaved_datasets': True, # If True, upsample smaller dataset (paths) to match larger (edges)
        'edge_only': False, # If True, train only on edges (no paths, no validation)
        'bias': False,
        
        # Optimization
        'learning_rate': 1e-3,
        'label_smoothing': 0,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Learning rate schedule
        # 'lr_scheduler': None, 'CosineLR', or 'ReduceLROnPlateau'
        # None = constant learning rate (no decay)
        'lr_scheduler': 'CosineLR',
        'warmup_frac': 0.10,  # Fraction of max_iters for warmup (CosineLR and ReduceLROnPlateau)
        'lr_decay_frac': 0.99,  # Fraction of max_iters for decay (CosineLR only)
        'min_lr': 6e-5,  # Minimum learning rate
        # ReduceLROnPlateau parameters (only used when lr_scheduler='ReduceLROnPlateau')
        'plateau_factor': 0.5,  # Factor to reduce LR by (new_lr = lr * factor)
        'plateau_patience': 10,  # Number of eval intervals with no improvement before reducing LR
        'plateau_threshold': 1e-4,  # Threshold for measuring improvement (relative)
        'plateau_cooldown': 0,  # Number of eval intervals to wait after reducing LR before resuming normal operation
        
        # System
        'device': 'auto',  # 'cuda', 'mps', 'cpu', or 'auto'
        'dtype': 'auto',  # 'float32', 'bfloat16', 'float16', or 'auto'
        'compile': True,
        'gpu_id': None,
        'experiment_name': None,
        # seed
        'seed': 1337,
        'predict_direction_for_edge_task': True,

        'console': LiveTrainingPanel.CONSOLE
    }

@torch.compile
def compute_per_token_loss_with_teacher_forcing(meta, logits, input, targets, token_positions_to_record, task_type='path'):
    """
    Compute per-token loss for specified token positions
    Assumes Teacher Forcing

    Args:
        task_type: 'edge' or 'path' to indicate which task type
    Returns a dictionary that is 1-indexed with the values being the loss of the token at that position
    with teacher forcing 
    """

    per_token_losses = {}
    
    # Compute context length per input based on task type
    # Use task tokens (PATH/EDGE prefix) to determine context length
    context_length_per_input = torch.where(
        input[:, 0] == meta['special_tokens']['EDGE'],
        torch.tensor(1 + (1 if meta['use_directional_tokens'] else 0) + 1, device=input.device),
        torch.where(
            input[:, 0] == meta['special_tokens']['PATH'],
            torch.tensor(1 + 1 + meta['num_pause_tokens'], device=input.device),
            torch.tensor(0, device=input.device)
        )
    ).unsqueeze(1)
    
    for token_pos in token_positions_to_record:
        y_idx = context_length_per_input + token_pos - 2 # TODO: I think there is an issue here
        y_idx = y_idx.squeeze(1)
        
        valid_idx_mask = y_idx < targets.size(1)
        invalid_idx_mask = y_idx >= targets.size(1)

        if invalid_idx_mask.any():
            raise ValueError("Issue with indexing in compute_per_token_loss_with_teacher_forcing")
        
        if valid_idx_mask.any():
            batch_size_local = logits.size(0)
            # Match dtype of logits to avoid dtype mismatch with mixed precision training
            logits_at_pos = torch.zeros(batch_size_local, logits.size(2), device=logits.device, dtype=logits.dtype)
            targets_at_pos = torch.full((batch_size_local,), -1, dtype=targets.dtype, device=targets.device)
            
            # Vectorized indexing instead of Python loop
            batch_indices = torch.arange(batch_size_local, device=logits.device)
            valid_batch_indices = batch_indices[valid_idx_mask]
            valid_y_idx = y_idx[valid_idx_mask]
            
            # Use advanced indexing to gather logits and targets
            # TODO: debug this for gods sake
            logits_at_pos[valid_idx_mask] = logits[valid_batch_indices, valid_y_idx, :]
            targets_at_pos[valid_idx_mask] = targets[valid_batch_indices, valid_y_idx]
            
            valid_mask = targets_at_pos != -1
            if valid_mask.any():
                logits_valid = logits_at_pos[valid_mask]
                targets_valid = targets_at_pos[valid_mask]
                
                token_loss = F.cross_entropy(logits_valid, targets_valid, reduction='mean')
                per_token_losses[token_pos] = (token_loss.item(), batch_size_local)
            else:
                raise ValueError("Issue at line 168")
        else:
            raise ValueError("Issue at line 170")
    
    return per_token_losses


def forward_with_scheduled_sampling(model, X, Y, meta, p_sub, label_smoothing=0.0, ctx=None):
    """
    Forward pass with scheduled sampling (autoregressive substitution) for PATH tasks.
    
    For PATH sequences, after the context portion, with probability p_sub at each position,
    substitute the ground truth input token with the model's prediction from the previous
    position. This helps the model learn to recover from its own prediction errors during
    autoregressive inference.
    
    EDGE sequences always use standard teacher forcing (no substitution).
    
    Loss masking logic:
        For sequence: x_root, ..., x_a, x_b, x_c, ..., x_leaf
        If x_b is substituted with model's prediction:
        - Loss for predicting x_b (given x_a) is MASKED - the target would be the model's
          own prediction, not ground truth, so computing loss is meaningless.
        - Loss for predicting x_c (given substituted x_b) is KEPT - we want the model
          to learn to predict correctly even when its input may be wrong.
    
    Args:
        model: GPT model
        X: input tokens (batch, seq_len) - teacher-forced inputs
        Y: target tokens (batch, seq_len) - targets with masking (-1 for ignored positions)
        meta: dataset metadata containing special_tokens and path_context_length
        p_sub: probability of using model's prediction instead of ground truth (0.0 to 1.0)
        label_smoothing: label smoothing for cross entropy loss
        ctx: autocast context manager (e.g., torch.amp.autocast)
    
    Returns:
        logits: (batch, seq_len, vocab_size)
        loss: scalar loss tensor
    
    Note:
        When p_sub > 0, this function processes the prediction portion step-by-step,
        which is slower than standard teacher forcing but produces more robust models.
        The time complexity is O(prediction_length) forward passes for PATH sequences.
    """
    
    if ctx is None:
        ctx = nullcontext()
    
    # If no scheduled sampling, use standard teacher forcing
    if p_sub <= 0:
        with ctx:
            return model(X, Y, label_smoothing=label_smoothing)
    
    batch_size, seq_len = X.shape
    device = X.device
    
    # Identify PATH vs EDGE sequences using the task token at position 0
    PATH_TOKEN = meta['special_tokens']['PATH']
    is_path = (X[:, 0] == PATH_TOKEN)
    
    # If no PATH sequences in batch, use standard teacher forcing
    if not is_path.any():
        with ctx:
            return model(X, Y, label_smoothing=label_smoothing)
    
    # Get context length for PATH tasks
    # Context = <PATH> + leaf + <PAUSE> tokens = 1 + 1 + num_pause_tokens
    path_context_length = meta['path_context_length']
    
    # Clone X and Y for modification
    # - X_modified: we'll substitute tokens for PATH sequences
    # - Y_modified: we'll mask out targets for substituted positions
    X_modified = X.clone()
    Y_modified = Y.clone()
    
    # Process step by step for positions after context
    # For each position pos in [path_context_length, seq_len):
    #   - Get model's prediction based on X_modified[:, :pos]
    #   - With probability p_sub (for PATH sequences only), substitute X_modified[:, pos] with prediction
    #   - When substituting, also mask Y_modified[pos-1] since the target is now model's prediction
    #
    # Indexing explanation:
    #   - X[:, pos] is the input token at position pos
    #   - Y[:, pos-1] is the target for predicting what goes at position pos (i.e., Y[pos-1] = ground_truth[pos])
    #   - When we substitute X[pos], Y[pos-1] should be masked because:
    #     * The model predicted some token at pos-1 to produce the substituted X[pos]
    #     * Computing loss against ground_truth[pos] doesn't make sense since we're using model's prediction
    #   - Y[pos] (predicting ground_truth[pos+1] given substituted X[pos]) is still valid and KEPT
    #     * We want the model to learn to predict correctly even from potentially wrong inputs
    
    for pos in range(path_context_length, seq_len):
        # Get predictions based on sequence up to position pos
        # The model will output logits for predicting the next token after each input position
        # We want the prediction at position pos-1 (which predicts what should go at position pos)
        with torch.no_grad():
            with ctx:
                logits_partial, _ = model(X_modified[:, :pos], targets=None)
        
        # logits_partial has shape (batch, pos, vocab_size)
        # The last position (pos-1 in 0-indexed) predicts the token at position pos
        predicted = logits_partial[:, -1, :].argmax(dim=-1)  # (batch,)
        
        # Determine which samples to substitute (only PATH sequences, with probability p_sub)
        should_substitute = (torch.rand(batch_size, device=device) < p_sub) & is_path
        
        # Substitute in X_modified for position pos
        X_modified[should_substitute, pos] = predicted[should_substitute]
        
        # Mask out the target Y[pos-1] for substituted samples
        # Y[pos-1] is the target for predicting X[pos], which we've replaced with model's prediction
        # Computing loss against ground truth here doesn't make sense anymore
        if pos > 0:
            Y_modified[should_substitute, pos - 1] = -1  # -1 is cross_entropy's ignore_index
    
    # Final forward pass with modified input and masked targets to compute loss
    with ctx:
        logits, loss = model(X_modified, Y_modified, label_smoothing=label_smoothing)
    
    return logits, loss


def create_attention_map_figures(model, X, itos, meta, num_samples=3):
    """
    Create attention map heatmaps for wandb logging.
    
    Args:
        model: the GPT model
        X: input tensor of shape (batch_size, seq_len)
        itos: index-to-string mapping for tokens
        meta: metadata dict
        num_samples: number of samples to visualize
        
    Returns:
        dict of wandb.Image objects keyed by layer/head
    """
    model.eval()
    with torch.no_grad():
        # Get attention maps for first num_samples
        X_subset = X[:num_samples]
        attention_maps = model.get_attention_maps(X_subset)
    model.train()
    
    images = {}
    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]
    seq_len = attention_maps[0].shape[2]
    
    # Get token labels for axes
    def get_token_label(token_id):
        token_id = int(token_id)
        if token_id in itos:
            return str(itos[token_id])[:8]  # Truncate long labels
        return str(token_id)
    
    # Create figures for each sample
    for sample_idx in range(num_samples):
        token_labels = [get_token_label(t) for t in X_subset[sample_idx].cpu().numpy()]
        
        # Option 1: Per-layer averaged across heads
        fig_layers, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        
        for layer_idx, attn in enumerate(attention_maps):
            # Average across heads for this sample
            attn_avg = attn[sample_idx].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
            
            im = axes[layer_idx].imshow(attn_avg, cmap='viridis', aspect='auto')
            axes[layer_idx].set_title(f'Layer {layer_idx}')
            axes[layer_idx].set_xlabel('Key Position')
            axes[layer_idx].set_ylabel('Query Position')
            
            # Add token labels if sequence is short enough
            if seq_len <= 20:
                axes[layer_idx].set_xticks(range(seq_len))
                axes[layer_idx].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=6)
                axes[layer_idx].set_yticks(range(seq_len))
                axes[layer_idx].set_yticklabels(token_labels, fontsize=6)
            
            plt.colorbar(im, ax=axes[layer_idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Maps (Sample {sample_idx}, Head-Averaged)')
        plt.tight_layout()
        images[f'attention/sample_{sample_idx}_layers'] = wandb.Image(fig_layers)
        plt.close(fig_layers)
        
        # Option 2: All heads for the last layer (usually most interpretable)
        last_layer_attn = attention_maps[-1][sample_idx]  # (n_heads, seq_len, seq_len)
        
        # Create grid of heads
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig_heads, axes_heads = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_heads == 1:
            axes_heads = np.array([[axes_heads]])
        axes_heads = np.atleast_2d(axes_heads)
        
        for head_idx in range(n_heads):
            row, col = head_idx // n_cols, head_idx % n_cols
            attn_head = last_layer_attn[head_idx].cpu().numpy()
            
            im = axes_heads[row, col].imshow(attn_head, cmap='viridis', aspect='auto')
            axes_heads[row, col].set_title(f'Head {head_idx}', fontsize=8)
            
            if seq_len <= 15:
                axes_heads[row, col].set_xticks(range(seq_len))
                axes_heads[row, col].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=5)
                axes_heads[row, col].set_yticks(range(seq_len))
                axes_heads[row, col].set_yticklabels(token_labels, fontsize=5)
        
        # Hide empty subplots
        for idx in range(n_heads, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes_heads[row, col].axis('off')
        
        plt.suptitle(f'Last Layer Attention Heads (Sample {sample_idx})')
        plt.tight_layout()
        images[f'attention/sample_{sample_idx}_last_layer_heads'] = wandb.Image(fig_heads)
        plt.close(fig_heads)
    
    return images


def evaluate_edge_memorization(ctx, model, meta, edges_data_np, device, batch_size=512):
    """
    Evaluate the percentage of edges memorized by the model.
    
    Args:
        ctx: autocast context
        model: the model to evaluate
        edges_data_np: numpy array of edge sequences (shape: [num_edges, seq_length])
        device: device to run evaluation on
        batch_size: batch size for evaluation
    
    Returns:
        edge_memorization_pct: percentage of edges where final token is correctly predicted
    """
    model.eval()
    
    num_edges = len(edges_data_np)
    num_batches = int(np.ceil(num_edges / batch_size))
    
    correct_predictions = 0
    total_predictions = 0
    
    # Debug: Print first batch details
    print(f"\n=== Edge Memorization Evaluation Debug ===")
    print(f"Total edges: {num_edges}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"use_directional_tokens: {meta.get('use_directional_tokens', False)}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_edges)
        
        # Get batch of edge sequences
        batch = torch.from_numpy(edges_data_np[start_idx:end_idx].astype(np.int64)).to(device)
        
        predict_dir = bool(meta.get('predict_direction_for_edge_task', False))
        if predict_dir:
            pos = 1 + 2  # EDGE token + u + v
        else:
            pos = 1 + (1 if meta.get('use_directional_tokens', False) else 0) + 1

        X = batch[:, :pos]
        Y_true = batch[:, pos]
        
        with ctx:
            with torch.no_grad():
                # Get model predictions
                logits, _ = model(X, None)  # No targets needed for inference
                
                # Get predictions for the last position (final token)
                final_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                predictions = torch.argmax(final_logits, dim=-1)  # Shape: [batch_size]
                
                
                # Count correct predictions
                correct = (predictions == Y_true).sum().item()
                correct_predictions += correct
                total_predictions += len(Y_true)
    
    model.train()
    
    # Calculate percentage
    edge_memorization_pct = (correct_predictions / total_predictions) * 100.0 if total_predictions > 0 else 0.0
    
    # print(f"\n=== Edge Memorization Results ===")
    # print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    # print(f"Edge memorization: {edge_memorization_pct:.2f}%")
    # print(f"=====================================\n")
    
    return edge_memorization_pct

def compute_per_token_accuracy_autoregressive(ctx, model, meta, val_data_batch, num_samples, device_local, print_samples=False):
    """Compute per-token accuracy using autoregressive generation"""
    sample_indices = np.random.choice(len(val_data_batch), size=min(num_samples, len(val_data_batch)), replace=False)
    
    # Context length: task token + leaf + pause tokens
    context_length = 2 + meta['num_pause_tokens']
    
    contexts = []
    ground_truths = []
    path_target_len = int(meta.get('path_target_length', meta['l']))
    
    for val_idx in sample_indices:
        full_sequence = val_data_batch[val_idx]
        context = full_sequence[:context_length]
        contexts.append(context)
        ground_truth = full_sequence[context_length:context_length + path_target_len]
        ground_truths.append(ground_truth)
    
    contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device_local)
    
    model.eval()
    with torch.no_grad():
        with ctx:
            generated_sequences = model.generate(contexts_batch, max_new_tokens=path_target_len, temperature=1.0, top_k=1)
            generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
            model_context_prediction_batch = generated_sequences[:, :context_length].cpu().numpy()
    model.train()
    
    # Generate formatted text for Live display instead of printing
    generated_text_output = None
    
    ground_truths_array = np.stack(ground_truths)
    if print_samples:
        itos = meta['itos']
        
        # We need to map the raw tokens to which set they belong to for coloring
        # We can implement a helper or reuse the logic. 
        # Since we are inside a function, let's use a simpler heuristic or passed config if possible.
        # But wait, meta has 'token_colors' which uses ANSI codes. Rich doesn't parse ANSI codes inside Text objects nicely unless we tell it to.
        # Alternatively, we can strip ANSI and use Rich styles. 
        # Or simpler: Just construct a string and let Rich process it? No, Rich works best with Text objects for partial coloring.
        
        # Let's fallback to string construction but compatible with Rich (or just standard string if we put it in a Panel)
        # Actually Panel accepts a string.
        
        lines = []
        lines.append("=== Generated vs Ground Truth ===")

        for sample_idx in range(len(generated_tokens_batch)):
            
            # Generated
            gen_tokens_str_list = []
            for token in generated_tokens_batch[sample_idx]:
                gen_tokens_str_list.append(get_rich_token_str(token, itos, meta))
            generated_str = " ".join(gen_tokens_str_list)
            
            # Ground Truth
            gt_tokens_str_list = []
            for token in ground_truths_array[sample_idx]:
                gt_tokens_str_list.append(get_rich_token_str(token, itos, meta))
            ground_truth_str = " ".join(gt_tokens_str_list)

            # Model Context
            if 'model_context_prediction_batch' in locals():
                mc_tokens_str_list = []
                for token in model_context_prediction_batch[sample_idx]:
                    mc_tokens_str_list.append(get_rich_token_str(token, itos, meta))
                model_context_str = " ".join(mc_tokens_str_list)
            else:
                model_context_str = "N/A"
            
            # Input Context
            ic_tokens_str_list = []
            for token in contexts[sample_idx]:
                ic_tokens_str_list.append(get_rich_token_str(token, itos, meta))
            input_context_str = " ".join(ic_tokens_str_list)
            
            if sample_idx < 5:
                lines.append(f"Sample {sample_idx}:")
                lines.append(f"  Generated:    {model_context_str} {generated_str}")
                lines.append(f"  Ground Truth: {input_context_str} {ground_truth_str}")
        
        generated_text_output = "\n".join(lines)
    
    per_token_accuracies = {}
    
    for token_pos in range(1, path_target_len + 1):
        idx = token_pos - 1
        if idx < generated_tokens_batch.shape[1] and idx < ground_truths_array.shape[1]:
            matches = generated_tokens_batch[:, idx] == ground_truths_array[:, idx]
            accuracy = np.mean(matches)
            per_token_accuracies[token_pos] = accuracy
        else:
            per_token_accuracies[token_pos] = float('nan')
    
    # Compute full path exact match accuracy (entire generated path matches ground truth)
    full_path_matches = np.all(generated_tokens_batch == ground_truths_array, axis=1)
    full_path_accuracy = np.mean(full_path_matches)
    
    return per_token_accuracies, generated_text_output, full_path_accuracy

# DONE 
def generate_samples_autoregressive(device, ctx, model, meta, data, data_size, split_name, num_samples=5, eval_batch_size=512):
    """
    Evaluate autoregressive generation on samples from a dataset.
    Assumes data is path-only (no filtering needed).
    
    Args:
        device: device name
        ctx: context 
        model: the model 
        meta: the dictionary of graph parameters and dataset parameters
        data: Dataset to sample from (assumed to be path-only)
        data_size: Size of the dataset
        split_name: Name of the split for logging ('train' or 'val')
        num_samples: Number of samples to evaluate
        eval_batch_size: Batch size for generation to avoid OOM (default: 512, optimized for RTX 3090)
    
    Returns:
        avg_accuracy: Average accuracy across all samples
    """
    num_samples = min(num_samples, data_size)
    eval_batch_size = min(eval_batch_size, num_samples)
    
    # Sample randomly without replacement (data is already path-only)
    sample_indices = np.random.choice(data_size, size=num_samples, replace=False)
    
    # Context length: task token + leaf + pause tokens
    context_length = 2 + meta['num_pause_tokens']
    contexts = []
    ground_truths = []
    path_target_len = int(meta.get('path_target_length', meta['l']))
    
    for idx in sample_indices:
        full_sequence = data[idx]
        context = full_sequence[:context_length]
        contexts.append(context)
        ground_truth = full_sequence[context_length:context_length + path_target_len]
        ground_truths.append(ground_truth)
    
    # Generate in batches to avoid OOM
    all_generated_tokens = []
    num_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
    
    model.eval()
    with torch.no_grad():
        with ctx:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * eval_batch_size
                end_idx = min(start_idx + eval_batch_size, num_samples)
                
                # Get batch of contexts
                batch_contexts = contexts[start_idx:end_idx]
                contexts_batch = torch.from_numpy(np.stack(batch_contexts).astype(np.int64)).to(device)
                
                # Generate for this batch
                generated_sequences = model.generate(contexts_batch, max_new_tokens=path_target_len, temperature=1.0, top_k=1)
                generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
                
                all_generated_tokens.append(generated_tokens_batch)
    model.train()
    
    # Concatenate all batches
    all_generated_tokens = np.concatenate(all_generated_tokens, axis=0)
    
    # Calculate accuracies
    LiveTrainingPanel.CONSOLE.print(f"\nAutoregressive generation on {num_samples} {split_name} samples:")
    accuracies = []
    for ground_truth, generated_tokens in zip(ground_truths, all_generated_tokens):
        # Calculate accuracy
        accuracy = np.mean(generated_tokens == ground_truth[:len(generated_tokens)])
        accuracies.append(accuracy)
    
    # Calculate average accuracy
    avg_accuracy = np.mean(accuracies)
    LiveTrainingPanel.CONSOLE.print(f"  Average accuracy: {avg_accuracy*100:.1f}%")
    LiveTrainingPanel.CONSOLE.print()  # Empty line for readability
    
    return avg_accuracy

# GOOD
def set_wandb_name(config):
    if config is not None:
        # Set custom run name for sweep runs
        if wandb.run is not None:
            utc_time = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            commit_id = get_git_commit_id()
            dir_label = "Udir" if config["use_undirected"] else "Dir"
            dt_label = 'Dt' if config['use_directional_tokens'] else ''
            ptgt_label = 'Pgt' if config.get('use_task_tokens_in_path', False) else ''
            ped_or_pet_label = 'Pd' if config['predict_direction_for_edge_task'] else 'Pe'
            wt_label = 'Wt' if config['weight_tying'] else ''
            wd_label = f"Wd{config['weight_decay']}" if config['weight_decay'] > 0 else ""

            model_bias_label = "Bias" if config["bias"] else ""
            model_ln_label = "Ln" if config["use_layernorm"] else ""
            model_mlp_label = "Mlp" if config["use_mlp"] else ""
            model_pos_label = "" if config.get("use_pos_embeddings", True) else "NoPos"
            activation = "A" + ((config["activation"] ) if config["activation"] else "").lower()
            # Include both dropout values if they differ, otherwise just one
            if config['dropout'] == config['embd_dropout']:
                dropout_label = f"D{config['dropout']}"
            else:
                dropout_label = f"D{config['dropout']}ED{config['embd_dropout']}"

            custom_name = (
                f"{utc_time}_"
                f"{commit_id}_"
                "DSET_"
                f"G{config['graph_d']}"
                f"L{config['graph_l']}"
                f"P{config['num_pause_tokens']}"
                f"{ped_or_pet_label}"
                f"{dir_label}"
                f"{dt_label}"
                f"{ptgt_label}"
                "_"
                f"L{config['n_layer']}"
                f"E{config['n_embd']}"
                f"H{config['n_head']}"
                f"{model_mlp_label}"
                f"{activation}"
                f"{model_ln_label}"
                f"{model_bias_label}"
                f"{model_pos_label}"
                f"{dropout_label}"
                f"{wd_label}"
                f"{wt_label}"
                f"Ep{config['epochs']}"
                f"Seed{config['seed']}"
            )
            wandb.run.name = custom_name
            print(f"Set sweep run name: {custom_name}")
            return custom_name


def initalize_model(device, meta, config, checkpoint_filename):
    # Model initialization
    model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=meta['block_size'],
        bias=config['bias'],
        vocab_size=None,
        dropout=config['dropout'],
        embd_dropout=config['embd_dropout'],
        use_pos_embeddings=config['use_pos_embeddings']
    )
    checkpoint = None
    iter_num = 0
    meta['best_val_loss'] = float('inf')
    meta['best_train_loss'] = float('inf')
    meta['best_mean_cosine_distance'] = float('inf')  # Track lowest mean cosine distance for edge_only mode
    if config['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        if meta['vocab_size'] is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta.get('vocab_size', 50304)
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], checkpoint_filename)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout', 'embd_dropout', 'use_pos_embeddings']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        meta['best_val_loss'] = checkpoint.get('best_val_loss', float('inf'))
        meta['best_train_loss'] = checkpoint.get('best_train_loss', float('inf'))
        meta['best_mean_cosine_distance'] = checkpoint.get('best_mean_cosine_distance', float('inf'))
    
    if meta['block_size'] < model.config.block_size:
        model.crop_block_size(meta['block_size'])
        model_args['block_size'] = meta['block_size']
    
    model.to(device)

    return model, model_args, checkpoint,  iter_num
    
def calculate_optimal_batch_size_for_training(model, block_size, vocab_size, device, dtype, 
                                    gradient_accumulation_steps, safety_factor=0.90, reserved_memory=0, target_batch_size=None):
    """
    Calculate maximum safe batch size based on available GPU memory.
    
    Memory breakdown:
    - Model parameters: N × bytes_per_param
    - Optimizer (AdamW): N × 2 × 4 bytes (momentum + variance in FP32)
    - Gradients: N × bytes_per_param
    - Activations: batch_size × memory_per_sample
    - Output logits: batch_size × seq_len × vocab_size (MAJOR memory consumer!)
    
    Args:
        safety_factor: Use 90% of available memory (conservative for torch.compile)
        reserved_memory: Memory reserved for datasets (in bytes). Will be subtracted from available memory.
        target_batch_size: Optional target batch size (hint) for CPU/optimality calculations.
    
    Returns:
        max_batch_size: Maximum safe batch size
    """
    # Handle device as string or torch.device object
    device_type = device if isinstance(device, str) else device.type
    if device_type != 'cuda':
        if target_batch_size is not None:
            # For CPU/MPS, cap batch size to dataset size to avoid batch_size > dataset_size
            # We maintain a minimum of 500 for efficiency on very small datasets.
            return min(3800, target_batch_size)
        return 3800  # Default for non-CUDA
    
    # Get GPU memory info
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = (total_memory - allocated_memory) * safety_factor - reserved_memory
    
    # Bytes per parameter based on dtype
    bytes_per_param = 2 if dtype in ['float16', 'bfloat16'] else 4
    
    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Memory components (all already allocated or will be)
    model_memory = num_params * bytes_per_param
    optimizer_memory = num_params * 2 * 4  # AdamW: 2 states in FP32
    gradient_memory = num_params * bytes_per_param
    
    static_memory = model_memory + optimizer_memory + gradient_memory
    
    # Per-sample activation memory estimation
    # Activations stored for backward pass in transformer:
    # 1. Token embeddings: seq_len × hidden_dim
    # 2. Per layer:
    #    - Layer input: seq_len × hidden_dim
    #    - Attention QKV: 3 × seq_len × hidden_dim
    #    - Attention scores: n_heads × seq_len × seq_len
    #    - Attention output: seq_len × hidden_dim
    #    - MLP intermediate: seq_len × 4 × hidden_dim
    # 3. Output logits: seq_len × vocab_size (CRITICAL with large vocab!)
    # 4. Gradients of all above (stored during backward)
    
    cfg = model.config
    seq_len = block_size
    hidden_dim = cfg.n_embd
    n_layers = cfg.n_layer
    n_heads = cfg.n_head
    
    # Conservative activation estimate per sample
    embeddings = seq_len * hidden_dim * bytes_per_param
    
    per_layer_activations = (
        seq_len * hidden_dim * 3 +           # Layer input/output + residual
        seq_len * hidden_dim * 3 +           # QKV projections
        n_heads * seq_len * seq_len +        # Attention weights
        seq_len * hidden_dim +               # Attention output
        seq_len * hidden_dim * 4 * 2         # MLP (fc + proj)
    ) * bytes_per_param
    
    total_layer_activations = per_layer_activations * n_layers
    
    # OUTPUT LOGITS - This is the MAJOR memory consumer with large vocab!
    # We need logits for forward (batch × seq × vocab) and their gradients
    output_logits_memory = seq_len * vocab_size * bytes_per_param * 2  # forward + backward
    
    # torch.compile overhead (empirically ~30% extra for intermediate buffers)
    compile_overhead = 1.3
    
    # Total per-sample memory
    activation_per_sample = (embeddings + total_layer_activations + output_logits_memory) * 2 * compile_overhead
    
    # With gradient accumulation: only 1 micro-batch in memory at a time
    # So we calculate max micro-batch size
    memory_for_batch = available_memory - static_memory
    
    if memory_for_batch <= 0:
        print(f"WARNING: Static memory ({static_memory/1e9:.2f}GB) exceeds available")
        return 500
    
    max_microbatch_size = int(memory_for_batch / activation_per_sample)
    
    # Apply reasonable bounds
    max_batch_size = max(500, min(max_microbatch_size, 5000))
    
    # Cap to target_batch_size if provided (avoid batch_size > dataset_size)
    if target_batch_size is not None:
        max_batch_size = min(max_batch_size, target_batch_size)
    
    # Diagnostic output
    print(f"\n=== Memory-Based Batch Size Calculation ===")
    print(f"GPU: {props.name}")
    print(f"Total VRAM: {total_memory/1e9:.2f} GB")
    print(f"Currently allocated: {allocated_memory/1e9:.2f} GB")
    if reserved_memory > 0:
        print(f"Reserved for datasets: {reserved_memory/1e9:.2f} GB")
    print(f"Available for batches: {memory_for_batch/1e9:.2f} GB")
    print(f"Static memory breakdown:")
    print(f"  - Model params: {model_memory/1e9:.2f} GB ({num_params:,} params)")
    print(f"  - Optimizer states: {optimizer_memory/1e9:.2f} GB")
    print(f"  - Gradients: {gradient_memory/1e9:.2f} GB")
    print(f"  - Total static: {static_memory/1e9:.2f} GB")
    print(f"Per-sample memory breakdown:")
    print(f"  - Embeddings + layers: {(embeddings + total_layer_activations) * 2 / 1e6:.2f} MB")
    print(f"  - Output logits (vocab={vocab_size}): {output_logits_memory / 1e6:.2f} MB")
    print(f"  - Total per sample (with compile overhead): {activation_per_sample/1e6:.2f} MB")
    print(f"Calculated max batch size: {max_batch_size}")
    print(f"With grad_accum={gradient_accumulation_steps}, effective: {max_batch_size * gradient_accumulation_steps}")
    print(f"===========================================\n")
    
    return max_batch_size


def check_for_nans(model, optimizer, loss, logits, X, Y, iter_num, phase='unknown'):
    """
    Comprehensive NaN detection and reporting.
    
    Returns:
        dict with NaN detection results, or None if no NaNs found
    """
    nan_report = {}
    has_nan = False
    
    # Check loss
    if torch.isnan(loss) or torch.isinf(loss):
        nan_report['loss'] = float(loss)
        has_nan = True
    
    # Check logits
    if logits is not None:
        if torch.isnan(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            nan_pct = 100 * nan_count / logits.numel()
            nan_report['logits_nan_count'] = nan_count
            nan_report['logits_nan_pct'] = nan_pct
            nan_report['logits_max'] = float(logits[~torch.isnan(logits)].max()) if nan_count < logits.numel() else float('nan')
            nan_report['logits_min'] = float(logits[~torch.isnan(logits)].min()) if nan_count < logits.numel() else float('nan')
            has_nan = True
        elif torch.isinf(logits).any():
            inf_count = torch.isinf(logits).sum().item()
            nan_report['logits_inf_count'] = inf_count
            has_nan = True
        else:
            # Log max/min even when healthy for tracking
            nan_report['logits_max'] = float(logits.max())
            nan_report['logits_min'] = float(logits.min())
    
    # Check inputs
    if torch.isnan(X).any() or torch.isnan(Y).any():
        nan_report['input_has_nan'] = True
        has_nan = True
    
    # Check model parameters
    param_nans = []
    param_infs = []
    for name, param in model.named_parameters():
        if param is not None and torch.isnan(param).any():
            param_nans.append(name)
            has_nan = True
        if param is not None and torch.isinf(param).any():
            param_infs.append(name)
            has_nan = True
    
    if param_nans:
        nan_report['param_nans'] = param_nans
    if param_infs:
        nan_report['param_infs'] = param_infs
    
    # Check gradients
    grad_nans = []
    grad_infs = []
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                grad_nans.append(name)
                has_nan = True
            if torch.isinf(param.grad).any():
                grad_infs.append(name)
                has_nan = True
            max_grad = max(max_grad, param.grad.abs().max().item())
    
    if grad_nans:
        nan_report['grad_nans'] = grad_nans
    if grad_infs:
        nan_report['grad_infs'] = grad_infs
    nan_report['max_grad'] = max_grad
    
    # Check optimizer state
    if optimizer is not None:
        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            if torch.isnan(value).any():
                                nan_report[f'optimizer_state_nan_group{group_idx}_param{param_idx}_{key}'] = True
                                has_nan = True
    
    if has_nan:
        nan_report['iter'] = iter_num
        nan_report['phase'] = phase
        return nan_report
    
    return None

def compute_path_similarities(paths, result_dict, meta, similarity_matrix, token_to_idx):
    """
    Compute pairwise similarities between nodes at different distances within paths.

    paths: list of lists , each list is a path from root to leaf i.e [root, n_1, n_2, ..., leaf]
    results_dict: dictionary to mutate to store the similarities by distance
    meta: metadata dictionary
    similarity_matrix: matrix of similarities between all nodes (num_nodes, num_nodes)
    token_to_idx: mapping from token to index in similarity matrix
    """
    for path in paths:
        assert len(path) == meta['l']
        # path is [root, n_1, n_2, ..., leaf] with length l
        for i in range(len(path)):
            for j in range(i, len(path)):
                dist = j - i  # Graph distance within the path
                token_i = path[i]
                token_j = path[j]
                
                if token_i in token_to_idx and token_j in token_to_idx:
                    idx_i = token_to_idx[token_i]
                    idx_j = token_to_idx[token_j]
                    sim = similarity_matrix[idx_i, idx_j].item()
                    result_dict[dist].append(sim)

def compute_mean_cosine_distance(model, meta):
    """
    Compute the mean pairwise cosine distance between all node embeddings.
    
    This metric tracks the global clustering of embeddings. As training progresses
    and "Geometric Memory" forms, embeddings organize into a lower-dimensional 
    structure (manifold), causing them to become more aligned and this metric to decrease.
    
    At initialization (random embeddings), cosine similarities ≈ 0, so distance ≈ 1.
    As learning progresses, embeddings align and the mean distance decreases.
    
    Mathematical Definition:
        Mean Cosine Distance = (1 / (N(N-1))) * Σ_i Σ_{j≠i} (1 - cos_sim(v_i, v_j))
    
    Args:
        model: The GPT model with embeddings
        meta: Metadata dict containing vertex information
        
    Returns:
        float: Mean cosine distance across all pairs of distinct node embeddings
    """
    model.eval()
    with torch.no_grad():
        # Get all node embeddings (excluding special tokens)
        E = model.transformer.wte.weight.detach().cpu()  # (vocab_size, n_embd)
        
        # Extract only the graph node embeddings (vertices)
        # meta['vertices'] contains the token IDs for actual graph nodes
        node_tokens = sorted(list(meta['vertices']))
        E_nodes = E[node_tokens]  # (num_nodes, n_embd)
        
        num_nodes = E_nodes.shape[0]
        
        # Edge case: need at least 2 nodes for pairwise distance
        if num_nodes < 2:
            return 0.0
        
        # Step 1: Normalize embeddings to unit vectors (L2 norm)
        E_norm = F.normalize(E_nodes, p=2, dim=1)  # (num_nodes, n_embd)
        
        # Step 2: Compute cosine similarity matrix: S = E_norm · E_norm^T
        S = torch.mm(E_norm, E_norm.t())  # (num_nodes, num_nodes)
        
        # Step 3: Compute cosine distance matrix: D = 1 - S
        D = 1.0 - S
        
        # Step 4: Mask diagonal (self-distances are 0 and shouldn't contribute)
        # We'll extract only the upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        
        # Step 5: Calculate mean of off-diagonal elements
        # Number of unique pairs: N(N-1)/2, but we compute over full N(N-1) by using upper triangle
        distances = D[mask]  # Extract upper triangle values
        mean_cosine_distance = distances.mean().item()
        
    model.train()
    return mean_cosine_distance


def plot_pairwise_cosine_similarity_matrix(model, meta, iter_num, config, out_dir='out'):
    """
    Plot the pairwise cosine similarity matrix of node embeddings as a heatmap.
    
    This diagnostic plot (similar to Figure 24 in PathStar paper Appendix C.3) helps
    distinguish between:
    - Associative memory: Heatmap looks like adjacency matrix (hot on edges only)
    - Geometric memory: Heatmap shows gradients/ripples (multi-hop similarity fading)
    
    Args:
        model: The GPT model with embeddings
        meta: Metadata dict containing vertex information and adjacency info
        iter_num: Current iteration number (for plot title and filename)
        config: Training configuration dict
        out_dir: Output directory for saving plots
        
    Returns:
        str: Path to saved plot file
    """
    model.eval()
    with torch.no_grad():
        # Get all node embeddings (excluding special tokens)
        E = model.transformer.wte.weight.detach().cpu()  # (vocab_size, n_embd)
        
        # Extract only the graph node embeddings (vertices)
        node_tokens = sorted(list(meta['vertices']))
        E_nodes = E[node_tokens]  # (num_nodes, n_embd)
        
        num_nodes = E_nodes.shape[0]
        
        # Normalize embeddings to unit vectors
        E_norm = F.normalize(E_nodes, p=2, dim=1)  # (num_nodes, n_embd)
        
        # Compute cosine similarity matrix: S = E_norm · E_norm^T
        S = torch.mm(E_norm, E_norm.t()).cpu().numpy()  # (num_nodes, num_nodes)
    
    model.train()
    
    # Create figure with appropriate size based on number of nodes
    # For large graphs, use smaller pixel-per-node ratio
    if num_nodes <= 50:
        figsize = (12, 10)
        show_ticks = True
    elif num_nodes <= 200:
        figsize = (14, 12)
        show_ticks = False
    else:
        figsize = (16, 14)
        show_ticks = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(S, cmap='viridis', aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=11)
    
    # Set title with iteration info
    current_epoch = iter_num / meta['batches_per_epoch']
    ax.set_title(f'Pairwise Cosine Similarity Matrix (Iter {iter_num}, Epoch {current_epoch:.1f})\n'
                 f'{num_nodes} nodes | Mean Distance: {1.0 - S[np.triu_indices(num_nodes, k=1)].mean():.4f}',
                 fontsize=13, pad=10)
    
    ax.set_xlabel('Node Index', fontsize=11)
    ax.set_ylabel('Node Index', fontsize=11)
    
    # Show tick labels only for small graphs
    if show_ticks and num_nodes <= 50:
        ax.set_xticks(range(0, num_nodes, max(1, num_nodes // 20)))
        ax.set_yticks(range(0, num_nodes, max(1, num_nodes // 20)))
    else:
        # Show fewer ticks for large graphs
        tick_spacing = max(1, num_nodes // 10)
        ax.set_xticks(range(0, num_nodes, tick_spacing))
        ax.set_yticks(range(0, num_nodes, tick_spacing))
    
    # Add grid for readability (optional, can be removed if too cluttered)
    if num_nodes <= 50:
        ax.grid(False)
    
    plt.tight_layout()
    
    # Save to file
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f'cosine_similarity_matrix_iter_{iter_num}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    
    LiveTrainingPanel.CONSOLE.print(f"[cyan]Saved cosine similarity matrix plot: {plot_path}[/cyan]")
    
    return plot_path


def analyze_embedding_geometry(model, meta, paths_data_np, val_data_np, iter_num, config, out_dir='out'):
    """
    Analyze if path structure is reflected in embedding space.
    
    Computes cosine similarities between node embeddings at different graph distances
    for both training and validation paths, using the actual path data from datasets.
    
    Args:
        model: The GPT model with embeddings
        meta: Metadata dict containing special tokens and path structure info
        paths_data_np: NumPy array of training path sequences (from paths.bin)
        val_data_np: NumPy array of validation path sequences (from val.bin)
        iter_num: Current iteration number (for plot title and filename)
        config: Training configuration dict
        out_dir: Output directory for saving plots
        
    Returns:
        dict: Statistics for train and val similarities by distance
    """
    model.eval()
    E = model.transformer.wte.weight.detach().cpu()
    
    # Calculate sequence dimensions from meta
    # NOTE: Use block_size_base since paths_data_np and val_data_np are raw stored data (WITHOUT pause tokens)
    seq_len = meta['block_size_base'] + 1  # Full sequence length (stored, without pause tokens)
    PATHS_DATASET_SIZE = meta['PATHS_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    # Reshape the flat numpy arrays to (num_samples, seq_len)
    paths_data = paths_data_np.reshape(PATHS_DATASET_SIZE, seq_len)
    val_data = val_data_np.reshape(VAL_DATASET_SIZE, seq_len)
    
    def extract_path_nodes(sequence, meta):
        """
        Extract just the graph node tokens from a path sequence.
        
        Path format (stored, WITHOUT pause tokens): [PATH?, leaf, root, (GT?), n_2, (GT?), ..., leaf]
        
        Returns list of node tokens: [root, n_2, ..., leaf] (length = l)
        """
        seq = [int(x) for x in sequence]
        
        # Calculate where path nodes start
        # NOTE: Stored sequences do NOT have pause tokens
        # Skip: PATH token + leaf
        path_start_idx = 2
        
        # Extract the path portion
        path_portion = seq[path_start_idx:]
        
        # Filter out special tokens (GT, PAD, etc.) to get just node tokens
        # Node tokens are >= num_special_tokens
        node_tokens = [t for t in path_portion if t in meta['vertices']]
        
        return node_tokens
    
    # Extract paths from train and val data
    train_paths = []
    for i in range(min(PATHS_DATASET_SIZE, 1000)):  # Limit to avoid memory issues
        path_nodes = extract_path_nodes(paths_data[i], meta)
        train_paths.append(path_nodes)
    
    val_paths = []
    for i in range(VAL_DATASET_SIZE):
        path_nodes = extract_path_nodes(val_data[i], meta)
        val_paths.append(path_nodes)
    
    # Get all unique node tokens for similarity matrix (sorted for deterministic ordering)
    all_node_tokens = set()
    for path in train_paths + val_paths:
        all_node_tokens.update(path)
    all_node_tokens = sorted(list(all_node_tokens))  # Sort to ensure deterministic ordering
    assert len(all_node_tokens) == meta['num_vertices']
    
    # Compute embeddings for all node tokens
    node_embeddings = E[all_node_tokens]  # (num_nodes, n_embd)
    
    # Normalize for cosine similarity
    node_embeddings_norm = F.normalize(node_embeddings, p=2, dim=1)
    
    # Cosine similarity matrix
    sim_matrix = torch.mm(node_embeddings_norm, node_embeddings_norm.t())
    
    # Create mapping from token to index in similarity matrix
    token_to_idx = {t: i for i, t in enumerate(all_node_tokens)}
    
    # Compute similarities by distance for train and val paths
    results = {
        'train': defaultdict(list),
        'val': defaultdict(list),
    }
    
    compute_path_similarities(train_paths, results['train'], meta, sim_matrix, token_to_idx)
    compute_path_similarities(val_paths, results['val'], meta, sim_matrix, token_to_idx)
    
    # Compute cross-path similarities (nodes at same position but different spokes)
    # This measures how similar nodes at the same depth are across different paths
    cross_path_sims_train = defaultdict(list)
    cross_path_sims_val = defaultdict(list)
    
    # Sample pairs from train paths
    for i in range(min(len(train_paths), 50)):
        for j in range(i + 1, min(len(train_paths), 50)):
            path_i = train_paths[i]
            path_j = train_paths[j]
            for pos in range(len(path_i)):
                token_i = path_i[pos]
                token_j = path_j[pos]
                if token_i in token_to_idx and token_j in token_to_idx:
                    idx_i = token_to_idx[token_i]
                    idx_j = token_to_idx[token_j]
                    sim = sim_matrix[idx_i, idx_j].item()
                    cross_path_sims_train[pos].append(sim)
    
    # Sample pairs from val paths
    for i in range(min(len(val_paths), 50)):
        for j in range(i + 1, min(len(val_paths), 50)):
            path_i = val_paths[i]
            path_j = val_paths[j]
            for pos in range(len(path_i)):
                token_i = path_i[pos]
                token_j = path_j[pos]
                if token_i in token_to_idx and token_j in token_to_idx:
                    idx_i = token_to_idx[token_i]
                    idx_j = token_to_idx[token_j]
                    sim = sim_matrix[idx_i, idx_j].item()
                    cross_path_sims_val[pos].append(sim)
    
    # Compute random baseline
    num_random_samples = min(500, len(all_node_tokens) * (len(all_node_tokens) - 1) // 2)
    random_sims = []
    for _ in range(num_random_samples):
        i, j = random.sample(range(len(all_node_tokens)), 2)
        random_sims.append(sim_matrix[i, j].item())
    
    # Create the plot with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Train within-path similarities by distance
    ax1 = axes[0, 0]
    train_distances = sorted(results['train'].keys())
    train_means = [np.mean(results['train'][d]) if results['train'][d] else 0 for d in train_distances]
    train_stds = [np.std(results['train'][d]) if results['train'][d] else 0 for d in train_distances]
    
    if train_distances:
        ax1.errorbar(train_distances, train_means, yerr=train_stds, 
                     marker='o', capsize=5, capthick=2, linewidth=2, 
                     color='blue', label='Train paths')
    
    if random_sims:
        ax1.axhline(y=np.mean(random_sims), color='gray', linestyle='--', 
                    label=f'Random baseline ({np.mean(random_sims):.3f})')
    
    ax1.set_xlabel('Graph Distance (within path)', fontsize=11)
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title(f'Train: Within-Path Similarities (iter {iter_num})', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.1)
    
    # Plot 2: Val within-path similarities by distance
    ax2 = axes[0, 1]
    val_distances = sorted(results['val'].keys())
    val_means = [np.mean(results['val'][d]) if results['val'][d] else 0 for d in val_distances]
    val_stds = [np.std(results['val'][d]) if results['val'][d] else 0 for d in val_distances]
    
    if val_distances:
        ax2.errorbar(val_distances, val_means, yerr=val_stds,
                     marker='s', capsize=5, capthick=2, linewidth=2,
                     color='red', label='Val paths (holdout)')
    
    if random_sims:
        ax2.axhline(y=np.mean(random_sims), color='gray', linestyle='--',
                    label=f'Random baseline ({np.mean(random_sims):.3f})')
    
    ax2.set_xlabel('Graph Distance (within path)', fontsize=11)
    ax2.set_ylabel('Cosine Similarity', fontsize=11)
    ax2.set_title(f'Val: Within-Path Similarities (iter {iter_num})', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 1.1)
    
    # Plot 3: Cross-path similarities by position (Train)
    ax3 = axes[1, 0]
    if cross_path_sims_train:
        positions = sorted(cross_path_sims_train.keys())
        cross_means = [np.mean(cross_path_sims_train[p]) for p in positions]
        cross_stds = [np.std(cross_path_sims_train[p]) for p in positions]
        
        ax3.errorbar(positions, cross_means, yerr=cross_stds,
                     marker='o', capsize=5, capthick=2, linewidth=2,
                     color='green', label='Cross-path (same depth)')
        
        if random_sims:
            ax3.axhline(y=np.mean(random_sims), color='gray', linestyle='--',
                        label=f'Random baseline ({np.mean(random_sims):.3f})')
    
    ax3.set_xlabel('Position in Path (0=root, l-1=leaf)', fontsize=11)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title(f'Train: Cross-Path Similarities (same depth)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.5, 1.1)
    
    # Plot 4: Cross-path similarities by position (Val)
    ax4 = axes[1, 1]
    if cross_path_sims_val:
        positions = sorted(cross_path_sims_val.keys())
        cross_means = [np.mean(cross_path_sims_val[p]) for p in positions]
        cross_stds = [np.std(cross_path_sims_val[p]) for p in positions]
        
        ax4.errorbar(positions, cross_means, yerr=cross_stds,
                     marker='s', capsize=5, capthick=2, linewidth=2,
                     color='purple', label='Cross-path (same depth)')
        
        if random_sims:
            ax4.axhline(y=np.mean(random_sims), color='gray', linestyle='--',
                        label=f'Random baseline ({np.mean(random_sims):.3f})')
    
    ax4.set_xlabel('Position in Path (0=root, l-1=leaf)', fontsize=11)
    ax4.set_ylabel('Cosine Similarity', fontsize=11)
    ax4.set_title(f'Val: Cross-Path Similarities (same depth)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, 1.1)
    
    plt.tight_layout()
    
    # Save to file
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f'embedding_geometry_iter_{iter_num}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    
    # Print summary to console
    summary_lines = []
    summary_lines.append(f"[bold cyan]Embedding Geometry Analysis (iter {iter_num})[/bold cyan]")
    summary_lines.append(f"  Plot saved: {plot_path}")
    summary_lines.append(f"  Train paths analyzed: {len(train_paths)}, Val paths: {len(val_paths)}")
    summary_lines.append(f"  [blue]Train within-path similarities:[/blue]")
    for dist in sorted(results['train'].keys())[:5]:  # Show first 5
        sims = results['train'][dist]
        if sims:
            summary_lines.append(f"    Distance {dist}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
    summary_lines.append(f"  [red]Val within-path similarities:[/red]")
    for dist in sorted(results['val'].keys())[:5]:
        sims = results['val'][dist]
        if sims:
            summary_lines.append(f"    Distance {dist}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
    if random_sims:
        summary_lines.append(f"  [dim]Random baseline: {np.mean(random_sims):.4f}[/dim]")
    
    LiveTrainingPanel.CONSOLE.print("\n".join(summary_lines))
    
    model.train()
    
    return {
        'train_similarities': dict(results['train']),
        'val_similarities': dict(results['val']),
        'cross_path_train': dict(cross_path_sims_train),
        'cross_path_val': dict(cross_path_sims_val),
        'random_baseline': np.mean(random_sims) if random_sims else 0,
        'plot_path': plot_path,
    }


def checkpoint_model(model, meta, config, iter_num, loss_value, loss_type='val', lr_scheduler_obj=None):
    """
    Save model checkpoint to disk.
    
    Args:
        model: The GPT model to checkpoint
        meta: Metadata dict containing optimizer, model_args, etc.
        config: Training configuration dict
        iter_num: Current iteration number
        loss_value: Loss value to log with checkpoint
        loss_type: Type of loss ('val' or 'train') for logging message
        lr_scheduler_obj: Optional LR scheduler to save state
    """
    # Exclude non-serializable objects (like console) from saved config
    serializable_config = {k: v for k, v in config.items() if k != 'console'}
    
    # Save only specific meta keys
    meta_keys_to_save = ['train_leaves', 'holdout_leaves', 'paths_by_leaf', 'd', 'l', 'vocab_size', 'special_tokens', 'root_vertex', 'itos', 'stoi']
    meta_subset = {k: meta[k] for k in meta_keys_to_save if k in meta}
    
    checkpoint_data = {
        'model': model.state_dict(),
        'optimizer': meta['optimizer'].state_dict(),
        'model_args': meta['model_args'],
        'iter_num': iter_num,
        'best_val_loss': meta.get('best_val_loss', float('inf')),
        'best_train_loss': meta.get('best_train_loss', float('inf')),
        'best_mean_cosine_distance': meta.get('best_mean_cosine_distance', float('inf')),
        'config': serializable_config,
        'meta': meta_subset,  # Save only the specified meta keys
    }
    
    # Save LR scheduler state if using ReduceLROnPlateau
    if lr_scheduler_obj is not None:
        checkpoint_data['lr_scheduler'] = lr_scheduler_obj.state_dict()
    
    checkpoint_path = os.path.join(config['out_dir'], meta['checkpoint_filename'])
    LiveTrainingPanel.CONSOLE.print(f"saving checkpoint to {checkpoint_path} ... {loss_type}_loss {loss_value:.6f}")
    torch.save(checkpoint_data, checkpoint_path)


def evaluate(estimate_metrics, config, meta, iter_num, lr, ctx, device, model, val_data_np, paths_data_np, edges_data_np, print_samples=False, live_panel=None, tokens_per_sec=None, batch_size=None, train_dataset_size=None, eval_dataset_size=None, lr_scheduler_obj=None):
    # Compute metrics for both splits
    val_metrics = estimate_metrics('val', print_samples)
    train_metrics = estimate_metrics('train', False) # Don't print train samples here
    losses = {**val_metrics, **train_metrics}
    
    # For PATH task token-level metrics, use the number of tokens generated after PATH context.
    graph_length = int(meta.get('path_target_length', meta['l']))
    PATHS_DATASET_SIZE = meta['PATHS_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    current_epoch = iter_num / meta['batches_per_epoch']

    # Evaluate autoregressive generation on validation and training samples
    # Use fewer samples during sweeps for faster evaluation
    is_sweep_mode = wandb.run is not None and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None
    autoregressive_eval_samples = 20 if is_sweep_mode else 100
    # val_avg_accuracy = generate_samples_autoregressive(
    #     device, ctx, model, meta, val_data_np, VAL_DATASET_SIZE, 'val',
    #     num_samples=min(VAL_DATASET_SIZE, autoregressive_eval_samples),
    #     eval_batch_size=int(config.get('eval_batch_size', 512)),
    # )
    # train_avg_accuracy = generate_samples_autoregressive(
    #     device, ctx, model, meta, paths_data_np, PATHS_DATASET_SIZE, 'train',
    #     num_samples=min(PATHS_DATASET_SIZE, autoregressive_eval_samples),
    #     eval_batch_size=int(config.get('eval_batch_size', 512)),
    # )
    
    # Compute mean cosine distance metric (lightweight, always computed)
    mean_cosine_distance = compute_mean_cosine_distance(model, meta)
    
    # Plot pairwise cosine similarity matrix (at every eval_interval)
    # This helps diagnose associative vs geometric memory (see PathStar paper Figure 24)
    cosine_similarity_plot_path = None
    try:
        cosine_similarity_plot_path = plot_pairwise_cosine_similarity_matrix(
            model, meta, iter_num, config,
            out_dir=config.get('out_dir', 'out')
        )
    except Exception as e:
        LiveTrainingPanel.CONSOLE.print(f"[yellow]Warning: Cosine similarity matrix plot failed: {e}[/yellow]")
    
    # Evaluate edge memorization
    edge_memorization_pct = None
    if config['show_edge_memorization_metrics']:
        edge_memorization_pct = evaluate_edge_memorization(
            ctx, model, meta, edges_data_np, device,
            batch_size=int(config.get('edge_eval_batch_size', 512)),
        )
    
    # Analyze embedding geometry when printing samples (every print_eval_interval)
    embedding_geometry_results = None
    if print_samples and config.get('analyze_embedding_geometry', False):
        try:
            embedding_geometry_results = analyze_embedding_geometry(
                model, meta, paths_data_np, val_data_np, iter_num, config, 
                out_dir=config.get('out_dir', 'out')
            )
        except Exception as e:
            LiveTrainingPanel.CONSOLE.print(f"[yellow]Warning: Embedding geometry analysis failed: {e}[/yellow]")
    
    # Print mean cosine distance to console
    LiveTrainingPanel.CONSOLE.print(f"[cyan]Mean Cosine Distance (embeddings): {mean_cosine_distance:.4f}[/cyan]")
    
    live_panel.update_metrics_table(
        losses,
        graph_length,
        iter_num,
        current_epoch,
        lr,
        meta,
        tokens_per_sec,
        batch_size,
        edge_memorization_pct,
        train_dataset_size=train_dataset_size,
        eval_dataset_size=eval_dataset_size,
        embedding_geometry_results=embedding_geometry_results,
        mean_cosine_distance=mean_cosine_distance,
    )
    
    # if 'val_per_token' in losses:
    #     # console.print("  Val per-token losses:")
    #     if graph_length <= 9:
    #         per_token_str = ", ".join([f"tok{i}: {losses['val_per_token'].get(i, float('nan')):.4f}"
    #                                    for i in range(1, graph_length + 1)])
    #     else:
    #         head = ", ".join([f"tok{i}: {losses['val_per_token'].get(i, float('nan')):.4f}"
    #                           for i in range(1, 10)])
    #         tail = f"tok{graph_length}: {losses['val_per_token'].get(graph_length, float('nan')):.4f}"
    #         # per_token_str = f"{head}, …, {tail}"
    #     # console.print(f"    {per_token_str}")
    
    # if 'val_per_token_accuracy' in losses:
    #     # console.print("  Val per-token accuracies (autoregressive):")
    #     if graph_length <= 9:
    #         per_token_acc_str = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'].get(i, float('nan'))*100:.1f}%"
    #                                        for i in range(1, graph_length + 1)])
        # else:
        #     head = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'].get(i, float('nan'))*100:.1f}%"
        #                       for i in range(1, 10)])
        #     tail = f"tok{graph_length}: {losses['val_per_token_accuracy'].get(graph_length, float('nan'))*100:.1f}%"
        #     per_token_acc_str = f"{head}, …, {tail}"
        # console.print(f"    {per_token_acc_str}")
    
    
    if config['wandb_log']:
        log_dict = {
            "iter": iter_num,
            'max_iters': meta['max_iters'],
            'warmup_iters': meta['warmup_iters'],
            "epoch": round(current_epoch, 4),
            "val/loss/overall": losses['val'],
            "train/loss/eval_paths": losses['train'],
            "lr": lr,
            #"gen/val_paths_avg_accuracy": val_avg_accuracy,
            # "gen/train_paths_avg_accuracy": train_avg_accuracy,
        }

        if edge_memorization_pct != None:
            log_dict["edge_memorization_pct"] = edge_memorization_pct
        
        if 'val_per_token' in losses:
            for token_pos in range(1, graph_length + 1):
                if token_pos == graph_length:
                    log_dict["val/loss/token_final"] = losses['val_per_token'][token_pos]
                else:
                    log_dict[f"val/loss/token_{token_pos}"] = losses['val_per_token'][token_pos]
        
        if 'val_per_token_accuracy' in losses:
            for token_pos in range(1, graph_length + 1):
                if token_pos == graph_length:
                    log_dict["val/accuracy/token_final"] = losses['val_per_token_accuracy'][token_pos]
                else:
                    log_dict[f"val/accuracy/token_{token_pos}"] = losses['val_per_token_accuracy'][token_pos]
        
        # Full path exact match accuracy (no teacher forcing)
        if 'val_full_path_accuracy' in losses:
            log_dict["val/accuracy/full_path"] = losses['val_full_path_accuracy']
        
        # Add embedding geometry metrics if available
        if embedding_geometry_results is not None:
            l = meta['l']
            train_sims = embedding_geometry_results.get('train_similarities', {})
            val_sims = embedding_geometry_results.get('val_similarities', {})
            
            # Self-similarity (distance = 0)
            if 0 in train_sims and train_sims[0]:
                log_dict['embedding_geometry/train_dist0_sim'] = np.mean(train_sims[0])
            if 0 in val_sims and val_sims[0]:
                log_dict['embedding_geometry/val_dist0_sim'] = np.mean(val_sims[0])
            
            # Adjacent nodes similarity (distance = 1)
            if 1 in train_sims and train_sims[1]:
                log_dict['embedding_geometry/train_dist1_sim'] = np.mean(train_sims[1])
            if 1 in val_sims and val_sims[1]:
                log_dict['embedding_geometry/val_dist1_sim'] = np.mean(val_sims[1])
            
            # Root-to-leaf similarity (distance = l-1)
            if l-1 in train_sims and train_sims[l-1]:
                log_dict['embedding_geometry/train_root_leaf_sim'] = np.mean(train_sims[l-1])
            if l-1 in val_sims and val_sims[l-1]:
                log_dict['embedding_geometry/val_root_leaf_sim'] = np.mean(val_sims[l-1])
            
            # Cross-path similarity at root position (should be high if root is shared)
            cross_train = embedding_geometry_results.get('cross_path_train', {})
            cross_val = embedding_geometry_results.get('cross_path_val', {})
            if 0 in cross_train and cross_train[0]:
                log_dict['embedding_geometry/train_cross_root_sim'] = np.mean(cross_train[0])
            if 0 in cross_val and cross_val[0]:
                log_dict['embedding_geometry/val_cross_root_sim'] = np.mean(cross_val[0])
            
            log_dict['embedding_geometry/random_baseline'] = embedding_geometry_results.get('random_baseline', 0)
        
        # Compute transformer weight norm (L2 norm of all parameters)
        total_norm = 0.0
        for p in model.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        log_dict['model/weight_norm'] = total_norm
        
        # Log mean cosine distance (global embedding clustering metric)
        log_dict['embedding_geometry/mean_cosine_distance'] = mean_cosine_distance
        
        # Log cosine similarity matrix plot to wandb
        if cosine_similarity_plot_path and os.path.exists(cosine_similarity_plot_path):
            log_dict['embedding_geometry/cosine_similarity_matrix'] = wandb.Image(cosine_similarity_plot_path)
        
        wandb.log(log_dict, step=iter_num)
    
    # Checkpointing logic for normal mode (with validation)
    # During sweeps, only save best checkpoint to reduce I/O overhead
    # In standalone mode, save based on always_save_checkpoint config
    save_checkpoint = False
    if losses['val'] < meta['best_val_loss']:
        meta['best_val_loss'] = losses['val']
        save_checkpoint = True
    elif not is_sweep_mode and config['always_save_checkpoint']:
        save_checkpoint = True
    
    if save_checkpoint and iter_num > 0:
        checkpoint_model(model, meta, config, iter_num, losses['val'], loss_type='val', lr_scheduler_obj=lr_scheduler_obj)
    
    # Return validation loss for LR schedulers like ReduceLROnPlateau
    return losses['val']

def determine_dataset_in_device_size(device, device_type, paths_data, edges_data, val_data, limit=0.1):
    # Calculate dataset memory requirements BEFORE batch size calculation
    # This ensures batch size accounts for dataset memory if it will be loaded to GPU
    dataset_reserved_memory = 0
    if device_type == 'cuda':
        paths_data_tensor_temp = torch.from_numpy(paths_data.astype(np.int64))
        edges_data_tensor_temp = torch.from_numpy(edges_data.astype(np.int64))
        val_data_tensor_temp = torch.from_numpy(val_data.astype(np.int64))
        
        paths_data_bytes = paths_data_tensor_temp.numel() * paths_data_tensor_temp.element_size()
        edges_data_bytes = edges_data_tensor_temp.numel() * edges_data_tensor_temp.element_size()
        val_data_bytes = val_data_tensor_temp.numel() * val_data_tensor_temp.element_size()
        total_dataset_bytes = paths_data_bytes + edges_data_bytes + val_data_bytes
        
        # Get available GPU memory (before model is fully loaded)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        props_temp = torch.cuda.get_device_properties(device)
        total_vram = props_temp.total_memory
        allocated_vram = torch.cuda.memory_allocated(device)
        available_vram = total_vram - allocated_vram
        
        # Policy: Only reserve memory if datasets use < 10% of available VRAM
        vram_limit = available_vram * limit
        
        if total_dataset_bytes <= vram_limit:
            # Datasets will be loaded to GPU, so reserve this memory
            dataset_reserved_memory = total_dataset_bytes
            print(f"\n=== Pre-calculation: Dataset Memory Check ===")
            print(f"Total dataset size: {total_dataset_bytes / 1e9:.3f} GB")
            print(f"VRAM limit (50%): {vram_limit / 1e9:.3f} GB")
            print(f"✓ Datasets will be loaded to GPU - reserving {dataset_reserved_memory / 1e9:.3f} GB for batch size calculation")
            print(f"============================================\n")
        else:
            print(f"\n=== Pre-calculation: Dataset Memory Check ===")
            print(f"Total dataset size: {total_dataset_bytes / 1e9:.3f} GB")
            print(f"VRAM limit (50%): {vram_limit / 1e9:.3f} GB")
            print(f"✗ Datasets will stay on CPU - no memory reservation needed")
            print(f"============================================\n")
        
        return dataset_reserved_memory
    return 0


def train(config=None):
    """
    Main training function that can be called standalone or by wandb sweep.
    
    Args:
        config: Optional dict of configuration overrides. If None, uses defaults and command-line args.
    """

    default_config = get_default_config()

    # Clear GPU memory at the start of training run
    clear_gpu_memory()
    
    # If config is provided (e.g., from wandb sweep), merge it with defaults
    if config is not None:
        default_config.update(config)
    
    # Store config in globals for configurator.py compatibility
    for k, v in default_config.items():
        globals()[k] = v
    
    # Execute configurator.py if running standalone (not in sweep mode)
    if config is None and os.path.exists('configurator.py'):
        # Only execute configurator if not in wandb sweep mode
        config_keys = list(default_config.keys())
        exec(open('configurator.py').read(), globals())
        # Update default_config with any overrides from configurator
        for k in config_keys:
            default_config[k] = globals()[k]

    # Set random seed and backend configurations
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(default_config['seed'])
        torch.cuda.manual_seed_all(default_config['seed'])  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    
    
    if default_config['init_from'] == 'resume':
        custom_name = default_config['wandb_run_name']
        wandb.run.name = custom_name
        print(f"Resuming wandb run: {custom_name}")
    else:
        custom_name = set_wandb_name(default_config)

    if default_config['wandb_run_name'] is None:
        default_config['wandb_run_name'] = custom_name

    # Validate vocab_size
    assert default_config['randomize_vocab_size'] == 'auto' or (default_config['randomize_vocab_size'] == default_config['graph_d'] * (default_config['graph_l'] - 1) + 1), \
        f"randomize_vocab_size must be >= graph_d * (graph_l - 1) + 1"
    
    # Generate/load dataset
    gen = InWeightsPathStar(
        d=default_config['graph_d'],
        l=default_config['graph_l'],
        randomize_vocab_size=default_config['randomize_vocab_size'],
        holdout_percentage=default_config['graph_holdout_percentage'],
    )

    # NOTE: num_pause_tokens is NOT passed here - pause tokens are added at runtime
    gen.generate_dataset_if_needed(
        use_undirected=default_config['use_undirected'],
        use_directional_tokens=default_config['use_directional_tokens'],
        predict_direction_for_edge_task=default_config['predict_direction_for_edge_task'],
        use_task_tokens_in_path=default_config.get('use_task_tokens_in_path', False),
    )
    
    meta, paths_data, edges_data, val_data = gen.load_dataset()
    
    # num_pause_tokens is a RUNTIME config parameter (not stored in dataset)
    # Add it to meta for use throughout training
    num_pause_tokens = default_config['num_pause_tokens']
    meta['num_pause_tokens'] = num_pause_tokens
    
    # Update block_size and path_context_length to include pause tokens
    # The dataset stores *_base values (without pause tokens)
    meta['block_size'] = meta['block_size_base'] + num_pause_tokens
    meta['path_context_length'] = meta['path_context_length_base'] + num_pause_tokens
    
    print(f"Runtime pause tokens: {num_pause_tokens}")
    print(f"  block_size_base (stored): {meta['block_size_base']}")
    print(f"  block_size (with pause): {meta['block_size']}")
    print(f"  path_context_length_base (stored): {meta['path_context_length_base']}")
    print(f"  path_context_length (with pause): {meta['path_context_length']}")
    
    # Only compute token colors if live display is enabled (saves compute)
    if default_config['live_display']:
        print("Precomputing token colors...")
        token_colors, RESET = compute_token_colors(paths_data, val_data, meta)
        meta['token_colors'] = token_colors
        meta['RESET_COLOR'] = RESET
    else:
        meta['token_colors'] = {}
        meta['RESET_COLOR'] = ''
    
    meta['randomize_vocab_size'] = gen.randomize_vocab_size
    # Extract graph parameters from metadata
    # graph_length is used for PATH task token-level metrics, so account for GT-interleaving.
    graph_length = int(meta.get('path_target_length', meta['l']))
    graph_spokes = meta['d']
    holdout_ratio = meta['holdout_percentage']

    num_holdout = math.ceil(graph_spokes * holdout_ratio)
    
    # Get dataset sizes from metadata
    paths_size = meta['PATHS_DATASET_SIZE']
    edges_size = meta['EDGES_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    if default_config['edge_only']:
        print(f"Training dataset composition (EDGE ONLY):")
        print(f"  Edges: {edges_size}")
        print(f"  Paths: SKIPPED (edge_only=True)")
        print(f"  Validation: DISABLED (edge_only=True)")
        print(f"  Total: {edges_size} samples (100% edges)")
    else:
        print(f"Training dataset composition (INTERLEAVED):")
        print(f"  Paths (original): {paths_size}")
        print(f"  Edges: {edges_size}")
        if default_config['balance_interleaved_datasets'] and paths_size < edges_size:
            print(f"  Paths (after balancing): {edges_size}")
            print(f"  Total Combined: {edges_size * 2} samples (50% paths, 50% edges)")
        else:
            print(f"  Total Combined: {paths_size + edges_size} samples ({paths_size} paths, {edges_size} edges)")
    
    # Log scheduled sampling configuration
    p_sub = default_config.get('p_autoregressive_substitution', 0.0)
    if p_sub > 0:
        print(f"\n=== Scheduled Sampling (Autoregressive Substitution) ===")
        print(f"  p_autoregressive_substitution: {p_sub}")
        print(f"  PATH sequences: With probability {p_sub}, teacher-forced tokens will be")
        print(f"                  substituted with model predictions during training.")
        print(f"  EDGE sequences: Always use pure teacher forcing (no substitution).")
        print(f"  Note: This may slow down training but improves inference robustness.")
        print(f"=========================================================\n")

    # Auto-detect device
    device, device_type, gpu_id = detect_device(default_config)
    
    # Enable PyTorch anomaly detection if requested (slow but thorough NaN debugging)
    if default_config.get('detect_anomaly', False):
        torch.autograd.set_detect_anomaly(True)
        LiveTrainingPanel.CONSOLE.print("[yellow]⚠️  Anomaly detection ENABLED - training will be slower but NaN sources will be caught[/yellow]")
    

    ptdtype, dtype = set_dtype(default_config)

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    os.makedirs(default_config['out_dir'], exist_ok=True)
    checkpoint_filename = f'ckpt_{custom_name}.pt' if custom_name else "ckpt.pt"
    meta["checkpoint_filename"] = checkpoint_filename
    print(f"Checkpoint will be saved as: {checkpoint_filename}")

    model, model_args, checkpoint, iter_num = initalize_model(device, meta, default_config, checkpoint_filename)
    meta['model_args'] = model_args
    meta['model'] = model

    # Calculate dataset structure from metadata (needed for memory calculation)
    
    # Calculate sequence lengths
    paths_seq_length = len(paths_data) // paths_size
    edges_seq_length = len(edges_data) // edges_size
    val_seq_length = len(val_data) // VAL_DATASET_SIZE
    
    # Reshape data for easier indexing (needed for memory calculation)
    paths_data = paths_data.reshape(paths_size, paths_seq_length)
    edges_data = edges_data.reshape(edges_size, edges_seq_length)
    val_data = val_data.reshape(VAL_DATASET_SIZE, val_seq_length)

    # Combine datasets for training
    if default_config['edge_only']:
        # Edge-only mode: use only edges, no paths
        # Make a copy to avoid read-only array issues during shuffling
        combined_data = edges_data.copy()
        combined_size = edges_size
        print(f"Using edge-only dataset: {combined_size} samples")
    else:
        # Interleaved mode: combine paths and edges
        # Optionally balance datasets by upsampling paths to match edges
        # Note: edges_size >= paths_size always holds for PathStar graphs
        # Uses deterministic duplication (tiling) rather than random sampling to avoid noise
        if default_config['balance_interleaved_datasets'] and paths_size < edges_size:
            # Tile the paths dataset: repeat full copies then take remainder from start
            num_full_copies = edges_size // paths_size
            remainder = edges_size % paths_size
            print(f"Balancing: Upsampling paths from {paths_size} to {edges_size} ({num_full_copies} full copies + {remainder} remainder)")
            indices = np.concatenate([
                np.tile(np.arange(paths_size), num_full_copies),
                np.arange(remainder)
            ])
            paths_data_balanced = paths_data[indices]
        else:
            if not default_config['balance_interleaved_datasets']:
                print(f"Skipping dataset balancing (paths: {paths_size}, edges: {edges_size})")
            paths_data_balanced = paths_data

        # Concatenate paths and edges (creates a new array)
        combined_data = np.concatenate((paths_data_balanced, edges_data), axis=0)
        combined_size = combined_data.shape[0]
    
    # Shuffle the combined data initially
    np.random.shuffle(combined_data)
    
    # Calculate memory for combined dataset
    # In edge_only mode, we don't load validation data to GPU
    if default_config['edge_only']:
        dataset_reserved_memory = determine_dataset_in_device_size(device, device_type, combined_data, np.array([]), np.array([]))
    else:
        dataset_reserved_memory = determine_dataset_in_device_size(device, device_type, combined_data, np.array([]), val_data)
    
    # Target batch size is the combined size
    target_bs_ref = combined_size

    train_batch_size = calculate_optimal_batch_size_for_training(
        model, meta['block_size'], meta['randomize_vocab_size'], device, dtype,
        default_config['gradient_accumulation_steps'],
        reserved_memory=dataset_reserved_memory,
        target_batch_size=target_bs_ref
    )

    # Optional user cap on training microbatch size (safer than overriding upward).
    # If you want a smaller batch to fit memory or to change optimization dynamics,
    # set `batch_size` in configurator.py or a wandb sweep config.
    user_batch_size = default_config.get('batch_size', None)
    if user_batch_size is not None:
        try:
            user_batch_size = int(user_batch_size)
        except (TypeError, ValueError):
            print(f"WARNING: Ignoring invalid batch_size={user_batch_size!r}; expected an int.")
        else:
            if user_batch_size <= 0:
                print(f"WARNING: Ignoring invalid batch_size={user_batch_size}; must be > 0.")
            else:
                if user_batch_size > train_batch_size:
                    print(
                        f"NOTE: batch_size cap {user_batch_size} > auto batch_size {train_batch_size}; "
                        f"keeping auto batch_size to avoid OOM."
                    )
                else:
                    train_batch_size = user_batch_size
                print(
                    f"Training microbatch size: {train_batch_size} "
                    f"(effective: {train_batch_size * default_config['gradient_accumulation_steps']})"
                )
    
    # Calculate training iteration parameters
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    # In interleaved mode, epoch is 1 pass over combined dataset
    batches_per_epoch = int(np.ceil(combined_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
    max_iters = default_config['epochs'] * batches_per_epoch
    
    print(f"\n=== Training Schedule (Interleaved) ===")
    print(f"Total samples: {combined_size}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total iterations: {max_iters}")
    print(f"=========================\n")

    meta['max_iters'] = max_iters
    meta['batches_per_epoch'] = batches_per_epoch
    
    val_batch_size = min(num_holdout, train_batch_size)
    eval_iters = int(np.ceil(VAL_DATASET_SIZE / val_batch_size))
    # Calculate learning rate schedule parameters
    warmup_iters = int(max_iters * default_config['warmup_frac'])
    lr_decay_iters = int(max_iters * default_config['lr_decay_frac'])
    meta['warmup_iters'] = warmup_iters
    meta['lr_decay_iters'] = lr_decay_iters
    
    # Initialize LR scheduler
    lr_scheduler_obj = initialize_lr_scheduler(
        default_config, 
        warmup_iters, 
        lr_decay_iters, 
        console=default_config.get('console')
    )
    
    
    # Initialize GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Optimizer
    optimizer = model.configure_optimizers(
        default_config['weight_decay'],
        default_config['learning_rate'],
        (default_config['beta1'], default_config['beta2']),
        device_type
    )
    if default_config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Load LR scheduler state if available and using ReduceLROnPlateau
        if lr_scheduler_obj is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler_obj.load_state_dict(checkpoint['lr_scheduler'])
            LiveTrainingPanel.CONSOLE.print(f"[cyan]Loaded LR scheduler state (current_lr={lr_scheduler_obj.current_lr:.2e})[/cyan]")

    meta['optimizer'] = optimizer
    
    # Compile model
    if default_config['compile']:
        if device_type == 'mps':
            print("WARNING: Disabling torch.compile on MPS due to known instability (Inductor backend issues).")
            default_config['compile'] = False
        else:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)
    
    # Initialize wandb (skip if already initialized by sweep agent)
    if default_config['wandb_log'] and wandb.run is None:
        wandb.init(
            project=default_config['wandb_project'],
            name=default_config['wandb_run_name'],
            config={k: v for k, v in default_config.items() if k != 'console'}
        )
    
    # Init tracking variables
    if default_config['init_from'] == 'resume':
        iter_num = checkpoint['iter_num']
    else:
        iter_num = 0

    # Calculate and log theoretical minimum loss
    if default_config['wandb_log'] and wandb.run is not None:
        # Only relevant when predicting edge endpoint (1 out of d options)
        # If predicting direction, the task is deterministic/binary so min loss is 0.
        predict_dir = default_config.get('predict_direction_for_edge_task', True)
        
        # Calculate N_paths and N_edges effective samples based on sampling strategy
        if default_config['edge_only']:
            n_path_samples = 0  # No paths in edge_only mode
            n_edge_samples = edges_size
        elif default_config.get('balance_interleaved_datasets', True) and paths_size < edges_size:
            n_path_samples = edges_size # Effective samples due to upsampling
            n_edge_samples = edges_size # Edges are never upsampled
        else:
            n_path_samples = paths_size
            n_edge_samples = edges_size # Edges are never upsampled
        
        # Calculate total tokens contributing to loss
        # Paths: 'path_target_length' tokens per sample (loss contribution is 0)
        # Edges: 1 token per sample (masked)
        path_target_len = int(meta.get('path_target_length', meta['l']))
        total_tokens = (n_path_samples * path_target_len) + (n_edge_samples * 1)
        
        # DEBUG: Count how many GT edges from root are actually in the dataset
        root_vertex = meta['root_vertex']
        GT_token = meta['special_tokens']['GT']
        EDGE_token = meta['special_tokens']['EDGE']
        num_gt_edges_from_root = 0
        edges_data_to_check = edges_data
        
        for i in range(len(edges_data_to_check)):
            seq = edges_data_to_check[i]
            # Check if this is an edge starting from root with GT direction
            # Format depends on predict_direction_for_edge_task:
            # - False: [EDGE, u, GT, v, ...]
            # - True: [EDGE, u, v, GT, ...]
            if seq[0] != EDGE_token:
                continue
            
            if predict_dir:
                # Format: [EDGE, u, v, GT/LT, ...]
                # Check if u == root and direction == GT
                if len(seq) >= 4 and seq[1] == root_vertex and seq[3] == GT_token:
                    num_gt_edges_from_root += 1
            else:
                # Format: [EDGE, u, GT/LT, v, ...]
                # Check if u == root and direction == GT  
                if len(seq) >= 4 and seq[1] == root_vertex and seq[2] == GT_token:
                    num_gt_edges_from_root += 1
        
        print(f"DEBUG: Found {num_gt_edges_from_root} GT edges from root in edge dataset (expected: {meta['d']})")
        
        # Calculate total entropy (Loss Mass)
        # Paths: Assumed perfect memorization -> 0 entropy.
        # Edges: 
        #   If predict_dir=True ([EDGE] u v -> GT/LT): Deterministic -> 0 entropy.
        #   If predict_dir=False ([EDGE] u GT/LT -> v): 
        #     - From Root (GT): d branches. Target is 1/d. Entropy = log(d). There are d such edges.
        #     - From Root (LT): Not possible (Root has no parent).
        #     - From Node (GT): 1 branch (linear path). Entropy = 0.
        #     - From Node (LT): 1 branch (parent). Entropy = 0.
        #     Total Entropy Mass = num_gt_edges_from_root * log(num_gt_edges_from_root)
        
        if predict_dir:
            optimal_loss = 0.0
        else:
            # Use actual count instead of meta['d']
            d_val = num_gt_edges_from_root if num_gt_edges_from_root > 0 else meta['d']
            entropy_mass = d_val * math.log(d_val) if d_val > 0 else 0.0
            optimal_loss = entropy_mass / total_tokens
            
        print(f"=== Theoretical Minimum Loss Calculation ===")
        print(f"  d (graph spokes): {meta['d']}")
        print(f"  l (path length): {meta['l']}")
        print(f"  GT edges from root: {num_gt_edges_from_root}")
        print(f"  Path samples (after upsampling): {n_path_samples}")
        print(f"  Edge samples: {n_edge_samples}")
        print(f"  Path tokens per sample: {path_target_len}")
        print(f"  Total path tokens: {n_path_samples * path_target_len}")
        print(f"  Total edge tokens: {n_edge_samples}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Entropy mass: {entropy_mass:.6f} ({d_val} * log({d_val}))")
        print(f"  Theoretical Minimum Loss: {optimal_loss:.8f}")
        print(f"=============================================")
        
        # Store for reference during training
        meta['theoretical_min_loss'] = optimal_loss
        meta['entropy_mass'] = entropy_mass
        meta['total_contributing_tokens'] = total_tokens
    
    meta_vocab_size = meta['randomize_vocab_size']
    print(f"found randomize_vocab_size= {meta_vocab_size}")
    
    if 'special_tokens' in meta:
        pause_token_id = meta['special_tokens'].get('PAUSE')
        pad_token_id = meta['special_tokens'].get('PAD')
    else:
        pause_token_id = meta.get('pause_token')
        pad_token_id = meta.get('pad_token')

    # Whether sequences include explicit task prefix tokens (PATH/EDGE).
    # This is required to reliably apply task-aware masking in the combined (interleaved) dataset.
    
    if pause_token_id is not None or pad_token_id is not None:
        print(f"Loaded special tokens: PAUSE={pause_token_id}, PAD={pad_token_id}")
        print("Note: PAD tokens will be masked in loss calculation (ignore_index=-1)")
    else:
        print("Warning: No special tokens found in metadata. PAD masking will be disabled.")
    
    itos = meta.get('itos', {})
    if itos:
        print(f"Loaded vocabulary mappings: {len(itos)} tokens")
    
    print(f"Dataset info:")
    print(f"  Paths: {paths_size} sequences of length {paths_seq_length} (stored, without pause tokens)")
    print(f"  Edges: {edges_size} sequences of length {edges_seq_length} (stored, without pause tokens)")
    print(f"  Val: {VAL_DATASET_SIZE} sequences of length {val_seq_length} (stored, without pause tokens)")
    print(f"  Block size (with pause tokens): {meta['block_size']}")
    
    assert paths_seq_length == edges_seq_length, f"Sequence length mismatch: paths={paths_seq_length}, edges={edges_seq_length}"
    
    # DETERMINE OPTIMAL STORAGE DTYPE
    # We use int32 or int16 for storage to save memory/bandwidth, but cast to Long (int64) for model input
    # This is safe because we only cast at the last moment
    max_vocab_idx = max(meta_vocab_size, max([x for x in meta['special_tokens'].values() if isinstance(x, int)] + [0]))
    if max_vocab_idx < 32767:
        storage_dtype = torch.int16
        print(f"Optimizing memory: Using int16 for dataset storage (max token: {max_vocab_idx})")
    elif max_vocab_idx < 2147483647:
        storage_dtype = torch.int32
        print(f"Optimizing memory: Using int32 for dataset storage (max token: {max_vocab_idx})")
    else:
        storage_dtype = torch.int64
        print(f"Using standard int64 for dataset storage (max token: {max_vocab_idx})")
        
    # Pre-process datasets: Create X (inputs) and Y (targets) tensors with optimized dtype
    # We apply masking to Y statically here to avoid doing it in the inner loop
    # 
    # PAUSE TOKENS: The stored dataset does NOT contain pause tokens. We add them here
    # at preprocessing time based on the runtime config (num_pause_tokens).
    
    num_pause_tokens = meta['num_pause_tokens']
    PATH_TOKEN = meta['special_tokens'].get('PATH')
    EDGE_TOKEN = meta['special_tokens'].get('EDGE')
    
    def preprocess_dataset(data_np, dataset_type='paths'):
        """
        Preprocess dataset by adding pause tokens and creating X/Y tensors.
        
        Args:
            data_np: numpy array of sequences (WITHOUT pause tokens)
            dataset_type: 'paths', 'edges', 'combined', or 'val'
                - 'paths'/'val': Insert pause tokens between leaf and path
                - 'edges': Add padding at the end
                - 'combined': Smart handling based on task token
        
        Returns:
            X, Y tensors with pause tokens inserted and masking applied
        """
        # Convert to tensor
        data_tensor = torch.from_numpy(data_np.astype(np.int64))
        
        # Add pause tokens based on dataset type
        if num_pause_tokens > 0:
            if dataset_type in ('paths', 'val'):
                # PATH sequences: insert pause tokens after [PATH, leaf]
                data_tensor = add_pause_tokens_to_batch(
                    data_tensor, num_pause_tokens, pause_token_id
                )
            elif dataset_type == 'edges':
                # EDGE sequences: add padding at the end
                data_tensor = add_pause_tokens_to_edges(
                    data_tensor, num_pause_tokens, pad_token_id
                )
            elif dataset_type == 'combined':
                # Combined: identify PATH vs EDGE sequences and handle separately
                if PATH_TOKEN is not None and EDGE_TOKEN is not None:
                    is_path = (data_tensor[:, 0] == PATH_TOKEN)
                    is_edge = (data_tensor[:, 0] == EDGE_TOKEN)
                    
                    # Process in batches by type for efficiency
                    result = torch.zeros(
                        (data_tensor.shape[0], data_tensor.shape[1] + num_pause_tokens),
                        dtype=data_tensor.dtype
                    )
                    
                    if is_path.any():
                        path_data = add_pause_tokens_to_batch(
                            data_tensor[is_path], num_pause_tokens, pause_token_id
                        )
                        result[is_path] = path_data
                    
                    if is_edge.any():
                        edge_data = add_pause_tokens_to_edges(
                            data_tensor[is_edge], num_pause_tokens, pad_token_id
                        )
                        result[is_edge] = edge_data
                    
                    data_tensor = result
                else:
                    raise ValueError("Combined dataset requires PATH/EDGE tokens")
        
        # Split into X and Y
        # X: 0 to L-1
        # Y: 1 to L
        X = data_tensor[:, :-1].to(storage_dtype)
        Y = data_tensor[:, 1:].clone()
        
        # Apply masking to Y
        if pad_token_id is not None:
            Y[Y == pad_token_id] = -1
        if pause_token_id is not None:
            Y[Y == pause_token_id] = -1
            
        # Cast Y to storage dtype
        Y = Y.to(storage_dtype)
        
        return X, Y

    # Create tensors and load to GPU if pre-calculated decision indicates they fit
    # NOTE: preprocess_dataset adds pause tokens at runtime based on config
    print(f"Pre-processing combined dataset (adding {num_pause_tokens} pause tokens)...")
    if default_config['edge_only']:
        # In edge_only mode, combined_data contains only edges
        combined_X, combined_Y = preprocess_dataset(combined_data, dataset_type='edges')
    else:
        # In interleaved mode, combined_data contains both paths and edges
        combined_X, combined_Y = preprocess_dataset(combined_data, dataset_type='combined')
    print(f"Created pre-processed combined tensors: X={combined_X.shape}, Y={combined_Y.shape}, dtype={storage_dtype}")
    
    # Store validation data with optimized dtype (skip in edge_only mode)
    # NOTE: Validation data also needs pause tokens added
    if default_config['edge_only']:
        val_data_tensor = None
        print("Skipping validation data preprocessing (edge_only=True)")
    else:
        val_data_with_pause = torch.from_numpy(val_data.astype(np.int64))
        if num_pause_tokens > 0:
            val_data_with_pause = add_pause_tokens_to_batch(
                val_data_with_pause, num_pause_tokens, pause_token_id
            )
        val_data_tensor = val_data_with_pause.to(storage_dtype)
    
    datasets_on_gpu = False
    if device_type == 'cuda':
        # Use pre-calculated decision: if reserved_memory > 0, datasets will be loaded to GPU
        if dataset_reserved_memory > 0:
            print(f"\n=== Loading Datasets to GPU ===")
            print(f"Reserved memory: {dataset_reserved_memory / 1e9:.3f} GB")
            print("✓ Loading datasets to GPU for faster training")
            combined_X = combined_X.pin_memory().to(device, non_blocking=True)
            combined_Y = combined_Y.pin_memory().to(device, non_blocking=True)
            if val_data_tensor is not None:
                val_data_tensor = val_data_tensor.pin_memory().to(device, non_blocking=True)
            datasets_on_gpu = True
            print(f"===================================\n")
        else:
            print(f"\n=== Dataset Loading Decision ===")
            print("✗ Datasets will stay on CPU (will transfer batches on-demand)")
            # Pin memory on CPU for faster transfers
            combined_X = combined_X.pin_memory()
            combined_Y = combined_Y.pin_memory()
            print(f"===================================\n")
            datasets_on_gpu = False
    else:
        # For non-CUDA devices, always keep on CPU or move to device as appropriate
        # On MPS, memory is unified, so .to(device) is practically zero-copy for large tensors
        if device_type != 'cpu':
            combined_X = combined_X.to(device)
            combined_Y = combined_Y.to(device)
            if val_data_tensor is not None:
                val_data_tensor = val_data_tensor.to(device)
            datasets_on_gpu = True
        else:
            datasets_on_gpu = False
    
    # Keep NumPy versions for evaluate_samples (will optimize separately)
    paths_data_np = paths_data
    edges_data_np = edges_data
    val_data_np = val_data
    
    # Initialize epoch indices for sampling without replacement
    combined_epoch_indices = np.arange(combined_size)
    # Perform initial shuffle once
    np.random.shuffle(combined_epoch_indices)
    
    combined_batch_idx = 0
    combined_epoch_completed = False
    
    # Initialize validation indices (skip in edge_only mode)
    if not default_config['edge_only']:
        val_epoch_indices = np.arange(VAL_DATASET_SIZE)
        # Perform initial shuffle for validation
        np.random.shuffle(val_epoch_indices)
        val_batch_idx = 0
        val_epoch_completed = False
    else:
        val_epoch_indices = None
        val_batch_idx = 0
        val_epoch_completed = False
    
    # DONE
    # Updated get_batch to use pre-processed tensors and handle dtype casting

    last_mask_debug_str = None

    def _format_mask_debug(x, y_before, y_after, dataset_label):
        """Return a compact debug string showing which Y positions are kept."""
        try:
            max_samples = int(default_config.get('debug_masking_samples', 2))
            max_len = int(default_config.get('debug_masking_max_len', 32))
        except Exception:
            max_samples, max_len = 2, 32

        def tok_str(t):
            if t == -1:
                return "<MASK>"
            # itos keys might be str or int depending on meta serialization
            return str(itos.get(int(t), t))

        x_cpu = x.detach().cpu()
        yb_cpu = y_before.detach().cpu()
        ya_cpu = y_after.detach().cpu()

        b = min(x_cpu.size(0), max_samples)
        lines = [f"Mask debug ({dataset_label})  keep=1 where Y_after!=-1"]
        for i in range(b):
            x_row = x_cpu[i].tolist()[:max_len]
            yb_row = yb_cpu[i].tolist()[:max_len]
            ya_row = ya_cpu[i].tolist()[:max_len]
            keep = [1 if t != -1 else 0 for t in ya_row]
            kept_count = sum(keep)
            lines.append(f"- sample {i}: kept {kept_count}/{len(keep)}")
            lines.append("  X : " + " ".join(tok_str(t) for t in x_row))
            lines.append("  Yb: " + " ".join(tok_str(t) for t in yb_row))
            lines.append("  m : " + " ".join(str(k) for k in keep))
            lines.append("  Ya: " + " ".join(tok_str(t) for t in ya_row))
        return "\n".join(lines)

    def apply_task_specific_target_mask(x, y, dataset):
        """
        Apply task-specific masking to targets (Y) using -1 as ignore_index.

        Correct masking behavior:
        - Edge task: only compute loss on the final supervised token.
          If predict_direction_for_edge_task=False (predict endpoint v):
            Sequence: [<EDGE>, u, <GT/LT (optional)>, v, <PAD>...]
            Targets Y: [u, <GT/LT (optional)>, v, <PAD>...]
            Mask: ignore everything except the v position.
          If predict_direction_for_edge_task=True (predict direction):
            Sequence: [<EDGE>, u, v, <GT/LT>, <PAD>...]
            Targets Y: [u, v, <GT/LT>, <PAD>...]
            Mask: ignore everything except the direction position.
        - Path task: ignore targets predicting <PAUSE> (handled elsewhere) and also ignore the
          first leaf target when <PATH> task prefix token is used.
          Sequence: [<PATH>, leaf, <PAUSE>x n, root, n2, ..., nℓ]
          Targets Y:   [leaf, <PAUSE>x n, root, n2, ..., nℓ]
          Mask: ignore leaf target and all <PAUSE> targets; keep loss from root onward.
        - Combined: apply per-sample based on task/direction tokens.
        """
        # y is (B, T) where T == x.size(1)
        use_directional_tokens = meta.get('use_directional_tokens', True)

        special = meta.get('special_tokens', {}) or {}
        EDGE = special.get('EDGE')
        PATH = special.get('PATH')
        GT = special.get('GT')
        LT = special.get('LT')

        def mask_edges(y_in):
            # Edge task: keep loss on exactly one supervised token.
            # - predict_direction_for_edge_task=False: predict v (final endpoint)
            #   x: [EDGE] u [GT/LT] v ...
            #   y:     u [GT/LT] v ...
            #   keep y index (dir? + 1)  == 2 when directional tokens are used
            # - predict_direction_for_edge_task=True: predict direction
            #   x: [EDGE] u v [GT/LT] ...
            #   y:     u v [GT/LT] ...
            #   keep y index 2
            predict_dir = bool(meta.get('predict_direction_for_edge_task', False))
            if predict_dir:
                target_idx = 2
            else:
                d = 1 if use_directional_tokens else 0
                target_idx = d + 1
            y_out = torch.full_like(y_in, -1)
            if 0 <= target_idx < y_in.size(1):
                y_out[:, target_idx] = y_in[:, target_idx]
            return y_out

        def mask_paths(x_in, y_in):
            y_out = y_in.clone()
            # PATH task token exists, the first target token is the leaf; ignore it.
            path_rows = (x_in[:, 0] == PATH)
            y_out[path_rows, 0] = -1  # ignore the first target token (leaf)
            return y_out

        if dataset == 'edges':
            return mask_edges(y)
        if dataset == 'paths':
            return mask_paths(x, y)
        if dataset == 'combined':
            # Need to determine per-sample task type using task tokens
            if EDGE is None or PATH is None:
                raise ValueError("Combined dataset requires EDGE and PATH tokens in metadata")
            is_edge = (x[:, 0] == EDGE)
            is_path = (x[:, 0] == PATH)

            y_out = y.clone()
            if is_edge.any():
                y_out[is_edge] = mask_edges(y_out[is_edge])
            if is_path.any():
                y_out[is_path] = mask_paths(x[is_path], y_out[is_path])
            return y_out

        return y

    def get_batch(dataset):
        """Sample a batch from the combined or validation dataset"""
        nonlocal val_batch_idx, val_epoch_indices, combined_batch_idx, combined_epoch_indices
        nonlocal combined_epoch_completed, val_epoch_completed
        nonlocal last_mask_debug_str

        if dataset == 'combined':
            batch_idx = combined_batch_idx
            epoch_indices = combined_epoch_indices
            dataset_size = combined_size
            X_source = combined_X
            Y_source = combined_Y
            epoch_completed = combined_epoch_completed
        elif dataset == 'val':
            if default_config['edge_only']:
                raise ValueError("Validation dataset not available in edge_only mode")
            # Validation logic remains largely same but we need to handle X/Y/Masking dynamically or pre-process it too.
            # For simplicity, we'll keep dynamic slicing for val since it's infrequent
            # But let's use the optimized storage tensor
            batch_idx = val_batch_idx
            epoch_indices = val_epoch_indices
            dataset_size = VAL_DATASET_SIZE
            X_source = None # Special case
            Y_source = None
            dataset_tensors = val_data_tensor
            epoch_completed = val_epoch_completed
        else:
            raise ValueError("This should not happen")
        
        # Smart shuffling strategy:
        # - Only shuffle at epoch boundaries (when batch_idx == 0 AND we've completed at least one epoch)
        # - This prevents constant reshuffling when batch_size >= dataset_size
        # - For dataset_size >> batch_size, this shuffles once per epoch (proper behavior)
        if batch_idx == 0 and epoch_completed:
            np.random.shuffle(epoch_indices)
        
        # Get batch indices
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, dataset_size)
        batch_seq_indices = epoch_indices[start_idx:end_idx]
        
        # Update batch index for next call
        # If we've exhausted the dataset, wrap to 0 and mark epoch as completed
        if end_idx >= dataset_size:
            batch_idx = 0
            epoch_completed = True
        else:
            batch_idx = batch_idx + 1

        if dataset == 'combined':
            combined_batch_idx = batch_idx
            combined_epoch_completed = epoch_completed
        elif dataset == 'val':
            val_batch_idx = batch_idx
            val_epoch_completed = epoch_completed
        else:
            raise ValueError("Invalid dataset type")


        if dataset == 'val':
            # Special handling for validation (dynamic)
            if datasets_on_gpu:
                sequences = dataset_tensors[batch_seq_indices]
            else:
                sequences = dataset_tensors[batch_seq_indices]
                if device_type in ['cuda', 'mps']:
                    sequences = sequences.to(device, non_blocking=True)
            
            # Cast to Long for model input (important for embedding)
            sequences = sequences.to(torch.long)
            
            x = sequences[:, :-1]
            y = sequences[:, 1:].clone() # Clone needed for masking
            if pad_token_id is not None: y[y == pad_token_id] = -1
            if pause_token_id is not None: y[y == pause_token_id] = -1
            # Validation set contains only path sequences; apply path-specific masking.
            if default_config.get('debug_masking') and (iter_num % default_config.get('log_interval', 100) == 0):
                y_before = y.clone()
                y_after = apply_task_specific_target_mask(x, y, 'paths')
                last_mask_debug_str = _format_mask_debug(x, y_before, y_after, "val(paths)")
                y = y_after
            else:
                y = apply_task_specific_target_mask(x, y, 'paths')
            return x, y
        
        # Standard training batch retrieval
        # Extract sequences (from GPU if available, otherwise from CPU and transfer)
        if datasets_on_gpu:
            x = X_source[batch_seq_indices]
            y = Y_source[batch_seq_indices]
        else:
            x = X_source[batch_seq_indices]
            y = Y_source[batch_seq_indices]
            if device_type in ['cuda', 'mps']:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
        
        # FINAL MILE CASTING: Ensure compatibility with GPU kernels (e.g. Embedding)
        # Casting here is cheap (on small batch) compared to storing full 64-bit dataset
        x = x.to(torch.long)
        y = y.to(torch.long)

        # Apply task-specific masking
        # In edge_only mode, all combined data is edges
        mask_dataset_type = 'edges' if (dataset == 'combined' and default_config['edge_only']) else dataset
        
        if default_config.get('debug_masking') and (iter_num % default_config.get('log_interval', 100) == 0):
            y_before = y.clone()
            y_after = apply_task_specific_target_mask(x, y, mask_dataset_type)
            last_mask_debug_str = _format_mask_debug(x, y_before, y_after, mask_dataset_type)
            y = y_after
        else:
            y = apply_task_specific_target_mask(x, y, mask_dataset_type)
        
        return x, y
    
    @torch.no_grad()
    def estimate_metrics(split, print_samples=False):
        """Estimate loss and metrics on validation or training split. Total and per token"""
        out = {}
        model.eval()
        
        # Determine data and size
        # NOTE: data_source_raw is stored data WITHOUT pause tokens
        # We add pause tokens on-the-fly when needed
        if split == 'val':
            nonlocal val_batch_idx, val_epoch_indices
            # Reset validation batch state for reproducible evaluation
            val_batch_idx = 0
            np.random.shuffle(val_epoch_indices)
            num_iters = eval_iters
            data_source_raw = val_data
            data_size = VAL_DATASET_SIZE
        else: # train
            # For training, we sample randomly from paths_data
            # We use a limited number of iterations similar to validation
            num_iters = eval_iters
            data_source_raw = paths_data
            data_size = paths_size
            
        token_losses = {i: [] for i in range(1, graph_length + 1)}
        batch_losses = torch.zeros(num_iters)
        
        for k in range(num_iters):
            if split == 'val':
                X, Y = get_batch('val')
            else:
                # Manual sampling for training paths to ensure we only evaluate on paths
                # (not edges) even though training uses combined dataset
                idx = np.random.randint(0, data_size, train_batch_size)
                batch = torch.from_numpy(data_source_raw[idx].astype(np.int64))
                # Add pause tokens to raw batch (stored data doesn't have them)
                if num_pause_tokens > 0:
                    batch = add_pause_tokens_to_batch(batch, num_pause_tokens, pause_token_id)
                batch = batch.to(device)
                X = batch[:, :-1].contiguous()
                Y = batch[:, 1:].contiguous()
                if pad_token_id is not None: Y[Y == pad_token_id] = -1
                if pause_token_id is not None: Y[Y == pause_token_id] = -1
                # Training metrics here are for paths; apply path-specific masking.
                Y = apply_task_specific_target_mask(X, Y, 'paths')

            with ctx:
                logits, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
            batch_losses[k] = loss.item()
            
            per_token_losses_in_batch = compute_per_token_loss_with_teacher_forcing(meta, logits, X, Y, range(1, graph_length + 1), task_type='path')
            for token_pos, (token_loss_sum, batch_size_local) in per_token_losses_in_batch.items():
                if not math.isnan(token_loss_sum):
                    token_losses[token_pos].append((token_loss_sum, batch_size_local))
        
        out[f'{split}'] = batch_losses.mean()
        
        if token_losses[1]:
            out[f'{split}_per_token'] = {
                token_pos: (
                    sum(loss_sum * batch_size for loss_sum, batch_size in losses_list) / 
                    sum(batch_size for _, batch_size in losses_list)
                ) if losses_list else float('nan')
                for token_pos, losses_list in token_losses.items()
            }
        
        # Compute per-token accuracy (autoregressive)
        # Use small number of samples for speed
        # NOTE: Need to add pause tokens to raw data before passing to this function
        num_samples_for_accuracy = min(100, data_size)
        data_source_with_pause = torch.from_numpy(data_source_raw.astype(np.int64))
        if num_pause_tokens > 0:
            data_source_with_pause = add_pause_tokens_to_batch(
                data_source_with_pause, num_pause_tokens, pause_token_id
            )
        per_token_accuracy, generated_text, full_path_accuracy = compute_per_token_accuracy_autoregressive(
            ctx, model, meta, data_source_with_pause.numpy(), num_samples_for_accuracy, device, print_samples
        )
        out[f'{split}_per_token_accuracy'] = per_token_accuracy
        out[f'{split}_full_path_accuracy'] = full_path_accuracy
        
        if print_samples and split == 'val':
            out['generated_text'] = generated_text
        
        model.train()
        return out
    
    
    # Setup live display
    live_panel = LiveTrainingPanel(default_config)

    
    # Track running average of training loss for comparison with theoretical minimum
    running_loss_sum = 0.0
    running_loss_count = 0

    # Initialize with first batch from combined dataset
    X, Y = get_batch('combined')
    

    with live_panel.context:
        # Training loop
        t0 = time.time()
        while True:
            # Set learning rate based on scheduler
            lr = get_lr(iter_num, warmup_iters, lr_decay_iters, default_config, lr_scheduler_obj=lr_scheduler_obj)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate
            if iter_num % default_config['eval_interval'] == 0:
                if not default_config['edge_only']:
                    # Full evaluation mode (paths + edges)
                    print_samples = iter_num % default_config['print_eval_interval'] == 0
                    # Calculate tokens_per_sec for display if available
                    current_tokens_per_sec = None
                    if 'dt' in locals() and dt > 0:
                         # Re-calculate or use stored value. We need 'steps' and 'block_size'
                         # 'steps' is defined below but used from previous iter effectively? 
                         # Actually 'steps' is defined in the loop. For iter_num > 0 it should be available.
                         if 'steps' in locals():
                             current_tokens_per_sec = (train_batch_size * steps * meta['block_size']) / dt
                    
                    train_total_dataset_size = combined_size
                    val_loss = evaluate(
                        estimate_metrics,
                        default_config,
                        meta,
                        iter_num,
                        lr,
                        ctx,
                        device,
                        model,
                        val_data_np,
                        paths_data_np,
                        edges_data_np,
                        print_samples,
                        live_panel=live_panel,
                        tokens_per_sec=current_tokens_per_sec,
                        batch_size=train_batch_size,
                        train_dataset_size=train_total_dataset_size,
                        eval_dataset_size=VAL_DATASET_SIZE,
                        lr_scheduler_obj=lr_scheduler_obj,
                    )
                    
                    # Update ReduceLROnPlateau scheduler if being used
                    if lr_scheduler_obj is not None:
                        lr_scheduler_obj.step(val_loss, iter_num)
                        
                        # Early termination if LR has dropped below threshold (LR exhausted)
                        if lr_scheduler_obj.is_lr_exhausted():
                            LiveTrainingPanel.CONSOLE.print(f"[yellow]Learning rate exhausted! LR={lr_scheduler_obj.current_lr:.2e} < 1e-8[/yellow]")
                            LiveTrainingPanel.CONSOLE.print(f"[yellow]Terminating training early at iter {iter_num}[/yellow]")
                            
                            # Log early termination event to wandb
                            if default_config['wandb_log'] and wandb.run is not None:
                                wandb.log({
                                    'early_termination/triggered': True,
                                    'early_termination/reason': 'lr_exhausted',
                                    'early_termination/iter': iter_num,
                                    'early_termination/final_lr': lr_scheduler_obj.current_lr,
                                    'early_termination/epoch': iter_num / meta['batches_per_epoch'],
                                }, step=iter_num)
                            
                            break
                    
                    # Early termination if validation loss falls below target threshold
                    if default_config['target_val_loss'] is not None and val_loss < default_config['target_val_loss']:
                        LiveTrainingPanel.CONSOLE.print(f"[green]Target validation loss achieved! val_loss={val_loss:.6f} < target={default_config['target_val_loss']:.6f}[/green]")
                        LiveTrainingPanel.CONSOLE.print(f"[green]Terminating training early at iter {iter_num}[/green]")
                        
                        # Log early termination event to wandb
                        if default_config['wandb_log'] and wandb.run is not None:
                            wandb.log({
                                'early_termination/triggered': True,
                                'early_termination/reason': 'target_val_loss_achieved',
                                'early_termination/iter': iter_num,
                                'early_termination/val_loss': val_loss,
                                'early_termination/target_val_loss': default_config['target_val_loss'],
                                'early_termination/epoch': iter_num / meta['batches_per_epoch'],
                            }, step=iter_num)
                        
                        break
                else:
                    # Edge-only mode: No validation set, but we can still evaluate edge memorization
                    # Compute mean cosine distance metric (lightweight, always computed)
                    mean_cosine_distance = compute_mean_cosine_distance(model, meta)
                    LiveTrainingPanel.CONSOLE.print(f"[cyan]Mean Cosine Distance (embeddings): {mean_cosine_distance:.4f}[/cyan]")
                    
                    # Check if this is a new lowest mean cosine distance and save special checkpoint
                    if mean_cosine_distance < meta['best_mean_cosine_distance']:
                        old_best = meta['best_mean_cosine_distance']
                        meta['best_mean_cosine_distance'] = mean_cosine_distance
                        LiveTrainingPanel.CONSOLE.print(f"[green]New lowest mean cosine distance: {mean_cosine_distance:.6f} (previous: {old_best:.6f})[/green]")
                        
                        # Save special checkpoint with "lowest_energy" suffix
                        # Create a modified checkpoint filename
                        base_checkpoint_filename = meta['checkpoint_filename']
                        if base_checkpoint_filename.endswith('.pt'):
                            lowest_energy_filename = base_checkpoint_filename[:-3] + '_lowest_energy.pt'
                        else:
                            lowest_energy_filename = base_checkpoint_filename + '_lowest_energy'
                        
                        # Temporarily modify meta to use special filename
                        original_checkpoint_filename = meta['checkpoint_filename']
                        meta['checkpoint_filename'] = lowest_energy_filename
                        
                        checkpoint_model(model, meta, default_config, iter_num, mean_cosine_distance, 
                                       loss_type='mean_cosine_distance', lr_scheduler_obj=lr_scheduler_obj)
                        
                        # Restore original checkpoint filename
                        meta['checkpoint_filename'] = original_checkpoint_filename
                    
                    # Plot pairwise cosine similarity matrix (at every eval_interval)
                    cosine_similarity_plot_path = None
                    try:
                        cosine_similarity_plot_path = plot_pairwise_cosine_similarity_matrix(
                            model, meta, iter_num, default_config,
                            out_dir=default_config.get('out_dir', 'out')
                        )
                    except Exception as e:
                        LiveTrainingPanel.CONSOLE.print(f"[yellow]Warning: Cosine similarity matrix plot failed: {e}[/yellow]")
                    
                    if default_config['show_edge_memorization_metrics']:
                        LiveTrainingPanel.CONSOLE.print(f"\n[cyan]Evaluating edge memorization (edge_only mode, iter {iter_num})...[/cyan]")
                        edge_memorization_pct = evaluate_edge_memorization(
                            ctx, model, meta, edges_data_np, device,
                            batch_size=int(default_config.get('edge_eval_batch_size', 512)),
                        )
                        
                        LiveTrainingPanel.CONSOLE.print(f"[cyan]Edge memorization: {edge_memorization_pct:.2f}%[/cyan]")
                        
                        # Log to wandb
                        if default_config['wandb_log']:
                            current_epoch = iter_num / meta['batches_per_epoch']
                            log_dict_edge = {
                                'edge_memorization_pct': edge_memorization_pct,
                                'embedding_geometry/mean_cosine_distance': mean_cosine_distance,
                                'iter': iter_num,
                                'epoch': round(current_epoch, 4),
                                'lr': lr,
                            }
                            # Add cosine similarity matrix plot
                            if cosine_similarity_plot_path and os.path.exists(cosine_similarity_plot_path):
                                log_dict_edge['embedding_geometry/cosine_similarity_matrix'] = wandb.Image(cosine_similarity_plot_path)
                            wandb.log(log_dict_edge, step=iter_num)
                    else:
                        # Still log mean cosine distance even if not showing edge memorization
                        if default_config['wandb_log']:
                            current_epoch = iter_num / meta['batches_per_epoch']
                            log_dict_edge = {
                                'embedding_geometry/mean_cosine_distance': mean_cosine_distance,
                                'iter': iter_num,
                                'epoch': round(current_epoch, 4),
                                'lr': lr,
                            }
                            # Add cosine similarity matrix plot
                            if cosine_similarity_plot_path and os.path.exists(cosine_similarity_plot_path):
                                log_dict_edge['embedding_geometry/cosine_similarity_matrix'] = wandb.Image(cosine_similarity_plot_path)
                            wandb.log(log_dict_edge, step=iter_num)
                        
                    # Update live panel with edge memorization info
                    if 'dt' in locals() and dt > 0 and 'steps' in locals():
                        current_tokens_per_sec = (train_batch_size * steps * meta['block_size']) / dt
                    else:
                        current_tokens_per_sec = None
                    
                    # Create a minimal metrics dict for edge_only mode
                    edge_only_metrics = {}
                    edge_memorization_pct = edge_memorization_pct if default_config['show_edge_memorization_metrics'] else None
                    live_panel.update_metrics_table(
                        edge_only_metrics,
                        graph_length,
                        iter_num,
                        current_epoch,
                        lr,
                        meta,
                        current_tokens_per_sec,
                        train_batch_size,
                        edge_memorization_pct,
                        train_dataset_size=combined_size,
                        eval_dataset_size=0,  # No validation in edge_only mode
                        mean_cosine_distance=mean_cosine_distance,
                    )
            
            if iter_num == 0 and default_config['eval_only']:
                break
            
            # Forward backward update with batch prefetching for better GPU utilization
            steps = default_config['gradient_accumulation_steps']

            # Track activation stats on the last micro_step only (to avoid overhead)
            track_stats = default_config.get('log_activation_stats', True)
            last_activation_stats = None
            last_logits = None  # For NaN debugging
            
            for micro_step in range(steps):
                # Use scheduled sampling for PATH tasks if p_autoregressive_substitution > 0
                p_sub = default_config.get('p_autoregressive_substitution', 0.0)
                if p_sub > 0:
                    # Scheduled sampling: substitute teacher-forced tokens with model predictions
                    # for PATH sequences (EDGE sequences still use pure teacher forcing)
                    logits_step, loss = forward_with_scheduled_sampling(
                        model, X, Y, meta, p_sub,
                        label_smoothing=default_config['label_smoothing'],
                        ctx=ctx
                    )
                    if micro_step == steps - 1:
                        last_logits = logits_step
                else:
                    # Standard teacher forcing
                    # Track activation stats on last micro_step only
                    should_track = track_stats and (micro_step == steps - 1)
                    with ctx:
                        result = model(X, Y, label_smoothing=default_config['label_smoothing'], 
                                       track_activation_stats=should_track)
                        if should_track:
                            logits_step, loss, last_activation_stats = result
                        else:
                            logits_step, loss = result
                    if micro_step == steps - 1:
                        last_logits = logits_step
                loss = loss / steps
                
                # CHECK FOR NaNs IMMEDIATELY AFTER FORWARD PASS (before backward!)
                # This catches NaNs before they crash the backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    LiveTrainingPanel.CONSOLE.print(f"[red]🔥 NaN/Inf DETECTED in loss at iter {iter_num}, micro_step {micro_step}![/red]")
                    LiveTrainingPanel.CONSOLE.print(f"[red]Loss value: {loss.item()}[/red]")
                    LiveTrainingPanel.CONSOLE.print(f"[red]Phase: combined[/red]")
                    # Run full diagnostic
                    nan_report = check_for_nans(model, optimizer, loss * steps, logits_step, X, Y, iter_num, 
                                               phase='combined')
                    if nan_report and default_config['wandb_log']:
                        wandb.log({f'nan_detection/{k}': v for k, v in nan_report.items() if not isinstance(v, list)})
                    raise ValueError(f"NaN/Inf detected in loss at iteration {iter_num}. Training stopped to prevent gradient corruption.")
                
                if logits_step is not None and (torch.isnan(logits_step).any() or torch.isinf(logits_step).any()):
                    LiveTrainingPanel.CONSOLE.print(f"[red]🔥 NaN/Inf DETECTED in logits at iter {iter_num}, micro_step {micro_step}![/red]")
                    nan_count = torch.isnan(logits_step).sum().item() if torch.isnan(logits_step).any() else 0
                    inf_count = torch.isinf(logits_step).sum().item() if torch.isinf(logits_step).any() else 0
                    LiveTrainingPanel.CONSOLE.print(f"[red]NaN count: {nan_count}, Inf count: {inf_count}[/red]")
                    LiveTrainingPanel.CONSOLE.print(f"[red]Phase: combined[/red]")
                    # Run full diagnostic
                    nan_report = check_for_nans(model, optimizer, loss * steps, logits_step, X, Y, iter_num, 
                                               phase='combined')
                    if nan_report and default_config['wandb_log']:
                        wandb.log({f'nan_detection/{k}': v for k, v in nan_report.items() if not isinstance(v, list)})
                    raise ValueError(f"NaN/Inf detected in logits at iteration {iter_num}. Training stopped to prevent gradient corruption.")
                
                # Prefetch next batch while backward pass runs (overlap I/O with compute)
                if micro_step < steps - 1:
                    X_next, Y_next = get_batch('combined')
                
                scaler.scale(loss).backward()
                
                # Move prefetched batch to current (if not last step)
                if micro_step < steps - 1:
                    X, Y = X_next, Y_next
            
            # Get batch for next iteration
            X, Y = get_batch('combined')
            
            # Clip gradients
            if default_config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), default_config['grad_clip'])
            
            # Check for NaNs BEFORE optimizer step (while gradients still exist)
            check_nan_interval = default_config.get('check_nan_interval', 0)
            cached_nan_report = None
            if check_nan_interval > 0 and iter_num % check_nan_interval == 0:
                cached_nan_report = check_for_nans(model, optimizer, loss * steps, last_logits, X, Y, iter_num, phase='combined')
            
            # Compute gradient statistics BEFORE optimizer step (for logging)
            # Cache these values since gradients will be cleared after optimizer step
            cached_grad_stats = {}
            if check_nan_interval > 0 and default_config['wandb_log']:
                max_grad = 0.0
                grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                cached_grad_stats['train/grad_max'] = max_grad
                cached_grad_stats['train/grad_norm'] = grad_norm
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Report NaNs if detected (using cached report from before zero_grad)
            if cached_nan_report is not None:
                LiveTrainingPanel.CONSOLE.print(f"[red]🔥 NaN DETECTED at iter {iter_num}![/red]")
                LiveTrainingPanel.CONSOLE.print(f"[red]Phase: {cached_nan_report.get('phase', 'unknown')}[/red]")
                
                if 'loss' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[red]  Loss: {cached_nan_report['loss']}[/red]")
                if 'logits_nan_count' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[red]  Logits: {cached_nan_report['logits_nan_count']} NaNs ({cached_nan_report['logits_nan_pct']:.2f}%)[/red]")
                if 'logits_max' in cached_nan_report and 'logits_min' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[yellow]  Logits range: [{cached_nan_report['logits_min']:.4f}, {cached_nan_report['logits_max']:.4f}][/yellow]")
                if 'param_nans' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[red]  Parameters with NaN: {cached_nan_report['param_nans'][:5]}{'...' if len(cached_nan_report['param_nans']) > 5 else ''}[/red]")
                if 'grad_nans' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[red]  Gradients with NaN: {cached_nan_report['grad_nans'][:5]}{'...' if len(cached_nan_report['grad_nans']) > 5 else ''}[/red]")
                if 'max_grad' in cached_nan_report:
                    LiveTrainingPanel.CONSOLE.print(f"[yellow]  Max gradient: {cached_nan_report['max_grad']:.6f}[/yellow]")
                
                # Log to wandb
                if default_config['wandb_log']:
                    wandb_nan_log = {f'nan_debug/{k}': v for k, v in cached_nan_report.items() if isinstance(v, (int, float, bool))}
                    wandb.log(wandb_nan_log, step=iter_num)
                
                LiveTrainingPanel.CONSOLE.print("[red]Training will continue but model may be unstable[/red]")
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            current_epoch = iter_num / meta['batches_per_epoch']
            
            # Track running average for comparison with theoretical minimum
            lossf = loss.item() * steps
            running_loss_sum += lossf
            running_loss_count += 1
            
            # Reset running average at epoch boundaries
            if iter_num > 0 and iter_num % meta['batches_per_epoch'] == 0:
                running_loss_sum = 0.0
                running_loss_count = 0
            
            if iter_num % default_config['log_interval'] == 0:
                tokens_per_sec = (X.numel() * steps) / dt
                
                # Compute mean cosine distance during training (at log intervals)
                train_mean_cosine_distance = compute_mean_cosine_distance(model, meta)
                
                # DEBUG: Count how many edge vs path samples in this batch
                EDGE_token = meta['special_tokens']['EDGE']
                PATH_token = meta['special_tokens']['PATH']
                num_edges_in_batch = (X[:, 0] == EDGE_token).sum().item()
                num_paths_in_batch = (X[:, 0] == PATH_token).sum().item()
                
                # Calculate running average
                running_avg_loss = running_loss_sum / running_loss_count if running_loss_count > 0 else 0.0

                if default_config['wandb_log']:
                    # Use cached gradient statistics (computed before zero_grad)
                    grad_stats = cached_grad_stats.copy()
                    
                    # Add logits statistics if available
                    if check_nan_interval > 0 and last_logits is not None:
                        grad_stats['train/logits_max'] = float(last_logits.max())
                        grad_stats['train/logits_min'] = float(last_logits.min())
                        grad_stats['train/logits_mean'] = float(last_logits.mean())
                    
                    wandb.log({
                        'train/loss/overall': lossf,
                        'train/loss/overall_remove_optimal': lossf - meta['theoretical_min_loss'],
                        'train/loss/running_avg_epoch': running_avg_loss - meta['theoretical_min_loss'],
                        'train/batch_composition/num_edges': num_edges_in_batch,
                        'train/batch_composition/num_paths': num_paths_in_batch,
                        'train/optimal_loss': meta['theoretical_min_loss'],
                        'train/embedding_geometry/mean_cosine_distance': train_mean_cosine_distance,
                        'iter': iter_num,
                        "epoch": round(current_epoch, 4),
                        'tokens_per_sec': tokens_per_sec,
                        **grad_stats,
                    }, step=iter_num)
                
                # Update training slice panel (only if live display and show_training_slices are enabled)
                # Only update every vis_interval to save sync/formatting time
                live_panel.update_train(X, Y, iter_num, meta, last_mask_debug_str=last_mask_debug_str, loss=lossf)
                
                # Checkpointing for edge_only mode (based on training loss)
                # In edge_only mode, there's no validation set, so checkpoint based on best training loss
                if default_config['edge_only'] and iter_num > 0:
                    is_sweep_mode = wandb.run is not None and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None
                    save_checkpoint = False
                    
                    # Check if this is a new best training loss
                    if lossf < meta['best_train_loss']:
                        meta['best_train_loss'] = lossf
                        save_checkpoint = True
                    elif not is_sweep_mode and default_config['always_save_checkpoint']:
                        # In standalone mode, save every checkpoint if always_save_checkpoint is True
                        save_checkpoint = True
                    
                    if save_checkpoint:
                        checkpoint_model(model, meta, default_config, iter_num, lossf, loss_type='train', lr_scheduler_obj=lr_scheduler_obj)
                
                # Log attention maps to wandb (expensive, so use separate interval)
                if default_config['wandb_log'] and default_config.get('log_attention_maps', False):
                    if iter_num % default_config.get('attention_map_interval', 500) == 0:
                        try:
                            attn_images = create_attention_map_figures(
                                model, X, itos, meta,
                                num_samples=default_config.get('attention_map_samples', 3)
                            )
                            wandb.log(attn_images, step=iter_num)
                        except Exception as e:
                            LiveTrainingPanel.CONSOLE.print(f"[yellow]Warning: Failed to log attention maps: {e}[/yellow]")
                
                # Log activation mean/variance per layer (collected during forward pass)
                if default_config['wandb_log'] and last_activation_stats is not None:
                    try:
                        activation_log = {f'activation/{k}': v for k, v in last_activation_stats.items()}
                        wandb.log(activation_log, step=iter_num)
                    except Exception as e:
                        LiveTrainingPanel.CONSOLE.print(f"[yellow]Warning: Failed to log activation stats: {e}[/yellow]")
            
            iter_num += 1
            
            if iter_num > max_iters:
                break
    
    # Cleanup and finalization
    LiveTrainingPanel.CONSOLE.print("Finalizing training run...")
    
    # Clear GPU memory before finishing
    if device_type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            LiveTrainingPanel.CONSOLE.print(f"Warning during GPU cleanup: {e}")
    
    # Only call wandb.finish() if we initialized wandb ourselves (not in sweep mode)
    # In sweep mode, the agent handles finishing the run
    if default_config['wandb_log'] and wandb.run is not None:
        # Check if we're in sweep mode
        if not hasattr(wandb.run, 'sweep_id') or wandb.run.sweep_id is None:
            # Standalone mode - we initialized it, so we finish it
            wandb.finish()
        # In sweep mode, don't call finish - let the agent handle it
    
    LiveTrainingPanel.CONSOLE.print("Training complete!")


def sweep_train():
    """Wrapper function for wandb sweeps"""
    # wandb.init() is called automatically by the sweep agent
    # We need to wait for it to complete before accessing config
    print("Running in wandb sweep mode")
    
    # Initialize wandb run if not already initialized by agent
    if wandb.run is None:
        wandb.init()
    
    # Now safely access config after init is complete
    # Convert wandb.config to a regular dict
    config_dict = {k: v for k, v in wandb.config.items()}
    
    train(config=config_dict)


if __name__ == '__main__':
    # Check if we're running in a wandb sweep
    if os.environ.get('WANDB_SWEEP_ID'):
        sweep_train()
    else:
        # Running standalone - use command-line args and configurator.py
        print("Running in standalone mode")
        train(config=None)

