"""
This training script runs on a single GPU and supports wandb sweeps.
This version handles separate edge and path datasets with interleaved training.

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
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from model import GPTConfig, GPT
from pathstar import InWeightsPathStar

# Rich imports
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.table import Table

console = Console()


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
        'analyze_embedding_geometry': False,  # If True, compute and log embedding geometry metrics during eval
        # Debugging
        'debug_masking': False,          # If True, show target masks applied to Y
        'debug_masking_samples': 2,      # How many batch rows to show
        'debug_masking_max_len': 32,     # Max tokens to show per row
        
        # Dataset generation parameters
        'graph_d': 1000,
        'graph_l': 5,
        'randomize_vocab_size': 'auto',
        'graph_holdout_percentage': 0.2,
        'num_pause_tokens': 5,
        'use_undirected': True,
        'use_directional_tokens': False,
        'use_task_tokens': False,
        # If True, PATH task sequences interleave GT tokens between edges:
        #   [PATH] leaf (PAUSE)xN root GT n2 GT n3 ... GT leaf
        'use_task_tokens_in_path': False,
        
        # Training parameters
        'gradient_accumulation_steps': 1,
        # If set, this caps the memory-based auto batch size.
        # This is the per-step microbatch size; effective batch size is
        #   batch_size * gradient_accumulation_steps.
        # Keep this <= the auto-computed value to avoid OOM.
        'batch_size': None,
        'edge_iterations_per_epoch': 10,  # Number of iterations on edges per epoch
        'path_iterations_per_epoch': 10,  # Number of iterations on paths per epoch
        'epochs': 1000,
        # Evaluation batch sizes (kept separate from training batch_size)
        'eval_batch_size': 512,
        'edge_eval_batch_size': 512,
        
        # Model architecture
        'n_layer': 3,
        'n_head': 8,
        'n_embd': 96,
        'dropout': 0.0,  # Dropout for attention, MLP, and residual connections
        'embd_dropout': 0.0,
        'holdout_percentage': 0.0, # Percentage of paths to hold out for validation
        'interleave_dataset': False, # If True, combines edges and paths into a single training dataset
        'bias': False,
        
        # Optimization
        'learning_rate': 1e-3,
        'label_smoothing': 0,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Learning rate schedule
        'decay_lr': True,
        'warmup_frac': 0.10,
        'lr_decay_frac': 0.99,
        'min_lr': 6e-5,
        
        # System
        'device': 'auto',  # 'cuda', 'mps', 'cpu', or 'auto'
        'dtype': 'auto',  # 'float32', 'bfloat16', 'float16', or 'auto'
        'compile': True,
        'gpu_id': None,
        'experiment_name': None,
        # seed
        'seed': 1337,
        'predict_direction_for_edge_task': True
    }

# TODO: check this 
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
    
    use_task_tokens = meta.get('use_task_tokens', True)
    
    # Compute context length per input based on task type and task tokens
    if use_task_tokens:
        # Use task tokens to determine context length
        context_length_per_input = torch.where(
            input[:, 0] == meta['special_tokens']['EDGE'],
            torch.tensor((1 if use_task_tokens else 0) + (1 if meta['use_directional_tokens'] else 0) + 1, device=input.device),
            torch.where(
                input[:, 0] == meta['special_tokens']['PATH'],
                torch.tensor((1 if use_task_tokens else 0) + 1 + meta['num_pause_tokens'], device=input.device),
                torch.tensor(0, device=input.device)
            )
        ).unsqueeze(1)
    else:
        # No task tokens - compute based on task_type
        if task_type == 'edge':
            # Edge: (directional token if present) + 1
            edge_context = (1 if meta['use_directional_tokens'] else 0) + 1
            context_length_per_input = torch.full((input.size(0), 1), edge_context, device=input.device, dtype=torch.long)
        else:  # path
            # Path: leaf + pause tokens
            path_context = 1 + meta['num_pause_tokens']
            context_length_per_input = torch.full((input.size(0), 1), path_context, device=input.device, dtype=torch.long)
    
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



def get_rich_token_str(token, itos, meta):
    """Helper to format a token with rich coloring based on training/validation splitting"""
    # Ensure dictionary lookups work with numpy scalar types
    token = int(token)
    token_str = itos.get(token, str(token))

    # meta['token_colors'] maps token_id -> ANSI escape code string
    ansi_color = meta.get('token_colors', {}).get(token, '')
    if not ansi_color:
        return token_str

    # Handle 256-color ANSI foreground codes like "\x1b[38;5;226m"
    if ansi_color.startswith('\x1b[38;5;') and ansi_color.endswith('m'):
        # Extract the color number between the prefix and suffix
        color_num_str = ansi_color[7:-1]  # Skip '\x1b[38;5;' (7 chars) and 'm' (1 char)
        try:
            n = int(color_num_str)
            return f"[color({n})]{token_str}[/]"
        except ValueError:
            return token_str

    # Handle basic bright ANSI colors like "\x1b[91m"
    if ansi_color.startswith('\x1b[') and ansi_color.endswith('m'):
        code_str = ansi_color[2:-1]  # Skip '\x1b[' (2 chars) and 'm' (1 char)
        try:
            code = int(code_str)
            basic = {91: "red", 92: "green", 93: "yellow", 94: "blue", 95: "magenta", 96: "cyan"}
            if code in basic:
                return f"[bold {basic[code]}]{token_str}[/]"
        except ValueError:
            return token_str

    return token_str

def format_training_slice(sequences, itos, meta, num_samples=10):
    """Format a batch of sequences for display in Rich panel"""
    lines = []
    num_samples = min(num_samples, len(sequences))
    
    # Check if we have tensors or numpy arrays
    if isinstance(sequences, torch.Tensor):
        sequences_np = sequences.detach().cpu().numpy()
    else:
        sequences_np = sequences
        
    for i in range(num_samples):
        seq = sequences_np[i]
        token_strs = []
        for token in seq:
            # Handle potential padding/special tokens if needed, but for now just display
            if token == -1: continue # Ignore masked tokens if any
            token_strs.append(get_rich_token_str(token, itos, meta))
        
        seq_str = " ".join(token_strs)
        lines.append(f"Sample {i}: {seq_str}")
        
    return "\n".join(lines)

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

def create_metrics_table(metrics, graph_length, iter_num, epoch, lr, tokens_per_sec=None, batch_size=None, edge_memorization_pct=None, train_dataset_size=None, eval_dataset_size=None, embedding_geometry=None):
    """Create a Rich Table for per-token metrics (Train vs Val)"""
    title = f"Per-Token Metrics (Iter {iter_num}, Epoch {epoch:.2f}, LR {lr:.2e}"
    if batch_size is not None:
        title += f", BS {batch_size}"
    if tokens_per_sec is not None:
        title += f", {tokens_per_sec:.2e} tok/s"
    if edge_memorization_pct is not None:
        title += f", Edge Mem: {edge_memorization_pct:.1f}%"
    if train_dataset_size is not None and eval_dataset_size is not None:
        title += f", Train N={train_dataset_size:,}, Eval N={eval_dataset_size:,}"
    title += ")"
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Pos", style="cyan", justify="center")
    table.add_column("Train Loss", style="red", justify="right")
    table.add_column("Val Loss", style="red", justify="right")
    table.add_column("Train Acc", style="green", justify="right")
    table.add_column("Val Acc", style="green", justify="right")
    
    # The live metrics panel has a fixed height in the UI; for long sequences the table will be
    # visually truncated. To make it obvious we still compute all tokens, we display:
    # - positions 1..9
    # - an ellipsis row (if needed)
    # - the final position (graph_length)
    if graph_length <= 10:
        display_positions = list(range(1, graph_length + 1))
    else:
        display_positions = list(range(1, 10)) + [graph_length]

    last_display = None
    for i in display_positions:
        if last_display is not None and i - last_display > 1:
            table.add_row("…", "…", "…", "…", "…")
        t_loss = metrics.get('train_per_token', {}).get(i, float('nan'))
        v_loss = metrics.get('val_per_token', {}).get(i, float('nan'))
        t_acc = metrics.get('train_per_token_accuracy', {}).get(i, float('nan'))
        v_acc = metrics.get('val_per_token_accuracy', {}).get(i, float('nan'))
        
        table.add_row(
            str(i),
            f"{t_loss:.4f}", f"{v_loss:.4f}",
            f"{t_acc*100:.1f}%", f"{v_acc*100:.1f}%"
        )
        last_display = i
    
    return table


def create_embedding_geometry_table(embedding_geometry, l):
    """Create a Rich Table for embedding geometry cosine similarities"""
    if embedding_geometry is None:
        return None
    
    train_sims = embedding_geometry.get('train_similarities', {})
    val_sims = embedding_geometry.get('val_similarities', {})
    random_baseline = embedding_geometry.get('random_baseline', 0)
    
    table = Table(title=f"Embedding Cosine Similarity by Distance (baseline: {random_baseline:.4f})", 
                  show_header=True, header_style="bold cyan")
    table.add_column("Dist", style="cyan", justify="center", width=5)
    table.add_column("Train", justify="center", width=10)
    table.add_column("Val", justify="center", width=10)
    table.add_column("Status", justify="left", width=12)
    
    # Get all distances, show up to l (path length)
    all_distances = sorted(set(train_sims.keys()) | set(val_sims.keys()))
    display_distances = [d for d in all_distances if d <= l]
    
    for dist in display_distances:
        t_sims = train_sims.get(dist, [])
        v_sims = val_sims.get(dist, [])
        
        t_mean = np.mean(t_sims) if t_sims else float('nan')
        v_mean = np.mean(v_sims) if v_sims else float('nan')
        
        # Format with color coding
        def format_sim(val, baseline):
            if np.isnan(val):
                return "-"
            if val > baseline + 0.1:
                return f"[green]{val:.4f}[/green]"
            elif val < baseline - 0.1:
                return f"[red]{val:.4f}[/red]"
            else:
                return f"[yellow]{val:.4f}[/yellow]"
        
        t_str = format_sim(t_mean, random_baseline)
        v_str = format_sim(v_mean, random_baseline)
        
        # Status indicator
        if dist == 0:
            status = "✓ self" if abs(t_mean - 1.0) < 0.01 else "✗ self!"
        elif dist == 1:
            if t_mean > random_baseline + 0.1:
                status = "✓ adjacent"
            else:
                status = "[red]✗ no struct[/red]"
        else:
            status = ""
        
        table.add_row(str(dist), t_str, v_str, status)
    
    return table

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
    print(f"use_task_tokens: {meta.get('use_task_tokens', False)}")
    print(f"use_directional_tokens: {meta.get('use_directional_tokens', False)}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_edges)
        
        # Get batch of edge sequences
        batch = torch.from_numpy(edges_data_np[start_idx:end_idx].astype(np.int64)).to(device)
        
        predict_dir = bool(meta.get('predict_direction_for_edge_task', False))
        if predict_dir:
            pos = (1 if meta.get('use_task_tokens', False) else 0) + 2
        else:
            pos = (1 if meta.get('use_task_tokens', False) else 0) + (1 if meta.get('use_directional_tokens', False) else 0) + 1

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
    
    # Calculate context length based on whether task tokens are used
    use_task_tokens = meta.get('use_task_tokens', True)
    if use_task_tokens:
        context_length = 2 + meta['num_pause_tokens']  # task token + leaf + pause tokens
    else:
        context_length = 1 + meta['num_pause_tokens']  # leaf + pause tokens
    
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
    
    return per_token_accuracies, generated_text_output

# DONE 
def evaluate_samples(device, ctx, model, meta, data, data_size, split_name, num_samples=5, eval_batch_size=512):
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
    
    # Calculate context length based on whether task tokens are used
    use_task_tokens = meta.get('use_task_tokens', True)
    if use_task_tokens:
        context_length = 2 + meta['num_pause_tokens']  # task token + leaf + pause tokens
    else:
        context_length = 1 + meta['num_pause_tokens']  # leaf + pause tokens
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
    console.print(f"\nAutoregressive generation on {num_samples} {split_name} samples:")
    accuracies = []
    for ground_truth, generated_tokens in zip(ground_truths, all_generated_tokens):
        # Calculate accuracy
        accuracy = np.mean(generated_tokens == ground_truth[:len(generated_tokens)])
        accuracies.append(accuracy)
    
    # Calculate average accuracy
    avg_accuracy = np.mean(accuracies)
    console.print(f"  Average accuracy: {avg_accuracy*100:.1f}%")
    console.print()  # Empty line for readability
    
    return avg_accuracy

def get_lr(it, warmup_iters, lr_decay_iters, default_config):
    """Learning rate decay scheduler (cosine with warmup)"""
    if it < warmup_iters:
        return default_config['learning_rate'] * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return default_config['min_lr']
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return default_config['min_lr'] + coeff * (default_config['learning_rate'] - default_config['min_lr'])

# GOOD
def clear_gpu_memory():
    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
            # Reset memory stats for clean monitoring
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception as e:
            print(f"Warning during GPU memory clearing: {e}")

# GOOD
def set_wandb_name(config):
    if config is not None:
        # Set custom run name for sweep runs
        if wandb.run is not None:
            utc_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            dir_label = "undir_" if config["use_undirected"] else "dir_"
            tt_label = "tt_" if config['use_task_tokens'] else 'nott_'
            dt_label = 'dt_' if config['use_directional_tokens'] else 'nodt_'
            ptgt_label = 'ptgt_' if config.get('use_task_tokens_in_path', False) else ''
            ped_or_pet_label = 'ped_' if config['predict_direction_for_edge_task'] else 'pet_'
            wt_label = 'wt_' if config['weight_tying'] else ''
            # Include both dropout values if they differ, otherwise just one
            if config['dropout'] == config['embd_dropout']:
                dropout_label = f"D{config['dropout']}_"
            else:
                dropout_label = f"D{config['dropout']}_ED{config['embd_dropout']}_"
            custom_name = (
                f"{utc_time}_"
                f"G{config['graph_d']},"
                f"{config['graph_l']}_"
                f"{ped_or_pet_label}"
                f"L{config['n_layer']}_"
                f"E{config['n_embd']}_"
                f"H{config['n_head']}_"
                f"{dropout_label}"
                f"p{config['num_pause_tokens']}_"
                f"{dir_label}"
                f"{tt_label}"
                f"{dt_label}"
                f"{ptgt_label}"
                f"{wt_label}"
                f"{config['epochs']}"
            )
            wandb.run.name = custom_name
            print(f"Set sweep run name: {custom_name}")
            return custom_name

# GOOD
def detect_device(config):
    if config['device'] == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = config['device']
    
    # Print device information
    if device == 'cuda':

        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using device: {device} (GPU {current_device}/{num_gpus-1}: {device_name})")
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print(f"Using device: {device}")
    # Determine GPU ID for checkpoint naming
    gpu_id = config.get('gpu_id')
    if gpu_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible is not None:
            gpu_id = cuda_visible.split(',')[0]
        elif torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
        else:
            gpu_id = 'cpu'
    
    
    device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
    return device, device_type, gpu_id

def set_dtype(config):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Auto-detect dtype with GPU-aware selection
    if config['dtype'] == 'auto':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).upper()
            # RTX 30-series: use FP16 (optimized tensor cores, BF16 is ~50% slower)
            # RTX 40-series, A100, H100: use BF16 (better numerical range)
            if any(x in gpu_name for x in ['RTX 30', '3090', '3080', '3070', '3060']):
                dtype = 'float16'
                print(f"Using FP16 for optimal performance on {gpu_name}")
            elif torch.cuda.is_bf16_supported():
                dtype = 'bfloat16'
                print(f"Using BF16 on {gpu_name}")
            else:
                dtype = 'float16'
        else:
            dtype = 'float16'
    else:
        dtype = config['dtype']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    return ptdtype, dtype

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
        embd_dropout=config['embd_dropout']
    )
    checkpoint = None
    iter_num = 0
    meta['best_val_loss'] = float('inf')
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
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout', 'embd_dropout']:
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
        meta['best_val_loss'] = checkpoint['best_val_loss']
    
    if meta['block_size'] < model.config.block_size:
        model.crop_block_size(meta['block_size'])
        model_args['block_size'] = meta['block_size']
    
    model.to(device)

    return model, model_args, checkpoint,  iter_num

def get_theoretical_loss(meta):
    # Calculate theoretical baseline for train/loss/token_1
    # This represents the expected loss for the first token prediction
    root_edges_in_dataset = meta['d']
    theoretical_token_1_loss = -np.log(
        (meta['total_edge_size'] - root_edges_in_dataset + meta['replicated_train_paths'] + 1) / meta['TRAIN_DATASET_SIZE']
    )
    print(f"Theoretical baseline for train/loss/token_1: {theoretical_token_1_loss:.4f}")
    return theoretical_token_1_loss
    
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
            return min(2000, target_batch_size)
        return 2000  # Default for non-CUDA
    
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
    
    # Get metadata
    l = meta['l']
    special_tokens = meta['special_tokens']
    num_special_tokens = len(special_tokens)
    use_task_tokens = meta.get('use_task_tokens', True)
    num_pause_tokens = meta.get('num_pause_tokens', 1)
    use_task_tokens_in_path = meta.get('use_task_tokens_in_path', False)
    
    # Calculate sequence dimensions from meta
    seq_len = meta['block_size'] + 1  # Full sequence length
    PATHS_DATASET_SIZE = meta['PATHS_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    # Reshape the flat numpy arrays to (num_samples, seq_len)
    paths_data = paths_data_np.reshape(PATHS_DATASET_SIZE, seq_len)
    val_data = val_data_np.reshape(VAL_DATASET_SIZE, seq_len)
    
    def extract_path_nodes(sequence, meta):
        """
        Extract just the graph node tokens from a path sequence.
        
        Path format: [PATH?, leaf, PAUSE, ..., PAUSE, root, (GT?), n_2, (GT?), ..., leaf]
        
        Returns list of node tokens: [root, n_2, ..., leaf] (length = l)
        """
        seq = [int(x) for x in sequence]
        
        # Calculate where path nodes start
        # Skip: PATH token (if present) + leaf + PAUSE tokens
        path_start_idx = (1 if use_task_tokens else 0) + 1 + num_pause_tokens
        
        # Extract the path portion
        path_portion = seq[path_start_idx:]
        
        # Filter out special tokens (GT, PAD, etc.) to get just node tokens
        # Node tokens are >= num_special_tokens
        node_tokens = [t for t in path_portion if t >= num_special_tokens]
        
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
    
    # Get all unique node tokens for similarity matrix
    all_node_tokens = set()
    for path in train_paths + val_paths:
        all_node_tokens.update(path)
    all_node_tokens = sorted(list(all_node_tokens))
    
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
    
    def compute_path_similarities(paths, result_dict):
        """Compute pairwise similarities between nodes at different distances within paths."""
        for path in paths:
            # path is [root, n_1, n_2, ..., leaf] with length l
            for i in range(len(path)):
                for j in range(i, len(path)):
                    dist = j - i  # Graph distance within the path
                    token_i = path[i]
                    token_j = path[j]
                    
                    if token_i in token_to_idx and token_j in token_to_idx:
                        idx_i = token_to_idx[token_i]
                        idx_j = token_to_idx[token_j]
                        sim = sim_matrix[idx_i, idx_j].item()
                        result_dict[dist].append(sim)
    
    compute_path_similarities(train_paths, results['train'])
    compute_path_similarities(val_paths, results['val'])
    
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
    
    console.print("\n".join(summary_lines))
    
    model.train()
    
    return {
        'train_similarities': dict(results['train']),
        'val_similarities': dict(results['val']),
        'cross_path_train': dict(cross_path_sims_train),
        'cross_path_val': dict(cross_path_sims_val),
        'random_baseline': np.mean(random_sims) if random_sims else 0,
        'plot_path': plot_path,
    }


# GOOD
def evaluate(estimate_metrics, config, meta, iter_num, lr, ctx, device, model, val_data_np, paths_data_np, edges_data_np, print_samples=False, eval_layout_component=None, metrics_layout_component=None, tokens_per_sec=None, batch_size=None, train_dataset_size=None, eval_dataset_size=None):
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
    val_avg_accuracy = evaluate_samples(
        device, ctx, model, meta, val_data_np, VAL_DATASET_SIZE, 'val',
        num_samples=min(VAL_DATASET_SIZE, autoregressive_eval_samples),
        eval_batch_size=int(config.get('eval_batch_size', 512)),
    )
    train_avg_accuracy = evaluate_samples(
        device, ctx, model, meta, paths_data_np, PATHS_DATASET_SIZE, 'train',
        num_samples=min(PATHS_DATASET_SIZE, autoregressive_eval_samples),
        eval_batch_size=int(config.get('eval_batch_size', 512)),
    )
    
    # Evaluate edge memorization
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
            console.print(f"[yellow]Warning: Embedding geometry analysis failed: {e}[/yellow]")
    
    # Update Live display if new samples were generated
    if 'generated_text' in losses and losses['generated_text'] and eval_layout_component:
        eval_layout_component.update(Panel(losses['generated_text'], title="Evaluation Examples", border_style="blue"))

    # Update metrics display
    if metrics_layout_component:
        metrics_table = create_metrics_table(
            losses,
            graph_length,
            iter_num,
            current_epoch,
            lr,
            tokens_per_sec,
            batch_size,
            edge_memorization_pct,
            train_dataset_size=train_dataset_size,
            eval_dataset_size=eval_dataset_size,
            embedding_geometry=embedding_geometry_results,
        )
        
        # Create embedding geometry table if available
        emb_table = create_embedding_geometry_table(embedding_geometry_results, meta['l']) if embedding_geometry_results else None
        
        # Combine tables using Group
        if emb_table:
            combined = Group(metrics_table, Text(""), emb_table)
        else:
            combined = metrics_table
        
        metrics_layout_component.update(Panel(Align.center(combined), title="Validation Metrics", border_style="magenta"))

    # # PRINTING
    # console.print(f"step {iter_num}: epoch {current_epoch:.2f}, val loss {losses['val']:.4f}, train loss {losses['train']:.4f}")
    
    if 'val_per_token' in losses:
        # console.print("  Val per-token losses:")
        if graph_length <= 9:
            per_token_str = ", ".join([f"tok{i}: {losses['val_per_token'].get(i, float('nan')):.4f}"
                                       for i in range(1, graph_length + 1)])
        else:
            head = ", ".join([f"tok{i}: {losses['val_per_token'].get(i, float('nan')):.4f}"
                              for i in range(1, 10)])
            tail = f"tok{graph_length}: {losses['val_per_token'].get(graph_length, float('nan')):.4f}"
            per_token_str = f"{head}, …, {tail}"
        # console.print(f"    {per_token_str}")
    
    if 'val_per_token_accuracy' in losses:
        # console.print("  Val per-token accuracies (autoregressive):")
        if graph_length <= 9:
            per_token_acc_str = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'].get(i, float('nan'))*100:.1f}%"
                                           for i in range(1, graph_length + 1)])
        else:
            head = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'].get(i, float('nan'))*100:.1f}%"
                              for i in range(1, 10)])
            tail = f"tok{graph_length}: {losses['val_per_token_accuracy'].get(graph_length, float('nan'))*100:.1f}%"
            per_token_acc_str = f"{head}, …, {tail}"
        # console.print(f"    {per_token_acc_str}")
    
    
    if config['wandb_log']:
        log_dict = {
            "iter": iter_num,
            'max_iters': meta['max_iters'],
            'warmup_iters': meta['warmup_iters'],
            "epoch": round(current_epoch, 4),
            "val/loss/overall": losses['val'],
            "train/loss/eval": losses['train'],
            "lr": lr,
            "gen/val_paths_avg_accuracy": val_avg_accuracy,
            "gen/train_paths_avg_accuracy": train_avg_accuracy,
            "edge_memorization_pct": edge_memorization_pct
        }
        
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
        
        wandb.log(log_dict, step=iter_num)
    
    # During sweeps, only save best checkpoint to reduce I/O overhead
    # In standalone mode, save based on always_save_checkpoint config
    save_checkpoint = False
    if losses['val'] < meta['best_val_loss']:
        meta['best_val_loss'] = losses['val']
        save_checkpoint = True
    elif not is_sweep_mode and config['always_save_checkpoint']:
        save_checkpoint = True
    
    if save_checkpoint and iter_num > 0:
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': meta['optimizer'].state_dict(),
            'model_args': meta['model_args'],
            'iter_num': iter_num,
            'best_val_loss': meta['best_val_loss'],
            'config': config,
        }
        console.print(f"saving checkpoint to {config['out_dir']}/{meta['checkpoint_filename']}")
        torch.save(checkpoint_data, os.path.join(config['out_dir'], meta['checkpoint_filename']))

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

def compute_token_colors(paths_data, val_data, meta):
    """Compute ANSI color codes for tokens based on their depth and train/val split"""
    train_tokens = set(np.unique(paths_data))
    val_tokens = set(np.unique(val_data))
    
    # Extract metadata for coloring
    root_vertex = meta['root_vertex']
    special_tokens = set(meta['special_tokens'].values())
    use_task_tokens = meta['use_task_tokens']
    
    # Build a mapping from each token to its distance from root
    token_to_depth = {}
    
    # Reshape data to get sequences (paths_data is a flat memmap, need to reshape)
    # Calculate sequence length from metadata
    block_size = meta['block_size']
    seq_length = block_size + 1  # block_size is context + targets - 1, so full sequence is block_size + 1
    
    # Reshape paths_data and val_data into sequences
    paths_sequences = paths_data.reshape(-1, seq_length)
    val_sequences = val_data.reshape(-1, seq_length)
    
    # Process training paths to determine depth
    for path_seq in paths_sequences:
        # Skip special tokens and find the actual path
        path_tokens = [t for t in path_seq[1+(1 if use_task_tokens else 0):] if t not in special_tokens]
        if len(path_tokens) > 0:
            # First token after special tokens should be leaf, last should be root
            for i, token in enumerate(path_tokens):
                # Distance from root: 0 for root, increases towards leaf
                depth = len(path_tokens) - 1 - i
                token_int = int(token)  # Convert to Python int
                if token_int not in token_to_depth:
                    token_to_depth[token_int] = depth
    
    # Process validation paths
    for path_seq in val_sequences:
        path_tokens = [t for t in path_seq[1+(1 if use_task_tokens else 0):] if t not in special_tokens]
        if len(path_tokens) > 0:
            for i, token in enumerate(path_tokens):
                depth = len(path_tokens) - 1 - i
                token_int = int(token)  # Convert to Python int
                if token_int not in token_to_depth:
                    token_to_depth[token_int] = depth
    
    # Determine max depth for normalization
    max_depth = max(token_to_depth.values()) if token_to_depth else 1
    
    # ANSI color codes - extended palette for finer gradients
    # Training path colors (RED at leaf -> YELLOW at root)
    RED = '\033[91m'           # Bright red
    ORANGE_RED = '\033[38;5;202m'  # Orange-red
    ORANGE = '\033[38;5;208m'      # Orange
    YELLOW_ORANGE = '\033[38;5;214m'  # Yellow-orange
    
    # Validation path colors (GREEN at leaf -> YELLOW at root)
    GREEN = '\033[92m'         # Bright green
    LIME = '\033[38;5;154m'    # Lime green
    YELLOW_GREEN = '\033[38;5;190m'  # Yellow-green
    LIGHT_YELLOW = '\033[38;5;226m'  # Light yellow
    
    YELLOW = '\033[93m'        # Yellow
    RESET = '\033[0m'
    
    token_colors = {}
    
    # Convert train_tokens and val_tokens to Python ints
    train_tokens_int = {int(t) for t in train_tokens}
    val_tokens_int = {int(t) for t in val_tokens}
    
    # Color each token based on its role and depth with fine-grained blending
    for token in train_tokens_int | val_tokens_int:
        # Skip special tokens (no color)
        if token in special_tokens:
            continue
        
        # Root is always yellow (both train and val)
        if token == root_vertex:
            token_colors[token] = YELLOW
        else:
            depth = token_to_depth.get(token, 0)
            # Normalize depth: 0.0 at root, 1.0 at leaf
            normalized_depth = 1.0 - (depth / max_depth if max_depth > 0 else 0.0)
            
            # Determine if this token appears in validation paths
            is_val_token = token in val_tokens_int
            
            # Fine-grained color blending based on depth
            # normalized_depth ranges from 0.0 (root) to 1.0 (leaf)
            
            if is_val_token:
                # Validation: GREEN (at leaf) -> YELLOW (at root)
                if normalized_depth >= 0.875:
                    token_colors[token] = GREEN  # Leaf - bright green
                elif normalized_depth >= 0.625:
                    token_colors[token] = LIME  # Lime green
                elif normalized_depth >= 0.375:
                    token_colors[token] = YELLOW_GREEN  # Yellow-green
                elif normalized_depth >= 0.125:
                    token_colors[token] = LIGHT_YELLOW  # Light yellow
            else:
                # Training: RED (at leaf) -> YELLOW (at root)
                if normalized_depth >= 0.875:
                    token_colors[token] = RED  # Leaf - bright red
                elif normalized_depth >= 0.625:
                    token_colors[token] = ORANGE_RED  # Orange-red
                elif normalized_depth >= 0.375:
                    token_colors[token] = ORANGE  # Orange
                elif normalized_depth >= 0.125:
                    token_colors[token] = YELLOW_ORANGE  # Yellow-orange
    
    return token_colors, RESET

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

    gen.generate_dataset_if_needed(
        num_pause_tokens=default_config['num_pause_tokens'],
        use_undirected=default_config['use_undirected'],
        use_directional_tokens=default_config['use_directional_tokens'],
        use_task_tokens=default_config['use_task_tokens'],
        predict_direction_for_edge_task=default_config['predict_direction_for_edge_task'],
        use_task_tokens_in_path=default_config.get('use_task_tokens_in_path', False),
    )
    
    meta, paths_data, edges_data, val_data = gen.load_dataset()
    
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
    
    if default_config['interleave_dataset']:
        print(f"Training dataset composition (INTERLEAVED):")
        print(f"  Paths: {paths_size}")
        print(f"  Edges: {edges_size}")
        print(f"  Total Combined: {paths_size + edges_size} samples")
    else:
        print(f"Training dataset composition:")
        print(f"  Paths: {paths_size} (no replication)")
        print(f"  Edges: {edges_size}")
        print(f"  Total: {paths_size + edges_size} samples")

    # Auto-detect device
    device, device_type, gpu_id = detect_device(default_config)

    # Set random seed and backend configurations
    torch.manual_seed(default_config['seed'])
    ptdtype, dtype = set_dtype(default_config)

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    os.makedirs(default_config['out_dir'], exist_ok=True)
    checkpoint_filename = f'ckpt_{custom_name}_{gpu_id}.pt' if custom_name else "ckpt.pt"
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

    # If interleaving, we will combine datasets later, but need to consider this for memory
    combined_data = None
    combined_size = 0
    if default_config['interleave_dataset']:
        # Balance datasets by upsampling the smaller one to match the larger one
        max_size = max(paths_size, edges_size)
        
        if paths_size < max_size:
            print(f"Balancing: Upsampling paths from {paths_size} to {max_size}")
            indices = np.random.choice(paths_size, max_size, replace=True)
            paths_data_balanced = paths_data[indices]
        else:
            paths_data_balanced = paths_data
            
        if edges_size < max_size:
            print(f"Balancing: Upsampling edges from {edges_size} to {max_size}")
            indices = np.random.choice(edges_size, max_size, replace=True)
            edges_data_balanced = edges_data[indices]
        else:
            edges_data_balanced = edges_data

        # Concatenate paths and edges
        combined_data = np.concatenate((paths_data_balanced, edges_data_balanced), axis=0)
        combined_size = combined_data.shape[0]
        # Shuffle the combined data initially
        np.random.shuffle(combined_data)
        
        # Calculate memory for combined dataset (pass empty array for edges to reuse function)
        dataset_reserved_memory = determine_dataset_in_device_size(device, device_type, combined_data, np.array([]), val_data)
        
        # Target batch size is the combined size
        target_bs_ref = combined_size
    else:
        dataset_reserved_memory = determine_dataset_in_device_size(device, device_type, paths_data, edges_data, val_data)
        target_bs_ref = edges_size

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
    
    if default_config['interleave_dataset']:
        # In interleaved mode, epoch is 1 pass over combined dataset
        batches_per_epoch = int(np.ceil(combined_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
        max_iters = default_config['epochs'] * batches_per_epoch
        
        print(f"\n=== Training Schedule (Interleaved) ===")
        print(f"Total samples: {combined_size}")
        print(f"Batches per epoch: {batches_per_epoch}")
        print(f"Total iterations: {max_iters}")
        print(f"=========================\n")
        
    else:
        # Calculate iterations per epoch for edges and paths
        edge_iterations_per_epoch = default_config['edge_iterations_per_epoch']
        path_iterations_per_epoch = default_config['path_iterations_per_epoch']
        
        # Calculate batches per dataset
        edge_batches_per_iteration = int(np.ceil(edges_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
        path_batches_per_iteration = int(np.ceil(paths_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
        
        print(f"\n=== Training Schedule ===")
        print(f"Edge batches per iteration: {edge_batches_per_iteration}")
        print(f"Path batches per iteration: {path_batches_per_iteration}")
        print(f"Edge iterations per epoch: {edge_iterations_per_epoch}")
        print(f"Path iterations per epoch: {path_iterations_per_epoch}")
        print(f"=========================\n")

        # One epoch = A edge iterations + B path iterations
        batches_per_epoch = edge_iterations_per_epoch * edge_batches_per_iteration + path_iterations_per_epoch * path_batches_per_iteration
        max_iters = default_config['epochs'] * batches_per_epoch

    meta['max_iters'] = max_iters
    meta['batches_per_epoch'] = batches_per_epoch
    
    val_batch_size = min(num_holdout, train_batch_size)
    eval_iters = int(np.ceil(VAL_DATASET_SIZE / val_batch_size))
    # Calculate learning rate schedule parameters
    warmup_iters = int(max_iters * default_config['warmup_frac'])
    lr_decay_iters = int(max_iters * default_config['lr_decay_frac'])
    meta['warmup_iters'] = warmup_iters
    meta['lr_decay_iters'] = lr_decay_iters
    
    # Skip theoretical loss calculation for separate datasets (not applicable)
    # theoretical_token_1_loss = get_theoretical_loss(meta)
    
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
    checkpoint = None

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
            config=default_config
        )
    
    # Init tracking variables
    iter_num = 0
    
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
    print(f"  Paths: {paths_size} sequences of length {paths_seq_length}")
    print(f"  Edges: {edges_size} sequences of length {edges_seq_length}")
    print(f"  Val: {VAL_DATASET_SIZE} sequences of length {val_seq_length}")
    print(f"  Block size: {meta['block_size']}")
    
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
    
    def preprocess_dataset(data_np):
        # Convert to tensor with optimized dtype
        data_inv = torch.from_numpy(data_np.astype(np.int64)) # Load as int64 first safely
        
        # Split into X and Y
        # X: 0 to L-1
        # Y: 1 to L
        X = data_inv[:, :-1].to(storage_dtype)
        Y = data_inv[:, 1:].clone() # Keep as int64 temporarily for masking if needed, or mask then cast
        
        # Apply masking to Y
        if pad_token_id is not None:
            Y[Y == pad_token_id] = -1
        if pause_token_id is not None:
            Y[Y == pause_token_id] = -1
            
        # Now cast Y to storage dtype. 
        # Note: -1 (mask) in int16/int32 is preserved as -1 (signed)
        Y = Y.to(storage_dtype)
        
        return X, Y

    # Create tensors and load to GPU if pre-calculated decision indicates they fit
    # If interleaving, combined_data is already prepared
    if default_config['interleave_dataset']:
        paths_X, paths_Y = None, None
        edges_X, edges_Y = None, None
        print("Pre-processing combined dataset...")
        combined_X, combined_Y = preprocess_dataset(combined_data)
        print(f"Created pre-processed combined tensors: X={combined_X.shape}, Y={combined_Y.shape}, dtype={storage_dtype}")
    else:
        print("Pre-processing separate datasets...")
        paths_X, paths_Y = preprocess_dataset(paths_data)
        edges_X, edges_Y = preprocess_dataset(edges_data)
        combined_X, combined_Y = None, None
        print(f"Created pre-processed path tensors: X={paths_X.shape}, Y={paths_Y.shape}")
    
    # Store validation data with optimized dtype too (though less critical)
    val_data_tensor = torch.from_numpy(val_data.astype(np.int64)).to(storage_dtype)
    
    datasets_on_gpu = False
    if device_type == 'cuda':
        # Use pre-calculated decision: if reserved_memory > 0, datasets will be loaded to GPU
        if dataset_reserved_memory > 0:
            print(f"\n=== Loading Datasets to GPU ===")
            print(f"Reserved memory: {dataset_reserved_memory / 1e9:.3f} GB")
            print("✓ Loading datasets to GPU for faster training")
            if default_config['interleave_dataset']:
                combined_X = combined_X.pin_memory().to(device, non_blocking=True)
                combined_Y = combined_Y.pin_memory().to(device, non_blocking=True)
            else:
                paths_X = paths_X.pin_memory().to(device, non_blocking=True)
                paths_Y = paths_Y.pin_memory().to(device, non_blocking=True)
                edges_X = edges_X.pin_memory().to(device, non_blocking=True)
                edges_Y = edges_Y.pin_memory().to(device, non_blocking=True)
            val_data_tensor = val_data_tensor.pin_memory().to(device, non_blocking=True)
            datasets_on_gpu = True
            print(f"===================================\n")
        else:
            print(f"\n=== Dataset Loading Decision ===")
            print("✗ Datasets will stay on CPU (will transfer batches on-demand)")
            # Should we pin memory on CPU? Yes, always good for transfer
            if default_config['interleave_dataset']:
                combined_X = combined_X.pin_memory()
                combined_Y = combined_Y.pin_memory()
            else:
                paths_X = paths_X.pin_memory()
                paths_Y = paths_Y.pin_memory()
                edges_X = edges_X.pin_memory()
                edges_Y = edges_Y.pin_memory()
            print(f"===================================\n")
            datasets_on_gpu = False
    else:
        # For non-CUDA devices, always keep on CPU or move to device as appropriate
        # On MPS, memory is unified, so .to(device) is practically zero-copy for large tensors?
        # Actually explicitly moving to mps device is good if it fits.
        if device_type != 'cpu':
            if default_config['interleave_dataset']:
                combined_X = combined_X.to(device)
                combined_Y = combined_Y.to(device)
            else:
                paths_X = paths_X.to(device)
                paths_Y = paths_Y.to(device)
                edges_X = edges_X.to(device)
                edges_Y = edges_Y.to(device)
            val_data_tensor = val_data_tensor.to(device)
            datasets_on_gpu = True
        else:
            datasets_on_gpu = False
    
    # Keep NumPy versions for evaluate_samples (will optimize separately)
    paths_data_np = paths_data
    edges_data_np = edges_data
    val_data_np = val_data
    
    # Initialize epoch indices for sampling without replacement
    if default_config['interleave_dataset']:
        paths_epoch_indices = None
        edges_epoch_indices = None 
        combined_epoch_indices = np.arange(combined_size)
        # Perform initial shuffle once
        np.random.shuffle(combined_epoch_indices)
    else:
        paths_epoch_indices = np.arange(paths_size)
        edges_epoch_indices = np.arange(edges_size)
        # Perform initial shuffle once for each dataset
        np.random.shuffle(paths_epoch_indices)
        np.random.shuffle(edges_epoch_indices)
        
    val_epoch_indices = np.arange(VAL_DATASET_SIZE)
    # Perform initial shuffle for validation
    np.random.shuffle(val_epoch_indices)
    
    paths_batch_idx = 0
    edges_batch_idx = 0
    combined_batch_idx = 0
    val_batch_idx = 0
    
    # Track whether we've completed at least one full pass through each dataset
    # This is used to determine shuffling strategy
    paths_epoch_completed = False
    edges_epoch_completed = False
    combined_epoch_completed = False
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
        use_task_tokens = meta.get('use_task_tokens', True)
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
            # If PATH task token exists, the first target token is the leaf; ignore it.
            if use_task_tokens:
                path_rows = (x_in[:, 0] == PATH)
                y_out[path_rows, 0] = -1 # ignore the first targent token leaf
            # no task token, first token in y is PAUSE or the root which is already masked correctly
            return y_out

        if dataset == 'edges':
            return mask_edges(y)
        if dataset == 'paths':
            return mask_paths(x, y)
        if dataset == 'combined':
            # Need to determine per-sample task type.
            if use_task_tokens:
                if EDGE is None or PATH is None:
                    raise ValueError("Combined dataset requires EDGE and PATH tokens in metadata when use_task_tokens=True")
                is_edge = (x[:, 0] == EDGE)
                is_path = (x[:, 0] == PATH)
            else:
                # Without task tokens, we can only disambiguate if directional tokens are present.
                if not use_directional_tokens:
                    raise ValueError("Cannot interleave edges/paths without task tokens or directional tokens (ambiguous sequences).")
                # Note: if predict_direction_for_edge_task=True, EDGE sequences do not contain GT/LT in X
                # (direction is the supervised token), so disambiguation is impossible without task tokens.
                if bool(meta.get('predict_direction_for_edge_task', False)):
                    raise ValueError("Cannot interleave edges/paths without task tokens when predict_direction_for_edge_task=True (ambiguous sequences).")
                # For predict_direction_for_edge_task=False, EDGE examples include GT/LT inside X.
                # Support both historical layouts:
                #   - [GT/LT, u, v, ...]  (direction first)
                #   - [u, GT/LT, v, ...]  (u first)
                is_edge = (x[:, 0] == GT) | (x[:, 0] == LT)
                if x.size(1) >= 2:
                    is_edge = is_edge | (x[:, 1] == GT) | (x[:, 1] == LT)
                is_path = ~is_edge

            y_out = y.clone()
            if is_edge.any():
                y_out[is_edge] = mask_edges(y_out[is_edge])
            if is_path.any():
                y_out[is_path] = mask_paths(x[is_path], y_out[is_path])
            return y_out

        return y

    def get_batch(dataset):
        """Sample a batch from the edge dataset"""
        nonlocal edges_batch_idx, edges_epoch_indices, paths_batch_idx, paths_epoch_indices, val_batch_idx, val_epoch_indices, combined_batch_idx, combined_epoch_indices
        nonlocal paths_epoch_completed, edges_epoch_completed, combined_epoch_completed, val_epoch_completed
        nonlocal last_mask_debug_str

        if dataset == 'edges':
            batch_idx = edges_batch_idx
            epoch_indices = edges_epoch_indices
            dataset_size = edges_size
            X_source = edges_X
            Y_source = edges_Y
            epoch_completed = edges_epoch_completed
        elif dataset == 'paths':
            batch_idx = paths_batch_idx
            epoch_indices = paths_epoch_indices
            dataset_size = paths_size
            X_source = paths_X
            Y_source = paths_Y
            epoch_completed = paths_epoch_completed
        elif dataset == 'combined':
            batch_idx = combined_batch_idx
            epoch_indices = combined_epoch_indices
            dataset_size = combined_size
            X_source = combined_X
            Y_source = combined_Y
            epoch_completed = combined_epoch_completed
        elif dataset == 'val':
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

        if dataset == 'edges':
            edges_batch_idx = batch_idx
            edges_epoch_completed = epoch_completed
        elif dataset == 'paths':
            paths_batch_idx = batch_idx
            paths_epoch_completed = epoch_completed
        elif dataset == 'combined':
            combined_batch_idx = batch_idx
            combined_epoch_completed = epoch_completed
        elif dataset == 'val':
            val_batch_idx = batch_idx
            val_epoch_completed = epoch_completed
        else:
            raise ValueError("This should not happen")


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

        # Apply task-specific masking (edge/path/combined).
        if default_config.get('debug_masking') and (iter_num % default_config.get('log_interval', 100) == 0):
            y_before = y.clone()
            y_after = apply_task_specific_target_mask(x, y, dataset)
            last_mask_debug_str = _format_mask_debug(x, y_before, y_after, dataset)
            y = y_after
        else:
            y = apply_task_specific_target_mask(x, y, dataset)
        
        return x, y
    
    @torch.no_grad()
    def estimate_metrics(split, print_samples=False):
        """Estimate loss and metrics on validation or training split"""
        out = {}
        model.eval()
        
        # Determine data and size
        if split == 'val':
            nonlocal val_batch_idx, val_epoch_indices
            # Reset validation batch state for reproducible evaluation
            val_batch_idx = 0
            np.random.shuffle(val_epoch_indices)
            num_iters = eval_iters
            data_source = val_data
            data_size = VAL_DATASET_SIZE
        else: # train
            # For training, we sample randomly from paths_data
            # We use a limited number of iterations similar to validation
            num_iters = eval_iters
            data_source = paths_data
            data_size = paths_size
            
        token_losses = {i: [] for i in range(1, graph_length + 1)}
        batch_losses = torch.zeros(num_iters)
        
        for k in range(num_iters):
            if split == 'val':
                X, Y = get_batch('val')
            else:
                # Manual sampling for training paths to safely handle interleaved case
                # and ensures we only evaluate on paths
                idx = np.random.randint(0, data_size, train_batch_size)
                batch = torch.from_numpy(data_source[idx].astype(np.int64)).to(device)
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
        num_samples_for_accuracy = min(100, data_size)
        per_token_accuracy, generated_text = compute_per_token_accuracy_autoregressive(
            ctx, model, meta, data_source, num_samples_for_accuracy, device, print_samples
        )
        out[f'{split}_per_token_accuracy'] = per_token_accuracy
        
        if print_samples and split == 'val':
            out['generated_text'] = generated_text
        
        model.train()
        return out
    
    # Training loop with interleaved edge and path training
    t0 = time.time()
    
    # Track which phase we're in (edge or path)
    if default_config['interleave_dataset']:
        current_phase = 'combined'
        X, Y = get_batch('combined')
    else:   
        current_phase = 'edge'  # Start with edges
        phase_iteration_count = 0
        
        # Initialize with first batch
        if current_phase == 'edge':
            X, Y = get_batch('edges')
        else:
            X, Y = get_batch('paths')
    
    # Live display for evaluation examples (optional - controlled by live_display config)
    use_live_display = default_config.get('live_display', True)
    
    if use_live_display:
        layout = Layout()
        show_training_slices = default_config.get('show_training_slices', False)
        show_debug_masking = default_config.get('debug_masking', False)
        
        # Build layout based on enabled features
        layout_components = [
            Layout(name="metrics", size=14),  # Fixed size for metrics table
            Layout(name="evaluation"),
        ]
        if show_training_slices:
            layout_components.append(Layout(name="training"))
        if show_debug_masking:
            layout_components.append(Layout(name="mask", size=10))
        
        layout.split_column(*layout_components)
        
        layout["metrics"].update(Panel("Waiting for first evaluation...", title="Validation Metrics", border_style="magenta"))
        layout["evaluation"].update(Panel("Waiting for first evaluation...", title="Evaluation Examples", border_style="blue"))
        if show_training_slices:
            layout["training"].update(Panel("Waiting for first training batch...", title="Training Slice (10 samples)", border_style="green"))
        if show_debug_masking:
            layout["mask"].update(Panel("Waiting for first mask debug...", title="Mask Debug", border_style="yellow"))
        live_context = Live(layout, console=console, refresh_per_second=4)
    else:
        layout = None
        live_context = nullcontext()

    with live_context:
        while True:
            # Set learning rate
            lr = get_lr(iter_num, warmup_iters, lr_decay_iters, default_config) if default_config['decay_lr'] else default_config['learning_rate']

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate
            if iter_num % default_config['eval_interval'] == 0:
                print_samples = iter_num % default_config['print_eval_interval'] == 0
                # Calculate tokens_per_sec for display if available
                current_tokens_per_sec = None
                if 'dt' in locals() and dt > 0:
                     # Re-calculate or use stored value. We need 'steps' and 'block_size'
                     # 'steps' is defined below but used from previous iter effectively? 
                     # Actually 'steps' is defined in the loop. For iter_num > 0 it should be available.
                     if 'steps' in locals():
                         current_tokens_per_sec = (train_batch_size * steps * meta['block_size']) / dt
                
                train_total_dataset_size = combined_size if default_config['interleave_dataset'] else (paths_size + edges_size)
                evaluate(
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
                    eval_layout_component=layout["evaluation"] if use_live_display else None,
                    metrics_layout_component=layout["metrics"] if use_live_display else None,
                    tokens_per_sec=current_tokens_per_sec,
                    batch_size=train_batch_size,
                    train_dataset_size=train_total_dataset_size,
                    eval_dataset_size=VAL_DATASET_SIZE,
                )
            
            if iter_num == 0 and default_config['eval_only']:
                break
            
            # Forward backward update with batch prefetching for better GPU utilization
            if default_config['interleave_dataset']:
                cur_batch_size = 1 # Not really used in this loop structure for steps calc 
                # For combined, we don't have "batches per iteration" concept in the same way
                # We just iterate
                # But steps likely refers to gradient accumulation steps
                pass
            else:
                cur_batch_size = ( edge_batches_per_iteration if current_phase == 'edge' else path_batches_per_iteration)   
                
            steps = min(default_config['gradient_accumulation_steps'], cur_batch_size) if not default_config['interleave_dataset'] else default_config['gradient_accumulation_steps']

            for micro_step in range(steps):
                with ctx:
                    _, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
                    loss = loss / steps
                
                # Prefetch next batch while backward pass runs (overlap I/O with compute)
                if micro_step < steps - 1:
                    if default_config['interleave_dataset']:
                        X_next, Y_next = get_batch('combined')
                    else: 
                        if current_phase == 'edge':
                            X_next, Y_next = get_batch('edges')
                        else:
                            X_next, Y_next = get_batch('paths')
                
                scaler.scale(loss).backward()
                
                # Move prefetched batch to current (if not last step)
                if micro_step < steps - 1:
                    X, Y = X_next, Y_next
            
            # Determine next batch based on interleaving schedule
            if not default_config['interleave_dataset']:
                # Check if we've completed the current phase's iterations
                if current_phase == 'edge':
                    phase_iteration_count += 1
                    if phase_iteration_count >= edge_iterations_per_epoch * edge_batches_per_iteration:
                        # Switch to path phase
                        current_phase = 'path'
                        phase_iteration_count = 0
                        # Reset batch indices for new phase
                        paths_batch_idx = 0
                else:  # path phase
                    phase_iteration_count += 1
                    if phase_iteration_count >= path_iterations_per_epoch * path_batches_per_iteration:
                        # Switch back to edge phase (new epoch)
                        current_phase = 'edge'
                        phase_iteration_count = 0
                        # Reset batch indices for new phase
                        edges_batch_idx = 0
            
            # Get batch for next iteration
            if default_config['interleave_dataset']:
                X, Y = get_batch('combined')
            else:
                if current_phase == 'edge':
                    X, Y = get_batch('edges')
                else:
                    X, Y = get_batch('paths')
            
            # Clip gradients
            if default_config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), default_config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            current_epoch = iter_num / meta['batches_per_epoch']
            if iter_num % default_config['log_interval'] == 0:
                lossf = loss.item() * steps
                tokens_per_sec = (X.numel() * steps) / dt
                if default_config['interleave_dataset']:
                    phase_label = "[COMBINED]"
                else:
                    phase_label = "[EDGE]" if current_phase == 'edge' else "[PATH]"
                # console.print(f"iter {iter_num}: {phase_label} loss {lossf:.4f}, time {dt*1000:.2f}ms, tok/sec {tokens_per_sec:.2f}")
                if default_config['wandb_log']:
                    wandb.log({
                        'train/loss/overall': lossf,
                        'dt': dt,
                        'iter': iter_num,
                        "epoch": round(current_epoch, 4),
                        'tokens_per_sec': tokens_per_sec,
                    }, step=iter_num)
                
                # Update training slice panel (only if live display and show_training_slices are enabled)
                # Only update every vis_interval to save sync/formatting time
                if use_live_display and iter_num % default_config['vis_interval'] == 0:
                    if default_config.get('show_training_slices', False):
                        # Reconstruct full sequence for visualization: X + last token of Y
                        # Note: Y has masking (-1) applied, so if the last token is masked, it won't show, 
                        # but for path tasks the last token (LEAF) is not masked.
                        full_batch = torch.cat([X, Y[:, -1:]], dim=1)
                        training_slice_str = format_training_slice(full_batch, itos, meta, num_samples=10)
                        layout["training"].update(Panel(training_slice_str, title=f"Training Slice (Iter {iter_num})", border_style="green"))
                    if default_config.get('debug_masking') and last_mask_debug_str is not None:
                        layout["mask"].update(Panel(last_mask_debug_str, title=f"Mask Debug (Iter {iter_num})", border_style="yellow"))
                
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
                            console.print(f"[yellow]Warning: Failed to log attention maps: {e}[/yellow]")
            
            iter_num += 1
            
            if iter_num > max_iters:
                break
    
    # Cleanup and finalization
    console.print("Finalizing training run...")
    
    # Clear GPU memory before finishing
    if device_type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            console.print(f"Warning during GPU cleanup: {e}")
    
    # Only call wandb.finish() if we initialized wandb ourselves (not in sweep mode)
    # In sweep mode, the agent handles finishing the run
    if default_config['wandb_log'] and wandb.run is not None:
        # Check if we're in sweep mode
        if not hasattr(wandb.run, 'sweep_id') or wandb.run.sweep_id is None:
            # Standalone mode - we initialized it, so we finish it
            wandb.finish()
        # In sweep mode, don't call finish - let the agent handle it
    
    console.print("Training complete!")


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

