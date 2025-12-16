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
import wandb
import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig, GPT
from pathstar import InWeightsPathStar

# Rich imports
from rich.console import Console
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
        
        # Dataset generation parameters
        'graph_d': 1000,
        'graph_l': 5,
        'randomize_vocab_size': 'auto',
        'graph_holdout_percentage': 0.2,
        'num_pause_tokens': 5,
        'use_undirected': True,
        'use_directional_tokens': False,
        'use_task_tokens': False,
        
        # Training parameters
        'gradient_accumulation_steps': 1,
        'edge_iterations_per_epoch': 10,  # Number of iterations on edges per epoch
        'path_iterations_per_epoch': 10,  # Number of iterations on paths per epoch
        'epochs': 1000,
        
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
        'seed': 1337
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
    token_str = itos.get(token, str(token))
    # meta['token_colors'] maps token_id -> ANSI code
    # We check for the specific ANSI codes used in train.py
    ansi_color = meta.get('token_colors', {}).get(token, '')
    if '\033[91m' in ansi_color: return f"[bold red]{token_str}[/]"
    if '\033[92m' in ansi_color: return f"[bold green]{token_str}[/]"
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

def create_metrics_table(metrics, graph_length, iter_num, epoch, lr):
    """Create a Rich Table for per-token metrics (Train vs Val)"""
    table = Table(title=f"Per-Token Metrics (Iter {iter_num}, Epoch {epoch:.2f}, LR {lr:.2e})", show_header=True, header_style="bold magenta")
    table.add_column("Pos", style="cyan", justify="center")
    table.add_column("Train Loss", style="red", justify="right")
    table.add_column("Val Loss", style="red", justify="right")
    table.add_column("Train Acc", style="green", justify="right")
    table.add_column("Val Acc", style="green", justify="right")
    
    for i in range(1, graph_length + 1):
        t_loss = metrics.get('train_per_token', {}).get(i, float('nan'))
        v_loss = metrics.get('val_per_token', {}).get(i, float('nan'))
        t_acc = metrics.get('train_per_token_accuracy', {}).get(i, float('nan'))
        v_acc = metrics.get('val_per_token_accuracy', {}).get(i, float('nan'))
        
        table.add_row(
            str(i),
            f"{t_loss:.4f}", f"{v_loss:.4f}",
            f"{t_acc*100:.1f}%", f"{v_acc*100:.1f}%"
        )
            
    return table

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
    
    for val_idx in sample_indices:
        full_sequence = val_data_batch[val_idx]
        context = full_sequence[:context_length]
        contexts.append(context)
        ground_truth = full_sequence[context_length:context_length + meta['l']]
        ground_truths.append(ground_truth)
    
    contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device_local)
    
    model.eval()
    with torch.no_grad():
        with ctx:
            generated_sequences = model.generate(contexts_batch, max_new_tokens=meta['l'], temperature=1.0, top_k=1)
            generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
            model_context_prediction_batch = generated_sequences[:, :context_length].cpu().numpy()
    model.train()
    
    # Generate formatted text for Live display instead of printing
    generated_text_output = None
    
    ground_truths_array = np.stack(ground_truths)
    if print_samples:
        itos = meta['itos']
        # Rich colors
        RICH_RED = "bold red"
        RICH_GREEN = "bold green"
        RICH_DEFAULT = "default"
        
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

        # Helper to get rich style
        def get_rich_style(token, token_str):
            # Check if token is in pure_train or pure_val sets
            # meta['token_colors'] has the ANSI codes. 
            # We can try to infer or just use the ANSI codes if we wrap in Text.from_ansi
            
            # Re-using the logic from meta['token_colors'] but adapting for Rich would be cleaner
            # but for now let's rely on Text.from_ansi if we can, or just manual coloring.
            
            # Let's inspect meta['token_colors'] again. It maps token_id -> ANSI code string.
            ansi_color = meta.get('token_colors', {}).get(token, '')
            if '\033[91m' in ansi_color: return "[bold red]" + token_str + "[/]"
            if '\033[92m' in ansi_color: return "[bold green]" + token_str + "[/]"
            return token_str

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
    
    for token_pos in range(1, meta['l'] + 1):
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
    
    for idx in sample_indices:
        full_sequence = data[idx]
        context = full_sequence[:context_length]
        contexts.append(context)
        ground_truth = full_sequence[context_length:context_length + meta['l']]
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
                generated_sequences = model.generate(contexts_batch, max_new_tokens=meta['l'], temperature=1.0, top_k=1)
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
    assert 0 <= decay_ratio <= 1
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
            # Include both dropout values if they differ, otherwise just one
            if config['dropout'] == config['embd_dropout']:
                dropout_label = f"D{config['dropout']}_"
            else:
                dropout_label = f"D{config['dropout']}_ED{config['embd_dropout']}_"
            custom_name = (
                f"{utc_time}_"
                f"G{config['graph_d']},"
                f"{config['graph_l']}_"
                f"L{config['n_layer']}_"
                f"E{config['n_embd']}_"
                f"H{config['n_head']}_"
                f"{dropout_label}"
                f"p{config['num_pause_tokens']}_"
                f"{dir_label}"
                f"{tt_label}"
                f"{dt_label}"
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
            # For CPU, we default to the dataset size (or close to it) if provided, 
            # to align with "one iteration means a complete iteration of that dataset".
            # We maintain a minimum of 2000 for efficiency on very small datasets.
            return max(2000, target_batch_size)
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

# GOOD
def evaluate(estimate_metrics, config, meta, iter_num, lr, ctx, device, model, val_data_np, paths_data_np, print_samples=False, eval_layout_component=None, metrics_layout_component=None):
    # Compute metrics for both splits
    val_metrics = estimate_metrics('val', print_samples)
    train_metrics = estimate_metrics('train', False) # Don't print train samples here
    losses = {**val_metrics, **train_metrics}
    
    graph_length = meta['l']
    PATHS_DATASET_SIZE = meta['PATHS_DATASET_SIZE']
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    current_epoch = iter_num / meta['batches_per_epoch']

    # Evaluate autoregressive generation on validation and training samples
    # Use fewer samples during sweeps for faster evaluation
    is_sweep_mode = wandb.run is not None and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None
    autoregressive_eval_samples = 20 if is_sweep_mode else 100
    val_avg_accuracy = evaluate_samples(device, ctx, model,  meta, val_data_np, VAL_DATASET_SIZE, 'val', num_samples=min(VAL_DATASET_SIZE, autoregressive_eval_samples))
    train_avg_accuracy = evaluate_samples(device, ctx, model, meta, paths_data_np, PATHS_DATASET_SIZE, 'train', num_samples=min(PATHS_DATASET_SIZE, autoregressive_eval_samples))
    
    # Update Live display if new samples were generated
    if 'generated_text' in losses and losses['generated_text'] and eval_layout_component:
        eval_layout_component.update(Panel(losses['generated_text'], title="Evaluation Examples", border_style="blue"))

    # Update metrics display
    if metrics_layout_component:
        metrics_table = create_metrics_table(losses, graph_length, iter_num, current_epoch, lr)
        metrics_layout_component.update(Panel(Align.center(metrics_table), title="Validation Metrics", border_style="magenta"))

    # PRINTING
    console.print(f"step {iter_num}: epoch {current_epoch:.2f}, val loss {losses['val']:.4f}, train loss {losses['train']:.4f}")
    
    if 'val_per_token' in losses:
        console.print("  Val per-token losses:")
        per_token_str = ", ".join([f"tok{i}: {losses['val_per_token'].get(i, float('nan')):.4f}" 
                                for i in range(1, min(graph_length + 1, 10))])
        console.print(f"    {per_token_str}")
    
    if 'val_per_token_accuracy' in losses:
        console.print("  Val per-token accuracies (autoregressive):")
        per_token_acc_str = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'].get(i, float('nan'))*100:.1f}%" 
                                    for i in range(1, min(graph_length + 1, 10))])
        console.print(f"    {per_token_acc_str}")
    
    
    if config['wandb_log']:
        log_dict = {
            "iter": iter_num,
            'max_iters': meta['max_iters'],
            'warmup_iters': meta['warmup_iters'],
            "epoch": round(current_epoch, 4),
            "val/loss/overall": losses['val'],
            "train/loss/overall": losses['train'],
            "lr": lr,
            "gen/val_paths_avg_accuracy": val_avg_accuracy,
            "gen/train_paths_avg_accuracy": train_avg_accuracy
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
        
        wandb.log(log_dict)
    
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
    )
    
    meta, paths_data, edges_data, val_data = gen.load_dataset()

    # Precompute token colors for visualization
    # Tokens exclusively in training set -> RED
    # Tokens exclusively in validation set -> GREEN
    # Shared tokens (Root, Special) -> No color
    print("Precomputing token colors...")
    train_tokens = set(np.unique(paths_data))
    val_tokens = set(np.unique(val_data))
    
    pure_train_tokens = train_tokens - val_tokens
    pure_val_tokens = val_tokens - train_tokens
    
    token_colors = {}
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    for t in pure_train_tokens:
        token_colors[t] = RED
    for t in pure_val_tokens:
        token_colors[t] = GREEN
    
    meta['token_colors'] = token_colors
    meta['RESET_COLOR'] = RESET

    meta['randomize_vocab_size'] = gen.randomize_vocab_size
    
    # Extract graph parameters from metadata
    graph_length = meta['l']
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
    
    # Create tensors and load to GPU if pre-calculated decision indicates they fit
    # Create tensors and load to GPU if pre-calculated decision indicates they fit
    # If interleaving, combined_data is already prepared
    if default_config['interleave_dataset']:
        paths_data_tensor = None # Unused in interleaved mode
        edges_data_tensor = None # Unused in interleaved mode
        combined_data_tensor = torch.from_numpy(combined_data.astype(np.int64))
        print(f"Created balanced interleaved dataset tensor with shape {combined_data_tensor.shape}")
    else:
        paths_data_tensor = torch.from_numpy(paths_data.astype(np.int64))
        edges_data_tensor = torch.from_numpy(edges_data.astype(np.int64))
        combined_data_tensor = None
    
    val_data_tensor = torch.from_numpy(val_data.astype(np.int64))
    
    datasets_on_gpu = False
    if device_type == 'cuda':
        # Use pre-calculated decision: if reserved_memory > 0, datasets will be loaded to GPU
        if dataset_reserved_memory > 0:
            print(f"\n=== Loading Datasets to GPU ===")
            print(f"Reserved memory: {dataset_reserved_memory / 1e9:.3f} GB")
            print("✓ Loading datasets to GPU for faster training")
            if default_config['interleave_dataset']:
                combined_data_tensor = combined_data_tensor.pin_memory().to(device, non_blocking=True)
            else:
                paths_data_tensor = paths_data_tensor.pin_memory().to(device, non_blocking=True)
                edges_data_tensor = edges_data_tensor.pin_memory().to(device, non_blocking=True)
            val_data_tensor = val_data_tensor.pin_memory().to(device, non_blocking=True)
            datasets_on_gpu = True
            print(f"===================================\n")
        else:
            print(f"\n=== Dataset Loading Decision ===")
            print("✗ Datasets will stay on CPU (will transfer batches on-demand)")
            print(f"===================================\n")
            datasets_on_gpu = False
    else:
        # For non-CUDA devices, always keep on CPU or move to device as appropriate
        if device_type != 'cpu':
            if default_config['interleave_dataset']:
                combined_data_tensor = combined_data_tensor.to(device)
            else:
                paths_data_tensor = paths_data_tensor.to(device)
                edges_data_tensor = edges_data_tensor.to(device)
            val_data_tensor = val_data_tensor.to(device)
            datasets_on_gpu = True
        else:
            datasets_on_gpu = False
    
    # Keep NumPy versions for evaluate_samples (will optimize separately)
    paths_data_np = paths_data
    val_data_np = val_data
    
    # Initialize epoch indices for sampling without replacement
    if default_config['interleave_dataset']:
        paths_epoch_indices = None
        edges_epoch_indices = None 
        combined_epoch_indices = np.arange(combined_size)
    else:
        paths_epoch_indices = np.arange(paths_size)
        edges_epoch_indices = np.arange(edges_size)
        
    val_epoch_indices = np.arange(VAL_DATASET_SIZE)
    paths_batch_idx = 0
    edges_batch_idx = 0
    combined_batch_idx = 0
    val_batch_idx = 0
    
    # DONE
    def get_batch(dataset):
        """Sample a batch from the edge dataset"""
        nonlocal edges_batch_idx, edges_epoch_indices, paths_batch_idx, paths_epoch_indices, val_batch_idx, val_epoch_indices, combined_batch_idx, combined_epoch_indices

        if dataset == 'edges':
            batch_idx = edges_batch_idx
            epoch_indices = edges_epoch_indices
            dataset_size = edges_size
            dataset_tensors = edges_data_tensor
        elif dataset == 'paths':
            batch_idx = paths_batch_idx
            epoch_indices = paths_epoch_indices
            dataset_size = paths_size
            dataset_tensors = paths_data_tensor
        elif dataset == 'combined':
            batch_idx = combined_batch_idx
            epoch_indices = combined_epoch_indices
            dataset_size = combined_size
            dataset_tensors = combined_data_tensor
        elif dataset == 'val':
            batch_idx = val_batch_idx
            epoch_indices = val_epoch_indices
            dataset_size = VAL_DATASET_SIZE
            dataset_tensors = val_data_tensor
        else:
            raise ValueError("This should not happen")
        
        # Check if we need to shuffle for new epoch
        if batch_idx == 0:
            np.random.shuffle(epoch_indices)
        
        # Get batch indices
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, dataset_size)
        batch_seq_indices = epoch_indices[start_idx:end_idx]
        
        # Update batch index for next call
        batch_idx = (batch_idx + 1) if end_idx < dataset_size else 0

        if dataset == 'edges':
            edges_batch_idx = batch_idx
        elif dataset == 'paths':
            paths_batch_idx = batch_idx
        elif dataset == 'combined':
            combined_batch_idx = batch_idx
        elif dataset == 'val':
            val_batch_idx = batch_idx
        else:
            raise ValueError("This should not happen")

        
        # Extract sequences (from GPU if available, otherwise from CPU and transfer)
        if datasets_on_gpu:
            sequences = dataset_tensors[batch_seq_indices]
        else:
            sequences = dataset_tensors[batch_seq_indices]
            if device_type in ['cuda', 'mps']:
                sequences = sequences.to(device, non_blocking=True)
        
        # Pad or truncate to block_size if needed
        if edges_seq_length < meta['block_size']:
            raise ValueError(f"Sequence length ({edges_seq_length}) is less than block_size ({meta['block_size']}). This should not happen.")
        elif edges_seq_length > meta['block_size']:
            raise ValueError(f"Sequence length ({edges_seq_length}) exceeds block_size ({meta['block_size']}). This should not happen.")
        
        # Create input (x) and target (y) by shifting
        x = sequences[:, :-1].clone()
        y = sequences[:, 1:].clone()
        
        # Mask PAD tokens in targets
        if pad_token_id is not None:
            y[y == pad_token_id] = -1

        # Mask PAUSE tokens in targets
        if pause_token_id is not None:
            y[y == pause_token_id] = -1
        
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
    
    # Live display for evaluation examples
    layout = Layout()
    layout.split_column(
        Layout(name="metrics", size=14), # Fixed size for metrics table
        Layout(name="evaluation"),
        Layout(name="training")
    )
    layout["metrics"].update(Panel("Waiting for first evaluation...", title="Validation Metrics", border_style="magenta"))
    layout["evaluation"].update(Panel("Waiting for first evaluation...", title="Evaluation Examples", border_style="blue"))
    layout["training"].update(Panel("Waiting for first training batch...", title="Training Slice (10 samples)", border_style="green"))

    with Live(layout, console=console, refresh_per_second=4) as live:
        while True:
            # Set learning rate
            lr = get_lr(iter_num, warmup_iters, lr_decay_iters, default_config) if default_config['decay_lr'] else default_config['learning_rate']

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate
            if iter_num % default_config['eval_interval'] == 0:
                print_samples = iter_num % default_config['print_eval_interval'] == 0
                evaluate(estimate_metrics, default_config, meta, iter_num, lr, ctx, device, model, val_data_np, paths_data_np, print_samples, eval_layout_component=layout["evaluation"], metrics_layout_component=layout["metrics"])
            
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
                if default_config['interleave_dataset']:
                    phase_label = "[COMBINED]"
                else:
                    phase_label = "[EDGE]" if current_phase == 'edge' else "[PATH]"
                console.print(f"iter {iter_num}: {phase_label} loss {lossf:.4f}, time {dt*1000:.2f}ms")
                if default_config['wandb_log']:
                    wandb.log({
                        'train/loss/overall': lossf,
                        'dt': dt,
                        'iter': iter_num,
                        "epoch": round(current_epoch, 4),
                    })
                
                # Update training slice panel
                # Reconstruct full sequence for visualization: X + last token of Y
                # Note: Y has masking (-1) applied, so if the last token is masked, it won't show, 
                # but for path tasks the last token (LEAF) is not masked.
                full_batch = torch.cat([X, Y[:, -1:]], dim=1)
                training_slice_str = format_training_slice(full_batch, itos, meta, num_samples=10)
                layout["training"].update(Panel(training_slice_str, title=f"Training Slice (Iter {iter_num})", border_style="green"))
            
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

