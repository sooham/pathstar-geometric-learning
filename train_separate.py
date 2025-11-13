"""
This training script runs on a single GPU and supports wandb sweeps.
This version handles separate edge and path datasets with interleaved training.

To run standalone:
$ python train_separate.py --batch_size=32 --compile=False

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


def get_default_config():
    """
    Returns default configuration for training.
    This can be overridden by command-line args, config files, or wandb sweep.
    """
    return {
        # I/O
        'init_from': 'scratch',  # 'scratch' or 'resume'
        'out_dir': 'out',
        'eval_interval': 100,
        'log_interval': 50,
        'eval_only': False,
        'always_save_checkpoint': True,
        
        # wandb logging
        'wandb_log': True,
        'wandb_project': 'pathstar_interleave',
        'wandb_run_name': None,  # Will be auto-generated
        
        # Dataset generation parameters
        'graph_d': 1000,
        'graph_l': 5,
        'graph_vocab_size': '1max',
        'graph_holdout_percentage': 0.2,
        'num_pause_tokens': 5,
        'use_undirected': True,
        'use_directional_tokens': False,
        'use_task_tokens': False,
        
        # Training parameters
        'gradient_accumulation_steps': 1,
        'edge_iterations_per_epoch': 10,  # Number of iterations on edges per epoch
        'path_iterations_per_epoch': 10,  # Number of iterations on paths per epoch
        
        # Model architecture
        'n_layer': 3,
        'n_head': 8,
        'n_embd': 96,
        'dropout': 0.0,
        'bias': False,
        
        # Optimization
        'learning_rate': 1e-3,
        'label_smoothing': 0.10,
        'epochs': 1000,
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

@torch.compile
def compute_per_token_loss_with_teacher_forcing(meta, logits, input, targets, token_positions, task_type='path', debug=False, split=''):
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
            input[:, 0] == meta['task_tokens']['EDGE'],
            torch.tensor((1 if use_task_tokens else 0) + (1 if meta['use_directional_tokens'] else 0) + 1, device=input.device),
            torch.where(
                input[:, 0] == meta['task_tokens']['PATH'],
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
    
    for token_pos in token_positions:
        y_idx = context_length_per_input + token_pos - 2
        y_idx = y_idx.squeeze(1)
        
        valid_idx_mask = y_idx < targets.size(1)
        
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
            logits_at_pos[valid_idx_mask] = logits[valid_batch_indices, valid_y_idx, :]
            targets_at_pos[valid_idx_mask] = targets[valid_batch_indices, valid_y_idx]
            
            valid_mask = targets_at_pos != -1
            if valid_mask.any():
                logits_valid = logits_at_pos[valid_mask]
                targets_valid = targets_at_pos[valid_mask]
                
                token_loss = F.cross_entropy(logits_valid, targets_valid, reduction='mean')
                per_token_losses[token_pos] = token_loss.item()
            else:
                per_token_losses[token_pos] = float('nan')
        else:
            per_token_losses[token_pos] = float('nan')
    
    return per_token_losses


def compute_per_token_accuracy_autoregressive(ctx, model, meta, val_data_batch, num_samples, device_local):
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
    
    with torch.no_grad():
        with ctx:
            generated_sequences = model.generate(contexts_batch, max_new_tokens=meta['l'], temperature=1.0, top_k=1)
            generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
    
    per_token_accuracies = {}
    ground_truths_array = np.stack(ground_truths)
    
    for token_pos in range(1, meta['l'] + 1):
        idx = token_pos - 1
        if idx < generated_tokens_batch.shape[1] and idx < ground_truths_array.shape[1]:
            matches = generated_tokens_batch[:, idx] == ground_truths_array[:, idx]
            accuracy = np.mean(matches)
            per_token_accuracies[token_pos] = accuracy
        else:
            per_token_accuracies[token_pos] = float('nan')
    
    return per_token_accuracies

def evaluate_samples(device, ctx, model, meta, data, data_size, split_name, num_samples=5, eval_batch_size=512):
    """
    Evaluate autoregressive generation on samples from a dataset.
    Assumes data is path-only (no filtering needed).
    
    Args:
        ctx: context 
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
    
    # Concatenate all batches
    all_generated_tokens = np.concatenate(all_generated_tokens, axis=0)
    
    # Calculate accuracies
    print(f"\nAutoregressive generation on {num_samples} {split_name} samples:")
    accuracies = []
    for ground_truth, generated_tokens in zip(ground_truths, all_generated_tokens):
        # Calculate accuracy
        accuracy = np.mean(generated_tokens == ground_truth[:len(generated_tokens)])
        accuracies.append(accuracy)
    
    # Calculate average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"  Average accuracy: {avg_accuracy*100:.1f}%")
    print()  # Empty line for readability
    
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

def set_wandb_name(config):
    if config is not None:
        # Set custom run name for sweep runs
        if wandb.run is not None:
            utc_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            custom_name = (
                f"{utc_time}_"
                f"G{config['graph_d']},"
                f"{config['graph_l']}_"
                f"L{config['n_layer']}_"
                f"E{config['n_embd']}_"
                f"H{config['n_head']}_"
                f"D{config['dropout']}_"
                f"p{config['num_pause_tokens']}_"
                f"{config['epochs']}"
            )
            wandb.run.name = custom_name
            print(f"Set sweep run name: {custom_name}")
            return custom_name

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
    
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    return device, device_type, gpu_id

def set_dtype(config):
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
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
        dropout=config['dropout']
    )
    checkpoint = None
    iter_num = 0
    best_val_loss = float('inf')
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
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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
        best_val_loss = checkpoint['best_val_loss']
    
    if meta['block_size'] < model.config.block_size:
        model.crop_block_size(meta['block_size'])
        model_args['block_size'] = meta['block_size']
    
    model.to(device)

    return model, model_args, checkpoint,  iter_num, best_val_loss

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
                                    gradient_accumulation_steps, safety_factor=0.90):
    """
    Calculate maximum safe batch size based on available GPU memory.
    
    Memory breakdown:
    - Model parameters: N × bytes_per_param
    - Optimizer (AdamW): N × 2 × 4 bytes (momentum + variance in FP32)
    - Gradients: N × bytes_per_param
    - Activations: batch_size × memory_per_sample
    - Output logits: batch_size × seq_len × vocab_size (MAJOR memory consumer!)
    
    Args:
        safety_factor: Use 70% of available memory (conservative for torch.compile)
    
    Returns:
        max_batch_size: Maximum safe batch size
    """
    # Handle device as string or torch.device object
    device_type = device if isinstance(device, str) else device.type
    if device_type != 'cuda':
        return 2000  # Default for non-CUDA
    
    # Get GPU memory info
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = (total_memory - allocated_memory) * safety_factor
    
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


def train(config=None):
    """
    Main training function that can be called standalone or by wandb sweep.
    
    Args:
        config: Optional dict of configuration overrides. If None, uses defaults and command-line args.
    """
    
    # Clear GPU memory at the start of training run
    clear_gpu_memory()
    
    # Get default config
    default_config = get_default_config()
    
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
    if default_config['graph_vocab_size'].endswith('max'):
        factor = int(default_config['graph_vocab_size'][:-3])
        default_config['graph_vocab_size'] = factor * ((default_config['graph_l'] - 1) * default_config['graph_d'] + 1)

    assert default_config['graph_vocab_size'] >= default_config['graph_d'] * (default_config['graph_l'] - 1) + 1, \
        f"graph_vocab_size must be >= graph_d * (graph_l - 1) + 1"
    
    # Generate/load dataset
    gen = InWeightsPathStar(
        d=default_config['graph_d'],
        l=default_config['graph_l'],
        vocab_size=default_config['graph_vocab_size'],
        holdout_percentage=default_config['graph_holdout_percentage'],
    )
    dataset = gen.generate_dataset_if_needed(
        num_pause_tokens=default_config['num_pause_tokens'],
        use_undirected=default_config['use_undirected'],
        use_directional_tokens=default_config['use_directional_tokens'],
        combine=False  # Always use separate datasets for train_separate.py
    )
    
    result = gen.load_dataset()
    if len(result) == 3:
        # Combined mode (shouldn't happen, but handle gracefully)
        meta, train_data, val_data = result
        raise ValueError("train_separate.py requires combine=False. Please regenerate dataset with --combine flag omitted.")
    else:
        # Separate mode
        meta, paths_data, edges_data, val_data = result
    
    # Verify combine is False
    if meta.get('combine', True):
        raise ValueError("train_separate.py requires combine=False in metadata. Please regenerate dataset.")
    
    # Extract graph parameters from metadata
    graph_length = meta['l']
    graph_spokes = meta['d']
    holdout_ratio = meta['holdout_percentage']

    num_holdout = round(graph_spokes * holdout_ratio)
    
    # Get dataset sizes from metadata
    paths_size = meta['PATHS_DATASET_SIZE']
    edges_size = meta['EDGES_DATASET_SIZE']
    
    print(f"Training dataset composition:")
    print(f"  Paths: {paths_size} (no replication)")
    print(f"  Edges: {edges_size}")
    print(f"  Total: {paths_size + edges_size} samples")

    # TODO: port model defintion here
    # Auto-detect device
    device, device_type, gpu_id = detect_device(default_config)

    # Set random seed and backend configurations
    torch.manual_seed(default_config['seed'])
    ptdtype, dtype = set_dtype(default_config)

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    os.makedirs(default_config['out_dir'], exist_ok=True)
    checkpoint_filename = f'ckpt_{custom_name}_{gpu_id}.pt' if custom_name else "ckpt.pt"
    print(f"Checkpoint will be saved as: {checkpoint_filename}")

    model, model_args, checkpoint, iter_num, best_val_loss = initalize_model(device, meta, default_config, checkpoint_filename)

    train_batch_size = calculate_optimal_batch_size_for_training(
        model, meta['block_size'], meta['vocab_size'], device, dtype,
        default_config['gradient_accumulation_steps']
    )
    
    # Calculate training iteration parameters
    VAL_DATASET_SIZE = meta['VAL_DATASET_SIZE']
    
    # Calculate iterations per epoch for edges and paths
    edge_iterations_per_epoch = default_config['edge_iterations_per_epoch']
    path_iterations_per_epoch = default_config['path_iterations_per_epoch']
    
    # Calculate batches per dataset
    edge_batches_per_iteration = int(np.ceil(edges_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
    path_batches_per_iteration = int(np.ceil(paths_size / (train_batch_size * default_config['gradient_accumulation_steps'])))
    
    # One epoch = A edge iterations + B path iterations
    batches_per_epoch = edge_iterations_per_epoch * edge_batches_per_iteration + path_iterations_per_epoch * path_batches_per_iteration
    max_iters = default_config['epochs'] * batches_per_epoch
    
    val_batch_size = min(num_holdout, train_batch_size)
    eval_iters = int(np.ceil(VAL_DATASET_SIZE / val_batch_size))
    # Calculate learning rate schedule parameters
    warmup_iters = int(max_iters * default_config['warmup_frac'])
    lr_decay_iters = int(max_iters * default_config['lr_decay_frac'])
    
    # Skip theoretical loss calculation for separate datasets (not applicable)
    # theoretical_token_1_loss = get_theoretical_loss(meta)
    
    
    # Calculate optimal batch size based on available GPU memory
    # Calculate and apply optimal batch size
    
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
    
    # Compile model
    if default_config['compile']:
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
    best_val_loss = 1e9
    
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}")
    
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
    
    # Calculate dataset structure from metadata
    val_size = meta.get('val_size', VAL_DATASET_SIZE)
    
    # Calculate sequence lengths
    paths_seq_length = len(paths_data) // paths_size
    edges_seq_length = len(edges_data) // edges_size
    val_seq_length = len(val_data) // val_size
    
    print(f"Dataset info:")
    print(f"  Paths: {paths_size} sequences of length {paths_seq_length}")
    print(f"  Edges: {edges_size} sequences of length {edges_seq_length}")
    print(f"  Val: {val_size} sequences of length {val_seq_length}")
    print(f"  Block size: {meta['block_size']}")
    
    assert paths_seq_length == edges_seq_length, f"Sequence length mismatch: paths={paths_seq_length}, edges={edges_seq_length}"
    assert val_size == VAL_DATASET_SIZE, f"Val size mismatch: {val_size} != {VAL_DATASET_SIZE}"
    
    # Reshape data for easier indexing
    paths_data = paths_data.reshape(paths_size, paths_seq_length)
    edges_data = edges_data.reshape(edges_size, edges_seq_length)
    val_data = val_data.reshape(val_size, val_seq_length)
    
    # Determine if datasets can fit in GPU memory (policy: use at most 50% of available VRAM)
    paths_data_tensor = torch.from_numpy(paths_data.astype(np.int64))
    edges_data_tensor = torch.from_numpy(edges_data.astype(np.int64))
    val_data_tensor = torch.from_numpy(val_data.astype(np.int64))
    
    datasets_on_gpu = False
    if device_type == 'cuda':
        # Calculate dataset memory requirements
        paths_data_bytes = paths_data_tensor.numel() * paths_data_tensor.element_size()
        edges_data_bytes = edges_data_tensor.numel() * edges_data_tensor.element_size()
        val_data_bytes = val_data_tensor.numel() * val_data_tensor.element_size()
        total_dataset_bytes = paths_data_bytes + edges_data_bytes + val_data_bytes
        
        # Get available GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory
        allocated_vram = torch.cuda.memory_allocated(device)
        available_vram = total_vram - allocated_vram
        
        # Policy: Only load to GPU if datasets use < 50% of available VRAM
        vram_limit = available_vram * 0.5
        
        print(f"\n=== Dataset Memory Policy Check ===")
        print(f"Paths dataset size: {paths_data_bytes / 1e9:.3f} GB ({paths_data_tensor.shape})")
        print(f"Edges dataset size: {edges_data_bytes / 1e9:.3f} GB ({edges_data_tensor.shape})")
        print(f"Val dataset size: {val_data_bytes / 1e9:.3f} GB ({val_data_tensor.shape})")
        print(f"Total dataset size: {total_dataset_bytes / 1e9:.3f} GB")
        print(f"Available VRAM: {available_vram / 1e9:.3f} GB")
        print(f"VRAM limit (50%): {vram_limit / 1e9:.3f} GB")
        
        if total_dataset_bytes <= vram_limit:
            print("✓ Datasets fit within memory limit - loading to GPU for faster training")
            paths_data_tensor = paths_data_tensor.pin_memory().to(device, non_blocking=True)
            edges_data_tensor = edges_data_tensor.pin_memory().to(device, non_blocking=True)
            val_data_tensor = val_data_tensor.pin_memory().to(device, non_blocking=True)
            datasets_on_gpu = True
        else:
            print("✗ Datasets exceed memory limit - keeping on CPU (will transfer batches on-demand)")
            datasets_on_gpu = False
        print(f"===================================\n")
    else:
        # For non-CUDA devices, always keep on CPU or move to device as appropriate
        if device_type != 'cpu':
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
    paths_epoch_indices = np.arange(paths_size)
    edges_epoch_indices = np.arange(edges_size)
    val_epoch_indices = np.arange(val_size)
    paths_batch_idx = 0
    edges_batch_idx = 0
    val_batch_idx = 0
    
    def get_edge_batch():
        """Sample a batch from the edge dataset"""
        nonlocal edges_batch_idx, edges_epoch_indices
        
        # Check if we need to shuffle for new epoch
        if edges_batch_idx == 0:
            np.random.shuffle(edges_epoch_indices)
        
        # Get batch indices
        start_idx = edges_batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, edges_size)
        batch_seq_indices = edges_epoch_indices[start_idx:end_idx]
        
        actual_batch_size = len(batch_seq_indices)
        
        # Update batch index for next call
        edges_batch_idx = (edges_batch_idx + 1) if end_idx < edges_size else 0
        
        # Extract sequences (from GPU if available, otherwise from CPU and transfer)
        if datasets_on_gpu:
            sequences = edges_data_tensor[batch_seq_indices]
        else:
            sequences = edges_data_tensor[batch_seq_indices]
            if device_type == 'cuda':
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
        
        return x, y
    
    def get_path_batch():
        """Sample a batch from the path dataset"""
        nonlocal paths_batch_idx, paths_epoch_indices
        
        # Check if we need to shuffle for new epoch
        if paths_batch_idx == 0:
            np.random.shuffle(paths_epoch_indices)
        
        # Get batch indices
        start_idx = paths_batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, paths_size)
        batch_seq_indices = paths_epoch_indices[start_idx:end_idx]
        
        actual_batch_size = len(batch_seq_indices)
        
        # Update batch index for next call
        paths_batch_idx = (paths_batch_idx + 1) if end_idx < paths_size else 0
        
        # Extract sequences (from GPU if available, otherwise from CPU and transfer)
        if datasets_on_gpu:
            sequences = paths_data_tensor[batch_seq_indices]
        else:
            sequences = paths_data_tensor[batch_seq_indices]
            if device_type == 'cuda':
                sequences = sequences.to(device, non_blocking=True)
        
        # Pad or truncate to block_size if needed
        if paths_seq_length < meta['block_size']:
            raise ValueError(f"Sequence length ({paths_seq_length}) is less than block_size ({meta['block_size']}). This should not happen.")
        elif paths_seq_length > meta['block_size']:
            raise ValueError(f"Sequence length ({paths_seq_length}) exceeds block_size ({meta['block_size']}). This should not happen.")
        
        # Create input (x) and target (y) by shifting
        x = sequences[:, :-1].clone()
        y = sequences[:, 1:].clone()
        
        # Mask PAD tokens in targets
        if pad_token_id is not None:
            y[y == pad_token_id] = -1
        
        return x, y
    
    def get_val_batch():
        """Sample a batch from the validation dataset"""
        nonlocal val_batch_idx, val_epoch_indices
        
        # Check if we need to shuffle for new epoch
        if val_batch_idx == 0:
            np.random.shuffle(val_epoch_indices)
        
        # Get batch indices
        start_idx = val_batch_idx * val_batch_size
        end_idx = min(start_idx + val_batch_size, val_size)
        batch_seq_indices = val_epoch_indices[start_idx:end_idx]
        
        actual_batch_size = len(batch_seq_indices)
        
        # Update batch index for next call
        val_batch_idx = (val_batch_idx + 1) if end_idx < val_size else 0
        
        # Extract sequences (from GPU if available, otherwise from CPU and transfer)
        if datasets_on_gpu:
            sequences = val_data_tensor[batch_seq_indices]
        else:
            sequences = val_data_tensor[batch_seq_indices]
            if device_type == 'cuda':
                sequences = sequences.to(device, non_blocking=True)
        
        # Pad or truncate to block_size if needed
        if val_seq_length < meta['block_size']:
            raise ValueError(f"Sequence length ({val_seq_length}) is less than block_size ({meta['block_size']}). This should not happen.")
        elif val_seq_length > meta['block_size']:
            raise ValueError(f"Sequence length ({val_seq_length}) exceeds block_size ({meta['block_size']}). This should not happen.")
        
        # Create input (x) and target (y) by shifting
        x = sequences[:, :-1].clone()
        y = sequences[:, 1:].clone()
        
        # Mask PAD tokens in targets
        if pad_token_id is not None:
            y[y == pad_token_id] = -1
        
        return x, y
    
    
    @torch.no_grad()
    def estimate_loss():
        """Estimate loss on validation split"""
        out = {}
        model.eval()
        
        val_token_losses = {i: [] for i in range(1, graph_length + 1)}
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_val_batch()
            with ctx:
                logits, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
            losses[k] = loss.item()
            
            batch_per_token = compute_per_token_loss_with_teacher_forcing(meta, logits, X, Y, range(1, graph_length + 1), task_type='path', debug=False, split='val')
            for token_pos, token_loss in batch_per_token.items():
                if not math.isnan(token_loss):
                    val_token_losses[token_pos].append(token_loss)
        
        out['val'] = losses.mean()
        
        if val_token_losses[1]:
            out['val_per_token'] = {
                token_pos: np.mean(losses_list) if losses_list else float('nan')
                for token_pos, losses_list in val_token_losses.items()
            }
        else:
            out['val_per_token'] = {token_pos: float('nan') for token_pos in range(1, graph_length + 1)}
        
        # Compute per-token accuracy
        num_samples_for_accuracy = min(100, val_size)
        val_per_token_accuracy = compute_per_token_accuracy_autoregressive(ctx, model, meta, val_data, num_samples_for_accuracy, device)
        out['val_per_token_accuracy'] = val_per_token_accuracy
        
        model.train()
        return out
    
    # Training loop with interleaved edge and path training
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    # Track which phase we're in (edge or path)
    current_phase = 'edge'  # Start with edges
    phase_iteration_count = 0
    
    # Initialize with first batch
    if current_phase == 'edge':
        X, Y = get_edge_batch()
    else:
        X, Y = get_path_batch()
    
    while True:
        # Set learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, default_config) if default_config['decay_lr'] else default_config['learning_rate']

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % default_config['eval_interval'] == 0:
            losses = estimate_loss()
            
            current_epoch = iter_num / batches_per_epoch
            
            print(f"step {iter_num}: epoch {current_epoch:.2f}, val loss {losses['val']:.4f}")
            
            if 'val_per_token' in losses:
                print("  Val per-token losses:")
                per_token_str = ", ".join([f"tok{i}: {losses['val_per_token'][i]:.4f}" 
                                          for i in range(1, min(graph_length + 1, 10))])
                print(f"    {per_token_str}")
                if graph_length > 9:
                    per_token_str_rest = ", ".join([f"tok{i}: {losses['val_per_token'][i]:.4f}" 
                                                    for i in range(10, graph_length + 1)])
                    print(f"    {per_token_str_rest}")
            
            if 'val_per_token_accuracy' in losses:
                print("  Val per-token accuracies (autoregressive):")
                per_token_acc_str = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'][i]*100:.1f}%" 
                                              for i in range(1, min(graph_length + 1, 10))])
                print(f"    {per_token_acc_str}")
                if graph_length > 9:
                    per_token_acc_str_rest = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'][i]*100:.1f}%" 
                                                        for i in range(10, graph_length + 1)])
                    print(f"    {per_token_acc_str_rest}")
            
            # Evaluate autoregressive generation on validation and training samples
            # Use fewer samples during sweeps for faster evaluation
            is_sweep_mode = wandb.run is not None and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None
            eval_samples = 20 if is_sweep_mode else 100
            val_avg_accuracy = evaluate_samples(device, ctx, model,  meta, val_data_np, val_size, 'val', num_samples=min(val_size, eval_samples))
            train_avg_accuracy = evaluate_samples(device, ctx, model, meta, paths_data_np, paths_size, 'train', num_samples=min(paths_size, eval_samples))
            
            if default_config['wandb_log']:
                log_dict = {
                    "iter": iter_num,
                    "epoch": round(current_epoch, 4),
                    "val/loss/overall": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "gen/val_avg_accuracy": val_avg_accuracy,
                    "gen/train_avg_accuracy": train_avg_accuracy
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
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint = True
            elif not is_sweep_mode and default_config['always_save_checkpoint']:
                save_checkpoint = True
            
            if save_checkpoint and iter_num > 0:
                checkpoint_data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': default_config,
                }
                print(f"saving checkpoint to {default_config['out_dir']}/{checkpoint_filename}")
                torch.save(checkpoint_data, os.path.join(default_config['out_dir'], checkpoint_filename))
        
        if iter_num == 0 and default_config['eval_only']:
            break
        
        # Forward backward update with batch prefetching for better GPU utilization
        for micro_step in range(default_config['gradient_accumulation_steps']):
            with ctx:
                logits, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
                loss = loss / default_config['gradient_accumulation_steps']
            
            # Prefetch next batch while backward pass runs (overlap I/O with compute)
            if micro_step < default_config['gradient_accumulation_steps'] - 1:
                if current_phase == 'edge':
                    X_next, Y_next = get_edge_batch()
                else:
                    X_next, Y_next = get_path_batch()
            
            scaler.scale(loss).backward()
            
            # Move prefetched batch to current (if not last step)
            if micro_step < default_config['gradient_accumulation_steps'] - 1:
                X, Y = X_next, Y_next
        
        # Determine next batch based on interleaving schedule
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
        if current_phase == 'edge':
            X, Y = get_edge_batch()
        else:
            X, Y = get_path_batch()
        
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
        if iter_num % default_config['log_interval'] == 0:
            lossf = loss.item() * default_config['gradient_accumulation_steps']
            if local_iter_num >= 5:
                mfu = model.estimate_mfu(train_batch_size * default_config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            phase_label = "[EDGE]" if current_phase == 'edge' else "[PATH]"
            print(f"iter {iter_num}: {phase_label} loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        
        iter_num += 1
        local_iter_num += 1
        
        if iter_num > max_iters:
            break
    
    # Cleanup and finalization
    print("Finalizing training run...")
    
    # Clear GPU memory before finishing
    if device_type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning during GPU cleanup: {e}")
    
    # Only call wandb.finish() if we initialized wandb ourselves (not in sweep mode)
    # In sweep mode, the agent handles finishing the run
    if default_config['wandb_log'] and wandb.run is not None:
        # Check if we're in sweep mode
        if not hasattr(wandb.run, 'sweep_id') or wandb.run.sweep_id is None:
            # Standalone mode - we initialized it, so we finish it
            wandb.finish()
        # In sweep mode, don't call finish - let the agent handle it
    
    print("Training complete!")


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
    # Running standalone - use command-line args and configurator.py
    print("Running in standalone mode")
    train(config=None)

