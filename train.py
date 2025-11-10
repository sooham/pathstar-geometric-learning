"""
This training script runs on a single GPU and supports wandb sweeps.

To run standalone:
$ python train.py --batch_size=32 --compile=False

To run with wandb sweep:
$ wandb sweep sweep_config.yaml
$ wandb agent <sweep_id>
"""

import wandb
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig, GPT
from pathstar import InWeightsPathStar


def generate_dataset_name(v, d, l, num_pause_tokens, use_undirected, use_directional_tokens):
    """
    Generate dataset directory name matching the naming convention in pathstar.py
    """
    dir_name = f'inweights_pathstar_v{v}_d{d}_l{l}_p{num_pause_tokens}_{"un" if use_undirected else ""}directed_{"dt" if use_directional_tokens else ""}'
    return dir_name


def check_dataset_exists(dataset_name, d, l, num_pause_tokens, use_undirected, use_directional_tokens, holdout_percentage):
    """
    Check if dataset exists and validate that metadata matches requested parameters.
    
    Returns:
        bool: True if dataset exists and parameters match, False otherwise
    """
    data_dir = os.path.join('data', dataset_name)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    
    # Check if all required files exist
    if not (os.path.exists(data_dir) and os.path.exists(meta_path) and 
            os.path.exists(train_path) and os.path.exists(val_path)):
        return False
    
    # Load metadata and validate parameters
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        # Check if all key parameters match
        params_match = (
            meta.get('d') == d and
            meta.get('l') == l and
            meta.get('num_pause_tokens') == num_pause_tokens and
            meta.get('use_undirected') == use_undirected and
            meta.get('use_directional_tokens') == use_directional_tokens and
            abs(meta.get('holdout_percentage', 0.0) - holdout_percentage) < 1e-6  # Float comparison
        )
        
        if not params_match:
            print(f"Dataset exists but parameters don't match:")
            print(f"  Existing: d={meta.get('d')}, l={meta.get('l')}, pause={meta.get('num_pause_tokens')}, "
                  f"undirected={meta.get('use_undirected')}, directional_tokens={meta.get('use_directional_tokens')}, "
                  f"holdout={meta.get('holdout_percentage')}")
            print(f"  Requested: d={d}, l={l}, pause={num_pause_tokens}, "
                  f"undirected={use_undirected}, directional_tokens={use_directional_tokens}, "
                  f"holdout={holdout_percentage}")
            print(f"  Will regenerate dataset...")
            return False
        
        return True
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False


def generate_dataset_if_needed(d, l, vocab_size, holdout_percentage, num_pause_tokens, 
                               use_undirected, use_directional_tokens):
    """
    Generate the dataset using InWeightsPathStar if it doesn't exist or parameters don't match.
    """
    # Validate vocab_size
    num_vertices = d * (l - 1) + 1
    if vocab_size < num_vertices:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be >= d * (l-1) + 1 = {num_vertices}"
        )
    
    # Generate dataset name
    dataset_name = generate_dataset_name(vocab_size, d, l, num_pause_tokens, use_undirected, use_directional_tokens)
    
    # Check if dataset exists and parameters match
    if check_dataset_exists(dataset_name, d, l, num_pause_tokens, use_undirected, 
                           use_directional_tokens, holdout_percentage):
        print(f"Dataset '{dataset_name}' exists with matching parameters. Using existing dataset.")
        return dataset_name
    
    # Dataset doesn't exist or needs regeneration
    print(f"\n{'='*80}")
    print(f"Generating dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Create InWeightsPathStar generator
    generator = InWeightsPathStar(
        d=d,
        l=l,
        vocab_size=vocab_size,
        holdout_percentage=holdout_percentage
    )
    
    # Generate and save dataset
    output_dir = generator.prepare(
        num_pause_tokens=num_pause_tokens,
        output_dir='./data',
        use_undirected=use_undirected,
        use_directional_tokens=use_directional_tokens
    )
    
    print(f"\n{'='*80}")
    print(f"Dataset generation complete: {output_dir}")
    print(f"{'='*80}\n")
    
    return dataset_name


def load_dataset(dataset):
    # Data loading setup
    data_dir = os.path.join('data', dataset)
    # Load data once as memory-mapped arrays
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # Load metadata once at initialization
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise ValueError(f"Metadata file not found at {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    return meta, train_data, val_data


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
        'log_interval': 10,
        'eval_only': False,
        'always_save_checkpoint': True,
        
        # wandb logging
        'wandb_log': True,
        'wandb_project': 'pathstar_small',
        'wandb_run_name': None,  # Will be auto-generated
        
        # Dataset generation parameters
        'graph_d': 250,
        'graph_l': 5,
        'graph_vocab_size': 2000,
        'graph_holdout_percentage': 0.2,
        'num_pause_tokens': 1,
        'use_undirected': True,
        'use_directional_tokens': True,
        
        # Training parameters
        'gradient_accumulation_steps': 1,
        
        # Model architecture
        'n_layer': 3,
        'n_head': 8,
        'n_embd': 96,
        'dropout': 0.0,
        'bias': False,
        
        # Optimization
        'learning_rate': 1e-3,
        'label_smoothing': 0.10,
        'epochs': 50000,
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
    }


def train(config=None):
    """
    Main training function that can be called standalone or by wandb sweep.
    
    Args:
        config: Optional dict of configuration overrides. If None, uses defaults and command-line args.
    """
    
    # Get default config
    default_config = get_default_config()
    
    # If config is provided (e.g., from wandb sweep), merge it with defaults
    if config is not None:
        default_config.update(config)
        
        # Set custom run name for sweep runs
        if wandb.run is not None:
            custom_name = (
                f"d{default_config['graph_d']}_"
                f"l{default_config['graph_l']}_"
                f"L{default_config['n_layer']}_"
                f"E{default_config['n_embd']}_"
                f"p{default_config['num_pause_tokens']}_"
                f"{default_config['epochs']}"
            )
            wandb.run.name = custom_name
            print(f"Set sweep run name: {custom_name}")
    
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
    
    # Validate vocab_size
    if default_config['graph_vocab_size'].endswith('max'):
        factor = int(default_config['graph_vocab_size'][:-3])
        default_config['graph_vocab_size'] = factor * ((default_config['graph_l'] - 1) * default_config['graph_d'] + 1)

    assert default_config['graph_vocab_size'] >= default_config['graph_d'] * (default_config['graph_l'] - 1) + 1, \
        f"graph_vocab_size must be >= graph_d * (graph_l - 1) + 1"
    
    # Generate/load dataset
    dataset = generate_dataset_if_needed(
        d=default_config['graph_d'],
        l=default_config['graph_l'],
        vocab_size=default_config['graph_vocab_size'],
        holdout_percentage=default_config['graph_holdout_percentage'],
        num_pause_tokens=default_config['num_pause_tokens'],
        use_undirected=default_config['use_undirected'],
        use_directional_tokens=default_config['use_directional_tokens']
    )
    
    meta, train_data, val_data = load_dataset(dataset)
    
    # Extract graph parameters from metadata
    graph_length = meta['l']
    graph_spokes = meta['d']
    holdout_ratio = meta['holdout_percentage']
    bidirectional = meta['use_undirected']
    pause_length = meta['num_pause_tokens']
    use_directional_tokens_actual = meta['use_directional_tokens']
    
    # Calculate dataset-dependent parameters
    total_edge_size = (2 if bidirectional else 1) * (graph_length - 1) * graph_spokes 
    num_holdout = round(graph_spokes * holdout_ratio)
    total_train_paths = (graph_spokes - num_holdout)
    
    # Account for path replication to balance classes
    replication_factor = total_edge_size // total_train_paths if total_train_paths > 0 else 1
    if replication_factor < 1:
        replication_factor = 1
    replicated_train_paths = total_train_paths * replication_factor
    
    print(f"Class balancing: {total_train_paths} training paths replicated by factor {replication_factor} → {replicated_train_paths} replicated paths")
    print(f"Training dataset composition: {replicated_train_paths} replicated paths + {total_edge_size} edges = {replicated_train_paths + total_edge_size} total samples")
    
    max_allowed_batch_size = total_edge_size + replicated_train_paths
    batch_size = max(min(max_allowed_batch_size, 512), min(total_edge_size, 512))
    effective_batch_size = default_config['gradient_accumulation_steps'] * batch_size
    block_size = graph_length + 2 + pause_length
    
    assert effective_batch_size <= max_allowed_batch_size, (
        f"Effective batch size ({effective_batch_size} = {default_config['gradient_accumulation_steps']} * {batch_size}) "
        f"exceeds total training dataset size ({max_allowed_batch_size} = {total_edge_size} + {replicated_train_paths}). "
        f"Reduce batch_size or gradient_accumulation_steps."
    )
    
    # Calculate training iteration parameters
    TRAIN_DATASET_SIZE = total_edge_size + replicated_train_paths 
    VAL_DATASET_SIZE = num_holdout
    batch_per_dataset = int(np.ceil(TRAIN_DATASET_SIZE / (batch_size * default_config['gradient_accumulation_steps'])))
    eval_iters = int(np.ceil(TRAIN_DATASET_SIZE / batch_size))
    max_iters = default_config['epochs'] * batch_per_dataset
    
    # Calculate learning rate schedule parameters
    warmup_iters = int(max_iters * default_config['warmup_frac'])
    lr_decay_iters = int(max_iters * default_config['lr_decay_frac'])
    
    # Generate wandb run name if not provided
    if default_config['wandb_run_name'] is None:
        default_config['wandb_run_name'] = f"{dataset}_L{default_config['n_layer']}H{default_config['n_head']}E{default_config['n_embd']}_lr{default_config['learning_rate']}_bs{batch_size}_ga{default_config['gradient_accumulation_steps']}_drop{default_config['dropout']}_ls{default_config['label_smoothing']}_{time.time()}"
    
    # Print configuration
    tokens_per_iter = default_config['gradient_accumulation_steps'] * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    os.makedirs(default_config['out_dir'], exist_ok=True)
    
    # Determine GPU ID for checkpoint naming
    gpu_id = default_config['gpu_id']
    if gpu_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible is not None:
            gpu_id = cuda_visible.split(',')[0]
        elif torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
        else:
            gpu_id = 'cpu'
    
    # Determine experiment name for checkpoint naming
    experiment_name = default_config['experiment_name']
    if experiment_name is None:
        experiment_name = default_config['wandb_run_name']
    
    checkpoint_filename = f'ckpt_exp_{experiment_name}_pl{pause_length}_bi{bidirectional}_gpu_{gpu_id}.pt'
    print(f"Checkpoint will be saved as: {checkpoint_filename}")
    
    torch.manual_seed(1337)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    # Auto-detect device
    if default_config['device'] == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = default_config['device']
    
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
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Auto-detect dtype
    if default_config['dtype'] == 'auto':
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    else:
        dtype = default_config['dtype']
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Init tracking variables
    iter_num = 0
    best_val_loss = 1e9
    
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}")
    
    # Load special token IDs
    edge_token = None
    path_token = None
    root_vertex = None
    
    if 'task_tokens' in meta:
        edge_token = meta['task_tokens']['EDGE']
        path_token = meta['task_tokens']['PATH']
    
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
    
    stoi = meta.get('stoi', {})
    itos = meta.get('itos', {})
    root_vertex = meta.get('root_vertex')
    if itos:
        print(f"Loaded vocabulary mappings: {len(itos)} tokens")
    
    # Calculate dataset structure from metadata
    train_size = meta.get('train_size', TRAIN_DATASET_SIZE)
    val_size = meta.get('val_size', VAL_DATASET_SIZE)
    train_seq_length = len(train_data) // train_size
    val_seq_length = len(val_data) // val_size
    
    print(f"Dataset info:")
    print(f"  Train: {train_size} sequences of length {train_seq_length}")
    print(f"  Val: {val_size} sequences of length {val_seq_length}")
    print(f"  Block size: {block_size}")
    
    assert train_size == TRAIN_DATASET_SIZE, f"Train size mismatch: {train_size} != {TRAIN_DATASET_SIZE}"
    assert val_size == VAL_DATASET_SIZE, f"Val size mismatch: {val_size} != {VAL_DATASET_SIZE}"
    
    # Reshape data for easier indexing
    train_data = train_data.reshape(train_size, train_seq_length)
    val_data = val_data.reshape(val_size, val_seq_length)
    
    # Initialize epoch indices for sampling without replacement
    train_epoch_indices = np.arange(train_size)
    val_epoch_indices = np.arange(val_size)
    train_batch_idx = 0
    val_batch_idx = 0
    
    def get_batch(split):
        """Sample a batch from the dataset"""
        nonlocal train_batch_idx, val_batch_idx, train_epoch_indices, val_epoch_indices
        
        # Select dataset and indices
        if split == 'train':
            data = train_data
            batch_idx = train_batch_idx
            epoch_indices = train_epoch_indices
            dataset_size = train_size
            seq_length = train_seq_length
        else:
            data = val_data
            batch_idx = val_batch_idx
            epoch_indices = val_epoch_indices
            dataset_size = val_size
            seq_length = val_seq_length
        
        # Check if we need to shuffle for new epoch
        if batch_idx == 0:
            np.random.shuffle(epoch_indices)
        
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_seq_indices = epoch_indices[start_idx:end_idx]
        
        actual_batch_size = len(batch_seq_indices)
        
        # Update batch index for next call
        if split == 'train':
            train_batch_idx = (batch_idx + 1) if end_idx < dataset_size else 0
        else:
            val_batch_idx = (batch_idx + 1) if end_idx < dataset_size else 0
        
        # Extract sequences
        sequences = data[batch_seq_indices]
        
        # Pad or truncate to block_size if needed
        if seq_length < block_size:
            pad_value = pad_token_id if pad_token_id is not None else 0
            padding = np.full((actual_batch_size, block_size - seq_length), pad_value, dtype=np.int64)
            sequences = np.concatenate([sequences.astype(np.int64), padding], axis=1)
        elif seq_length > block_size:
            print(f"WARNING: Sequence length ({seq_length}) exceeds block_size ({block_size}). Truncating sequences.")
            sequences = sequences[:, :block_size].astype(np.int64)
        else:
            sequences = sequences.astype(np.int64)
        
        # Create input (x) and target (y) by shifting
        x = torch.from_numpy(sequences[:, :-1])
        y = torch.from_numpy(sequences[:, 1:])
        
        # Mask PAD tokens in targets
        if pad_token_id is not None:
            y[y == pad_token_id] = -1
        
        # Move to device
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        
        return x, y
    
    def compute_per_token_loss(logits, input, targets, token_positions, debug=False, split=''):
        """Compute per-token loss for specified token positions"""
        per_token_losses = {}
        
        # Compute context length per input based on task token
        context_length_per_input = torch.where(
            input[:, 0] == edge_token,
            torch.tensor(3 if use_directional_tokens_actual else 2, device=input.device),
            torch.where(
                input[:, 0] == path_token,
                torch.tensor(2 + pause_length, device=input.device),
                torch.tensor(0, device=input.device)
            )
        ).unsqueeze(1)
        
        for token_pos in token_positions:
            y_idx = context_length_per_input + token_pos - 2
            y_idx = y_idx.squeeze(1)
            
            valid_idx_mask = y_idx < targets.size(1)
            
            if valid_idx_mask.any():
                batch_size_local = logits.size(0)
                logits_at_pos = torch.zeros(batch_size_local, logits.size(2), device=logits.device)
                targets_at_pos = torch.zeros(batch_size_local, dtype=targets.dtype, device=targets.device)
                
                valid_batch_indices = []
                
                for b in range(batch_size_local):
                    if valid_idx_mask[b]:
                        idx = y_idx[b].item()
                        logits_at_pos[b] = logits[b, idx, :]
                        targets_at_pos[b] = targets[b, idx]
                        valid_batch_indices.append(b)
                    else:
                        targets_at_pos[b] = -1
                
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
    
    def compute_per_token_accuracy_autoregressive(val_data_batch, num_samples, device_local):
        """Compute per-token accuracy using autoregressive generation"""
        sample_indices = np.random.choice(len(val_data_batch), size=min(num_samples, len(val_data_batch)), replace=False)
        
        context_length = 2 + pause_length
        
        contexts = []
        ground_truths = []
        
        for val_idx in sample_indices:
            full_sequence = val_data_batch[val_idx]
            context = full_sequence[:context_length]
            contexts.append(context)
            ground_truth = full_sequence[context_length:context_length + graph_length]
            ground_truths.append(ground_truth)
        
        contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device_local)
        
        with torch.no_grad():
            with ctx:
                generated_sequences = model.generate(contexts_batch, max_new_tokens=graph_length, temperature=1.0, top_k=1)
                generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
        
        per_token_accuracies = {}
        ground_truths_array = np.stack(ground_truths)
        
        for token_pos in range(1, graph_length + 1):
            idx = token_pos - 1
            if idx < generated_tokens_batch.shape[1] and idx < ground_truths_array.shape[1]:
                matches = generated_tokens_batch[:, idx] == ground_truths_array[:, idx]
                accuracy = np.mean(matches)
                per_token_accuracies[token_pos] = accuracy
            else:
                per_token_accuracies[token_pos] = float('nan')
        
        return per_token_accuracies
    
    def evaluate_samples(data, data_size, split_name, num_samples=5, eval_batch_size=256):
        """
        Evaluate autoregressive generation on samples from a dataset.
        
        Args:
            data: Dataset to sample from (train_data or val_data)
            data_size: Size of the dataset
            split_name: Name of the split for logging ('train' or 'val')
            num_samples: Number of samples to evaluate
            eval_batch_size: Batch size for generation to avoid OOM (default: 256)
        
        Returns:
            avg_accuracy: Average accuracy across all samples
        """
        num_samples = min(num_samples, data_size)
        
        # For train data, filter to only PATH tasks
        if split_name == 'train':
            # Find indices where first token is path_token
            path_indices = []
            for idx in range(data_size):
                if data[idx, 0] == path_token:
                    path_indices.append(idx)
            
            if len(path_indices) == 0:
                print(f"Warning: No PATH tasks found in {split_name} data")
                return 0.0
            
            # Sample from valid PATH indices without replacement
            num_samples = min(num_samples, len(path_indices))
            sample_indices = np.random.choice(path_indices, size=num_samples, replace=False)
        else:
            # For val data, sample randomly without replacement
            sample_indices = np.random.choice(data_size, size=num_samples, replace=False)
        
        # Prepare contexts and ground truths
        context_length = 2 + pause_length
        contexts = []
        ground_truths = []
        
        for idx in sample_indices:
            full_sequence = data[idx]
            context = full_sequence[:context_length]
            contexts.append(context)
            ground_truth = full_sequence[context_length:context_length + graph_length]
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
                    generated_sequences = model.generate(contexts_batch, max_new_tokens=graph_length, temperature=1.0, top_k=1)
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
    
    @torch.no_grad()
    def estimate_loss():
        """Estimate loss on train/val splits"""
        out = {}
        model.eval()
        iters_dict = {'train': eval_iters, 'val': int(np.ceil(num_holdout / batch_size))}
        
        train_token_losses = {1: []}
        val_token_losses = {i: [] for i in range(1, graph_length + 1)}
        
        for split in ['train', 'val']:
            losses = torch.zeros(iters_dict[split])
            for k in range(iters_dict[split]):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
                losses[k] = loss.item()
                
                if split == 'train':
                    batch_per_token = compute_per_token_loss(logits, X, Y, [1], debug=False, split='train')
                    if 1 in batch_per_token:
                        train_token_losses[1].append(batch_per_token[1])
                else:
                    batch_per_token = compute_per_token_loss(logits, X, Y, range(1, graph_length + 1), debug=False, split='val')
                    for token_pos, token_loss in batch_per_token.items():
                        if not math.isnan(token_loss):
                            val_token_losses[token_pos].append(token_loss)
            
            out[split] = losses.mean()
        
        if train_token_losses[1]:
            out['train_per_token'] = {1: np.mean(train_token_losses[1])}
        else:
            out['train_per_token'] = {1: float('nan')}
        
        if val_token_losses[1]:
            out['val_per_token'] = {
                token_pos: np.mean(losses_list) if losses_list else float('nan')
                for token_pos, losses_list in val_token_losses.items()
            }
        else:
            out['val_per_token'] = {token_pos: float('nan') for token_pos in range(1, graph_length + 1)}
        
        # Compute per-token accuracy
        num_samples_for_accuracy = val_size
        val_per_token_accuracy = compute_per_token_accuracy_autoregressive(val_data, num_samples_for_accuracy, device)
        out['val_per_token_accuracy'] = val_per_token_accuracy
        
        model.train()
        return out
    
    def get_lr(it):
        """Learning rate decay scheduler (cosine with warmup)"""
        if it < warmup_iters:
            return default_config['learning_rate'] * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return default_config['min_lr']
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return default_config['min_lr'] + coeff * (default_config['learning_rate'] - default_config['min_lr'])
    
    # Model initialization
    model_args = dict(
        n_layer=default_config['n_layer'],
        n_head=default_config['n_head'],
        n_embd=default_config['n_embd'],
        block_size=block_size,
        bias=default_config['bias'],
        vocab_size=None,
        dropout=default_config['dropout']
    )
    
    if default_config['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif default_config['init_from'] == 'resume':
        print(f"Resuming training from {default_config['out_dir']}")
        ckpt_path = os.path.join(default_config['out_dir'], checkpoint_filename)
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
    
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    
    model.to(device)
    
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
        unoptimized_model = model
        model = torch.compile(model)
    
    # Initialize wandb (skip if already initialized by sweep agent)
    if default_config['wandb_log'] and wandb.run is None:
        wandb.init(
            project=default_config['wandb_project'],
            name=default_config['wandb_run_name'],
            config=default_config
        )
    
    # Training loop
    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # Set learning rate
        lr = get_lr(iter_num) if default_config['decay_lr'] else default_config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % default_config['eval_interval'] == 0:
            losses = estimate_loss()
            
            samples_processed = iter_num * batch_size * default_config['gradient_accumulation_steps']
            current_epoch = samples_processed / TRAIN_DATASET_SIZE
            
            print(f"step {iter_num}: epoch {current_epoch:.2f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if 'train_per_token' in losses and 1 in losses['train_per_token']:
                print(f"  Train 1st token loss: {losses['train_per_token'][1]:.4f}")
            
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
            val_avg_accuracy = evaluate_samples(val_data, val_size, 'val', num_samples=val_size)
            train_avg_accuracy = evaluate_samples(train_data, train_size, 'train', num_samples=replicated_train_paths)
            
            if default_config['wandb_log']:
                log_dict = {
                    "iter": iter_num,
                    "train/loss/overall": losses['train'],
                    "val/loss/overall": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "gen/val_avg_accuracy": val_avg_accuracy,
                    "gen/train_avg_accuracy": train_avg_accuracy
                }
                
                if 'train_per_token' in losses and 1 in losses['train_per_token']:
                    log_dict["train/loss/token_1"] = losses['train_per_token'][1]
                
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
            
            if losses['val'] < best_val_loss or default_config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
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
        
        # Forward backward update
        for micro_step in range(default_config['gradient_accumulation_steps']):
            with ctx:
                logits, loss = model(X, Y, label_smoothing=default_config['label_smoothing'])
                loss = loss / default_config['gradient_accumulation_steps']
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        
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
                mfu = model.estimate_mfu(batch_size * default_config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1
        
        if iter_num > max_iters:
            break
    
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

