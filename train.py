"""
This training script runs on a single GPU.

To run:
$ python train.py --batch_size=32 --compile=False
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

# -----------------------------------------------------------------------------
# default config values designed to train a customized small GPT
# I/O
init_from = 'scratch' # 'scratch' or 'resume'
out_dir = 'out'
eval_interval = 50
log_interval = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'pathstar_small'
# data
###################################################
#dataset = 'inweights_pathstar_d500_l9_p3_undirected_dt'
dataset = 'inweights_pathstar_d5_l5_p1_undirected_dt'

meta, train_data, val_data = load_dataset(dataset)
gradient_accumulation_steps = 1 # used to simulate larger batch sizes (effective batch = 512 * 8 = 4096)
graph_length = meta['l']  # this can be determined by meta.pkl
graph_spokes = meta['d'] # this can be determined by meta.pkl
holdout_ratio = meta['holdout_percentage'] # this can be determined by meta.pkl
bidirectional = meta['use_undirected'] # this can be determined by meta.pkl
pause_length = meta['num_pause_tokens'] # this can be determined by meta.pkl
use_directional_tokens = meta['use_directional_tokens']
total_edge_size = (2 if bidirectional else 1) * (graph_length - 1) * graph_spokes 
num_holdout = round(graph_spokes * holdout_ratio)
total_train_paths =  (graph_spokes - num_holdout)
batch_size = 43  #850 # 8250 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = graph_length + 2 + pause_length
effective_batch_size = gradient_accumulation_steps * batch_size
max_allowed_batch_size = total_edge_size + total_train_paths
assert effective_batch_size <= max_allowed_batch_size, (
    f"Effective batch size ({effective_batch_size} = {gradient_accumulation_steps} * {batch_size}) "
    f"exceeds total training dataset size ({max_allowed_batch_size} = {total_edge_size} + {total_train_paths}). "
    f"Reduce batch_size or gradient_accumulation_steps."
)
###################################################
# model
n_layer = 12 # the default is 12
n_head = 8  # default is 8
n_embd =  384   # defulat is 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # max learning rate
epochs = 10000 # 20000
TRAIN_DATASET_SIZE = total_edge_size + total_train_paths 
VAL_DATASET_SIZE = num_holdout
batch_per_dataset = int(np.ceil(TRAIN_DATASET_SIZE / (batch_size * gradient_accumulation_steps)))
eval_iters = int(np.ceil(TRAIN_DATASET_SIZE / batch_size))  # eval for one epoch, no gradient accumulation
max_iters = epochs * batch_per_dataset # total number of training iterations
weight_decay = 0.01 # weight decay for AdamW optimizer
beta1 = 0.9  # AdamW optimizer beta1 parameter (exponential decay rate for first moment estimates)
beta2 = 0.95  # AdamW optimizer beta2 parameter (exponential decay rate for second moment estimates)
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
wandb_run_name = f"{dataset}_L{n_layer}H{n_head}E{n_embd}_lr{learning_rate}_bs{batch_size}_ga{gradient_accumulation_steps}_{time.time()}"

# learning rate decay settings
# Learning rate schedule with cosine decay and linear warmup:
# 1. Linear warmup: LR increases linearly from 0 to learning_rate over warmup_iters steps
# 2. Cosine decay: LR decays following a cosine curve from learning_rate to min_lr over (lr_decay_iters - warmup_iters) steps
# 3. Constant minimum: LR stays at min_lr for all iterations beyond lr_decay_iters
# Example: learning_rate=1e-3, min_lr=6e-5 means LR goes 0 -> 1e-3 (warmup) -> 6e-5 (decay) -> 6e-5 (constant)
decay_lr = True # whether to decay the learning rate (cosine decay with linear warmup)
warmup_iters = int(max_iters * 0.10) # how many steps to warm up for (linear increase from 0 to learning_rate)
lr_decay_iters = int(max_iters * 0.99) # total steps for warmup + decay phase; should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate after decay completes, should be ~= learning_rate/10 per Chinchilla
# system
# Auto-detect device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# Print device information with more details for multi-GPU setups
if device == 'cuda':
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Using device: {device} (GPU {current_device}/{num_gpus-1}: {device_name})")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print(f"Using device: {device}")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# GPU ID for parallel experiments (can be overridden via command line)
gpu_id = None  # If None, will use CUDA_VISIBLE_DEVICES or default to 0
# Experiment name for distinguishing parallel runs (can be overridden via command line)
experiment_name = None  # If None, will use wandb_run_name
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)

# Determine GPU ID for checkpoint naming
if gpu_id is None:
    # Try to get GPU ID from CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        # Use the first GPU in CUDA_VISIBLE_DEVICES
        gpu_id = cuda_visible.split(',')[0]
    elif torch.cuda.is_available():
        # Use current device
        gpu_id = torch.cuda.current_device()
    else:
        gpu_id = 'cpu'

# Determine experiment name for checkpoint naming
if experiment_name is None:
    experiment_name = wandb_run_name

# Create unique checkpoint filename based on experiment and GPU
checkpoint_filename = f'ckpt_exp_{experiment_name}_pl{pause_length}_bi{bidirectional}_gpu_{gpu_id}.pt'
print(f"Checkpoint will be saved as: {checkpoint_filename}")

torch.manual_seed(1337)
torch.backends.cudnn.conv.fp32_precision = 'tf32' # allow tf32 on cudnn convolutions
torch.backends.cuda.matmul.fp32_precision = 'tf32' # allow tf32 on matmul
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

meta_vocab_size = meta['vocab_size']
print(f"found vocab_size = {meta_vocab_size}")

# Load special token IDs for masking in loss calculation
edge_token = None
path_token = None
root_vertex = None

if 'task_tokens' in meta:
    edge_token = meta['task_tokens']['EDGE']
    path_token = meta['task_tokens']['PATH']

if 'special_tokens' in meta:
    # InContextPathStar format
    pause_token_id = meta['special_tokens'].get('PAUSE')
    pad_token_id = meta['special_tokens'].get('PAD')
else:
    # InWeightsPathStar format
    pause_token_id = meta.get('pause_token')
    pad_token_id = meta.get('pad_token')

if pause_token_id is not None or pad_token_id is not None:
    print(f"Loaded special tokens: PAUSE={pause_token_id}, PAD={pad_token_id}")
    print("Note: PAD tokens will be masked in loss calculation (ignore_index=-1)")
else:
    print("Warning: No special tokens found in metadata. PAD masking will be disabled.")

# Load stoi and itos for token visualization
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

# Verify dimensions
assert train_size == TRAIN_DATASET_SIZE, f"Train size mismatch: {train_size} != {TRAIN_DATASET_SIZE}"
assert val_size == VAL_DATASET_SIZE, f"Val size mismatch: {val_size} != {VAL_DATASET_SIZE}"

# Reshape data for easier indexing: [num_sequences, seq_length]
train_data = train_data.reshape(train_size, train_seq_length)
val_data = val_data.reshape(val_size, val_seq_length)

# Initialize epoch indices for sampling without replacement
train_epoch_indices = np.arange(train_size)
val_epoch_indices = np.arange(val_size)
train_batch_idx = 0
val_batch_idx = 0

def get_batch(split):
    """
    Simple, performant batch sampler that:
    1. Samples without replacement within each epoch
    2. Loads data efficiently from pre-loaded memmap
    3. Handles sequence padding/truncation cleanly
    """
    global train_batch_idx, val_batch_idx, train_epoch_indices, val_epoch_indices
    
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
    
    # Get batch indices (without replacement)
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, dataset_size)
    batch_seq_indices = epoch_indices[start_idx:end_idx]
    
    # Handle last batch being smaller
    actual_batch_size = len(batch_seq_indices)
    
    # Update batch index for next call
    if split == 'train':
        train_batch_idx = (batch_idx + 1) if end_idx < dataset_size else 0
    else:
        val_batch_idx = (batch_idx + 1) if end_idx < dataset_size else 0
    
    # Extract sequences: [actual_batch_size, seq_length]
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
    x = torch.from_numpy(sequences[:, :-1])  # All but last token
    y = torch.from_numpy(sequences[:, 1:])   # All but first token
    
    # Mask PAD tokens in targets (ignored in loss)
    # Note: PAUSE tokens are NOT masked - the model should learn to predict them
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

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, checkpoint_filename)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

def compute_per_token_loss(logits, input,  targets, token_positions, debug=False, split=''):
    """
    Compute per-token loss for specified token positions.
    
    Args:
        logits: Model logits of shape (batch_size, seq_length, vocab_size)
        input: Input tokens of shape (batch_size, seq_length) that will be used to predict the targets
        targets: Target tokens of shape (batch_size, seq_length)
        token_positions: List or range of token positions (1-indexed) to compute loss for based on context
        debug: If True, print debug information about misclassifications
        split: Dataset split name ('train' or 'val') for debug messages

    The context is automatically determined by looking at the TASK token used : either PATH or EDGE
    
    Returns:
        Dictionary mapping token_pos -> loss value for that position
    """
    per_token_losses = {}
    # Compute context length per input based on task token
    # Shape: (batch_size,) where each element is either 2 (EDGE) or (2 + pause_length) (PATH)
    # 3 for EDGE depending on if directional tokens are used
    context_length_per_input = torch.where(
        input[:, 0] == edge_token,
        torch.tensor(3 if use_directional_tokens else 2, device=input.device), # (EDGE , u)
        torch.where(
            input[:, 0] == path_token,
            torch.tensor(2 + pause_length, device=input.device),
            torch.tensor(0, device=input.device)  # fallback for unexpected tokens
        )
    ).unsqueeze(1)  # Shape: (batch_size, 1)
    
    for token_pos in token_positions:
        # token_pos=1 corresponds to the first token after context
        # In the shifted targets Y, this is at index: context_length + token_pos - 2
        # Example: context_length=3, token_pos=1 → y_idx=2 (ROOT_Y in Y)
        # y_idx is now a tensor of shape (batch_size, 1) since context_length_per_input is (batch_size, 1)
        y_idx = context_length_per_input + token_pos - 2  # Shape: (batch_size, 1)
        y_idx = y_idx.squeeze(1)  # Shape: (batch_size,)
        
        # Check which samples have valid indices (within bounds)
        valid_idx_mask = y_idx < targets.size(1)
        
        if valid_idx_mask.any():
            # For each sample in the batch, gather the logits and targets at the appropriate position
            batch_size = logits.size(0)
            logits_at_pos = torch.zeros(batch_size, logits.size(2), device=logits.device)
            targets_at_pos = torch.zeros(batch_size, dtype=targets.dtype, device=targets.device)
            
            # Track which batch indices are valid for debug purposes
            valid_batch_indices = []
            
            for b in range(batch_size):
                if valid_idx_mask[b]:
                    idx = y_idx[b].item()
                    logits_at_pos[b] = logits[b, idx, :]
                    targets_at_pos[b] = targets[b, idx]
                    valid_batch_indices.append(b)
                else:
                    print("Warning: masking happening in compute per token loss")
                    targets_at_pos[b] = -1  # Mark as invalid
            
            # Filter out masked tokens (ignore_index=-1)
            valid_mask = targets_at_pos != -1
            if valid_mask.any():
                logits_valid = logits_at_pos[valid_mask]
                targets_valid = targets_at_pos[valid_mask]
                
                # Debug: check predictions vs ground truth
                if debug and split == 'train':
                    predicted_classes = torch.argmax(logits_valid, dim=1)
                    mismatches = predicted_classes != targets_valid
                    num_mismatches = mismatches.sum().item()
                    total_samples = len(targets_valid)
                    accuracy = (total_samples - num_mismatches) / total_samples if total_samples > 0 else 0.0
                    
                    # Compute probabilities for detailed analysis
                    probs = F.softmax(logits_valid, dim=1)
                    
                    print(f"\n[DEBUG {split.upper()}] Token position {token_pos}:")
                    print(f"  Accuracy: {accuracy*100:.2f}% ({total_samples - num_mismatches}/{total_samples})")
                    print(f"  Mismatches: {num_mismatches}")
                    
                    # Count mismatches where the context node is the root node
                    if root_vertex is not None and num_mismatches > 0:
                        root_mismatches = 0
                        mismatch_indices = torch.where(mismatches)[0]
                        for idx in mismatch_indices:
                            original_batch_idx = valid_batch_indices[idx.item()]
                            # Get the node from context (second token in context for EDGE tasks)
                            context_node = input[original_batch_idx, 1].item()
                            if context_node == root_vertex:
                                root_mismatches += 1
                        print(f"  Root node mismatches: {root_mismatches}/{num_mismatches} ({root_mismatches/num_mismatches*100:.1f}%)")
                    
                    # Show first few mismatches
                    if num_mismatches > 0:
                        mismatch_indices = torch.where(mismatches)[0][:10]  # Show up to 10 mismatches
                        print(f"  First {min(num_mismatches, 10)} misclassifications:")
                        for i, idx in enumerate(mismatch_indices):
                            pred = predicted_classes[idx].item()
                            truth = targets_valid[idx].item()
                            
                            # Get probabilities
                            pred_prob = probs[idx, pred].item()  # Probability of predicted class
                            truth_prob = probs[idx, truth].item()  # Probability of correct class
                            individual_loss = -np.log(max(truth_prob, 1e-10))  # Avoid log(0)
                            
                            # Convert to token strings if available
                            pred_str = itos.get(pred, f"<{pred}>") if itos else str(pred)
                            truth_str = itos.get(truth, f"<{truth}>") if itos else str(truth)
                            
                            # Get the original batch index to access the input context
                            original_batch_idx = valid_batch_indices[idx.item()]
                            input_context = input[original_batch_idx, :context_length_per_input[original_batch_idx]].cpu().numpy()
                            # Convert input context to readable strings
                            context_str = ' '.join([itos.get(int(tok), f"<{int(tok)}>") if itos else str(int(tok)) for tok in input_context])
                            
                            print(f"    [{i+1}] Input Context: {context_str}")
                            print(f"        Predicted:\t\t{pred_str} (ID={pred}, prob={pred_prob:.4f})")
                            print(f"        Ground Truth:\t\t{truth_str} (ID={truth}, prob={truth_prob:.6f})")
                            print(f"        Individual Loss: {individual_loss:.4f}")
                # Compute cross-entropy loss for this token position
                token_loss = F.cross_entropy(logits_valid, targets_valid, reduction='mean')
                per_token_losses[token_pos] = token_loss.item()
            else:
                per_token_losses[token_pos] = float('nan')
        else:
            per_token_losses[token_pos] = float('nan')
    
    return per_token_losses


def compute_per_token_accuracy_autoregressive(val_data_batch, num_samples, device):
    """
    Compute per-token accuracy using autoregressive generation (no teacher forcing).
    
    Args:
        val_data_batch: Validation data array of shape (val_size, seq_length)
        num_samples: Number of validation samples to evaluate
        device: Device to run computations on
    
    Returns:
        Dictionary mapping token_pos -> accuracy for that position (1-indexed)
    """
    # Sample random validation indices
    sample_indices = np.random.choice(len(val_data_batch), size=min(num_samples, len(val_data_batch)), replace=False)
    
    # Context for validation: <PATH> LEAF_NODE <PAUSE> <PAUSE> ... <PAUSE>
    context_length = 2 + pause_length
    
    # Prepare batched contexts for parallel generation
    contexts = []
    ground_truths = []
    
    for val_idx in sample_indices:
        full_sequence = val_data_batch[val_idx]
        context = full_sequence[:context_length]
        contexts.append(context)
        ground_truth = full_sequence[context_length:context_length + graph_length]
        ground_truths.append(ground_truth)
    
    # Stack contexts into a batch: [num_samples, context_length]
    contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device)
    
    # Generate for all samples in parallel using greedy decoding
    with torch.no_grad():
        with ctx:
            # Generate graph_length tokens using greedy decoding (temperature=1.0, top_k=1)
            generated_sequences = model.generate(contexts_batch, max_new_tokens=graph_length, temperature=1.0, top_k=1)
            # Extract only the newly generated tokens (exclude context): [num_samples, graph_length]
            generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
    
    # Compute per-token accuracy
    per_token_accuracies = {}
    ground_truths_array = np.stack(ground_truths)  # [num_samples, graph_length]
    
    for token_pos in range(1, graph_length + 1):
        # token_pos is 1-indexed, array index is 0-indexed
        idx = token_pos - 1
        if idx < generated_tokens_batch.shape[1] and idx < ground_truths_array.shape[1]:
            # Compare predicted token at position i with ground truth at position i
            matches = generated_tokens_batch[:, idx] == ground_truths_array[:, idx]
            accuracy = np.mean(matches)
            per_token_accuracies[token_pos] = accuracy
        else:
            per_token_accuracies[token_pos] = float('nan')
    
    return per_token_accuracies


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    # no graident accumulation
    out = {}
    model.eval()
    iters = {'train': eval_iters, 'val': int(np.ceil(num_holdout / batch_size)) }
    
    # Context length for all datasets
    # val_context_length = 2 + pause_length
    # train_context_length = 2 # issue: this is incorrect
    
    # Track per-token losses:
    # - Training: only 1st token loss
    # - Validation: all token positions from 1 to graph_length
    train_token_losses = {1: []}
    val_token_losses = {i: [] for i in range(1, graph_length + 1)}
    
    for split in ['train', 'val']:
        losses = torch.zeros(iters[split])
        for k in range(iters[split]):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            
            # Compute per-token losses based on split
            if split == 'train':
                # Only compute 1st token loss for training
                # Enable debug for the first batch only to avoid flooding the output
                batch_per_token = compute_per_token_loss(logits, X, Y, [1], debug=False, split='train')
                if 1 in batch_per_token:
                    train_token_losses[1].append(batch_per_token[1])
            else:  # val
                # Compute all token positions for validation
                batch_per_token = compute_per_token_loss(logits, X, Y, range(1, graph_length + 1), debug=False, split='val')
                for token_pos, token_loss in batch_per_token.items():
                    if not math.isnan(token_loss):
                        val_token_losses[token_pos].append(token_loss)
        
        out[split] = losses.mean()
    
    # Average per-token losses across all batches
    # Training: 1st token loss
    if train_token_losses[1]:
        out['train_per_token'] = {1: np.mean(train_token_losses[1])}
    else:
        out['train_per_token'] = {1: float('nan')}
    
    # Validation: all token positions
    if val_token_losses[1]:
        out['val_per_token'] = {
            token_pos: np.mean(losses_list) if losses_list else float('nan')
            for token_pos, losses_list in val_token_losses.items()
        }
    else:
        out['val_per_token'] = {token_pos: float('nan') for token_pos in range(1, graph_length + 1)}
    
    # Compute per-token accuracy using autoregressive generation (no teacher forcing)
    # Use all validation samples for comprehensive evaluation
    num_samples_for_accuracy = val_size
    val_per_token_accuracy = compute_per_token_accuracy_autoregressive(val_data, num_samples_for_accuracy, device)
    out['val_per_token_accuracy'] = val_per_token_accuracy
    
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        
        # Calculate current epoch (accounting for gradient accumulation)
        # Each epoch processes TRAIN_DATASET_SIZE samples
        # Each iter processes batch_size * gradient_accumulation_steps samples
        samples_processed = iter_num * batch_size * gradient_accumulation_steps
        current_epoch = samples_processed / TRAIN_DATASET_SIZE
        
        print(f"step {iter_num}: epoch {current_epoch:.2f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Print 1st token loss for training
        if 'train_per_token' in losses and 1 in losses['train_per_token']:
            print(f"  Train 1st token loss: {losses['train_per_token'][1]:.4f}")
        
        # Print per-token validation losses
        if 'val_per_token' in losses:
            print("  Val per-token losses:")
            per_token_str = ", ".join([f"tok{i}: {losses['val_per_token'][i]:.4f}" 
                                       for i in range(1, min(graph_length + 1, 10))])
            print(f"    {per_token_str}")
            if graph_length > 9:
                per_token_str_rest = ", ".join([f"tok{i}: {losses['val_per_token'][i]:.4f}" 
                                                for i in range(10, graph_length + 1)])
                print(f"    {per_token_str_rest}")
        
        # Print per-token validation accuracies (autoregressive, no teacher forcing)
        if 'val_per_token_accuracy' in losses:
            print("  Val per-token accuracies (autoregressive):")
            per_token_acc_str = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'][i]*100:.1f}%" 
                                          for i in range(1, min(graph_length + 1, 10))])
            print(f"    {per_token_acc_str}")
            if graph_length > 9:
                per_token_acc_str_rest = ", ".join([f"tok{i}: {losses['val_per_token_accuracy'][i]*100:.1f}%" 
                                                    for i in range(10, graph_length + 1)])
                print(f"    {per_token_acc_str_rest}")
        
        # Autoregressive generation on validation samples (parallelized)
        model.eval()
        def evaluate_samples(data, data_size, split_name, num_samples=5):
            """
            Evaluate autoregressive generation on samples from a dataset.
            
            Args:
                data: Dataset to sample from (train_data or val_data)
                data_size: Size of the dataset
                split_name: Name of the split for logging ('train' or 'val')
                num_samples: Number of samples to evaluate
            
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
            
            # Helper function to convert token IDs to readable strings
            def tokens_to_str(token_ids):
                if itos:
                    return [itos.get(int(tid), f"<{tid}>") for tid in token_ids]
                else:
                    return token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
            
            # Prepare batched contexts for parallel generation
            context_length = 2 + pause_length
            contexts = []
            ground_truths = []
            full_sequences = []
            
            for idx in sample_indices:
                full_sequence = data[idx]
                full_sequences.append(full_sequence)
                context = full_sequence[:context_length]
                contexts.append(context)
                ground_truth = full_sequence[context_length:context_length + graph_length]
                ground_truths.append(ground_truth)
            
            # Stack contexts into a batch: [num_samples, context_length]
            contexts_batch = torch.from_numpy(np.stack(contexts).astype(np.int64)).to(device)
            
            # Generate for all samples in parallel using greedy decoding
            with torch.no_grad():
                with ctx:
                    # Generate graph_length tokens using greedy decoding (temperature=1.0, top_k=1)
                    generated_sequences = model.generate(contexts_batch, max_new_tokens=graph_length, temperature=1.0, top_k=1)
                    # Extract only the newly generated tokens (exclude context): [num_samples, graph_length]
                    generated_tokens_batch = generated_sequences[:, context_length:].cpu().numpy()
            
            # Calculate accuracies and prepare wandb data
            print(f"\nAutoregressive generation on {num_samples} {split_name} samples:")
            accuracies = []
            for sample_idx, (idx, full_sequence, ground_truth, generated_tokens) in enumerate(
                zip(sample_indices, full_sequences, ground_truths, generated_tokens_batch)
            ):
                # Calculate accuracy
                accuracy = np.mean(generated_tokens == ground_truth[:len(generated_tokens)])
                accuracies.append(accuracy)
                
                # Prepare strings for display and logging
                context_str = tokens_to_str(full_sequence[:context_length])
                ground_truth_str = tokens_to_str(ground_truth)
                generated_str = tokens_to_str(generated_tokens)
                
                print(f"  Sample {sample_idx+1} (idx={idx}):")
                print(f"    Context:      {context_str}")
                print(f"    Ground truth: {ground_truth_str}")
                print(f"    Generated:    {generated_str}")
                print(f"    Accuracy: {accuracy*100:.1f}%")
            
            # Calculate average accuracy
            avg_accuracy = np.mean(accuracies)
            print(f"  Average accuracy: {avg_accuracy*100:.1f}%")
            print()  # Empty line for readability
            
            return avg_accuracy
        
        # Evaluate on validation samples
        val_avg_accuracy = evaluate_samples(val_data, val_size, 'val', num_samples=5)
        
        # Evaluate on training samples (PATH tasks only)
        train_avg_accuracy = evaluate_samples(train_data, train_size, 'train', num_samples=5)
        
        model.train()
        if wandb_log:
            # Merge all generation data into a single dict
            log_dict = {
                "iter": iter_num,
                "train/loss/overall": losses['train'],
                "val/loss/overall": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "gen/avg_accuracy": avg_accuracy,
            }
            
            # Add training 1st token loss to wandb
            if 'train_per_token' in losses and 1 in losses['train_per_token']:
                log_dict["train/loss/token_1"] = losses['train_per_token'][1]
            
            # Add per-token validation losses to wandb
            if 'val_per_token' in losses:
                for token_pos in range(1, graph_length + 1):
                    if token_pos == graph_length:
                        # Use "token_final" for the last token position
                        log_dict["val/loss/token_final"] = losses['val_per_token'][token_pos]
                    else:
                        log_dict[f"val/loss/token_{token_pos}"] = losses['val_per_token'][token_pos]
            
            # Add per-token validation accuracies (autoregressive) to wandb
            if 'val_per_token_accuracy' in losses:
                for token_pos in range(1, graph_length + 1):
                    if token_pos == graph_length:
                        # Use "token_final" for the last token position
                        log_dict["val/accuracy/token_final"] = losses['val_per_token_accuracy'][token_pos]
                    else:
                        log_dict[f"val/accuracy/token_{token_pos}"] = losses['val_per_token_accuracy'][token_pos]
            
            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}/{checkpoint_filename}")
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_filename))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break