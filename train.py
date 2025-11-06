"""
This training script runs on a single GPU.

To run:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a customized small GPT
# I/O
out_dir = 'out'
eval_interval = 50
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'pathstar'
# data
###################################################
dataset = 'inweights_pathstar_d500_l9'
wandb_run_name = f"{dataset}_{time.time()}"
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
vocab_size = 8000
meta_vocab_size = vocab_size + 11
graph_length = 9 
graph_spokes = 500
holdout_ratio = 0.5
pause_length = 1
batch_size = 256 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = graph_length + 2 + pause_length
###################################################
# model
n_layer = 1
n_head = 8 
n_embd =  96  
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # max learning rate
epochs = 500
DATASET_SIZE = 2 * (graph_length - 1) * graph_spokes + holdout_ratio *  graph_spokes
batch_per_dataset = DATASET_SIZE // batch_size + 1
max_iters = epochs * batch_per_dataset # total number of training iterations
weight_decay = 0.01 # weight decay for AdamW optimizer
beta1 = 0.9  # AdamW optimizer beta1 parameter (exponential decay rate for first moment estimates)
beta2 = 0.95  # AdamW optimizer beta2 parameter (exponential decay rate for second moment estimates)
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
# Learning rate schedule with cosine decay and linear warmup:
# 1. Linear warmup: LR increases linearly from 0 to learning_rate over warmup_iters steps
# 2. Cosine decay: LR decays following a cosine curve from learning_rate to min_lr over (lr_decay_iters - warmup_iters) steps
# 3. Constant minimum: LR stays at min_lr for all iterations beyond lr_decay_iters
# Example: learning_rate=1e-3, min_lr=6e-5 means LR goes 0 -> 1e-3 (warmup) -> 6e-5 (decay) -> 6e-5 (constant)
decay_lr = True # whether to decay the learning rate (cosine decay with linear warmup)
warmup_iters = int(max_iters * 0.334) # how many steps to warm up for (linear increase from 0 to learning_rate)
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
print(f"Using device: {device}")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset and load special tokens
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
pause_token_id = None
pad_token_id = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # Load special token IDs for masking in loss calculation
    # Support both formats: 'special_tokens' dict (InContextPathStar) and direct keys (InWeightsPathStar)
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
        print("Note: PAUSE and PAD tokens will be masked in loss calculation (ignore_index=-1)")
    else:
        print("Warning: No special tokens found in metadata. PAUSE/PAD masking will be disabled.")

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Determine sequence length from metadata
    # The data is stored as [num_sequences, seq_length] flattened to 1D
    # We need to sample complete sequences, not arbitrary windows
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Calculate sequence length based on dataset structure
        # For InWeightsPathStar: seq_len = 1 (task_token) + 1 (leaf/x) + num_pause_tokens + l (path length) or 1 (y for edges)
        # We can infer it from the total data size and known number of sequences
        if split == 'train':
            total_sequences = meta.get('train_size', len(data) // block_size)
        else:
            total_sequences = meta.get('val_size', len(data) // block_size)
        
        seq_length = len(data) // total_sequences
    else:
        # Fallback: assume sequences are block_size length
        seq_length = block_size
        total_sequences = len(data) // seq_length
    
    # Sample random complete sequences
    # Each sequence starts at index: seq_idx * seq_length
    seq_indices = torch.randint(0, total_sequences, (batch_size,))
    
    # Extract sequences and pad/truncate to block_size
    sequences = []
    for seq_idx in seq_indices:
        start_idx = seq_idx * seq_length
        end_idx = start_idx + seq_length
        seq = data[start_idx:end_idx].astype(np.int64)
        
        # Pad or truncate to block_size
        if len(seq) < block_size:
            # Pad with pad_token_id if available, otherwise use 0
            pad_value = pad_token_id if pad_token_id is not None else 0
            seq = np.pad(seq, (0, block_size - len(seq)), constant_values=pad_value)
        elif len(seq) > block_size:
            seq = seq[:block_size]
        
        sequences.append(seq) 
    
    # Stack into batch
    x_batch = np.stack(sequences, axis=0)
    
    # Create targets: shift by 1 position (next token prediction)
    # x: [tok0, tok1, tok2, ..., tok_{n-1}]
    # y: [tok1, tok2, tok3, ..., tok_n]
    x = torch.from_numpy(x_batch[:, :-1])  # All but last token
    y = torch.from_numpy(x_batch[:, 1:])   # All but first token
    
    # Pad x and y back to block_size if needed
    if x.shape[1] < block_size:
        pad_value = pad_token_id if pad_token_id is not None else 0
        x = torch.cat([x, torch.full((batch_size, block_size - x.shape[1]), pad_value, dtype=torch.long)], dim=1)
        y = torch.cat([y, torch.full((batch_size, block_size - y.shape[1]), -1, dtype=torch.long)], dim=1)
    
    # Mask out PAUSE and PAD tokens in targets
    # The model should not be trained to predict these tokens
    # They will be ignored in the loss calculation (ignore_index=-1 in cross_entropy)
    if pause_token_id is not None:
        y[y == pause_token_id] = -1
    if pad_token_id is not None:
        y[y == pad_token_id] = -1
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
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
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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
    import wandb
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
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
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