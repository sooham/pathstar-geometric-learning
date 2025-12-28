import torch
import subprocess
import os
import numpy as np

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

def get_git_commit_id():
    """Get the short git commit ID, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return 'unknown'

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

def compute_token_colors(paths_data, val_data, meta):
    """Compute ANSI color codes for tokens based on their depth and train/val split"""
    train_tokens = set(np.unique(paths_data))
    val_tokens = set(np.unique(val_data))
    
    # Extract metadata for coloring
    root_vertex = meta['root_vertex']
    special_tokens = set(meta['special_tokens'].values())
    
    # Build a mapping from each token to its distance from root
    token_to_depth = {}
    
    # Reshape data to get sequences (paths_data is a flat memmap, need to reshape)
    # Calculate sequence length from metadata
    # NOTE: Use block_size_base (stored length WITHOUT pause tokens) since we're working with raw stored data
    block_size_base = meta['block_size_base']
    seq_length = block_size_base + 1  # block_size is context + targets - 1, so full sequence is block_size + 1
    
    # Reshape paths_data and val_data into sequences
    paths_sequences = paths_data.reshape(-1, seq_length)
    val_sequences = val_data.reshape(-1, seq_length)
    
    # Process training paths to determine depth
    for path_seq in paths_sequences:
        # Skip PATH token and leaf, find the actual path
        path_tokens = [t for t in path_seq[2:] if t not in special_tokens]
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
        path_tokens = [t for t in path_seq[2:] if t not in special_tokens]
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