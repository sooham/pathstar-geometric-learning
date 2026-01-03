"""
Example script showing how to create GIFs of embeddings evolution during training.

There are two main approaches:

1. Create GIF from saved checkpoints (after training)
2. Save plots during training and create GIF afterwards
"""

from model import GPT, GPTConfig
from umap_utils import create_embedding_gif
import glob
import os
import torch
import matplotlib.pyplot as plt


# ============================================================================
# Approach 1: Create GIF from saved checkpoints (RECOMMENDED)
# ============================================================================
def create_gif_from_checkpoints():
    """
    Create a GIF by loading multiple checkpoint files.
    This is the easiest approach - just point to your saved checkpoints.
    """
    # Get all checkpoint files (adjust pattern to match your naming)
    checkpoint_dir = 'out'
    checkpoints = sorted(glob.glob(f'{checkpoint_dir}/ckpt_epoch_*.pt'))
    
    # Filter to every 10th epoch if you have many checkpoints
    # For example: epochs 0, 10, 20, 30, ...
    checkpoints = [ckpt for ckpt in checkpoints if 
                   any(f'epoch_{i}' in ckpt for i in range(0, 1000, 10))]
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return
    
    # Create the GIF
    GPT.create_embedding_gif_from_checkpoints(
        checkpoint_paths=checkpoints,
        output_path='embeddings_evolution.gif',
        duration=800,  # 800ms per frame
        loop=0,  # Infinite loop
        include_root=False,
        include_special=False,
        num_paths=5
    )


# ============================================================================
# Approach 2: During training - save plots every N epochs
# ============================================================================
def training_loop_with_embedding_plots():
    """
    Example of how to save embedding plots during training.
    Add this to your training loop in train.py
    """
    # Pseudocode for training loop
    
    # model = GPT(config, meta=meta)
    # embedding_plot_interval = 10  # Save plot every 10 epochs
    # embedding_save_dir = 'out/embedding_plots'
    # os.makedirs(embedding_save_dir, exist_ok=True)
    
    # for epoch in range(num_epochs):
    #     # ... training code ...
    #     
    #     # Save embedding plot every 10 epochs
    #     if epoch % embedding_plot_interval == 0:
    #         save_path = f'{embedding_save_dir}/epoch_{epoch:04d}.png'
    #         fig = model.plot_embeddings_umap(
    #             save_path=save_path,
    #             epoch=epoch,
    #             iteration=iter_num,
    #             num_paths=5
    #         )
    #         plt.close(fig)  # Clean up memory
    
    # After training, create GIF from saved plots
    # image_paths = sorted(glob.glob(f'{embedding_save_dir}/epoch_*.png'))
    # create_embedding_gif(
    #     image_paths=image_paths,
    #     output_path='embeddings_training.gif',
    #     duration=500
    # )
    
    pass


# ============================================================================
# Approach 3: Generate plots on-the-fly (without saving to disk first)
# ============================================================================
def create_gif_from_checkpoints_in_memory():
    """
    Create GIF by generating plots in memory (no intermediate PNG files).
    This is memory-intensive but cleaner if you don't need the individual plots.
    """
    import matplotlib.pyplot as plt
    
    checkpoint_dir = 'out'
    checkpoints = sorted(glob.glob(f'{checkpoint_dir}/ckpt_epoch_*.pt'))
    
    # Filter to every 10 epochs
    checkpoints = [ckpt for ckpt in checkpoints if 
                   any(f'epoch_{i}' in ckpt for i in range(0, 1000, 10))]
    
    print(f"Generating plots from {len(checkpoints)} checkpoints...")
    
    figures = []
    for i, ckpt_path in enumerate(checkpoints):
        print(f"  [{i+1}/{len(checkpoints)}] Processing {os.path.basename(ckpt_path)}")
        
        # Load checkpoint and extract info
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_args = checkpoint['model_args']
        meta = checkpoint.get('meta', {})
        
        # Create model
        config = GPTConfig(**model_args)
        model = GPT(config, meta=meta)
        
        # Load weights
        state_dict = checkpoint['model']
        # Remove potential _orig_mod. prefix
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Remove neighborhood tensors
        for k in ['neighborhood_tensor', 'neighborhood_sizes_tensor', 'inv_neighborhood_sizes_tensor']:
            state_dict.pop(k, None)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Generate plot
        epoch = checkpoint.get('epoch', None)
        iter_num = checkpoint.get('iter_num', None)
        
        fig = model.plot_embeddings_umap(
            save_path=None,  # Don't save to disk
            epoch=epoch,
            iteration=iter_num,
            num_paths=5
        )
        figures.append(fig)
    
    # Create GIF
    print("Creating GIF...")
    create_embedding_gif(
        figures=figures,
        output_path='embeddings_evolution.gif',
        duration=800,
        loop=0
    )
    
    # Clean up
    for fig in figures:
        plt.close(fig)
    
    print("Done!")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create embedding evolution GIF')
    parser.add_argument('--checkpoint-dir', type=str, default='out',
                       help='Directory containing checkpoint files')
    parser.add_argument('--pattern', type=str, default='ckpt_epoch_*.pt',
                       help='Pattern to match checkpoint files')
    parser.add_argument('--output', type=str, default='embeddings_evolution.gif',
                       help='Output GIF path')
    parser.add_argument('--duration', type=int, default=800,
                       help='Duration per frame in milliseconds')
    parser.add_argument('--epoch-interval', type=int, default=10,
                       help='Use every Nth epoch (default: 10)')
    parser.add_argument('--num-paths', type=int, default=5,
                       help='Number of paths to highlight')
    
    args = parser.parse_args()
    
    # Get checkpoints
    checkpoint_pattern = os.path.join(args.checkpoint_dir, args.pattern)
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    # Filter by interval
    if args.epoch_interval > 1:
        checkpoints = [ckpt for ckpt in checkpoints if 
                      any(f'epoch_{i}' in ckpt or f'epoch_{i:04d}' in ckpt 
                          for i in range(0, 10000, args.epoch_interval))]
    
    print(f"Found {len(checkpoints)} checkpoints matching pattern")
    
    if len(checkpoints) == 0:
        print(f"No checkpoints found matching: {checkpoint_pattern}")
        exit(1)
    
    # Create GIF using the static method
    GPT.create_embedding_gif_from_checkpoints(
        checkpoint_paths=checkpoints,
        output_path=args.output,
        duration=args.duration,
        loop=0,
        num_paths=args.num_paths
    )

