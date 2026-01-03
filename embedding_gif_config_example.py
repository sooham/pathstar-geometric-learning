"""
Example configuration snippet for enabling embedding GIF generation during training.

Add these lines to your configurator.py file to enable automatic embedding visualization
and GIF creation during training.

For full documentation, see: EMBEDDING_GIF_TRAINING.md
"""

# ============================================================================
# EMBEDDING GIF CONFIGURATION
# ============================================================================

# Enable automatic embedding GIF generation
# When True: saves embedding plots during training and creates GIF at end/interrupt
output_embedding_gif = True

# How often to save embedding plots (in iterations)
# Recommendation: Use multiple of eval_interval for efficiency
# - Small graphs (d < 1000): 50-100
# - Medium graphs (d < 5000): 100-200  
# - Large graphs (d > 5000): 200-500
embedding_plot_interval = 100

# GIF animation settings
embedding_gif_duration = 500  # Milliseconds per frame (lower = faster animation)
                               # 200ms = 5 FPS, 500ms = 2 FPS, 1000ms = 1 FPS

embedding_gif_num_paths = 5   # Number of training paths to highlight in plots
                               # More paths = more colorful but busier visualization

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

# Example 1: Fast training, smooth GIF
# output_embedding_gif = True
# embedding_plot_interval = 50   # Save every 50 iters (many frames)
# embedding_gif_duration = 300    # Fast playback (3.3 FPS)
# embedding_gif_num_paths = 5

# Example 2: Long training, memory-efficient
# output_embedding_gif = True
# embedding_plot_interval = 500   # Save every 500 iters (fewer frames)
# embedding_gif_duration = 800    # Slower playback (1.25 FPS)
# embedding_gif_num_paths = 8

# Example 3: High-quality visualization
# output_embedding_gif = True
# embedding_plot_interval = 100
# embedding_gif_duration = 500
# embedding_gif_num_paths = 10    # Show more paths

# ============================================================================
# USAGE NOTES
# ============================================================================

# 1. The GIF will be saved to: out/embedding_evolution_RUNNAME.gif
# 2. Individual plots saved to: out/embedding_plots/
# 3. Press Ctrl+C once during training for graceful shutdown with GIF creation
# 4. Press Ctrl+C twice to force exit (no GIF creation)
# 5. Requires dataset with 'paths_by_leaf' metadata
# 6. Works in both normal and edge_only modes
# 7. GIF is automatically logged to wandb if wandb_log=True

# ============================================================================

