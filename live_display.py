# Rich imports
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.table import Table

import torch
import numpy as np
from contextlib import nullcontext

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

class LiveTrainingPanel:
    """
    Manages the Rich Live display for training visualization.
    
    Encapsulates layout creation, updates, and provides convenient access to layout sections.
    """
    CONSOLE = Console()
    
    def __init__(self, config):
        """
        Initialize the LiveTrainingPanel with configuration.
        
        Args:
            config: Configuration dictionary containing display settings
        """
        # Store relevant config values
        self.use_live_display = config.get('live_display', True)
        self.show_training_slices = config.get('show_training_slices', False)
        self.show_debug_masking = config.get('debug_masking', False)
        self.vis_interval = config.get('vis_interval', 100)
        
        # Create layout if enabled
        if self.use_live_display:
            self._setup_layout(config)
        else:
            self.layout = None
            self.context = nullcontext()
    
    def _setup_layout(self, config):
        """
        Sets up the Rich Live display layout based on configuration.
        
        Args:
            config: Configuration dictionary containing display settings
        """
        # Initialize architecture panel with config info (static, doesn't change during training)
        arch_info = self.create_architecture_info_panel(config)
        
        self.layout = Layout()
        # Build layout based on enabled features
        layout_components = [
            Layout(name="architecture", size=3),  # Compact architecture info
            Layout(name="metrics", size=14),  # Fixed size for metrics table
            Layout(name="evaluation"),
        ]
        if self.show_training_slices:
            layout_components.append(Layout(name="training"))
        if self.show_debug_masking:
            layout_components.append(Layout(name="mask", size=10))
        
        self.layout.split_column(*layout_components)
        
        self.layout["architecture"].update(Panel(Align.center(arch_info), title="Architecture", border_style="cyan"))
        self.layout["metrics"].update(Panel("Waiting for first evaluation...", title="Validation Metrics", border_style="magenta"))
        self.layout["evaluation"].update(Panel("Waiting for first evaluation...", title="Evaluation Examples", border_style="blue"))
        if self.show_training_slices:
            self.layout["training"].update(Panel("Waiting for first training batch...", title="Training Slice (10 samples)", border_style="green"))
        if self.show_debug_masking:
            self.layout["mask"].update(Panel("Waiting for first mask debug...", title="Mask Debug", border_style="yellow"))
        
        if self.use_live_display:
            self.context = Live(self.layout, console=self.CONSOLE, refresh_per_second=0.1)
        else:
            self.context = nullcontext()
    
    def update_train(self, X, Y, iter_num, meta, last_mask_debug_str=None):
        """
        Update training slice and mask debug panels in the live display.
        
        Args:
            X: Input tensor (batch_size, seq_len)
            Y: Target tensor (batch_size, seq_len) with masking applied
            iter_num: Current iteration number
            meta: Metadata dictionary containing itos and other info
            last_mask_debug_str: Debug string for mask visualization (optional)
        """
        # Early return if live display is disabled or not time to update
        if not self.use_live_display or iter_num % self.vis_interval != 0:
            return
        
        # Early return if layout is not available
        if self.layout is None:
            return
        
        # Get itos from meta
        itos = meta.get('itos', {})
        
        # Update training slice panel if enabled
        if self.show_training_slices:
            # Reconstruct full sequence for visualization: X + last token of Y
            # Note: Y has masking (-1) applied, so if the last token is masked, it won't show, 
            # but for path tasks the last token (LEAF) is not masked.
            full_batch = torch.cat([X, Y[:, -1:]], dim=1)
            training_slice_str = self.format_training_slice(full_batch, itos, meta, num_samples=10)
            self.layout["training"].update(Panel(training_slice_str, title=f"Training Slice (Iter {iter_num})", border_style="green"))
        
        # Update mask debug panel if enabled
        if self.show_debug_masking and last_mask_debug_str is not None:
            self.layout["mask"].update(Panel(last_mask_debug_str, title=f"Mask Debug (Iter {iter_num})", border_style="yellow"))
    
    @property
    def evaluation(self):
        """Access the evaluation layout section."""
        return self.layout["evaluation"] if self.layout else None
    
    @property
    def metrics(self):
        """Access the metrics layout section."""
        return self.layout["metrics"] if self.layout else None
    
    def get_layout(self):
        """Get the full layout object for use with Rich Live display."""
        return self.layout
    
    def is_enabled(self):
        """Check if live display is enabled."""
        return self.use_live_display
    
    def create_architecture_info_panel(self, config):
        """Create a Rich Text display for architecture info"""
        # Extract architecture parameters
        n_layer = config.get('n_layer', '?')
        n_embd = config.get('n_embd', '?')
        n_head = config.get('n_head', '?')

        num_pause = config.get('num_pause_tokens', 0)
        activation = config.get('activation', 'GELU')
        use_ln = config.get('use_layernorm', True)
        use_bias = config.get('bias', False)
        use_mlp = config.get('use_mlp', True)
        dropout = config.get('dropout', 0.0)
        embd_dropout = config.get('embd_dropout', 0.0)
        
        # Format labels
        ln_label = "[green]LN[/green]" if use_ln else "[dim]no-LN[/dim]"
        bias_label = "[green]Bias[/green]" if use_bias else "[dim]no-Bias[/dim]"
        mlp_label = "[green]MLP[/green]" if use_mlp else "[dim]no-MLP[/dim]"

        seed_label = config.get('seed', '?')
        
        # Dropout display
        if dropout == embd_dropout:
            dropout_str = f"D={dropout}"
        else:
            dropout_str = f"D={dropout}/ED={embd_dropout}"
        
        info_str = (
            f"[bold cyan]Layers:[/bold cyan] {n_layer}  "
            f"[bold cyan]Embd:[/bold cyan] {n_embd}  "
            f"[bold cyan]Heads:[/bold cyan] {n_head}  "
            f"[bold cyan]Pause:[/bold cyan] {num_pause}  "
            f"[bold cyan]Act:[/bold cyan] {activation}  "
            f"{mlp_label}  {ln_label}  {bias_label}  "
            f"[dim]{dropout_str}[/dim] "
            f"[dim]seed={seed_label}[/dim]"
        )
        
        return Text.from_markup(info_str)


    def update_metrics_table(self, metrics, graph_length, iter_num, epoch, lr, meta, tokens_per_sec=None, batch_size=None, edge_memorization_pct=None, train_dataset_size=None, eval_dataset_size=None, embedding_geometry_results=None):
        """Create a Rich Table for per-token metrics (Train vs Val)"""

        if 'generated_text' in metrics and metrics['generated_text'] and self.evaluation is not None:
            self.evaluation.update(Panel(metrics['generated_text'], title="Evaluation Examples", border_style="blue"))

        if self.metrics is None:
            return None 


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

        metrics_table = Table(title=title, show_header=True, header_style="bold magenta")
        metrics_table.add_column("Pos", style="cyan", justify="center")
        metrics_table.add_column("Train Loss (TF)", style="red", justify="right")
        metrics_table.add_column("Val Loss (TF)", style="red", justify="right")
        metrics_table.add_column("Train Acc (Autoregressive)", style="green", justify="right")
        metrics_table.add_column("Val Acc (Autoregressive)", style="green", justify="right")
        
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
                metrics_table.add_row("…", "…", "…", "…", "…")
            t_loss = metrics.get('train_per_token', {}).get(i, float('nan'))
            v_loss = metrics.get('val_per_token', {}).get(i, float('nan'))
            t_acc = metrics.get('train_per_token_accuracy', {}).get(i, float('nan'))
            v_acc = metrics.get('val_per_token_accuracy', {}).get(i, float('nan'))
            
            metrics_table.add_row(
                str(i),
                f"{t_loss:.4f}", f"{v_loss:.4f}",
                f"{t_acc*100:.1f}%", f"{v_acc*100:.1f}%"
            )
            last_display = i
        
        emb_table = self.create_embedding_geometry_table(embedding_geometry_results, meta['l']) if embedding_geometry_results else None

        if emb_table:
            combined = Group(metrics_table, Text(""), emb_table)
        else:
            combined = metrics_table
        
        self.metrics.update(Panel(Align.center(combined), title="Validation Metrics", border_style="magenta"))
        
        
    def create_embedding_geometry_table(self, embedding_geometry, l):
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