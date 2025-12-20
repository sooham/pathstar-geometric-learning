"""
Learning rate schedulers for training.

Supports:
- None: Constant learning rate
- CosineLR: Cosine annealing with warmup
- ReduceLROnPlateau: Reduce LR when validation loss plateaus
"""

import math


def get_cosine_lr(it, warmup_iters, lr_decay_iters, config):
    """
    Learning rate decay scheduler (cosine with warmup).
    
    Args:
        it: Current iteration number
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations for LR decay
        config: Configuration dictionary with 'learning_rate' and 'min_lr'
    
    Returns:
        Current learning rate
    """
    if it < warmup_iters:
        return config['learning_rate'] * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return config['min_lr']
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])


class ReduceLROnPlateauScheduler:
    """
    Learning rate scheduler that reduces LR when validation loss plateaus.
    Similar to torch.optim.lr_scheduler.ReduceLROnPlateau but stateless-friendly.
    """
    def __init__(self, config, warmup_iters, console=None):
        """
        Initialize the scheduler.
        
        Args:
            config: Configuration dictionary with LR scheduler parameters
            warmup_iters: Number of warmup iterations
            console: Rich console for printing (optional)
        """
        self.learning_rate = config['learning_rate']
        self.min_lr = config['min_lr']
        self.factor = config['plateau_factor']
        self.patience = config['plateau_patience']
        self.threshold = config['plateau_threshold']
        self.cooldown = config['plateau_cooldown']
        self.warmup_iters = warmup_iters
        self.console = console
        
        # State variables
        self.current_lr = self.learning_rate
        self.best_loss = float('inf')
        self.num_bad_evals = 0
        self.cooldown_counter = 0
        self.last_update_iter = -1
    
    def get_lr(self, it):
        """
        Get current learning rate.
        
        Args:
            it: Current iteration number
        
        Returns:
            Current learning rate
        """
        # During warmup, use linear warmup
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        
        # After warmup, return the current plateau-adjusted LR
        return self.current_lr
    
    def step(self, val_loss, iter_num):
        """
        Update the scheduler based on validation loss.
        Should be called after each evaluation.
        
        Args:
            val_loss: Current validation loss
            iter_num: Current iteration number
        
        Returns:
            True if LR was reduced, False otherwise
        """
        # Don't update during warmup
        if iter_num < self.warmup_iters:
            return False
        
        # Avoid duplicate updates for the same iteration
        if iter_num == self.last_update_iter:
            return False
        self.last_update_iter = iter_num
        
        # If in cooldown, decrement and return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Check if this is an improvement (relative threshold)
        is_improvement = val_loss < self.best_loss * (1 - self.threshold)
        
        if is_improvement:
            self.best_loss = val_loss
            self.num_bad_evals = 0
        else:
            self.num_bad_evals += 1
        
        # Check if we should reduce LR
        if self.num_bad_evals >= self.patience:
            old_lr = self.current_lr
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                self.current_lr = new_lr
                self.num_bad_evals = 0
                self.cooldown_counter = self.cooldown
                if self.console:
                    self.console.print(f"[yellow]ReduceLROnPlateau: reducing LR from {old_lr:.2e} to {new_lr:.2e}[/yellow]")
                else:
                    print(f"ReduceLROnPlateau: reducing LR from {old_lr:.2e} to {new_lr:.2e}")
                return True
        
        return False
    
    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_lr': self.current_lr,
            'best_loss': self.best_loss,
            'num_bad_evals': self.num_bad_evals,
            'cooldown_counter': self.cooldown_counter,
            'last_update_iter': self.last_update_iter,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_lr = state_dict['current_lr']
        self.best_loss = state_dict['best_loss']
        self.num_bad_evals = state_dict['num_bad_evals']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.last_update_iter = state_dict['last_update_iter']


def get_lr(it, warmup_iters, lr_decay_iters, config, lr_scheduler_obj=None):
    """
    Learning rate scheduler dispatcher.
    
    Args:
        it: Current iteration number
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations for LR decay (CosineLR only)
        config: Configuration dictionary
        lr_scheduler_obj: ReduceLROnPlateauScheduler object (only for ReduceLROnPlateau)
    
    Returns:
        Current learning rate
    """
    scheduler_type = config.get('lr_scheduler', None)
    
    # None or 'None' means no scheduling - use constant learning rate
    if scheduler_type is None or scheduler_type == 'None':
        return config['learning_rate']
    elif scheduler_type == 'CosineLR':
        return get_cosine_lr(it, warmup_iters, lr_decay_iters, config)
    elif scheduler_type == 'ReduceLROnPlateau':
        if lr_scheduler_obj is None:
            raise ValueError("lr_scheduler_obj must be provided for ReduceLROnPlateau")
        return lr_scheduler_obj.get_lr(it)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_type}. Must be None, 'CosineLR', or 'ReduceLROnPlateau'")


def initialize_lr_scheduler(config, warmup_iters, lr_decay_iters, console=None):
    """
    Initialize and return the appropriate LR scheduler based on config.
    
    Args:
        config: Configuration dictionary with lr_scheduler and related parameters
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations for LR decay (CosineLR only)
        console: Rich console for printing (optional)
    
    Returns:
        lr_scheduler_obj: ReduceLROnPlateauScheduler object or None
    """
    lr_scheduler_type = config.get('lr_scheduler', None)
    lr_scheduler_obj = None
    
    if lr_scheduler_type is None or lr_scheduler_type == 'None':
        msg = f"Using constant learning rate: {config['learning_rate']:.2e}"
        if console:
            console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler_obj = ReduceLROnPlateauScheduler(config, warmup_iters, console=console)
        msg = f"Using ReduceLROnPlateau scheduler (factor={config['plateau_factor']}, patience={config['plateau_patience']})"
        if console:
            console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)
    elif lr_scheduler_type == 'CosineLR':
        msg = f"Using CosineLR scheduler (warmup={warmup_iters}, decay_iters={lr_decay_iters})"
        if console:
            console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_type}. Must be None, 'CosineLR', or 'ReduceLROnPlateau'")
    
    return lr_scheduler_obj
