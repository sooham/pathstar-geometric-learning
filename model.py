"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection to residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, output_attentions=False):
        """
        Input x: (batch_size, sequence_length, n_embd)
        Args:
            output_attentions: if True, also return attention weights (disables flash attention)
        Returns:
            y: output tensor
            attn_weights: attention weights (B, nh, T, T) if output_attentions=True, else None
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) # each one is (batch_size, sequence_length, n_embd)
        # expand last dim to (n_head, residual_stream_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        attn_weights = None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and not output_attentions:
            # efficient attention using Flash Attention CUDA kernels
            # TODO: look into custom attention masks via attn_mask
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention (required when output_attentions=True)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Create causal mask on the fly if we don't have bias buffer
            if hasattr(self, 'bias'):
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            else:
                # Create causal mask for flash attention path
                causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                att = att.masked_fill(causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            if output_attentions:
                attn_weights = att.detach()  # Save before dropout
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side (B, T, nh*hs=C)

        # output projection
        y = self.resid_dropout(self.c_proj(y)) # (B, T, nh*hs=C) the output 
        
        if output_attentions:
            return y, attn_weights
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        if config.activation == 'GELU':
            self.activation    = nn.GELU()  # Gaussian Error Linear Unit: smooth, non-monotonic activation function
        elif config.activation == 'RELU':
            self.activation = nn.RELU()
        else:
            raise ValueError("Unsupported activation")
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.use_layernorm:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.ln_1 = nn.Identity()
        self.attn = CausalSelfAttention(config)

        if config.use_layernorm:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.ln_2 = nn.Identity()
        
        if config.use_mlp:
            self.mlp = MLP(config)
        else:
            self.mlp = nn.Identity()

    def forward(self, x, output_attentions=False):
        if output_attentions:
            attn_out, attn_weights = self.attn(self.ln_1(x), output_attentions=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # aka. context length
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    base: float = 100.0 # base for positional encodings

    # regularization
    dropout: float = 0.0  # Dropout for attention, MLP, and residual connections
    embd_dropout: float = 0.0  # Dropout applied after embedding layer (tok_emb + pos_emb)
    weight_tying: bool = True

    # ML features
    activation: str = 'GELU'
    use_layernorm: bool = True
    use_mlp: bool = True
    use_pos_embeddings: bool = True  # True: use positional embeddings. False: no positional information
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config, meta=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.meta = meta

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.embd_dropout),  # Dropout layer applied to the sum of token and position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias) if config.use_layernorm else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Create sinusoidal position embeddings (if enabled)
        if config.use_pos_embeddings:
            base = config.base if config.base else 10_000
            self.register_buffer("pos_emb", self._create_sinusoidal_embeddings(config.block_size, config.n_embd, base=base))
        else:
            self.pos_emb = None

        # Precompute neighborhood information for efficient loss computation
        self.use_neighborhood_loss = False
        # Note: neighborhood_tensor, neighborhood_sizes_tensor, and inv_neighborhood_sizes_tensor
        # are registered as buffers in _precompute_neighborhood_info() if needed
        if meta is not None:
            self._precompute_neighborhood_info()

        # init all weights
        self.apply(self._init_weights)
        # weight tying
        if config.weight_tying:
            self.lm_head.weight = self.transformer.wte.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _precompute_neighborhood_info(self):
        """
        Precompute neighborhood information for efficient KL divergence loss computation.
        
        Creates tensors for fast lookup of neighbors and neighborhood sizes during forward pass.
        This is only used when use_directional_tokens=False and predict_direction_for_edge_task=False.
        """
        if self.meta is None:
            return
        
        # Check if we should use neighborhood-based loss
        use_directional_tokens = self.meta.get('use_directional_tokens', False)
        predict_dir = self.meta.get('predict_direction_for_edge_task', False)
        self.use_neighborhood_loss = (not use_directional_tokens) and (not predict_dir)
        
        if not self.use_neighborhood_loss:
            return
        
        adj_list = self.meta.get('adj_list')
        if adj_list is None:
            raise ValueError("adj_list not found in meta, but use_neighborhood_loss is True")
        
        vocab_size = self.config.vocab_size
        max_neighbors = max(len(neighbors) for neighbors in adj_list.values()) if adj_list else 0
        
        # Create tensors for neighbor lookup (padded to max_neighbors)
        # Shape: (vocab_size, max_neighbors) - padded with -1 for nodes with fewer neighbors
        neighborhood_tensor = torch.full((vocab_size, max_neighbors), -1, dtype=torch.long)
        neighborhood_sizes_tensor = torch.zeros(vocab_size, dtype=torch.long)
        
        for node, neighbors in adj_list.items():
            node_idx = int(node)
            if node_idx >= vocab_size:
                raise ValueError(f"Node index {node_idx} is greater than vocab size {vocab_size}")
            neighbor_list = sorted(list(neighbors))
            num_neighbors = len(neighbor_list)
            neighborhood_sizes_tensor[node_idx] = num_neighbors
            if num_neighbors > 0:
                neighborhood_tensor[node_idx, :num_neighbors] = torch.tensor(neighbor_list, dtype=torch.long)
        
        # Precompute inverse of neighborhood sizes for efficient KL divergence computation
        # Use float type since we'll use this in division
        inv_neighborhood_sizes_tensor = torch.zeros(vocab_size, dtype=torch.float32)
        # Only compute inverse for nodes with neighbors (avoid division by zero)
        nonzero_mask = neighborhood_sizes_tensor > 0
        inv_neighborhood_sizes_tensor[nonzero_mask] = 1.0 / neighborhood_sizes_tensor[nonzero_mask].float()
        
        # Register as buffers so they move to the correct device with the model
        self.register_buffer('neighborhood_tensor', neighborhood_tensor)
        self.register_buffer('neighborhood_sizes_tensor', neighborhood_sizes_tensor)
        self.register_buffer('inv_neighborhood_sizes_tensor', inv_neighborhood_sizes_tensor)
        
        print(f"Precomputed neighborhood info: {len(adj_list)} nodes, max {max_neighbors} neighbors per node")
        print(f"Using neighborhood-based KL divergence loss for EDGE tasks")

    def _create_sinusoidal_embeddings(self, max_len, d_model, base=100):
        """
        Create sinusoidal position embeddings.
        Args:
            max_len: maximum sequence length
            d_model: embedding dimension
        Returns:
            position embeddings of shape (max_len, d_model)
        """
        assert d_model % 2 == 0, "d_model must be even for sinusoidal embeddings"
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # Note: sinusoidal embeddings are not parameters, so no need to subtract them
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt((2 if self.config.use_mlp else 1) * self.config.n_layer))


    def forward(self, idx, targets=None, label_smoothing=0.0, track_activation_stats=False, importance_weights=None):
        """
        Assumption: targets are already shifted to account for the correct prediction of 
        of target[i] based on idx[0:i]

        idx: tokenized vector of shape (batch, sequence_length)
        track_activation_stats: if True, also return dict with mean/variance of activations per layer
        importance_weights: optional tensor of shape (batch_size,) with importance weights for each sample
        
        Returns:
            logits, loss  (if track_activation_stats=False)
            logits, loss, activation_stats  (if track_activation_stats=True)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Optional activation stats tracking
        activation_stats = {} if track_activation_stats else None

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.use_pos_embeddings:
            pos_emb = self.pos_emb[pos] # sinusoidal position embeddings of shape (t, n_embd)
            # Apply dropout to the combined embeddings for regularization during training.
            # This randomly zeros out some elements with probability config.dropout,
            # helping prevent overfitting by making the model more robust.
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # No positional embeddings - only token embeddings with dropout
            x = self.transformer.drop(tok_emb)
        
        if track_activation_stats:
            activation_stats['embedding_mean'] = x.mean().item()
            activation_stats['embedding_var'] = x.var().item()
        
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if track_activation_stats:
                activation_stats[f'layer_{i}_mean'] = x.mean().item()
                activation_stats[f'layer_{i}_var'] = x.var().item()
        
        if self.config.use_layernorm:
            x = self.transformer.ln_f(x)
        
        if track_activation_stats:
            activation_stats['final_mean'] = x.mean().item()
            activation_stats['final_var'] = x.var().item()

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # shape (batch, sequence_length, vocab_size)
            
            # Compute loss with optional neighborhood-based KL divergence for EDGE tasks
            if self.use_neighborhood_loss and self.meta is not None:
                loss = self._compute_mixed_loss(logits, targets, idx, label_smoothing, importance_weights)
            else:
                # Standard cross-entropy loss for all tasks
                if importance_weights is not None:
                    # Compute per-sample loss, weight by importance, then average
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1),
                        ignore_index=-1,
                        label_smoothing=label_smoothing,
                        reduction='none'
                    )
                    # Reshape to (batch_size, seq_len) and sum over sequence
                    loss_per_sample = loss.view(targets.size(0), targets.size(1)).sum(dim=1)  # (batch_size,)
                    # Apply importance weighting and average
                    loss = (loss_per_sample * importance_weights).mean()
                else:
                    # Standard scalar loss computation
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1),
                        ignore_index=-1,
                        label_smoothing=label_smoothing,
                        reduction='mean'
                    )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if track_activation_stats:
            return logits, loss, activation_stats
        return logits, loss

    def _compute_mixed_loss(self, logits, targets, idx, label_smoothing, importance_weights=None):
        """
        Compute mixed loss: KL divergence for EDGE tasks, cross-entropy for PATH tasks.
        
        Precondition: This function is only called when use_neighborhood_loss=True,
        which means use_directional_tokens=False and predict_direction_for_edge_task=False.
        
        Args:
            logits: (batch, seq_len, vocab_size) - model predictions
            targets: (batch, seq_len) - target tokens (with -1 for masked positions)
            idx: (batch, seq_len) - input tokens
            label_smoothing: label smoothing factor for cross-entropy
            importance_weights: optional tensor of shape (batch_size,) with importance weights
            
        Returns:
            loss: scalar loss tensor
        """
        device = logits.device
        batch_size = logits.size(0)
        
        # Get special tokens
        EDGE_token = self.meta['special_tokens']['EDGE']
        PATH_token = self.meta['special_tokens']['PATH']
        
        # Identify EDGE vs PATH tasks
        is_edge = (idx[:, 0] == EDGE_token)
        is_path = (idx[:, 0] == PATH_token)
        
        # Prediction position for EDGE tasks
        # Since use_directional_tokens=False: [EDGE, u, v, ...] -> predict at position 1 (for token at position 2)
        edge_pred_pos = 1
        
        total_loss = 0.0
        loss_count = 0
        
        # Compute KL divergence loss for EDGE tasks
        if is_edge.any():
            edge_indices = torch.where(is_edge)[0]
            
            # Extract source nodes u (position 1 after EDGE token)
            source_nodes = idx[edge_indices, 1]  # Shape: (num_edge_samples,)
            
            # Get logits for the prediction position
            edge_logits = logits[edge_indices, edge_pred_pos, :]  # Shape: (num_edge_samples, vocab_size)
            
            # Get neighborhoods for each source node
            neighborhoods = self.neighborhood_tensor[source_nodes]  # Shape: (num_edge_samples, max_neighbors)
            neighborhood_sizes = self.neighborhood_sizes_tensor[source_nodes]  # Shape: (num_edge_samples,)
            inv_neighborhood_sizes = self.inv_neighborhood_sizes_tensor[source_nodes]  # Shape: (num_edge_samples,)
            
            # Compute model's distribution Q from logits
            Q = F.softmax(edge_logits, dim=1)  # Shape: (num_edge_samples, vocab_size)
            
            # Compute KL divergence for all edge samples (vectorized)
            # Check for zero neighbors (vectorized)
            if (neighborhood_sizes == 0).any():
                bad_nodes = source_nodes[neighborhood_sizes == 0]
                raise ValueError(f"Nodes {bad_nodes.tolist()} appear in EDGE task but have zero neighbors")
            
            # Create mask for valid neighbors
            valid_mask = (neighborhoods >= 0)  # Shape: (num_edge_samples, max_neighbors)
            
            # Clamp neighborhoods to valid range for safe indexing
            neighborhoods_clamped = torch.clamp(neighborhoods, min=0)
            
            # Batch gather Q probabilities using advanced indexing
            batch_idx = torch.arange(len(edge_indices), device=device).unsqueeze(1).expand_as(neighborhoods_clamped)
            q_all_neighbors = Q[batch_idx, neighborhoods_clamped]  # (num_edge_samples, max_neighbors)
            
            # Apply epsilon for numerical stability ONLY to valid neighbors
            # Then compute log and mask invalid entries to 0
            epsilon = 1e-10
            log_q_valid = torch.log(q_all_neighbors + epsilon)  # Apply epsilon to all (including clamped)
            log_q_masked = log_q_valid.masked_fill(~valid_mask, 0.0)  # Zero out invalid neighbors
            
            # Sum log(Q) over valid neighbors only
            sum_log_q = log_q_masked.sum(dim=1)  # Shape: (num_edge_samples,)
            
            # Use precomputed inverse of neighborhood sizes (used twice in KL formula)
            # inv_neighborhood_sizes already loaded from buffer above
            
            # Vectorized KL divergence: KL = log(1/N) - (1/N) * sum log(Q(x))
            # Note: log(1/N) is negative (entropy term)
            # The loss will be MINIMIZED during training, making Q approach P
            log_p_uniform = torch.log(inv_neighborhood_sizes)  # (num_edge_samples,)
            kl_losses = log_p_uniform - inv_neighborhood_sizes * sum_log_q  # (num_edge_samples,)
            
            # Apply importance weighting if provided
            if importance_weights is not None:
                edge_importance = importance_weights[edge_indices]  # (num_edge_samples,)
                kl_losses = kl_losses * edge_importance
            
            # Aggregate losses
            total_loss += kl_losses.sum()
            loss_count += len(edge_indices)
        
        # Compute cross-entropy loss for PATH tasks
        if is_path.any():
            path_indices = torch.where(is_path)[0]
            
            # Extract logits and targets for PATH tasks
            path_logits = logits[path_indices]  # Shape: (num_path_samples, seq_len, vocab_size)
            path_targets = targets[path_indices]  # Shape: (num_path_samples, seq_len)
            
            if importance_weights is not None:
                # Compute per-sample loss with importance weighting
                path_loss_per_token = F.cross_entropy(
                    path_logits.reshape(-1, path_logits.size(-1)),
                    path_targets.reshape(-1),
                    ignore_index=-1,
                    label_smoothing=label_smoothing,
                    reduction='none'
                )
                # Reshape to (num_path_samples, seq_len)
                path_loss_per_token = path_loss_per_token.view(path_targets.size(0), path_targets.size(1))
                
                # Sum over sequence for each sample
                path_loss_per_sample = path_loss_per_token.sum(dim=1)  # (num_path_samples,)
                
                # Apply importance weighting
                path_importance = importance_weights[path_indices]  # (num_path_samples,)
                path_loss_weighted = path_loss_per_sample * path_importance
                
                # Count non-masked tokens
                path_loss_count = (path_targets != -1).sum().item()
                
                if path_loss_count > 0:
                    total_loss += path_loss_weighted.sum()
                    loss_count += path_loss_count
            else:
                # Standard unweighted loss
                path_loss = F.cross_entropy(
                    path_logits.reshape(-1, path_logits.size(-1)),
                    path_targets.reshape(-1),
                    ignore_index=-1,
                    label_smoothing=label_smoothing,
                    reduction='sum'
                )
                
                # Count non-masked tokens
                path_loss_count = (path_targets != -1).sum().item()
                
                if path_loss_count > 0:
                    total_loss += path_loss
                    loss_count += path_loss_count
        
        # Return average loss
        if loss_count > 0:
            return total_loss / loss_count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def get_attention_maps(self, idx):
        """
        Extract attention maps from all layers for visualization.
        
        Args:
            idx: input token indices of shape (batch_size, sequence_length)
            
        Returns:
            attention_maps: list of attention weights, one per layer
                           Each has shape (batch_size, n_head, seq_len, seq_len)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward through embeddings
        tok_emb = self.transformer.wte(idx)
        if self.config.use_pos_embeddings:
            pos_emb = self.pos_emb[pos]
            x = tok_emb + pos_emb  # No dropout during attention extraction
        else:
            x = tok_emb  # No positional embeddings
        
        attention_maps = []
        for block in self.transformer.h:
            x, attn_weights = block(x, output_attentions=True)
            attention_maps.append(attn_weights)
        
        return attention_maps

    def crop_block_size(self, block_size):
        # TODO: not really an use for this right now, maybe deal with this later
        # model surgery to decrease the block size if necessary
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Crop sinusoidal position embeddings (if they exist)
        if self.pos_emb is not None:
            self.pos_emb = self.pos_emb[:block_size]
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of device peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Auto-detect device and get peak FLOPS
        flops_promised = self._get_device_peak_flops()
        
        # express our flops throughput as ratio of device peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        mfu = flops_achieved / flops_promised
        return mfu

    def _get_device_peak_flops(self):
        """
        Get peak FLOPS for the current device.
        Returns peak FLOPS in FLOPS (not TFLOPS).
        """
        # Check if model parameters exist and get device from them
        device = next(self.parameters()).device
        device_type = device.type
        
        if device_type == 'cuda':
            # Try to identify the specific GPU
            try:
                gpu_name = torch.cuda.get_device_name(device.index or 0).upper()
                
                # Common GPU peak FLOPS (FP32 CUDA cores for realistic MFU on small models)
                # Note: Using FP32 instead of FP16 tensor cores since small models
                # don't fully utilize tensor cores due to overhead
                if 'A100' in gpu_name:
                    return 19.5e12  # 19.5 TFLOPS FP32 for A100 (was 312 TFLOPS FP16 tensor)
                elif 'H100' in gpu_name:
                    return 67e12  # 67 TFLOPS FP32 for H100
                elif 'V100' in gpu_name:
                    return 125e12  # 125 TFLOPS for V100
                elif 'A6000' in gpu_name or 'RTX A6000' in gpu_name:
                    return 38.7e12  # 38.7 TFLOPS FP32 for RTX A6000
                elif 'RTX 4090' in gpu_name or '4090' in gpu_name:
                    return 82.58e12  # 82.58 TFLOPS FP32 for RTX 4090 (was 330 TFLOPS FP16 tensor)
                elif 'RTX 3090' in gpu_name or '3090' in gpu_name:
                    return 35.58e12  # 35.58 TFLOPS FP32 for RTX 3090 (was 142 TFLOPS FP16 tensor)
                elif 'RTX 5060 TI' in gpu_name or '5060 TI' in gpu_name:
                    return 70e12  # 70 TFLOPS FP32 for RTX 5060 Ti (estimated)
                elif 'A10' in gpu_name:
                    return 31.2e12  # 31.2 TFLOPS FP32 for A10
                elif 'T4' in gpu_name:
                    return 8.1e12   # 8.1 TFLOPS FP32 for T4
                else:
                    # Default to A100 if unknown CUDA GPU
                    print(f"Unknown GPU: {gpu_name}, defaulting to A100 FP32 peak FLOPS")
                    return 19.5e12
            except:
                # If we can't get GPU name, default to A100
                return 19.5e12
                
        elif device_type == 'mps':
            # Apple Silicon MPS
            # M1/M2/M3 vary, but rough estimates:
            # M1: ~10-11 TFLOPS, M2: ~13-15 TFLOPS, M3: ~15-18 TFLOPS
            # Using a conservative estimate for M2 as baseline
            return 13e12  # 13 TFLOPS (conservative estimate for Apple Silicon)
            
        elif device_type == 'cpu':
            # CPU peak FLOPS varies widely, use a conservative estimate
            # Modern high-end CPUs: 1-2 TFLOPS
            return 1e12  # 1 TFLOPS (conservative estimate)
            
        else:
            # Unknown device, default to A100
            print(f"Unknown device type: {device_type}, defaulting to A100 peak FLOPS")
            return 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            if temperature == 0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train(was_training)
        return idx

    @torch.no_grad()
    def compute_token_entropy(self, idx):
        """
        Compute the token-level entropy for a given input sequence.
        Args:
            idx: LongTensor of shape (b, t) containing the input token indices
        Returns:
            entropy: Tensor of shape (b, t) containing the entropy at each token position
                    Entropy is computed as H(p) = -∑(p_i * log(p_i)) where p_i are the 
                    probabilities over the vocabulary at each position.
        Note:
            Make sure to be in model.eval() mode for this operation.
            Higher entropy indicates more uncertainty/uniformity in the model's predictions.
        """
        # if the sequence context is too long, crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # forward the model to get the logits for all positions in the sequence
        logits, _ = self(idx_cond)
        
        # convert logits to probabilities using softmax
        # logits shape: (b, t, vocab_size)
        probs = F.softmax(logits, dim=-1)
        
        # compute entropy: H(p) = -∑(p_i * log(p_i))
        # add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # shape: (b, t)
        
        return entropy

    @torch.no_grad()
    def plot_embeddings_umap(self, save_path=None, epoch=None, iteration=None,
                            include_root=False, include_special=False, num_paths=5,
                            figsize=(10, 8), reference_reducer=None):
        """
        Plot token embeddings in 2D with path structure highlighted.
        
        Creates a simple 2D UMAP visualization showing sampled training paths
        with distinct colors and arrows. Tracks epoch and iteration in the plot title.
        
        Supports "anchored UMAP" for smooth animations:
        - On first call (reference_reducer=None): fits UMAP and returns (fig, reducer)
        - On subsequent calls: uses reference_reducer for consistent coordinate system
        
        Args:
            save_path: Path to save the plot (if None, returns figure without saving)
            epoch: Optional epoch number to display in title
            iteration: Optional iteration number to display in title
            include_root: Whether to include root vertex in UMAP (default: False)
            include_special: Whether to include special tokens in UMAP (default: False)
            num_paths: Number of paths to highlight (default: 5)
            figsize: Figure size as (width, height) in inches (default: (10, 8))
            reference_reducer: Optional pre-fitted UMAP reducer for anchored projections.
                              If provided, embeddings will be projected into the same space.
            
        Returns:
            If reference_reducer is None: Tuple of (fig, fitted_reducer)
            If reference_reducer is provided: fig only
            
        Note:
            Requires metadata (self.meta) with 'paths_by_leaf' for path visualization.
        """
        from visualize_embeddings_umap import plot_embeddings_2d_with_paths
        
        if self.meta is None or 'paths_by_leaf' not in self.meta:
            raise ValueError("Metadata with 'paths_by_leaf' is required for embedding visualization")
        
        # Extract embeddings from the model
        embeddings = self.transformer.wte.weight.detach().cpu().numpy()
        
        # Create visualization (returns fig or (fig, reducer) depending on reference_reducer)
        result = plot_embeddings_2d_with_paths(
            embeddings=embeddings,
            meta=self.meta,
            save_path=save_path,
            epoch=epoch,
            iteration=iteration,
            include_root=include_root,
            include_special=include_special,
            num_paths=num_paths,
            figsize=figsize,
            reference_reducer=reference_reducer
        )
        
        return result

    @staticmethod
    def create_embedding_gif_from_checkpoints(checkpoint_paths, output_path='embeddings.gif',
                                             duration=500, loop=0, include_root=False,
                                             include_special=False, num_paths=5, device='cpu'):
        """
        Create an animated GIF from embeddings across multiple checkpoint files.
        
        This is useful for visualizing how embeddings evolve during training.
        
        Args:
            checkpoint_paths: List of paths to checkpoint files (.pt), in order
            output_path: Path to save the output GIF (default: 'embeddings.gif')
            duration: Duration of each frame in milliseconds (default: 500ms)
            loop: Number of loops (0 = infinite loop, default: 0)
            include_root: Whether to include root vertex in UMAP (default: False)
            include_special: Whether to include special tokens in UMAP (default: False)
            num_paths: Number of paths to highlight (default: 5)
            device: Device to load checkpoints on (default: 'cpu')
            
        Returns:
            None (saves GIF to disk)
            
        Example:
            >>> checkpoints = [f'out/ckpt_epoch_{i}.pt' for i in range(0, 100, 10)]
            >>> GPT.create_embedding_gif_from_checkpoints(
            ...     checkpoints,
            ...     output_path='training_evolution.gif',
            ...     duration=800
            ... )
        """
        from umap_utils import create_embedding_gif
        
        figures = []
        
        print(f"Creating embedding GIF from {len(checkpoint_paths)} checkpoints...")
        
        for i, ckpt_path in enumerate(checkpoint_paths):
            print(f"  [{i+1}/{len(checkpoint_paths)}] Loading {ckpt_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            model_args = checkpoint['model_args']
            meta = checkpoint.get('meta', {})
            
            if meta is None or 'paths_by_leaf' not in meta:
                raise ValueError(f"Checkpoint {ckpt_path} missing required metadata")
            
            # Create model
            config = GPTConfig(**model_args)
            model = GPT(config, meta=meta)
            
            # Load state dict
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            # Remove neighborhood tensors if present
            for k in ['neighborhood_tensor', 'neighborhood_sizes_tensor', 'inv_neighborhood_sizes_tensor']:
                state_dict.pop(k, None)
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            # Extract epoch/iter info from checkpoint
            epoch = checkpoint.get('epoch', None)
            iter_num = checkpoint.get('iter_num', None)
            
            # Generate plot
            fig = model.plot_embeddings_umap(
                save_path=None,
                epoch=epoch,
                iteration=iter_num,
                include_root=include_root,
                include_special=include_special,
                num_paths=num_paths
            )
            
            figures.append(fig)
        
        # Create GIF from figures
        print(f"Generating GIF...")
        create_embedding_gif(
            figures=figures,
            output_path=output_path,
            duration=duration,
            loop=loop
        )
        
        # Clean up figures
        for fig in figures:
            plt.close(fig)
        
        print(f"Done! Saved to: {output_path}")
