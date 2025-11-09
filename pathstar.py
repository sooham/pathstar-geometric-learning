import random 
import torch
import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx

class InContextPathStar:
    def __init__(self, d=5, l=5, vocab_size=2000):
        """
        Generator for in-context path-star task where each example has a fresh,
        randomly-labeled path-star graph.
        
        Args:
            d: Number of spokes/paths in the path-star
            l: Length of each path (number of nodes from root to leaf)
            vocab_size: Size of vocabulary for node labels (excludes special tokens)
        """
        self.d = d
        self.l = l
        self.vocab_size = vocab_size + 9 # account for special tokens below
        
        # Define special tokens
        self.SPECIAL_TOKENS = {
            'PAD': 0,
            'PAUSE': 1,
            'GT': 2,  # > directional token
            'LT': 3,  # < directional token
            'SEP': 4,  # separator token
            'START': 5,  # start marker
            'GOAL': 6,   # goal marker
            'PATH_START': 7,  # path start marker
            'EOS': 8,  # end of sequence
        }
        
        # Effective vocabulary size: |V| = |nodes| + 9
        self.effective_vocab_size = vocab_size + 9
        
        # Graph structure constants
        self.total_nodes = d * (l - 1) + 1  # Total nodes in path-star
        self.total_edges = d * (l - 1)  # Total edges in path-star
    
    def _generate_random_mapping(self):
        """
        Generate a random mapping from canonical node IDs to vocabulary tokens.
        
        Returns:
            dict: Mapping from canonical node ID (0 to total_nodes-1) to random token
        """
        # Sample random tokens from vocabulary (without replacement)
        available_tokens = list(range(9, self.vocab_size + 9))
        selected_tokens = random.sample(available_tokens, self.total_nodes)
        
        # Create mapping: canonical_id -> random_token
        mapping = {i: selected_tokens[i] for i in range(self.total_nodes)}
        return mapping
    
    def _generate_canonical_pathstar(self):
        """
        Generate the canonical path-star structure before applying random mapping.
        
        The canonical structure uses node IDs 0 to (d*(l-1)):
        - Node 0 is the root
        - Nodes 1 to d*(l-1) form the spokes
        
        Returns:
            tuple: (adj_list, paths_by_leaf, root, leaves)
                - adj_list: dict mapping node_id -> list of neighbors
                - paths_by_leaf: dict mapping leaf_id -> path from root to leaf
                - root: root node ID (always 0 in canonical form)
                - leaves: list of leaf node IDs
        """
        adj_list = {}
        paths_by_leaf = {}
        root = 0
        leaves = []
        
        # Root connects to first node of each spoke
        # Spokes start at: 1, l, 2l-1, 3l-2, ... = 1+(l-1)*k for k in [0, d-1]
        adj_list[root] = [1 + (self.l - 1) * k for k in range(self.d)]
        
        # Generate each spoke
        for path_idx in range(self.d):
            path_list = [root]
            
            for i in range(1, self.l):
                # Calculate node_id: first node of spoke path_idx is at 1+(l-1)*path_idx
                # then increment by 1 for each subsequent node
                node_id = 1 + (self.l - 1) * path_idx + (i - 1)
                
                if i < self.l - 1:
                    # Intermediate node
                    adj_list[node_id] = [node_id + 1]
                else:
                    # Leaf node
                    adj_list[node_id] = []
                    leaves.append(node_id)
                
                path_list.append(node_id)
            
            # Store path for this leaf
            paths_by_leaf[path_list[-1]] = path_list
        
        return adj_list, paths_by_leaf, root, leaves
    
    def generate_training_example(self, use_directional_tokens=False, num_pause_tokens=1):
        """
        Generate a single in-context path-star training example.
        
        Each example consists of:
        Prefix: [edge_pairs (shuffled adjacency list), PAUSE tokens, start_node, goal_node]
        Target: [path from start to goal]
        
        Args:
            use_directional_tokens: If True, use > and < tokens to indicate edge direction
            num_pause_tokens: Number of PAUSE tokens between adjacency list and query
        
        Returns:
            dict with keys:
                - 'prefix': list of tokens forming the prefix
                - 'target': list of tokens forming the target path
                - 'full_sequence': concatenated prefix + target
                - 'mapping': the random node mapping used
                - 'root': the mapped root node
                - 'goal': the mapped goal node
        """
        # Generate canonical path-star structure
        adj_list, paths_by_leaf, canonical_root, canonical_leaves = self._generate_canonical_pathstar()
        
        # Generate random mapping
        mapping = self._generate_random_mapping()
        
        # Apply mapping to adjacency list
        mapped_adj_list = []
        for u in adj_list:
            for v in adj_list[u]:
                mapped_u = mapping[u]
                mapped_v = mapping[v]
                if use_directional_tokens:
                    # Add edge as: (u, >, v)
                    mapped_adj_list.extend([mapped_u, self.SPECIAL_TOKENS['GT'], mapped_v])
                else:
                    # Add edge as: (u, v)
                    mapped_adj_list.extend([mapped_u, mapped_v])
        
        # Shuffle the adjacency list
        if use_directional_tokens:
            # Shuffle in chunks of 3 (u, >, v)
            edge_triples = [mapped_adj_list[i:i+3] for i in range(0, len(mapped_adj_list), 3)]
            random.shuffle(edge_triples)
            adjacency_sequence = [token for triple in edge_triples for token in triple]
        else:
            # Shuffle in chunks of 2 (u, v)
            edge_pairs = [mapped_adj_list[i:i+2] for i in range(0, len(mapped_adj_list), 2)]
            random.shuffle(edge_pairs)
            adjacency_sequence = [token for pair in edge_pairs for token in pair]
        
        # Sample a random goal (leaf) node
        canonical_goal = random.choice(canonical_leaves)
        
        # Get the canonical path from root to goal
        canonical_path = paths_by_leaf[canonical_goal]
        
        # Apply mapping to root, goal, and path
        mapped_root = mapping[canonical_root]
        mapped_goal = mapping[canonical_goal]
        mapped_path = [mapping[node] for node in canonical_path]
        
        # Construct prefix: [adjacency_list, PAUSE, ..., PAUSE, root, goal]
        pause_tokens = [self.SPECIAL_TOKENS['PAUSE']] * num_pause_tokens
        prefix = adjacency_sequence + pause_tokens + [mapped_root, mapped_goal]
        
        # Target is the full path from root to goal
        target = mapped_path
        
        # Full sequence
        full_sequence = prefix + target
        
        return {
            'prefix': prefix,
            'target': target,
            'full_sequence': full_sequence,
            'mapping': mapping,
            'root': mapped_root,
            'goal': mapped_goal,
            'adjacency_sequence': adjacency_sequence,
            'path': mapped_path
        }
    
    def generate_training_set(self, size, use_directional_tokens=False, num_pause_tokens=1, 
                             return_tensors=True, pad_to_length=None):
        """
        Generate a training set of in-context path-star examples.
        
        Args:
            size: Number of training examples to generate
            use_directional_tokens: If True, use directional tokens in adjacency list
            num_pause_tokens: Number of PAUSE tokens between adjacency list and query
            return_tensors: If True, return PyTorch tensors; otherwise return lists
            pad_to_length: If specified, pad sequences to this length with PAD tokens
        
        Returns:
            dict with keys:
                - 'prefixes': tensor/list of prefix sequences
                - 'targets': tensor/list of target paths
                - 'full_sequences': tensor/list of full sequences (prefix + target)
                - 'prefix_lengths': tensor/list of prefix lengths (for masking)
                - 'target_lengths': tensor/list of target lengths
        """
        examples = []
        for _ in range(size):
            example = self.generate_training_example(
                use_directional_tokens=use_directional_tokens,
                num_pause_tokens=num_pause_tokens
            )
            examples.append(example)
        
        # Extract components
        prefixes = [ex['prefix'] for ex in examples]
        targets = [ex['target'] for ex in examples]
        full_sequences = [ex['full_sequence'] for ex in examples]
        prefix_lengths = [len(ex['prefix']) for ex in examples]
        target_lengths = [len(ex['target']) for ex in examples]
        
        # Apply padding if requested
        if pad_to_length is not None:
            pad_token = self.SPECIAL_TOKENS['PAD']
            
            prefixes = [p + [pad_token] * (pad_to_length - len(p)) if len(p) < pad_to_length else p[:pad_to_length] 
                       for p in prefixes]
            targets = [t + [pad_token] * (pad_to_length - len(t)) if len(t) < pad_to_length else t[:pad_to_length]
                      for t in targets]
            full_sequences = [s + [pad_token] * (pad_to_length - len(s)) if len(s) < pad_to_length else s[:pad_to_length]
                             for s in full_sequences]
        
        # Convert to tensors if requested
        if return_tensors:
            # Find max lengths if no padding specified
            if pad_to_length is None:
                max_prefix_len = max(prefix_lengths)
                max_target_len = max(target_lengths)
                max_full_len = max(len(s) for s in full_sequences)
                
                # Pad to max lengths
                pad_token = self.SPECIAL_TOKENS['PAD']
                prefixes = [p + [pad_token] * (max_prefix_len - len(p)) for p in prefixes]
                targets = [t + [pad_token] * (max_target_len - len(t)) for t in targets]
                full_sequences = [s + [pad_token] * (max_full_len - len(s)) for s in full_sequences]
            
            prefixes = torch.tensor(prefixes, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
            full_sequences = torch.tensor(full_sequences, dtype=torch.long)
            prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        return {
            'prefixes': prefixes,
            'targets': targets,
            'full_sequences': full_sequences,
            'prefix_lengths': prefix_lengths,
            'target_lengths': target_lengths
        }
    
    def prepare(self, train_size=100000, val_size=10000, use_directional_tokens=False, 
                num_pause_tokens=1, output_dir='./data'):
        """
        Prepare and save training and validation datasets to disk.
        
        Args:
            train_size: Number of training examples
            val_size: Number of validation examples
            use_directional_tokens: If True, use directional tokens in adjacency list
            num_pause_tokens: Number of PAUSE tokens between adjacency list and query
            output_dir: Base directory for output (default: './data')
        """
        # Create output directory with parameters in name
        dir_name = f'incontext_pathstar_d{self.d}_l{self.l}_v{self.vocab_size}_p{num_pause_tokens}{"_dir" if use_directional_tokens else ""}'
        full_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"Preparing InContextPathStar dataset...")
        print(f"  Parameters: d={self.d}, l={self.l}, vocab_size={self.vocab_size}")
        print(f"  Output directory: {full_output_dir}")
        print(f"  Use directional tokens: {use_directional_tokens}")
        print(f"  Number of pause tokens: {num_pause_tokens}")
        
        # Generate training set
        print("\nGenerating training set...")
        train_data = self.generate_training_set(
            size=train_size,
            use_directional_tokens=use_directional_tokens,
            num_pause_tokens=num_pause_tokens,
            return_tensors=True
        )
        
        # Generate validation set
        print("Generating validation set...")
        val_data = self.generate_training_set(
            size=val_size,
            use_directional_tokens=use_directional_tokens,
            num_pause_tokens=num_pause_tokens,
            return_tensors=True
        )
        
        # Save training data
        train_path = os.path.join(full_output_dir, 'train.bin')
        print(f"\nSaving training data to {train_path}...")
        train_sequences = train_data['full_sequences'].numpy().astype(np.uint16)
        train_sequences.tofile(train_path)
        print(f"  Saved {train_sequences.shape[0]} sequences of length {train_sequences.shape[1]}")
        
        # Save validation data
        val_path = os.path.join(full_output_dir, 'val.bin')
        print(f"Saving validation data to {val_path}...")
        val_sequences = val_data['full_sequences'].numpy().astype(np.uint16)
        val_sequences.tofile(val_path)
        print(f"  Saved {val_sequences.shape[0]} sequences of length {val_sequences.shape[1]}")
        
        # Create vocabulary mappings
        itos = {}
        stoi = {}
        
        # Add regular vocabulary tokens
        for i in range(self.vocab_size):
            itos[i] = f'NODE_{i}'
            stoi[f'NODE_{i}'] = i
        
        # Add special tokens
        for name, token_id in self.SPECIAL_TOKENS.items():
            itos[token_id] = f'<{name}>'
            stoi[f'<{name}>'] = token_id
        
        # Save metadata
        meta = {
            'vocab_size': self.effective_vocab_size,
            'itos': itos,
            'stoi': stoi,
            'd': self.d,
            'l': self.l,
            'base_vocab_size': self.vocab_size,
            'special_tokens': self.SPECIAL_TOKENS,
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'use_directional_tokens': use_directional_tokens,
            'num_pause_tokens': num_pause_tokens,
        }
        
        meta_path = os.path.join(full_output_dir, 'meta.pkl')
        print(f"\nSaving metadata to {meta_path}...")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"\nDataset preparation complete!")
        print(f"  Effective vocab size: {self.effective_vocab_size}")
        print(f"  Total tokens (train): {train_sequences.size}")
        print(f"  Total tokens (val): {val_sequences.size}")
        
        return full_output_dir
    
    def __str__(self):
        """String representation for debugging"""
        lines = []
        lines.append(f"InContextPathStar(d={self.d}, l={self.l}, vocab_size={self.vocab_size})")
        lines.append(f"  Total nodes per graph: {self.total_nodes}")
        lines.append(f"  Total edges per graph: {self.total_edges}")
        lines.append(f"  Effective vocabulary size: {self.effective_vocab_size}")
        lines.append(f"  Special tokens:")
        for name, token_id in self.SPECIAL_TOKENS.items():
            lines.append(f"    {name}: {token_id}")
        return "\n".join(lines)


class InWeightsPathStar:
    def __init__(self, d=5, l=5, vocab_size=None, holdout_percentage=0.0):
        """
        Generator instance for a pathstar graph with d spokes
        of length l
        
        Args:
            d: Number of spokes/paths in the path-star
            l: Length of each path (number of nodes from root to leaf)
            vocab_size: Optional vocabulary mapping size
            mapping: Optional mapping from canonical node IDs to vocabulary tokens
            holdout_percentage: Percentage of paths to hold out (0.0 to 1.0)
        """

        self.d = d
        self.l = l
        self.vocab_size = vocab_size

        self.adj_list = {}
        self.num_vertices = d * (l-1) + 1
        # Sample random tokens from vocabulary without replacement
        canonical_nodes = list(range(self.num_vertices))
        vocab_tokens = random.sample(range(vocab_size), self.num_vertices)
        self.mapping = None
        if vocab_size:
            self.mapping = dict(zip(canonical_nodes, vocab_tokens))

        # populate the vertices with d * (l-1) + 1 vertices
        # 0 jis the root node
        # Spokes start at: 1, l, 2l-1, 3l-2, ... = 1+(l-1)*k for k in [0, d-1]
        self.vertices = list(range(d * (l-1) + 1))
        self.v_root = 0
        self.total_vert = len(self.vertices)
        self.v_leaf = [1 + (self.l-1)*(k+1) - 1 for k in range(self.d)]  # Last node of each spoke

        # add the root to the adjacency list
        self.paths_by_leaf = {}

        self.adj_list[0] = [1+(self.l-1)*k for k in range(self.d)]
        for pi in range(d):
            path_list = [0]  # Start with root
            
            for i in range(1, l):
                # Calculate node_id: first node of spoke pi is at 1+(l-1)*pi
                # then increment by 1 for each subsequent node
                node_val = 1 + (self.l-1)*pi + (i-1)
                
                if i != l-1:
                    self.adj_list[node_val] = [node_val+1]
                else:
                    self.adj_list[node_val] = []
                path_list.append(node_val)
            self.paths_by_leaf[node_val] = path_list
        

        if self.mapping is not None:
            # modify the mapping to accomodate for the special tokens 
            self.mapping = {k:v+11 for k,v in self.mapping.items()}
            assert len(self.mapping) == self.total_vert and set(self.mapping.keys()) == set(self.vertices)
            self._apply_mapping()
        
        # Define special tokens
        self.SPECIAL_TOKENS = {
            'PAD': 0,
            'PAUSE': 1,
            'GT': 2,  # > directional token means parent > child
            'LT': 3,  # < directional token child < parent
            'SEP': 4,  # separator token
            'START': 5,  # start marker
            'GOAL': 6,   # goal marker
            'PATH_START': 7,  # path start marker
            'EOS': 8,  # end of sequence
            'PATH': 9,
            'EDGE': 10,
        }
        # Determine pause token based on mapping
        self.pause_token = self.SPECIAL_TOKENS['PAUSE']
        self.pad_token = self.SPECIAL_TOKENS['PAD']
        
        # Define task prefix tokens (after pad token)
        self.TASK_TOKENS = {
            'PATH': self.SPECIAL_TOKENS['PATH'],
            'EDGE': self.SPECIAL_TOKENS['EDGE'],
        }
        
        # Set up holdout paths
        self.holdout_percentage = holdout_percentage
        self._setup_holdout_paths()
    
    def _apply_mapping(self):
        """
        Apply the mapping to adjacency list, paths_by_leaf, and other vertex-related attributes
        """
        # Map adjacency list
        mapped_adj_list = {}
        for u in self.adj_list:
            mapped_u = self.mapping[u]
            mapped_adj_list[mapped_u] = [self.mapping[v] for v in self.adj_list[u]]
        self.adj_list = mapped_adj_list
        
        # Map paths_by_leaf
        mapped_paths_by_leaf = {}
        for leaf_node, path in self.paths_by_leaf.items():
            mapped_leaf = self.mapping[leaf_node]
            mapped_path = [self.mapping[node] for node in path]
            mapped_paths_by_leaf[mapped_leaf] = mapped_path
        self.paths_by_leaf = mapped_paths_by_leaf
        
        # Map vertices
        self.vertices = [self.mapping[v] for v in self.vertices]
        
        # Map root and leaf vertices
        self.v_root = self.mapping[self.v_root]
        self.v_leaf = [self.mapping[leaf] for leaf in self.v_leaf]
    
    def _setup_holdout_paths(self):
        """
        Set up holdout paths based on holdout_percentage.
        Randomly selects a subset of leaf nodes (and their paths) to hold out.
        """
        if not (0.0 <= self.holdout_percentage <= 1.0):
            raise ValueError(f"holdout_percentage must be between 0.0 and 1.0, got {self.holdout_percentage}")
        
        all_leaf_nodes = list(self.paths_by_leaf.keys())
        num_holdout = round(self.d * self.holdout_percentage)
        
        if num_holdout > 0:
            # Randomly select holdout leaves
            self.holdout_leaves = set(random.sample(all_leaf_nodes, num_holdout))
            self.train_leaves = set(all_leaf_nodes) - self.holdout_leaves
        else:
            self.holdout_leaves = set()
            self.train_leaves = set(all_leaf_nodes)
        
        # Convert to lists for easier sampling
        self.holdout_leaves = list(self.holdout_leaves)
        self.train_leaves = list(self.train_leaves)
    
    def __str__(self):
        """
        String representation for debugging
        """
        lines = []
        lines.append(f"InWeightsPathStar(d={self.d}, l={self.l}, holdout_percentage={self.holdout_percentage})")
        lines.append(f"  Root vertex: {self.v_root}")
        lines.append(f"  Total vertices: {self.total_vert}")
        lines.append(f"  Leaf vertices: {self.v_leaf}")
        lines.append(f"  Train leaves: {sorted(self.train_leaves)} ({len(self.train_leaves)} paths)")
        lines.append(f"  Holdout leaves: {sorted(self.holdout_leaves)} ({len(self.holdout_leaves)} paths)")
        lines.append(f"  Pause token: {self.pause_token}")
        lines.append(f"  Pad token: {self.pad_token}")
        lines.append(f"  Task tokens: PATH={self.TASK_TOKENS['PATH']}, EDGE={self.TASK_TOKENS['EDGE']}")
        lines.append(f"  Vertices: {sorted(self.vertices) if isinstance(self.vertices, set) else self.vertices}")
        lines.append(f"\n  Adjacency List:")
        for node in sorted(self.adj_list.keys()):
            lines.append(f"    {node} -> {self.adj_list[node]}")
        lines.append(f"\n  Paths by Leaf:")
        if isinstance(self.paths_by_leaf, dict):
            for leaf, path in sorted(self.paths_by_leaf.items()):
                holdout_marker = " [HOLDOUT]" if leaf in self.holdout_leaves else ""
                lines.append(f"    Leaf {leaf}: {path}{holdout_marker}")
        else:
            lines.append(f"    {self.paths_by_leaf}")
        return "\n".join(lines)
    
    def generate_adjacency_list(self):
        """
        Generate an adjacency list as a shuffled list of edge pairs
        """
        # total nodes N = D * (P -1 ) + 1
        # total edges  total edges (P-1)*D

        adjacency_pairs_list = []
        for u in self.adj_list:
            for v in self.adj_list[u]:
                adjacency_pairs_list.append((u, v))
        
        random.shuffle(adjacency_pairs_list)

        return adjacency_pairs_list
    
    def generate_paths_by_leaf(self):
        """
        Generate paths by leaf (returns a copy of the internal paths_by_leaf)
        """
        return dict(self.paths_by_leaf)
    
    def visualize(self, output_path=None, figsize=(12, 10), show_labels=True):
        """
        Visualize the path-star graph structure.
        
        Args:
            output_path: If provided, save the figure to this path. Otherwise, display it.
            figsize: Figure size as (width, height) tuple
            show_labels: If True, show node labels
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all edges from adjacency list
        for u in self.adj_list:
            for v in self.adj_list[u]:
                G.add_edge(u, v)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout - use a radial layout for path-star structure
        # Root at center, spokes radiating outward
        pos = {}
        
        # Position root at center
        pos[self.v_root] = (0, 0)
        
        # Position each spoke
        angle_step = 2 * np.pi / self.d
        for spoke_idx in range(self.d):
            angle = spoke_idx * angle_step
            
            # Get the path for this spoke
            leaf = self.v_leaf[spoke_idx]
            path = self.paths_by_leaf[leaf]
            
            # Position nodes along the spoke
            for i, node in enumerate(path[1:], 1):  # Skip root
                radius = i  # Distance from center
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        # Determine node colors based on type and holdout status
        node_colors = []
        for node in G.nodes():
            if node == self.v_root:
                node_colors.append('#FF6B6B')  # Red for root
            elif node in self.holdout_leaves:
                node_colors.append('#FFD93D')  # Yellow for holdout leaves
            elif node in self.train_leaves:
                node_colors.append('#6BCB77')  # Green for training leaves
            elif node in self.v_leaf:
                node_colors.append('#4D96FF')  # Blue for other leaves
            else:
                node_colors.append('#95E1D3')  # Light teal for intermediate nodes
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              arrowstyle='->', width=2, alpha=0.6, ax=ax)
        
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=8, 
                                   font_weight='bold', ax=ax)
        
        # Add title and legend
        title = f"PathStar Graph: d={self.d}, l={self.l}"
        if self.holdout_percentage > 0:
            title += f", holdout={self.holdout_percentage:.0%}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label=f'Root (node {self.v_root})'),
            Patch(facecolor='#6BCB77', label=f'Training Leaves ({len(self.train_leaves)})'),
        ]
        if len(self.holdout_leaves) > 0:
            legend_elements.append(
                Patch(facecolor='#FFD93D', label=f'Holdout Leaves ({len(self.holdout_leaves)})')
            )
        legend_elements.append(
            Patch(facecolor='#95E1D3', label='Intermediate Nodes')
        )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add statistics text
        stats_text = (
            f"Vertices: {self.total_vert}\n"
            f"Edges: {self.d * (self.l - 1)}\n"
            f"Paths: {self.d}\n"
            f"Train: {len(self.train_leaves)}\n"
            f"Holdout: {len(self.holdout_leaves)}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_edge_memorization_training_set(self, size, undirected=True, use_directional_tokens=True):
        """
        Generate a training set of edges sampled randomly from the path-star graph.
        
        Args:
            size: Number of samples (K) to generate
            undirected: If True, also include reverse edges (y -> x) in the sampling pool
        
        Returns:
            edges: shape (size, 2) or (size, 3) if use directional otkesn 
        """
        # Collect all edges from the adjacency list
        edges = []
        for u in self.adj_list:
            for v in self.adj_list[u]:
                if use_directional_tokens:
                    edges.append([self.SPECIAL_TOKENS['GT'], u, v])
                else:
                    edges.append([u, v])
                if undirected:
                    if use_directional_tokens:
                        edges.append([self.SPECIAL_TOKENS['LT'], v, u])
                    else:
                        edges.append([v, u])
        
        # Validate size
        max_edges = len(edges)
        if size > max_edges:
            raise ValueError(
                f"Requested size ({size}) exceeds the total number of available edges ({max_edges}). "
                f"Graph has {self.total_edges} directed edges"
                + (f" or {2 * self.total_edges} undirected edges." if undirected else ".")
            )
        
        # Shuffle edges and take the first k
        random.shuffle(edges)
        sampled_edges = edges[:size]
        
        # Return as torch tensor
        return torch.tensor(sampled_edges, dtype=torch.long)

    def generate_path_prediction_training_set(self, size, num_pause_tokens=1, 
                                             obey_holdout=True, holdout_only=False):
        """
        Generate a path-finding training set for the in-weights path memorization objective.
        
        Each training example has the format:
        Input: [<PATH>, leaf, <PAUSE>, <PAUSE>, ..., <PAUSE>, root, n_2, n_3, ..., n_ℓ]
               where <PATH> is a task prefix token and the number of <PAUSE> tokens 
               is controlled by num_pause_tokens
        Target: predict each next token left-to-right
        
        Args:
            size: Number of samples (K) to generate
            num_pause_tokens: Number of <PAUSE> tokens to insert between leaf and path (default: 1)
            obey_holdout: If True, only sample from training leaves (default: True)
            holdout_only: If True, only sample from holdout leaves (default: False)
        
        Returns:
            sequences: torch tensor of shape [size, l+2+num_pause_tokens] containing full sequences
                      (<PATH>, leaf, pause_1, ..., pause_n, root, n_2, ..., n_ℓ)
        """
        # Determine which leaf nodes to sample from
        if holdout_only:
            if len(self.holdout_leaves) == 0:
                raise ValueError("Cannot generate holdout_only data: no holdout paths available")
            leaf_nodes = self.holdout_leaves
        elif obey_holdout:
            if len(self.train_leaves) == 0:
                raise ValueError("Cannot generate training data with obey_holdout=True: no training paths available")
            if len(self.train_leaves) < size:
                raise ValueError("This should not happen you want to generate holdouts, the training set size should be the same as the holdout leaves")
            leaf_nodes = self.train_leaves
        else:
            # Use all leaf nodes
            leaf_nodes = list(self.paths_by_leaf.keys())
        
        # Validate size
        max_paths = len(leaf_nodes)
        if size > max_paths:
            mode_str = "holdout" if holdout_only else ("training" if obey_holdout else "all")
            raise ValueError(
                f"Requested size ({size}) exceeds the number of available {mode_str} paths ({max_paths}). "
                f"Graph has {len(self.train_leaves)} training paths and {len(self.holdout_leaves)} holdout paths."
            )
        
        # Sample leaf nodes uniformly without replacement (ensures unique paths)
        sampled_leaves = random.sample(leaf_nodes, size)
        
        sequences = []
        for leaf in sampled_leaves:
            # Get the path from root to leaf
            path = self.paths_by_leaf[leaf]
            
            # Construct sequence: [<PATH>, leaf, <PAUSE>, ..., <PAUSE>, root, n_2, ..., n_ℓ]
            pause_tokens = [self.pause_token] * num_pause_tokens
            sequence = [self.TASK_TOKENS['PATH'], leaf] + pause_tokens + path
            sequences.append(sequence)
        
        # Convert to tensor
        sequences = torch.tensor(sequences, dtype=torch.long)
        
        return sequences
    
    def prepare(self, num_pause_tokens=1, output_dir='./data', 
                use_undirected=True, use_directional_tokens=True):
        """
        Prepare and save training and validation datasets to disk for in-weights path-star.
        
        Dataset structure:
        - Training set: All training paths (self.train_leaves) + All edges (mixed and shuffled)
        - Validation set: Only holdout paths (self.holdout_leaves, no edges)
        
        Dataset size is automatically calculated based on graph structure:
        - Number of edges: (l-1) * d
        - Training paths: determined by holdout_percentage (train_leaves)
        - Validation paths: determined by holdout_percentage (holdout_leaves)
        
        Args:
            num_pause_tokens: Number of PAUSE tokens between leaf and path
            output_dir: Base directory for output (default: './data')
            use_undirected: If True, use undirected edges (both x->y and y->x) (default: True)
            use_directional_tokens: If True, use special tokens to demarcate edge directions in the edge training set
        """
        # Calculate dataset sizes based on graph structure
        num_edges = self.d * (self.l - 1)
        num_train_path_samples = len(self.train_leaves)  # Training paths
        num_val_path_samples = len(self.holdout_leaves)  # Validation paths (holdout)
        
        # Calculate edge dataset size
        num_edge_samples = (2 if use_undirected else 1) * num_edges
        
        # Calculate replication factor for path sequences to balance classes
        replication_factor = num_edge_samples // num_train_path_samples if num_train_path_samples > 0 else 1
        if replication_factor < 1:
            replication_factor = 1
        replicated_path_samples = num_train_path_samples * replication_factor
        
        # Training set: replicated train paths + all edges
        train_size = replicated_path_samples + num_edge_samples
        
        # Validation set: only holdout paths (no edges)
        val_size = num_val_path_samples
        
        # Create output directory with parameters in name
        dir_name = f'inweights_pathstar_d{self.d}_l{self.l}_p{num_pause_tokens}_{"un" if use_undirected else ""}directed_{"dt" if use_directional_tokens else ""}'
        full_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"Preparing InWeightsPathStar dataset...")
        print(f"  Parameters: d={self.d}, l={self.l}")
        print(f"  Graph structure:")
        print(f"    Total vertices: {self.total_vert}")
        print(f"    Total edges: {num_edges}")
        print(f"    Total paths (spokes): {self.d}")
        print(f"    Training paths: {num_train_path_samples}")
        print(f"    Validation paths (holdout): {num_val_path_samples}")
        print(f"  Dataset composition:")
        print(f"    Edge samples: {num_edge_samples} ({'undirected' if use_undirected else 'directed'})")
        print(f"    Training path samples (original): {num_train_path_samples}")
        print(f"    Training path samples (replicated): {replicated_path_samples} (factor: {replication_factor})")
        print(f"    Validation path samples: {num_val_path_samples}")
        print(f"  Final dataset sizes:")
        print(f"    Training set: {train_size} ({replicated_path_samples} replicated paths + {num_edge_samples} edges)")
        print(f"    Validation set: {val_size} (holdout paths only, no edges)")
        print(f"    2d dimension of training set is : {self.l + num_pause_tokens + 2}")
        print(f"  Output directory: {full_output_dir}")
        print(f"  Pause token: {self.pause_token}")
        print(f"  Pad token: {self.pad_token}")
        print(f"  EDGE token: {self.SPECIAL_TOKENS['EDGE']}")
        print(f"  PATH token: {self.SPECIAL_TOKENS['PATH']}")
        print(f"  Number of pause tokens: {num_pause_tokens}")
        print(f"  Holdout percentage: {self.holdout_percentage}")
        
        # Print paths_by_leaf in a pretty manner
        print(f"\n  Paths by leaf node:")
        for leaf, path in sorted(self.paths_by_leaf.items()):
            path_str = ' -> '.join(map(str, path))
            is_train = leaf in self.train_leaves
            is_holdout = leaf in self.holdout_leaves
            status = "TRAIN" if is_train else ("HOLDOUT" if is_holdout else "UNKNOWN")
            print(f"    Leaf {leaf} [{status}]: {path_str}")
        
        # Generate training set: paths + edges
        print("\nGenerating training set (training paths + edges)...")
        
        # Generate path sequences for training (uses self.train_leaves)
        train_path_sequences = self.generate_path_prediction_training_set(
            size=num_train_path_samples,
            num_pause_tokens=num_pause_tokens,
            obey_holdout=True  # Uses train_leaves
        )
        
        # Generate edge sequences
        edges = self.generate_edge_memorization_training_set(
            size=num_edge_samples,
            undirected=use_undirected,
            use_directional_tokens=use_directional_tokens
        )
        # Convert edge pairs to sequences: [<EDGE>,<optional direction token>, x, y]
        edge_task_tokens = torch.full((num_edge_samples, 1), self.TASK_TOKENS['EDGE'], dtype=torch.long)
        edge_sequences = torch.cat([edge_task_tokens, edges], dim=1)
        
        # Pad edge sequences to match path sequence length using <PAD> token
        path_seq_len = train_path_sequences.shape[1]
        edge_seq_len = edge_sequences.shape[1]
        
        if edge_seq_len < path_seq_len:
            padding = torch.full(
                (num_edge_samples, path_seq_len - edge_seq_len),
                self.pad_token,
                dtype=torch.long
            )
            edge_sequences = torch.cat([edge_sequences, padding], dim=1)
        
        # Replicate path sequences to account for class imbalance
        # The imbalance factor is approximately (num_edge_samples / num_train_path_samples)
        # which is roughly (l-1) for directed edges or 2*(l-1) for undirected
        if num_train_path_samples > 0:
            replication_factor = num_edge_samples // num_train_path_samples
            if replication_factor > 1:
                print(f"  Replicating path sequences by factor of {replication_factor} to balance classes")
                train_path_sequences = train_path_sequences.repeat(replication_factor, 1)
                print(f"  Path sequences after replication: {train_path_sequences.shape[0]}")
        
        # Concatenate and shuffle training sequences
        train_sequences = torch.cat([train_path_sequences, edge_sequences], dim=0)
        train_indices = torch.randperm(train_sequences.shape[0])
        train_sequences = train_sequences[train_indices]
        
        # Generate validation set: only holdout paths (no edges)
        print("Generating validation set (holdout paths only, no edges)...")
        val_sequences = self.generate_path_prediction_training_set(
            size=num_val_path_samples,
            num_pause_tokens=num_pause_tokens,
            holdout_only=True  # Uses holdout_leaves
        )
        
        # Debug: Print train and val sequences
        print(f"\nDebug - Train sequences (numpy):")
        print(train_sequences.numpy())
        print(f"\nDebug - Val sequences (numpy):")
        print(val_sequences.numpy())
        
        # Save training data
        train_path = os.path.join(full_output_dir, 'train.bin')
        print(f"\nSaving training data to {train_path}...")
        train_data = train_sequences.numpy().astype(np.uint16)
        train_data.tofile(train_path)
        print(f"  Saved {train_data.shape[0]} sequences of length {train_data.shape[1]}")
        
        # Save validation data
        val_path = os.path.join(full_output_dir, 'val.bin')
        print(f"Saving validation data to {val_path}...")
        val_data = val_sequences.numpy().astype(np.uint16)
        val_data.tofile(val_path)
        print(f"  Saved {val_data.shape[0]} sequences of length {val_data.shape[1]}")
        
        # Create vocabulary mappings
        # Vocab includes all vertices plus the pause token, pad token, and task tokens
        all_tokens = sorted(set(self.vertices) | set(self.SPECIAL_TOKENS.values()))
        # vocab_size must be max_token_id + 1 for PyTorch embedding layers
        # also add <PAUSE> <PAD> <PATH> <EDGE> into consideration
        vocab_size = self.vocab_size + 11
        
        itos = {}
        stoi = {}
        
        for token in all_tokens:
            if token == self.SPECIAL_TOKENS['PAUSE']:
                itos[token] = '<PAUSE>'
                stoi['<PAUSE>'] = token
            elif token == self.SPECIAL_TOKENS['PAD']:
                itos[token] = '<PAD>'
                stoi['<PAD>'] = token
            elif token == self.SPECIAL_TOKENS['PATH']:
                itos[token] = '<PATH>'
                stoi['<PATH>'] = token
            elif token == self.SPECIAL_TOKENS['EDGE']:
                itos[token] = '<EDGE>'
                stoi['<EDGE>'] = token
            elif token == self.SPECIAL_TOKENS['GT']:
                itos[token] = '>'
                stoi['>'] = token
            elif token == self.SPECIAL_TOKENS['LT']:
                itos[token] = '<'
                stoi['<'] = token
            elif token == self.v_root:
                itos[token] = f'ROOT_{token}'
                stoi[f'ROOT_{token}'] = token
            elif token in self.v_leaf:
                itos[token] = f'LEAF_{token}'
                stoi[f'LEAF_{token}'] = token
            else:
                itos[token] = f'NODE_{token}'
                stoi[f'NODE_{token}'] = token
        
        # Save metadata
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
            'd': self.d,
            'l': self.l,
            'total_vertices': self.total_vert,
            'total_edges': num_edges,
            'pause_token': self.SPECIAL_TOKENS["PAUSE"],
            'pad_token': self.SPECIAL_TOKENS["PAD"],
            'special_tokens': self.SPECIAL_TOKENS,
            'task_tokens': self.TASK_TOKENS,
            'num_pause_tokens': num_pause_tokens,
            'root_vertex': self.v_root,
            'leaf_vertices': self.v_leaf,
            'vertices': self.vertices,
            'holdout_percentage': self.holdout_percentage,
            'train_leaves': self.train_leaves,
            'holdout_leaves': self.holdout_leaves,
            'use_undirected': use_undirected,
            'use_directional_tokens': use_directional_tokens,
            'train_size': train_size,
            'val_size': val_size,
            'num_train_path_samples': num_train_path_samples,
            'num_val_path_samples': num_val_path_samples,
            'num_edge_samples': num_edge_samples,
        }
        
        meta_path = os.path.join(full_output_dir, 'meta.pkl')
        print(f"\nSaving metadata to {meta_path}...")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"\nDataset preparation complete!")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Total tokens (train): {train_data.size}")
        print(f"  Total tokens (val): {val_data.size}")
        
        return full_output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PathStar datasets')
    parser.add_argument('--mode', type=str, required=True, choices=['incontext', 'inweights'],
                        help='Dataset mode: incontext or inweights')
    parser.add_argument('--d', type=int, default=10,
                        help='Number of spokes/paths in the path-star (default: 10)')
    parser.add_argument('--l', type=int, default=100,
                        help='Length of each path (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=2000,
                        help='Vocabulary size (default: 2000)')
    parser.add_argument('--train_size', type=int, default=100000,
                        help='Number of incontext training examples (default: 100000)')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of incontext validation examples (default: 10000)')
    parser.add_argument('--num_pause_tokens', type=int, default=1,
                        help='Number of PAUSE tokens used (default: 1)')
    parser.add_argument('--use_directional_tokens', action='store_true',
                        help='Use directional tokens (> and <) in incontext mode')
    parser.add_argument('--use_directed', action='store_true',
                        help='Use directed edges for inweights mode (default: undirected)')
    parser.add_argument('--train_val_split', type=float, default=0.9,
                        help='Train/validation split ratio for inweights mode (default: 0.9)')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize the graph structure before generating dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for datasets (default: ./data)')
    
    args = parser.parse_args()
    
    if args.mode == 'incontext':
        print(f"Generating InContextPathStar dataset...")
        generator = InContextPathStar(d=args.d, l=args.l, vocab_size=args.vocab_size)
        generator.prepare(
            train_size=args.train_size,
            val_size=args.val_size,
            use_directional_tokens=args.use_directional_tokens,
            num_pause_tokens=args.num_pause_tokens,
            output_dir=args.output_dir
        )
    elif args.mode == 'inweights':
        print(f"Generating InWeightsPathStar dataset...")
        
        # Calculate holdout_percentage from train_val_split
        # train_val_split=0.9 means 90% train, 10% validation
        # so holdout_percentage = 1 - train_val_split = 0.1
        holdout_percentage = 1.0 - args.train_val_split
        
        # Create randomized mapping from canonical node IDs to vocabulary tokens
        num_vertices = args.d * (args.l - 1) + 1
        if num_vertices > args.vocab_size:
            raise ValueError(
                f"Graph requires {num_vertices} vertices but vocab_size is only {args.vocab_size}. "
                f"Please increase vocab_size to at least {num_vertices}."
            )
        
        
        generator = InWeightsPathStar(
            d=args.d, 
            l=args.l, 
            holdout_percentage=holdout_percentage,
            vocab_size=args.vocab_size
        )
        
        # Visualize the graph if requested
        if args.viz:
            print("\nVisualizing graph structure...")
            viz_output = os.path.join(args.output_dir, f'pathstar_d{args.d}_l{args.l}_viz.png')
            os.makedirs(args.output_dir, exist_ok=True)
            generator.visualize(output_path=viz_output)
        
        generator.prepare(
            num_pause_tokens=args.num_pause_tokens,
            output_dir=args.output_dir,
            use_undirected=not args.use_directed,
            use_directional_tokens=args.use_directional_tokens
        )