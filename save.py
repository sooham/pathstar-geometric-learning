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
