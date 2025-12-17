import random 
import torch
import os
import pickle
import numpy as np
import argparse
import math

# PATH TASK
# <PATH> x_leaf <PAUSE> ... <PAUSE> x_root, x_1, x_2, ... , x_leaf
# input:      <PATH>  x_leaf   <PAUSE> . <PAUSE> x_root x_1 x_2  ... 
# output:     x_leaf <PAUSE> . <PAUSE>   x_root   x_1,  x_2 ... x_leaf
# eval loss:    0      0 .       0         1       1    1        1
# the input x_1 depends on
# <PATH> x_leaf <PAUSE> ... <PAUSE> x_root
# attention mechanism will compute on (q_path, k_path), (q_leaf, k_leaf), (q_pause, k_pause), (q_root, k_root)
# so v_path, v_x_leaf v_pause, v_x_root are the embeddings that will in the circuit
# the only thing that appears in EDGE task is x_leaf and x_root
# so need to predict x_root+1 in direction of x_leaf from this 
# direction that matter is LT (aka. towards the root)
# x_leaf should contain x_leaf-1 which should contain x_leaf-2 ... contain x_root
# WANT: x_leaf should contain x_leaf-1 and GT (cause moving away from root)

# --- CURRENT -- 
# EDGE TASK  (> / GT means away from root, < LT means towards root , root index is 0)
# [EDGE] GT x_leaf-1 x_leaf
# input  [EDGE]   GT        x_leaf-1   potentially give GT
# output   GT     x_leaf-1  x_leaf
# mask     0 .    0 .         1
#  PROPERY : x_leaf contains x_leaf-1 and GT

# [EDGE] LT x_leaf x_leaf-1
# input  [EDGE]   LT        x_leaf
# output   LT     x_leaf   x_leaf-1
# mask     0 .    0 .         1
# PROPERTY: x_leaf-1 contains x_leaf and LT

# issue x_leaf needs to somehow store the vector for x_leaf-1 in it 
# in the above schema, training only happens on last step , so loss is 
# L(cross_entropy(one_hot(p(x | [EDGE GT x_leaf-1]))) =  -log(p(x_leaf| [EDGE] GT x_leaf-1))
# L(cross_entropy(one_hot(p(x | [EDGE LT x_leaf]))) =  -log(p(x_leaf-1| [EDGE] LT x_leaf))
# minimize L = -log( output_projection @ W_O @ (attention score for v_x_leaf on sequence)^T(v_edge + v_LT + v_x_leaf) ) produces output for v_x_leaf-1

# idea 1.
# versus if you give [EDGE] x_leaf LT x_leaf-1
# or [EDGE] x_leaf-1 GT x_leaf
# the attention score for LT on [EDGE and x_leaf is considered ] to produce the output v that leads to x_leaf-1
# the attention score for GT on [EDGE and x_leaf-1 is considered ] to produce the output v to x_leaf
# PROPERTIES so x_leaf-1 contains x_leaf and LT
# PROPERY x_leaf contains x_leaf-1 and GT
# okay so this is actually consistent
# but chainable [PATH] x_leaf [PAUSE] x_root GT x_root+1 GT x_root+2 ... GT x_leaf-1 GT x_leaf

# idea 2.
# we do [EDGE] x_leaf x_leaf-1 LT
# or    [EDGE] x_leaf-1 x_leaf GT
# here we are predicting the direction which is an easier problem
# no dependency betwene x_leaf-1 and x_leaf, but prediction needs to be determinable by embeddings

class InWeightsPathStar:
    def __init__(self, d=5, l=5, randomize_vocab_size=None, holdout_percentage=0.0):
        """
        Generator instance for a pathstar graph with d spokes
        of length l.
        
        Args:
            d: Number of spokes/paths in the path-star
            l: Length of each path (number of nodes from root to leaf)
            randomize_vocab_size: Optional vocabulary mapping size we want to randomize vertices with 
            holdout_percentage: Percentage of paths to hold out (0.0 to 1.0)
        """

        self.d = d
        self.l = l
        self.randomize_vocab_size = randomize_vocab_size

        self.adj_list = {}
        self.num_vertices = d * (l-1) + 1
        self.num_graph_edges = d * (l-1)
        # populate the vertices with d * (l-1) + 1 vertices
        # 0 is the root node
        # Spokes start at: 1, l, 2l-1, 3l-2, ... = 1+(l-1)*k for k in [0, d-1]
        # d = 3
        # g = 5
        # 0 1 2 3 4 
        #   5 6 7 8
        #   9 10 11 12
        # start index is d*(l-1)+1 = 0*4+1 = 1 , 1*4+1 = 5 , 2*4+1 = 9
        # end index is d*(l-1)+(l-1) = (d+1)*(l-1) 
        self.v_root = 0
        self.v_leaf = [(self.l-1)*(k+1) for k in range(self.d)]  # Last node of each spoke
        self.vertices = list(range(d * (l-1) + 1))

        # Sample random tokens from vocabulary without replacement
        canonical_nodes = list(range(self.num_vertices))

        if randomize_vocab_size == 'auto':
            print(f"Using auto vocab_size of {self.num_vertices}")
            self.randomize_vocab_size = self.num_vertices
            
        if self.randomize_vocab_size and self.num_vertices > self.randomize_vocab_size:
            raise ValueError(
                f"Graph requires {self.num_vertices} vertices but vocab_size is only {self.randomize_vocab_size}. "
                f"Please increase vocab_size to at least {self.num_vertices}."
            )

        if self.randomize_vocab_size and self.randomize_vocab_size >= self.num_vertices:
            vocab_tokens = random.sample(range(self.randomize_vocab_size), self.num_vertices)
        else:
            vocab_tokens = list(range(self.num_vertices))

        self.mapping = dict(zip(canonical_nodes, vocab_tokens))


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
        
        # Define special tokens
        self.SPECIAL_TOKENS = {
            'PAD': 0,
            'PAUSE': 1,
            'GT': 2,  # > directional token means parent > child
            'LT': 3,  # < directional token child < parent
            'PATH': 4,
            'EDGE': 5
        }
        self.pause_token = self.SPECIAL_TOKENS['PAUSE']
        self.pad_token = self.SPECIAL_TOKENS['PAD']
        self.num_special_tokens = len(self.SPECIAL_TOKENS)

        # modify the mapping to accomodate for the special tokens 
        self.mapping = {k:v+self.num_special_tokens for k,v in self.mapping.items()}
        self._apply_mapping()
        
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
        num_holdout = math.ceil(self.d * self.holdout_percentage)
        
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
        lines.append(f"  Total vertices: {self.num_vertices}")
        lines.append(f"  Leaf vertices: {self.v_leaf}")
        lines.append(f"  Train leaves: {sorted(self.train_leaves)} ({len(self.train_leaves)} paths)")
        lines.append(f"  Holdout leaves: {sorted(self.holdout_leaves)} ({len(self.holdout_leaves)} paths)")
        lines.append(f"  Pause token: {self.pause_token}")
        lines.append(f"  Pad token: {self.pad_token}")
        lines.append(f"  Task tokens: PATH={self.SPECIAL_TOKENS['PATH']}, EDGE={self.SPECIAL_TOKENS['EDGE']}")
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
    
    def _generate_adjacency_list(self):
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
    
    def _generate_paths_by_leaf(self):
        """
        Generate paths by leaf (returns a copy of the internal paths_by_leaf)
        """
        return dict(self.paths_by_leaf)
    
    def _generate_edge_memorization_training_set(self, size, undirected=True, use_directional_tokens=True, use_task_tokens=True, predict_direction_for_edge_task=True):
        """
        Generate a training set of edges sampled randomly from the path-star graph.
        
        Args:
            size: Number of samples (K) to generate
            undirected: If True, also include reverse edges (y -> x) in the sampling pool
            use_directional_tokens: If true uses GT and LT tokens to show direction
            use_task_tokens: If true uses EDGE token to show the task
            predict_direction_for_edge_task: If True the format is [EDGE] u v direction , otherwise it is [EDGE] direction u v
        Returns:
            edges: shape (size, 2+A+B) where A == 1 if use_directional_tokens is true and B == 1 if use_task_tokens is true, otherwise 0 
        """
        # Collect all edges from the adjacency list
        def add_edge(u, v):
            # assumption u is before v from root
            # first edge (u, v)
            if use_directional_tokens:
                if predict_direction_for_edge_task:
                    edges.append([u, v, self.SPECIAL_TOKENS['GT']]) # GT means  away from root
                else:
                    edges.append([self.SPECIAL_TOKENS['GT'], u, v]) # GT means  away from root
            else:
                edges.append([u, v])

            if undirected: # add the reverse edge (v, u)
                if use_directional_tokens:
                    if predict_direction_for_edge_task:
                        edges.append([v, u, self.SPECIAL_TOKENS['LT']]) # LT means toward root
                    else:
                        edges.append([self.SPECIAL_TOKENS['LT'], v, u]) # LT means  toward root
                else:
                    edges.append([v, u])

        edges = []
        for u in self.adj_list:
            for v in self.adj_list[u]:
                add_edge(u, v)
        
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
        edges =  torch.tensor(sampled_edges, dtype=torch.long)
        # Convert edge pairs to sequences: [<EDGE>,<optional direction token>, x, y] or [<optional direction token>, x, y]
        if use_task_tokens:
            edge_task_tokens = torch.full((size, 1), self.SPECIAL_TOKENS['EDGE'], dtype=torch.long)
            edge_sequences = torch.cat([edge_task_tokens, edges], dim=1)
        else:
            edge_sequences = edges
        
        return edge_sequences

    def _generate_path_prediction_training_set(self, size, split, num_pause_tokens=1, use_task_tokens=True):
        """
        Generate a path-finding training set for the in-weights path memorization objective.
        
        Each training example has the format:
        Input: [<optional PATH>, leaf, <PAUSE>, <PAUSE>, ..., <PAUSE>, root, n_2, n_3, ..., n_ℓ]
               where <PATH> is a task prefix token and the number of <PAUSE> tokens 
               is controlled by num_pause_tokens
        Target: predict each next token left-to-right
        
        Args:
            size: Number of samples (K) to generate
            num_pause_tokens: Number of <PAUSE> tokens to insert between leaf and path (default: 1)
            split: either 'train' (training leaves only), 'val' (holdout leaves) or all (both)
            use_task_tokens: If True, include <PATH> task prefix token (default: True)
        
        Returns:
            sequences: torch tensor of shape [size, l+1+num_pause_tokens+t] where t is if task_token is used or not, containing full sequences
                      <PATH>, leaf, pause_1, ..., pause_n, root, n_2, ..., n_ℓ) if use_task_tokens=True
                      or (leaf, pause_1, ..., pause_n, root, n_2, ..., n_ℓ) if use_task_tokens=False
        """
        # Determine which leaf nodes to sample from
        if split == 'val':
            if len(self.holdout_leaves) == 0:
                raise ValueError("Cannot generate holdout_only data: no holdout paths available")
            leaf_nodes = self.holdout_leaves
        elif split == 'train':
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
            raise ValueError(
                f"Requested size ({size}) exceeds the number of available {split} paths ({max_paths}). "
                f"Graph has {len(self.train_leaves)} training paths and {len(self.holdout_leaves)} holdout paths."
            )
        
        # Sample leaf nodes uniformly without replacement (ensures unique paths)
        sampled_leaves = random.sample(leaf_nodes, size)
        
        sequences = []
        for leaf in sampled_leaves:
            # Get the path from root to leaf
            path = self.paths_by_leaf[leaf]
            
            # Construct sequence with or without task prefix token
            pause_tokens = [self.pause_token] * num_pause_tokens
            if use_task_tokens:
                # With task token: [<PATH>, leaf, <PAUSE>, ..., <PAUSE>, root, n_2, ..., n_ℓ]
                sequence = [self.SPECIAL_TOKENS['PATH'], leaf] + pause_tokens + path
            else:
                # Without task token: [leaf, <PAUSE>, ..., <PAUSE>, root, n_2, ..., n_ℓ]
                sequence = [leaf] + pause_tokens + path
            sequences.append(sequence)
        
        # Convert to tensor
        sequences = torch.tensor(sequences, dtype=torch.long)
        
        return sequences
    
    def prepare(self, num_pause_tokens=1, output_dir='./data', 
                use_undirected=True, use_directional_tokens=True, use_task_tokens=True, predict_direction_for_edge_task=True):
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
            use_task_tokens: If True, include PATH and EDGE task prefix tokens in sequences (default: True)
            predict_direction_for_edge_task: If True, the EDGE task will be made to predict the direction LT or GT rather than edge
        """
        # Safety: predicting direction requires directional tokens to exist.
        if predict_direction_for_edge_task and not use_directional_tokens:
            raise ValueError("Invalid config: predict_direction_for_edge_task=True requires use_directional_tokens=True")

        # Calculate dataset sizes based on graph structure
        num_train_path_samples = len(self.train_leaves)  # Training paths
        num_val_path_samples = len(self.holdout_leaves)  # Validation paths (holdout)
        
        # Calculate edge dataset size
        num_edge_samples = (2 if use_undirected else 1) * self.num_graph_edges
        
        # Validation set: only holdout paths (no edges)
        
        # Create output directory with parameters in name
        dir_name = self._generate_dataset_name(num_pause_tokens, use_undirected, use_directional_tokens, use_task_tokens, predict_direction_for_edge_task)
        full_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"Preparing InWeightsPathStar dataset...")
        print(f"  Parameters: d={self.d}, l={self.l}")
        print(f"  Graph structure:")
        print(f"    Total vertices: {self.num_vertices}")
        print(f"    Total edges: {self.num_graph_edges}")
        print(f"    Total paths (spokes): {self.d}")
        print(f"    Training paths: {num_train_path_samples}")
        print(f"    Validation paths (holdout): {num_val_path_samples}")
        print(f"  Dataset composition:")
        print(f"    Holdout percentage: {self.holdout_percentage}")
        print(f"    Number of pause tokens: {num_pause_tokens}")
        print(f"    Edge samples: {num_edge_samples} ({'undirected' if use_undirected else 'directed'})")
        print(f"    Edge prediction: ({'direction' if predict_direction_for_edge_task else 'edge'})")
        print(f"    Training path samples (original): {num_train_path_samples}")
        print(f"    Validation path samples: {num_val_path_samples}")
        print(f"  Final dataset sizes:")
        print(f"    Path dataset: {num_train_path_samples} (no replication)")
        print(f"    Edge dataset: {num_edge_samples}")
        print(f"    Validation set: {num_val_path_samples} (holdout paths only, no edges)")
        print(f"    2d dimension of training set is : {self.l + num_pause_tokens + 1 + (1 if use_task_tokens else 0)}")
        print(f"  Output directory: {full_output_dir}")
        print(f"  Pause token: {self.pause_token}")
        print(f"  Pad token: {self.pad_token}")
        print(f"  EDGE token: {self.SPECIAL_TOKENS['EDGE']}")
        print(f"  PATH token: {self.SPECIAL_TOKENS['PATH']}")
        
        # Print paths_by_leaf in a pretty manner
        print(f"\n  Paths by leaf node:")
        for leaf, path in sorted(self.paths_by_leaf.items()):
            path_str = ' -> '.join(map(str, path))
            is_train = leaf in self.train_leaves
            is_holdout = leaf in self.holdout_leaves
            status = "TRAIN" if is_train else ("HOLDOUT" if is_holdout else "UNKNOWN")
            print(f"    Leaf {leaf} [{status}]: {path_str}")
        
        # Generate training set: paths + edges
        # print("\nGenerating training set (training paths + edges)...")
        
        # Generate path sequences for training (uses self.train_leaves)
        train_path_sequences = self._generate_path_prediction_training_set(
            size=num_train_path_samples,
            split='train',
            num_pause_tokens=num_pause_tokens,
            use_task_tokens=use_task_tokens
        )
        
        # Generate edge sequences
        edge_sequences = self._generate_edge_memorization_training_set(
            size=num_edge_samples,
            undirected=use_undirected,
            use_directional_tokens=use_directional_tokens,
            use_task_tokens=use_task_tokens
            , predict_direction_for_edge_task=predict_direction_for_edge_task
        )
        
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
        
        
        # Generate validation set: only holdout paths (no edges)
        print("Generating validation set (holdout paths only, no edges)...")
        val_sequences = self._generate_path_prediction_training_set(
            size=num_val_path_samples,
            split='val',
            num_pause_tokens=num_pause_tokens,
            use_task_tokens=use_task_tokens
        )
        
        # Debug: Print train and val sequences
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        print(f"\nDebug - Path sequences (numpy):")
        print(train_path_sequences.numpy())
        print(f"\nDebug - Edge sequences (numpy):")
        print(edge_sequences.numpy())
        print(f"\nDebug - Val sequences (numpy):")
        print(val_sequences.numpy())
        
        # Save paths and edges separately
        paths_path = os.path.join(full_output_dir, 'paths.bin')
        print(f"\nSaving path data to {paths_path}...")
        paths_data = train_path_sequences.numpy().astype(np.uint16)
        paths_data.tofile(paths_path)
        print(f"  Saved {paths_data.shape[0]} sequences of length {paths_data.shape[1]}")
        
        edges_path = os.path.join(full_output_dir, 'edges.bin')
        print(f"Saving edge data to {edges_path}...")
        edges_data = edge_sequences.numpy().astype(np.uint16)
        edges_data.tofile(edges_path)
        print(f"  Saved {edges_data.shape[0]} sequences of length {edges_data.shape[1]}")
        
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
        if self.randomize_vocab_size:
            vocab_size = self.randomize_vocab_size + self.num_special_tokens 
        else:
            vocab_size = self.num_vertices + self.num_special_tokens
        
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
        
        # Calculate context lengths
        # edge_context_length: number of tokens provided to predict the supervised target on EDGE task.
        # - predict_direction_for_edge_task=True:  [EDGE] u v -> predict direction, so context is (task) + 2
        # - predict_direction_for_edge_task=False: [EDGE] dir u -> predict v, so context is (task) + (dir) + 1
        if predict_direction_for_edge_task:
            edge_context_length = (1 if use_task_tokens else 0) + 2
        else:
            edge_context_length = (1 if use_task_tokens else 0) + (1 if use_directional_tokens else 0) + 1
        # path_context_length = PATH prefix (conditional) + leaf + pause tokens
        path_context_length = (1 if use_task_tokens else 0) + 1 + num_pause_tokens
        
        # Note: path_seq_len was already calculated above (line 424) after padding
        # It represents the full sequence length: path_context_length + l (context + path tokens)
        # Edge sequences are padded to match this length
        
        # Save metadata
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
            'd': self.d,
            'l': self.l,
            'total_vertices': self.num_vertices,
            'total_edges': self.num_graph_edges,
            'pause_token': self.SPECIAL_TOKENS["PAUSE"],
            'pad_token': self.SPECIAL_TOKENS["PAD"],
            'special_tokens': self.SPECIAL_TOKENS,
            'num_pause_tokens': num_pause_tokens,
            'root_vertex': self.v_root,
            'leaf_vertices': self.v_leaf,
            'vertices': self.vertices,
            'holdout_percentage': self.holdout_percentage,
            'train_leaves': self.train_leaves,
            'holdout_leaves': self.holdout_leaves,
            'use_undirected': use_undirected,
            'use_directional_tokens': use_directional_tokens,
            'use_task_tokens': use_task_tokens,
            'edge_context_length': edge_context_length,
            'path_context_length': path_context_length,
            'block_size': path_seq_len - 1,  # Use actual sequence length (path_context_length + l), not just context length TODO: if you want EOS or to use full context, you need to remove the -1.
            'num_train_path_samples': num_train_path_samples,
            'num_val_path_samples': num_val_path_samples,
            'total_edge_size': num_edge_samples,
            'predict_direction_for_edge_task': predict_direction_for_edge_task
        }
        
        meta['PATHS_DATASET_SIZE'] = paths_data.shape[0]
        meta['EDGES_DATASET_SIZE'] = edges_data.shape[0]
        meta['VAL_DATASET_SIZE'] = val_data.shape[0]
        
        meta_path = os.path.join(full_output_dir, 'meta.pkl')
        print(f"\nSaving metadata to {meta_path}...")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"\nDataset preparation complete!")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Total tokens (paths): {paths_data.size}")
        print(f"  Total tokens (edges): {edges_data.size}")
        print(f"  Total tokens (val): {val_data.size}")
        
        return full_output_dir

    def load_dataset(self):
        dataset = self.dir_name
        # Data loading setup
        data_dir = os.path.join('data', dataset)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file not found at {meta_path}")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Load separate paths.bin and edges.bin
        paths_data = np.memmap(os.path.join(data_dir, 'paths.bin'), dtype=np.uint16, mode='r')
        edges_data = np.memmap(os.path.join(data_dir, 'edges.bin'), dtype=np.uint16, mode='r')
        return meta, paths_data, edges_data, val_data

    def _generate_dataset_name(self, num_pause_tokens, use_undirected, use_directional_tokens, use_task_tokens=True, predict_direction_for_edge_task=True):
        """
        Generate dataset directory name matching the naming convention in pathstar.py
        """
        self.num_pause_tokens = num_pause_tokens
        self.use_undirected = use_undirected
        self.use_directional_tokens = use_directional_tokens
        self.use_task_tokens = use_task_tokens
        self.predict_direction_for_edge_task = predict_direction_for_edge_task
        task_token_suffix = "_tt" if use_task_tokens else "_nott"
        predict_edge_or_direction = "_ped" if predict_direction_for_edge_task else "_pet"
        randomize = (f"_v{self.randomize_vocab_size}" if self.randomize_vocab_size else "")
        self.dir_name = f'inweights_pathstar{randomize}{predict_edge_or_direction}_d{self.d}_l{self.l}_p{num_pause_tokens}_{"un" if use_undirected else ""}directed_{"dt" if use_directional_tokens else ""}{task_token_suffix}'
        return self.dir_name

    def _check_dataset_exists(self):
        """
        Check if dataset exists and validate that metadata matches requested parameters.
        
        Returns:
            bool: True if dataset exists and parameters match, False otherwise
        """
        dataset_name = self.dir_name
        data_dir = os.path.join('data', dataset_name)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        val_path = os.path.join(data_dir, 'val.bin')
        edges_path = os.path.join(data_dir, 'edges.bin')
        paths_path = os.path.join(data_dir, 'paths.bin')
        
        # Check if metadata and val files exist
        if not (os.path.exists(data_dir) and os.path.exists(meta_path) and os.path.exists(val_path)):
            return False

        if not (os.path.exists(paths_path) and os.path.exists(edges_path)):
            return False
        
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return False
        
        # Check if all key parameters match
        params_match = (
            meta.get('d') == self.d and
            meta.get('l') == self.l and
            meta.get('num_pause_tokens') == self.num_pause_tokens and
            meta.get('use_undirected') == self.use_undirected and
            meta.get('use_directional_tokens') == self.use_directional_tokens and
            meta.get('use_task_tokens', True) == self.use_task_tokens and  # Default to True for backward compatibility
            meta.get('predict_direction_for_edge_task', True) == self.predict_direction_for_edge_task and  # Default True for backward compatibility
            abs(meta.get('holdout_percentage', 0.0) - self.holdout_percentage) < 1e-6  # Float comparison
        )
        
        if not params_match:
            print(f"Dataset exists but parameters don't match:")
            print(f"  Existing: d={meta.get('d')}, l={meta.get('l')}, pause={meta.get('num_pause_tokens')}, "
                f"undirected={meta.get('use_undirected')}, directional_tokens={meta.get('use_directional_tokens')}, "
                f"task_tokens={meta.get('use_task_tokens', True)}, predict_direction_for_edge_task={meta.get('predict_direction_for_edge_task', True)}, holdout={meta.get('holdout_percentage')}")
            print(f"  Requested: d={self.d}, l={self.l}, pause={self.num_pause_tokens}, "
                f"undirected={self.use_undirected}, directional_tokens={self.use_directional_tokens}, "
                f"task_tokens={self.use_task_tokens}, predict_direction_for_edge_task={self.predict_direction_for_edge_task}, holdout={self.holdout_percentage}")
            print(f"  Will regenerate dataset...")
            return False
        
        return True


    def generate_dataset_if_needed(self, num_pause_tokens, use_undirected, use_directional_tokens, use_task_tokens=True, predict_direction_for_edge_task=True):
        """
        Generate the dataset using InWeightsPathStar if it doesn't exist or parameters don't match.
        """
        # Validate vocab_size
        if self.randomize_vocab_size != 'auto' and self.randomize_vocab_size < self.num_vertices:
            raise ValueError(
                f"vocab_size ({self.randomize_vocab_size}) must be >= d * (l-1) + 1 = {self.num_vertices}"
            )
        if predict_direction_for_edge_task and not use_directional_tokens:
            raise ValueError("Is an invalid config prediction directions requires use_directional_tokens")
        
        # Generate dataset name
        dataset_name = self._generate_dataset_name(num_pause_tokens, use_undirected, use_directional_tokens, use_task_tokens, predict_direction_for_edge_task)
        
        # Check if dataset exists and parameters match
        if self._check_dataset_exists():
            print(f"Dataset '{dataset_name}' exists with matching parameters. Using existing dataset.")
            return dataset_name
        
        # Dataset doesn't exist or needs regeneration
        print(f"\n{'='*80}")
        print(f"Generating dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        # Create InWeightsPathStar generator
        generator = self
        
        # Generate and save dataset
        output_dir = generator.prepare(
            num_pause_tokens=num_pause_tokens,
            output_dir='./data',
            use_undirected=use_undirected,
            use_directional_tokens=use_directional_tokens,
            use_task_tokens=use_task_tokens,
            predict_direction_for_edge_task=predict_direction_for_edge_task
        )
        
        print(f"\n{'='*80}")
        print(f"Dataset generation complete: {output_dir}")
        print(f"{'='*80}\n")
        
        return dataset_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PathStar datasets')
    parser.add_argument('--d', type=int, default=100,
                        help='Number of spokes/paths in the path-star (default: 100)')
    parser.add_argument('--l', type=int, default=5,
                        help='Length of each path (default: 5)')
    parser.add_argument('--randomize_vocab_size', type=str, default=None,
                        help='Vocabulary size to randomize on, "auto" will set it based on d and l. (default: None)')
    parser.add_argument('--num_pause_tokens', type=int, default=1,
                        help='Number of PAUSE tokens used (default: 1)')
    parser.add_argument('--use_directional_tokens', action='store_true',
                        help='Use directional tokens (> and <)')
    parser.add_argument('--use_task_tokens', action='store_true',
                        help='Use task prefix tokens (PATH and EDGE) in sequences (default: False)')
    parser.add_argument('--use_directed', action='store_true',
                        help='Use directed edges for inweights mode (default: undirected)')
    parser.add_argument('--holdout_percentage', type=float, default=0.2,
                        help='validation split ratio (default: 0.2)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for datasets (default: ./data)')
    
    args = parser.parse_args()
    
    print(f"Generating InWeightsPathStar dataset...")
    
    # Create randomized mapping from canonical node IDs to vocabulary tokens
    num_vertices = args.d * (args.l - 1) + 1

    if args.randomize_vocab_size and args.randomize_vocab_size != "auto":
        args.randomize_vocab_size = int(args.randomize_vocab_size)
    
    generator = InWeightsPathStar(
        d=args.d, 
        l=args.l, 
        holdout_percentage=args.holdout_percentage,
        randomize_vocab_size=args.randomize_vocab_size,
    )
    
    generator.prepare(
        num_pause_tokens=args.num_pause_tokens,
        output_dir=args.output_dir,
        use_undirected=not args.use_directed,
        use_directional_tokens=args.use_directional_tokens,
        use_task_tokens=args.use_task_tokens,
    )
