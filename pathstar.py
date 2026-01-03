import random 
import torch
import os
import pickle
import numpy as np
import argparse
import math
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigsh
from filelock import FileLock

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

# idea 2. (TRYING THIS - NOT WORKING WELL)
# we do [EDGE] x_leaf x_leaf-1 LT
# or    [EDGE] x_leaf-1 x_leaf GT
# here we are predicting the direction which is an easier problem
# no dependency betwene x_leaf-1 and x_leaf, but prediction needs to be determinable by embeddings

# idea 3. In larger batchsizes, all the paths need to have their edges related data too 
# otherwise the gradient updates are not stable for embeddings

# idea 4. consoldiate the LT and GT  (not great)
# into one token
# we do [EDGE] x_leaf LT x_leaf-1 
# and   [EDGE] x_leaf-1 GT x_leaf  (remove this)

# idea 5. GT and LT are tokens that are not related semantically
# do [EDGE] x_leaf LT x_leaf-1 LT x_leaf-2 
# do [EDGE] x_leaf LT x_leaf-1 GT x_leaf




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
    
    def _get_path_index_for_edge(self, u, v):
        """
        Determine which path (spoke index) an edge belongs to.
        
        Args:
            u: source node
            v: destination node
        
        Returns:
            path index in [0, d-1]
        """
        # Root is connected to all paths
        if u == self.v_root or v == self.v_root:
            # Determine which spoke based on the non-root vertex
            node = v if u == self.v_root else u
            # Find which path this node belongs to
            for path_idx in range(self.d):
                if node in self.paths_by_leaf[self.v_leaf[path_idx]]:
                    return path_idx
            raise ValueError(f"Node {node} not found in any path")
        
        # Non-root edge: find which path it belongs to
        for path_idx, leaf in enumerate(self.v_leaf):
            path = self.paths_by_leaf[leaf]
            if u in path and v in path:
                return path_idx
        
        raise ValueError(f"Edge ({u}, {v}) not found in any path")
    
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
    
    def compute_laplacian_eigen(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute all eigenvalues and eigenvectors of the Laplacian matrix of the PathStar graph.
        
        The Laplacian matrix is L = D - A, where D is the degree matrix and A is
        the adjacency matrix.
        
        Returns:
            tuple: (eigenvalues, eigenvectors) where:
                - eigenvalues: 1D array of shape (n,) sorted in ascending order
                - eigenvectors: 2D array of shape (n, n) where column i is the 
                  eigenvector corresponding to eigenvalues[i]
        """
        
        # Build undirected graph from adjacency list
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        for u, neighbors in self.adj_list.items():
            for v in neighbors:
                G.add_edge(u, v)
        
        # Get sparse Laplacian matrix, convert to dense for full eigen computation
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L)  # eigh for symmetric matrices
        
        # Sort by eigenvalue (eigh returns sorted, but ensure consistency)
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def compute_fiedler_value(self, use_undirected=True) -> float:
        """
        Compute the Fiedler value (algebraic connectivity) of the PathStar graph.
        
        The Fiedler value is the second-smallest eigenvalue of the Laplacian matrix L = D - A.
        For a connected graph, this value is always positive.
        
        Uses scipy.sparse.linalg.eigsh for efficient computation on sparse symmetric matrices.
        
        Returns:
            float: The Fiedler value (second-smallest Laplacian eigenvalue).
        """
        
        # Build undirected graph from adjacency list
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        for u, neighbors in self.adj_list.items():
            for v in neighbors:
                G.add_edge(u, v)
                if use_undirected:
                    G.add_edge(v, u)

        
        # Get sparse Laplacian matrix
        L = nx.laplacian_matrix(G).astype(float)
        
        # Compute only the 2 smallest eigenvalues using eigsh (sparse symmetric)
        # which='SM' for smallest magnitude eigenvalues
        eigenvalues, _ = eigsh(L, k=2, which='SM')
        return float(np.sort(eigenvalues)[1])
    
    def compute_fiedler_vector(self, use_undirected=True) -> np.ndarray:
        """
        Compute the Fiedler vector of the PathStar graph.
        
        The Fiedler vector is the eigenvector corresponding to the second-smallest 
        eigenvalue of the Laplacian matrix L = D - A.
        
        Uses scipy.sparse.linalg.eigsh for efficient computation on sparse symmetric matrices.
        
        Returns:
            np.ndarray: The Fiedler vector (1D array of shape (n,)).
        """
        
        # Build undirected graph from adjacency list
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        for u, neighbors in self.adj_list.items():
            for v in neighbors:
                G.add_edge(u, v)
                if use_undirected:
                    G.add_edge(v, u)
        
        # Get sparse Laplacian matrix
        L = nx.laplacian_matrix(G).astype(float)
        
        # Compute the 2 smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
        
        # Sort and return the eigenvector for the second-smallest eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvectors[:, idx[1]]
    
    def _generate_edge_memorization_training_set(self, size, undirected=True, use_directional_tokens=True, predict_direction_for_edge_task=True):
        """
        Generate a training set of edges sampled randomly from the path-star graph.
        
        Args:
            size: Number of samples (K) to generate
            undirected: If True, also include reverse edges (y -> x) in the sampling pool
            use_directional_tokens: If true uses GT and LT tokens to show direction
            predict_direction_for_edge_task:
                - If True (predict direction), the format is [EDGE] u v direction
                - If False (predict endpoint), the format is [EDGE] u direction v
        Returns:
            edges: shape (size, 3+A) where A == 1 if use_directional_tokens is true
            edge_path_membership: array of path indices for each edge
        """
        # Track which path each edge belongs to
        edge_path_membership = []
        
        # Collect all edges from the adjacency list
        def add_edge(u, v, path_idx):
            # assumption u is before v from root
            # first edge (u, v)
            if use_directional_tokens:
                if predict_direction_for_edge_task:
                    edges.append([u, v, self.SPECIAL_TOKENS['GT']]) # GT means  away from root
                else:
                    edges.append([u, self.SPECIAL_TOKENS['GT'], v]) # GT means  away from root
            else:
                edges.append([u, v])
            edge_path_membership.append(path_idx)

            if undirected: # add the reverse edge (v, u)
                if use_directional_tokens:
                    if predict_direction_for_edge_task:
                        edges.append([v, u, self.SPECIAL_TOKENS['LT']]) # LT means toward root
                    else:
                        edges.append([v, self.SPECIAL_TOKENS['LT'], u]) # LT means  toward root
                else:
                    edges.append([v, u])
                edge_path_membership.append(path_idx)

        edges = []
        for u in self.adj_list:
            for v in self.adj_list[u]:
                # Determine which path this edge belongs to
                path_idx = self._get_path_index_for_edge(u, v)
                add_edge(u, v, path_idx)
        
        # Validate size
        max_edges = len(edges)
        if size > max_edges:
            raise ValueError(
                f"Requested size ({size}) exceeds the total number of available edges ({max_edges}). "
                f"Graph has {self.total_edges} directed edges"
                + (f" or {2 * self.total_edges} undirected edges." if undirected else ".")
            )
        
        # Shuffle edges and membership together to maintain correspondence
        combined = list(zip(edges, edge_path_membership))
        random.shuffle(combined)
        sampled_combined = combined[:size]
        sampled_edges, sampled_membership = zip(*sampled_combined)
        
        # Return as torch tensor
        edges = torch.tensor(sampled_edges, dtype=torch.long)
        # Convert edge pairs to sequences with EDGE task token:
        # - predict_direction_for_edge_task=False: [<EDGE>, u, <optional GT/LT>, v]
        # - predict_direction_for_edge_task=True:  [<EDGE>, u, v, <GT/LT>]
        edge_task_tokens = torch.full((size, 1), self.SPECIAL_TOKENS['EDGE'], dtype=torch.long)
        edge_sequences = torch.cat([edge_task_tokens, edges], dim=1)
        
        # Convert membership to numpy array
        sampled_membership = np.array(sampled_membership, dtype=np.int32)
        
        return edge_sequences, sampled_membership

    def _generate_path_prediction_training_set(self, size, split, use_directional_tokens_in_path=False):
        """
        Generate a path-finding training set for the in-weights path memorization objective.
        
        NOTE: Pause tokens are NOT included in the stored dataset. They should be added
        at runtime using add_pause_tokens_to_batch() based on the training config.
        
        Each training example has the format (WITHOUT pause tokens):
        Input (default): [<PATH>, leaf, root, n_2, n_3, ..., n_ℓ]
        Input (with use_directional_tokens_in_path=True):
               [<PATH>, leaf, root, GT, n_2, GT, n_3, ..., GT, n_ℓ]
        
        Target: predict each next token left-to-right
        
        Args:
            size: Number of samples (K) to generate
            split: either 'train' (training leaves only), 'val' (holdout leaves) or all (both)
            use_directional_tokens_in_path: If True, interleave GT tokens between path edges (default: False)
        
        Returns:
            sequences: torch tensor containing full sequences (WITHOUT pause tokens).
                      If use_directional_tokens_in_path=False, sequence length is:
                        1 + 1(leaf) + l(path vertices)
                      If use_directional_tokens_in_path=True, sequence length is:
                        1 + 1(leaf) + (2*l - 1)  (root plus GT+vertex for each edge)
            path_membership: numpy array of path indices for each sequence
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
        path_membership = []
        for leaf in sampled_leaves:
            # Get the path from root to leaf
            path = self.paths_by_leaf[leaf]

            # Optionally interleave GT tokens between edges in the path.
            # path is [root, n2, ..., leaf] of length l.
            if use_directional_tokens_in_path:
                expanded_path = [path[0]]
                for node in path[1:]:
                    expanded_path.extend([self.SPECIAL_TOKENS['GT'], node])
                path_tokens = expanded_path
            else:
                path_tokens = path
            
            # Construct sequence with PATH task prefix token (NO pause tokens)
            # Format: [<PATH>, leaf, root, n_2, ..., n_ℓ]
            sequence = [self.SPECIAL_TOKENS['PATH'], leaf] + path_tokens
            sequences.append(sequence)
            
            # Track which path this sequence belongs to
            path_idx = self.v_leaf.index(leaf)
            path_membership.append(path_idx)
        
        # Convert to tensor
        sequences = torch.tensor(sequences, dtype=torch.long)
        path_membership = np.array(path_membership, dtype=np.int32)
        
        return sequences, path_membership
    
    def prepare(self, output_dir='./data',
                use_undirected=True, use_directional_tokens=True,
                predict_direction_for_edge_task=False, use_directional_tokens_in_path=False):
        """
        Prepare and save training and validation datasets to disk for in-weights path-star.
        
        NOTE: Pause tokens are NOT stored in the dataset. They should be added at runtime
        using add_pause_tokens_to_batch() based on the training configuration.
        
        Dataset structure:
        - Training set: All training paths (self.train_leaves) + All edges (mixed and shuffled)
        - Validation set: Only holdout paths (self.holdout_leaves, no edges)
        
        Dataset size is automatically calculated based on graph structure:
        - Number of edges: (l-1) * d
        - Training paths: determined by holdout_percentage (train_leaves)
        - Validation paths: determined by holdout_percentage (holdout_leaves)
        
        Args:
            output_dir: Base directory for output (default: './data')
            use_undirected: If True, use undirected edges (both x->y and y->x) (default: True)
            use_directional_tokens: If True, use special tokens to demarcate edge directions in the edge training set
            predict_direction_for_edge_task: If True, the EDGE task will be made to predict the direction LT or GT rather than edge
            use_directional_tokens_in_path: If True, interleave GT tokens between edges in the PATH task sequences
        """
        # Safety: predicting direction requires directional tokens to exist.
        if predict_direction_for_edge_task and not use_directional_tokens:
            raise ValueError("Invalid config: predict_direction_for_edge_task=True requires use_directional_tokens=True")
        # Safety: PATH interleaving uses GT token, so require directional tokens to be enabled.
        if use_directional_tokens_in_path and not use_directional_tokens:
            raise ValueError("Invalid config: use_directional_tokens_in_path=True requires use_directional_tokens=True")

        # Calculate dataset sizes based on graph structure
        num_train_path_samples = len(self.train_leaves)  # Training paths
        num_val_path_samples = len(self.holdout_leaves)  # Validation paths (holdout)
        
        # Calculate edge dataset size
        num_edge_samples = (2 if use_undirected else 1) * self.num_graph_edges
        
        # Validation set: only holdout paths (no edges)
        
        # Create output directory with parameters in name (NO pause tokens in name)
        dir_name = self._generate_dataset_name(
            use_undirected, use_directional_tokens,
            predict_direction_for_edge_task, use_directional_tokens_in_path
        )
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
        print(f"    Edge samples: {num_edge_samples} ({'undirected' if use_undirected else 'directed'})")
        print(f"    Edge prediction: ({'direction' if predict_direction_for_edge_task else 'edge'})")
        print(f"    Training path samples (original): {num_train_path_samples}")
        print(f"    Validation path samples: {num_val_path_samples}")
        print(f"  Final dataset sizes:")
        print(f"    Path dataset: {num_train_path_samples} (no replication)")
        print(f"    Edge dataset: {num_edge_samples}")
        print(f"    Validation set: {num_val_path_samples} (holdout paths only, no edges)")
        path_suffix_len = (2 * self.l - 1) if use_directional_tokens_in_path else self.l
        # Sequence length WITHOUT pause tokens: PATH token + leaf + path_suffix_len
        print(f"    Sequence length (without pause): {1 + 1 + path_suffix_len}")
        print(f"  Output directory: {full_output_dir}")
        print(f"  Pause token ID: {self.pause_token} (added at runtime)")
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
        # NOTE: No pause tokens - they are added at runtime
        train_path_sequences, train_path_membership = self._generate_path_prediction_training_set(
            size=num_train_path_samples,
            split='train',
            use_directional_tokens_in_path=use_directional_tokens_in_path,
        )
        
        # Generate edge sequences
        edge_sequences, edge_path_membership = self._generate_edge_memorization_training_set(
            size=num_edge_samples,
            undirected=use_undirected,
            use_directional_tokens=use_directional_tokens,
            predict_direction_for_edge_task=predict_direction_for_edge_task,
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
        # NOTE: No pause tokens - they are added at runtime
        print("Generating validation set (holdout paths only, no edges)...")
        val_sequences, val_path_membership = self._generate_path_prediction_training_set(
            size=num_val_path_samples,
            split='val',
            use_directional_tokens_in_path=use_directional_tokens_in_path,
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
        
        # Save path membership arrays
        print(f"\nSaving path membership arrays...")
        np.save(os.path.join(full_output_dir, 'paths_membership.npy'), train_path_membership)
        print(f"  Saved paths_membership.npy: {train_path_membership.shape}")
        np.save(os.path.join(full_output_dir, 'edges_membership.npy'), edge_path_membership)
        print(f"  Saved edges_membership.npy: {edge_path_membership.shape}")
        np.save(os.path.join(full_output_dir, 'val_membership.npy'), val_path_membership)
        print(f"  Saved val_membership.npy: {val_path_membership.shape}")
        
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
        
        # Calculate context lengths (WITHOUT pause tokens - those are added at runtime)
        # edge_context_length: number of tokens provided to predict the supervised target on EDGE task.
        # - predict_direction_for_edge_task=True:  [EDGE] u v -> predict direction, so context is 1 + 2
        # - predict_direction_for_edge_task=False: [EDGE] u (dir) -> predict v, so context is 1 + (dir) + 1
        if predict_direction_for_edge_task:
            edge_context_length = 3  # EDGE + u + v
        else:
            edge_context_length = 2 + (1 if use_directional_tokens else 0) # EDGE + u + (dir optional)
        
        # path_context_length_base = PATH prefix (conditional) + leaf (NO pause tokens)
        # The full path_context_length = path_context_length_base + num_pause_tokens (added at runtime)
        path_context_length_base = 2  # PATH + leaf
        
        # Note: path_seq_len is the stored sequence length (WITHOUT pause tokens).
        # At runtime, the actual sequence length = path_seq_len + num_pause_tokens
        
        # Create bidirectional adjacency list if use_undirected is True
        adj_list_bidirectional = self.adj_list
        if use_undirected:
            adj_list_bidirectional = {}
            # First copy the original directed edges
            for u, neighbors in self.adj_list.items():
                adj_list_bidirectional[u] = list(neighbors)  # Make a copy
            
            # Add reverse edges to make it bidirectional
            for u, neighbors in self.adj_list.items():
                for v in neighbors:
                    if v not in adj_list_bidirectional:
                        adj_list_bidirectional[v] = []
                    if u not in adj_list_bidirectional[v]:
                        adj_list_bidirectional[v].append(u)
        
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

            # NOTE: num_pause_tokens is NOT stored - it's a runtime config parameter
            'root_vertex': self.v_root,
            'leaf_vertices': self.v_leaf,

            'vertices': self.vertices,
            'adj_list': adj_list_bidirectional,  # Full adjacency list (bidirectional if use_undirected=True)
            'holdout_percentage': self.holdout_percentage,
            'train_leaves': self.train_leaves,
            'holdout_leaves': self.holdout_leaves,
            'paths_by_leaf': self.paths_by_leaf,  # Full mapping from leaf to path (after randomization)
            'use_undirected': use_undirected,
            'use_directional_tokens': use_directional_tokens,
            'use_directional_tokens_in_path': use_directional_tokens_in_path,
            # Layout note (important for consumers like train.py tests / disambiguation):
            # - predict_direction_for_edge_task=False: [EDGE] u (GT/LT) v
            # - predict_direction_for_edge_task=True:  [EDGE] u v (GT/LT)
            'edge_task_layout': 'u_dir_v' if (use_directional_tokens and not predict_direction_for_edge_task) else ('u_v_dir' if (use_directional_tokens and predict_direction_for_edge_task) else 'u_v'),
            # PATH layout note (stored WITHOUT pause tokens):
            # - Stored: [PATH] leaf root n2 ... nℓ  (pause tokens added at runtime)
            # - At runtime: [PATH] leaf (PAUSE)xN root n2 ... nℓ
            'path_task_layout': 'root_gt_nodes' if use_directional_tokens_in_path else 'root_nodes',
            # How many tokens the model should generate after the PATH context to reproduce the target suffix.
            'path_target_length': (2 * self.l - 1) if use_directional_tokens_in_path else self.l,
            'edge_context_length': edge_context_length,
            # path_context_length_base does NOT include pause tokens
            'path_context_length_base': path_context_length_base,
            # block_size_base is WITHOUT pause tokens - actual block_size = block_size_base + num_pause_tokens
            # we derive the correct block size based on the full sequence in train.py
            'block_size_base': path_seq_len - 1,
            'num_train_path_samples': num_train_path_samples,
            'num_val_path_samples': num_val_path_samples,
            'total_edge_size': num_edge_samples,
            'predict_direction_for_edge_task': predict_direction_for_edge_task
        }
        # 3 saved datasets
        # 1. paths   (training)
        # 2. edges
        # 3. paths (validation)
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

    def _generate_dataset_name(self, use_undirected, use_directional_tokens,
                               predict_direction_for_edge_task=True, use_directional_tokens_in_path=False):
        """
        Generate dataset directory name matching the naming convention in pathstar.py.
        
        NOTE: num_pause_tokens is NOT part of the dataset name because pause tokens
        are added at runtime, not stored in the dataset.
        """
        self.use_undirected = use_undirected
        self.use_directional_tokens = use_directional_tokens
        self.predict_direction_for_edge_task = predict_direction_for_edge_task
        self.use_directional_tokens_in_path = use_directional_tokens_in_path
        predict_edge_or_direction = "_ped" if predict_direction_for_edge_task else "_pet"
        randomize = (f"_v{self.randomize_vocab_size}" if self.randomize_vocab_size else "")
        # IMPORTANT: bump dataset name when sequence layout changes to avoid silently loading stale .bin files.
        # edge_layout_v2 corresponds to EDGE endpoint prediction layout: [EDGE] u (GT/LT) v
        # v3: pause tokens removed from stored dataset (added at runtime)
        # v4: use_task_tokens is now always True (task tokens always included)
        edge_layout_suffix = "_elv2"
        path_layout_suffix = "_plgt" if use_directional_tokens_in_path else "_plplain"
        # NOTE: No _p{num_pause_tokens} in name - pause tokens are a runtime config
        self.dir_name = f'inweights_pathstar_v4{randomize}{predict_edge_or_direction}{edge_layout_suffix}{path_layout_suffix}_d{self.d}_l{self.l}_{"un" if use_undirected else ""}directed_{"dt" if use_directional_tokens else ""}_tt'
        return self.dir_name

    def _check_dataset_exists(self):
        """
        Check if dataset exists and validate that metadata matches requested parameters.
        
        NOTE: num_pause_tokens is NOT validated because pause tokens are added at runtime,
        not stored in the dataset.
        
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
        
        # Check if all key parameters match (NO num_pause_tokens - it's a runtime config)
        params_match = (
            meta.get('d') == self.d and
            meta.get('l') == self.l and
            # NOTE: num_pause_tokens is NOT checked - it's added at runtime
            meta.get('use_undirected') == self.use_undirected and
            meta.get('use_directional_tokens') == self.use_directional_tokens and
            meta.get('predict_direction_for_edge_task', True) == self.predict_direction_for_edge_task and  # Default True for backward compatibility
            meta.get('use_directional_tokens_in_path', False) == getattr(self, 'use_directional_tokens_in_path', False) and
            # If missing, assume legacy layout; regenerated datasets will include edge_task_layout.
            meta.get('edge_task_layout', None) in (None, 'u_dir_v', 'u_v_dir', 'u_v') and
            abs(meta.get('holdout_percentage', 0.0) - self.holdout_percentage) < 1e-6  # Float comparison
        )
        
        if not params_match:
            print(f"Dataset exists but parameters don't match:")
            print(f"  Existing: d={meta.get('d')}, l={meta.get('l')}, "
                f"undirected={meta.get('use_undirected')}, directional_tokens={meta.get('use_directional_tokens')}, "
                f"predict_direction_for_edge_task={meta.get('predict_direction_for_edge_task', True)}, holdout={meta.get('holdout_percentage')}")
            print(f"  Requested: d={self.d}, l={self.l}, "
                f"undirected={self.use_undirected}, directional_tokens={self.use_directional_tokens}, "
                f"predict_direction_for_edge_task={self.predict_direction_for_edge_task}, holdout={self.holdout_percentage}")
            print(f"  Will regenerate dataset...")
            return False
        
        return True


    def generate_dataset_if_needed(self, use_undirected, use_directional_tokens,
                                   predict_direction_for_edge_task=False, use_directional_tokens_in_path=False):
        """
        Generate the dataset using InWeightsPathStar if it doesn't exist or parameters don't match.
        
        NOTE: num_pause_tokens is NOT a parameter here because pause tokens are added
        at runtime, not stored in the dataset. This allows using the same dataset with
        different pause token counts without regeneration.
        
        Uses file locking to prevent race conditions when multiple GPU processes
        call this method simultaneously.
        """
        # Validate vocab_size
        if self.randomize_vocab_size != 'auto' and self.randomize_vocab_size < self.num_vertices:
            raise ValueError(
                f"vocab_size ({self.randomize_vocab_size}) must be >= d * (l-1) + 1 = {self.num_vertices}"
            )
        if predict_direction_for_edge_task and not use_directional_tokens:
            raise ValueError("Is an invalid config prediction directions requires use_directional_tokens")
        if use_directional_tokens_in_path and not use_directional_tokens:
            raise ValueError("Is an invalid config PATH directions requires use_directional_tokens")
        
        # Generate dataset name (NO num_pause_tokens - it's a runtime config)
        dataset_name = self._generate_dataset_name(
            use_undirected, use_directional_tokens,
            predict_direction_for_edge_task, use_directional_tokens_in_path
        )
        
        # Ensure data directory exists for lock file
        os.makedirs('data', exist_ok=True)
        
        # Use file lock to prevent race condition when multiple GPUs try to generate simultaneously
        lock_path = os.path.join('data', f'{dataset_name}.lock')
        lock = FileLock(lock_path, timeout=3600)  # 1 hour timeout for large datasets
        
        with lock:
            # Re-check if dataset exists AFTER acquiring lock (another process may have created it)
            if self._check_dataset_exists():
                print(f"Dataset '{dataset_name}' exists with matching parameters. Using existing dataset.")
                return dataset_name
            
            # Dataset doesn't exist or needs regeneration
            print(f"\n{'='*80}")
            print(f"Generating dataset: {dataset_name}")
            print(f"{'='*80}\n")
            
            # Create InWeightsPathStar generator
            generator = self
            
            # Generate and save dataset (NO num_pause_tokens)
            output_dir = generator.prepare(
                output_dir='./data',
                use_undirected=use_undirected,
                use_directional_tokens=use_directional_tokens,
                predict_direction_for_edge_task=predict_direction_for_edge_task,
                use_directional_tokens_in_path=use_directional_tokens_in_path,
            )
            
            print(f"\n{'='*80}")
            print(f"Dataset generation complete: {output_dir}")
            print(f"{'='*80}\n")
        
        return dataset_name


def add_pause_tokens_to_batch(batch, num_pause_tokens, pause_token_id):
    """
    Add pause tokens to a batch of PATH sequences at runtime.
    
    This function inserts pause tokens between the leaf and the path (root -> leaf).
    
    Input format (stored in dataset, WITHOUT pause tokens):
        [PATH, leaf, root, n2, ..., leaf]
    
    Output format (after adding pause tokens):
        [PATH, leaf, PAUSE, ..., PAUSE, root, n2, ..., leaf]
    
    Args:
        batch: torch.Tensor of shape (batch_size, seq_len) - sequences WITHOUT pause tokens
        num_pause_tokens: Number of pause tokens to insert
        pause_token_id: The token ID for PAUSE
    
    Returns:
        torch.Tensor of shape (batch_size, seq_len + num_pause_tokens)
    """
    if num_pause_tokens == 0:
        return batch
    
    batch_size, seq_len = batch.shape
    device = batch.device
    dtype = batch.dtype
    
    # Position where pause tokens should be inserted: after [PATH, leaf]
    insert_pos = 2
    
    # Create the pause token tensor
    pause_tokens = torch.full(
        (batch_size, num_pause_tokens), 
        pause_token_id, 
        dtype=dtype, 
        device=device
    )
    
    # Split the batch at the insertion point and concatenate with pause tokens
    prefix = batch[:, :insert_pos]  # [PATH, leaf] or [leaf]
    suffix = batch[:, insert_pos:]  # [root, n2, ..., leaf]
    
    # Concatenate: prefix + pause_tokens + suffix
    result = torch.cat([prefix, pause_tokens, suffix], dim=1)
    
    return result


def add_pause_tokens_to_edges(batch, num_pause_tokens, pad_token_id):
    """
    Pad edge sequences to match the length of path sequences with pause tokens.
    
    Edge sequences don't need pause tokens inserted, but they need to be padded
    to the same length as path sequences (which have pause tokens added).
    
    Args:
        batch: torch.Tensor of shape (batch_size, seq_len) - edge sequences
        num_pause_tokens: Number of pause tokens added to paths (determines padding)
        pad_token_id: The token ID for PAD
    
    Returns:
        torch.Tensor of shape (batch_size, seq_len + num_pause_tokens)
    """
    if num_pause_tokens == 0:
        return batch
    
    batch_size, seq_len = batch.shape
    device = batch.device
    dtype = batch.dtype
    
    # Add padding at the end
    padding = torch.full(
        (batch_size, num_pause_tokens),
        pad_token_id,
        dtype=dtype,
        device=device
    )
    
    return torch.cat([batch, padding], dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PathStar datasets')
    parser.add_argument('--d', type=int, default=100,
                        help='Number of spokes/paths in the path-star (default: 100)')
    parser.add_argument('--l', type=int, default=5,
                        help='Length of each path (default: 5)')
    parser.add_argument('--randomize_vocab_size', type=str, default=None,
                        help='Vocabulary size to randomize on, "auto" will set it based on d and l. (default: None)')
    parser.add_argument('--use_directional_tokens', action='store_true',
                        help='Use directional tokens (> and <)')
    parser.add_argument('--use_directional_tokens_in_path', action='store_true',
                        help='Interleave GT tokens between path edges in PATH task sequences (requires --use_directional_tokens)')
    parser.add_argument('--use_directed', action='store_true',
                        help='Use directed edges for inweights mode (default: undirected)')
    parser.add_argument('--holdout_percentage', type=float, default=0.2,
                        help='validation split ratio (default: 0.2)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for datasets (default: ./data)')
    
    args = parser.parse_args()
    
    print(f"Generating InWeightsPathStar dataset...")
    print(f"NOTE: Pause tokens are NOT stored in dataset - they are added at runtime")
    
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
        output_dir=args.output_dir,
        use_undirected=not args.use_directed,
        use_directional_tokens=args.use_directional_tokens,
        use_directional_tokens_in_path=args.use_directional_tokens_in_path,
    )
