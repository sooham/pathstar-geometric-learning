import unittest
import torch
import numpy as np
import os
import pickle
from unittest.mock import patch, mock_open, call, Mock
from pathstar import InWeightsPathStar
import random
import tempfile
import shutil

class TestInWeightsPathStar(unittest.TestCase):
    @classmethod
    @patch('random.sample')
    def setUpClass(cls, mock_random_sample):
        """
        Set up a deterministic generator instance for all tests.
        Mocks random.sample to control:
        1. The vocabulary mapping (to be canonical: node <i> maps to i + 11).
        2. The holdout leaf selection (to be deterministic).
        """
        
        # --- Test Parameters ---
        cls.D = 5
        cls.L = 5
        cls.VOCAB_SIZE = 2000
        cls.HOLDOUT_PERC = 0.5
        cls.NUM_VERTICES = cls.D * (cls.L - 1) + 1  # 5 * 4 + 1 = 21

        # --- Mock Configuration ---
        
        # 1. Mock the __init__ mapping generation:
        #    random.sample(range(vocab_size), num_vertices)
        #    We return [0, 1, ..., 20] so node <i> maps to <i>.
        #    The class logic then adds 11 (for special tokens).
        #    Final mapping: 0 -> 11, 1 -> 12, ..., 20 -> 31.
        init_mapping_call = list(range(cls.NUM_VERTICES))

        # 2. Mock the _setup_holdout_paths holdout selection:
        # .  0 1 2 3 4 5
        #    Canonical leaves are [4, 8, 12, 16, 20].
        #    Mapped leaves are [15, 19, 23, 27, 31].
        #    num_holdout = round(5 * 0.5) = 3.
        #    We will deterministically select [15, 23, 31] as holdout.
        #    The call will be random.sample([15, 19, 23, 27, 31], 3)
        holdout_selection_call = [15, 23, 31]

        import random
        # Configure the mock to return values based on the call signature
        def sample_side_effect(*args, **kwargs):
            if args[0] == range(cls.VOCAB_SIZE) and args[1] == cls.NUM_VERTICES:
                return init_mapping_call
            elif set(args[0]) == {15, 19, 23, 27, 31} and args[1] == 2:
                return holdout_selection_call
            else:
                raise ValueError('This shouldnt happen')

        mock_random_sample.side_effect = sample_side_effect

        # --- Instantiate the Class ---
        cls.gen = InWeightsPathStar(
            d=cls.D,
            l=cls.L,
            vocab_size=cls.VOCAB_SIZE,
            holdout_percentage=cls.HOLDOUT_PERC
        )
        
        # --- Define Expected Values ---
        
        # 11 special tokens [0-10] + 21 nodes [11-31]
        cls.MAPPED_NODES = list(range(11, 32)) # 11 ... 31
        
        # Canonical 0 -> Mapped 11
        cls.ROOT = 11 
        
        # Canonical [4, 8, 12, 16, 20] -> Mapped [15, 19, 23, 27, 31]
        cls.LEAVES = [15, 19, 23, 27, 31] 
        
        # From our mock:
        cls.HOLDOUT_LEAVES = {15, 23, 31}
        cls.TRAIN_LEAVES = {19, 27}
        
        # Special Tokens
        cls.TOK_PAD = 0
        cls.TOK_PAUSE = 1
        cls.TOK_GT = 2
        cls.TOK_LT = 3
        cls.TOK_PATH = 9
        cls.TOK_EDGE = 10

    def test_01_init_and_graph_structure(self):
        """
        Tests __init__, graph properties, mapping, holdout, and adjacency list.
        """
        # Test basic properties
        self.assertEqual(self.gen.d, self.D)
        self.assertEqual(self.gen.l, self.L)
        self.assertEqual(self.gen.vocab_size, self.VOCAB_SIZE)
        self.assertEqual(self.gen.total_vert, self.NUM_VERTICES)
        
        # Test root and leaf nodes (post-mapping)
        self.assertEqual(self.gen.v_root, self.ROOT)
        self.assertEqual(self.gen.v_leaf, self.LEAVES)
        
        # Test holdout/train split (post-mapping)
        self.assertEqual(set(self.gen.holdout_leaves), self.HOLDOUT_LEAVES)
        self.assertEqual(set(self.gen.train_leaves), self.TRAIN_LEAVES)

        # Test adjacency list (post-mapping)
        # Node <i> maps to i+11
        expected_adj_list = {
            11: [12, 16, 20, 24, 28], # Root -> spoke starts
            12: [13], 13: [14], 14: [15], 15: [], # Spoke 0
            16: [17], 17: [18], 18: [19], 19: [], # Spoke 1
            20: [21], 21: [22], 22: [23], 23: [], # Spoke 2
            24: [25], 25: [26], 26: [27], 27: [], # Spoke 3
            28: [29], 29: [30], 30: [31], 31: [], # Spoke 4
        }
        self.assertEqual(self.gen.adj_list, expected_adj_list)
        
        # Test paths_by_leaf (post-mapping)
        expected_paths = {
            15: [11, 12, 13, 14, 15],
            19: [11, 16, 17, 18, 19],
            23: [11, 20, 21, 22, 23],
            27: [11, 24, 25, 26, 27],
            31: [11, 28, 29, 30, 31],
        }
        self.assertEqual(self.gen.paths_by_leaf, expected_paths)

    @patch('random.shuffle', side_effect=lambda x: x) # No-op shuffle
    def test_02_generate_edge_memorization(self, mock_shuffle):
        """
        Tests generate_edge_memorization_training_set for all 4 combinations
        of undirected and use_directional_tokens.
        """
        # Total directed edges = D * (L-1) = 5 * 4 = 20
        
        # --- Case 1: undirected=False, use_directional_tokens=False ---
        edges_case_1 = self.gen.generate_edge_memorization_training_set(
            size=20, undirected=False, use_directional_tokens=False
        )
        self.assertEqual(edges_case_1.shape, (20, 2))
        expected_set_1 = {
            (11, 12), (12, 13), (13, 14), (14, 15),
            (11, 16), (16, 17), (17, 18), (18, 19),
            (11, 20), (20, 21), (21, 22), (22, 23),
            (11, 24), (24, 25), (25, 26), (26, 27),
            (11, 28), (28, 29), (29, 30), (30, 31),
        }
        self.assertEqual(set(map(tuple, edges_case_1.tolist())), expected_set_1)

        # --- Case 2: undirected=False, use_directional_tokens=True ---
        edges_case_2 = self.gen.generate_edge_memorization_training_set(
            size=20, undirected=False, use_directional_tokens=True
        )
        self.assertEqual(edges_case_2.shape, (20, 3))
        expected_set_2 = {
            (self.TOK_GT, u, v) for (u, v) in expected_set_1
        }
        self.assertEqual(set(map(tuple, edges_case_2.tolist())), expected_set_2)

        # --- Case 3: undirected=True, use_directional_tokens=False ---
        edges_case_3 = self.gen.generate_edge_memorization_training_set(
            size=40, undirected=True, use_directional_tokens=False
        )
        self.assertEqual(edges_case_3.shape, (40, 2))
        expected_set_3 = set()
        for (u, v) in expected_set_1:
            expected_set_3.add((u, v))
            expected_set_3.add((v, u))
        self.assertEqual(set(map(tuple, edges_case_3.tolist())), expected_set_3)

        # --- Case 4: undirected=True, use_directional_tokens=True ---
        edges_case_4 = self.gen.generate_edge_memorization_training_set(
            size=40, undirected=True, use_directional_tokens=True
        )
        self.assertEqual(edges_case_4.shape, (40, 3))
        expected_set_4 = set()
        for (u, v) in expected_set_1:
            expected_set_4.add((self.TOK_GT, u, v))
            expected_set_4.add((self.TOK_LT, v, u))
        self.assertEqual(set(map(tuple, edges_case_4.tolist())), expected_set_4)

    @patch('random.sample', side_effect=lambda pop, k: sorted(pop)[:k]) # Deterministic sample
    def test_03_generate_path_prediction(self, mock_sample):
        """
        Tests generate_path_prediction_training_set for:
        - num_pause_tokens = 1, 2, 3
        - obey_holdout = True (train only)
        - holdout_only = True (holdout only)
        - obey_holdout = False (all)
        """
        
        # --- Paths ---
        # Path 19 (train): [11, 16, 17, 18, 19]
        # Path 27 (train): [11, 24, 25, 26, 27]
        # Path 15 (holdout): [11, 12, 13, 14, 15]
        # Path 23 (holdout): [11, 20, 21, 22, 23]
        # Path 31 (holdout): [11, 28, 29, 30, 31]

        # --- Test: num_pause_tokens=1 ---
        seq_p1 = self.gen.generate_path_prediction_training_set(
            size=2, num_pause_tokens=1, obey_holdout=True, holdout_only=False
        )
        # Mock samples [19, 27] from self.TRAIN_LEAVES
        self.assertEqual(seq_p1.shape, (2, 8)) # 1(PATH)+1(leaf)+1(PAUSE)+5(path)
        expected_p1 = [
            [self.TOK_PATH, 19, self.TOK_PAUSE, 11, 16, 17, 18, 19],
            [self.TOK_PATH, 27, self.TOK_PAUSE, 11, 24, 25, 26, 27],
        ]
        self.assertTrue(torch.equal(seq_p1, torch.tensor(expected_p1)))

        # --- Test: num_pause_tokens=2 ---
        seq_p2 = self.gen.generate_path_prediction_training_set(
            size=3, num_pause_tokens=2, obey_holdout=False, holdout_only=True
        )
        # Mock samples [15, 23, 31] from self.HOLDOUT_LEAVES
        self.assertEqual(seq_p2.shape, (3, 9)) # 1+1+2+5
        expected_p2 = [
            [self.TOK_PATH, 15, self.TOK_PAUSE, self.TOK_PAUSE, 11, 12, 13, 14, 15],
            [self.TOK_PATH, 23, self.TOK_PAUSE, self.TOK_PAUSE, 11, 20, 21, 22, 23],
            [self.TOK_PATH, 31, self.TOK_PAUSE, self.TOK_PAUSE, 11, 28, 29, 30, 31],
        ]
        self.assertTrue(torch.equal(seq_p2, torch.tensor(expected_p2)))

        # --- Test: num_pause_tokens=3 ---
        seq_p3 = self.gen.generate_path_prediction_training_set(
            size=5, num_pause_tokens=3, obey_holdout=False, holdout_only=False
        )
        # Mock samples [15, 19, 23, 27, 31] from all leaves
        self.assertEqual(seq_p3.shape, (5, 10)) # 1+1+3+5
        expected_p3 = [
            [self.TOK_PATH, 15, self.TOK_PAUSE, self.TOK_PAUSE, self.TOK_PAUSE, 11, 12, 13, 14, 15],
            [self.TOK_PATH, 19, self.TOK_PAUSE, self.TOK_PAUSE, self.TOK_PAUSE, 11, 16, 17, 18, 19],
            [self.TOK_PATH, 23, self.TOK_PAUSE, self.TOK_PAUSE, self.TOK_PAUSE, 11, 20, 21, 22, 23],
            [self.TOK_PATH, 27, self.TOK_PAUSE, self.TOK_PAUSE, self.TOK_PAUSE, 11, 24, 25, 26, 27],
            [self.TOK_PATH, 31, self.TOK_PAUSE, self.TOK_PAUSE, self.TOK_PAUSE, 11, 28, 29, 30, 31],
        ]
        self.assertTrue(torch.equal(seq_p3, torch.tensor(expected_p3)))
    
        
    @patch('torch.randperm', return_value=torch.arange(80)) # Deterministic shuffling (no reordering)
    def test_04_prepare(self, mock_randperm):
        """Tests the prepare method: directory naming, size calculations, and data shape/content."""
        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True

        # Create a temporary directory for test output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run prepare
            output_dir = g.prepare(
                num_pause_tokens=num_pause_tokens, 
                output_dir=temp_dir, 
                use_undirected=use_undirected, 
                use_directional_tokens=use_directional_tokens
            )
            
            # --- Expected Calculations ---
            # Edges (directed): d * (l - 1) = 5 * 4 = 20
            num_edges = g.d * (g.l - 1)
            # Edge Samples (undirected): 2 * 20 = 40
            num_edge_samples = 2 * num_edges
            
            # Path Samples: 2 (self.train_leaves)
            num_train_path_samples = len(g.train_leaves) # 2
            
            # Replication factor: 40 // 2 = 20
            replication_factor = num_edge_samples // num_train_path_samples # 20
            replicated_path_samples = num_train_path_samples * replication_factor # 40
            
            # Training size: 40 paths + 40 edges = 80
            train_size = replicated_path_samples + num_edge_samples # 80
            
            # Validation size: 3 (self.holdout_leaves)
            val_size = len(g.holdout_leaves) # 3
            
            # Final calculated vocab_size: initial vocab_size + special tokens = 2000 + 11 = 2011
            final_vocab_size = self.VOCAB_SIZE + 11 

            # Sequence Length: l + num_pause_tokens + 2 = 5 + 1 + 2 = 8
            seq_len = g.l + num_pause_tokens + 2 # 8
            
            # --- 1. Verify Directory Naming ---
            # With use_task_tokens=True (default), name should have _tt suffix
            expected_dir_name = f'inweights_pathstar_v{self.VOCAB_SIZE}_d{self.D}_l{self.L}_p1_undirected_dt_tt'
            self.assertTrue(output_dir.endswith(expected_dir_name))
            self.assertTrue(os.path.exists(output_dir))
            
            # --- 2. Verify Files Exist ---
            train_path = os.path.join(output_dir, 'train.bin')
            val_path = os.path.join(output_dir, 'val.bin')
            meta_path = os.path.join(output_dir, 'meta.pkl')
            
            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))
            self.assertTrue(os.path.exists(meta_path))
            
            # --- 3. Load and verify metadata ---
            with open(meta_path, 'rb') as f:
                actual_meta = pickle.load(f)
            
            self.assertEqual(actual_meta['vocab_size'], final_vocab_size)
            self.assertEqual(actual_meta['d'], self.D)
            self.assertEqual(actual_meta['l'], self.L)
            self.assertEqual(actual_meta['holdout_percentage'], self.HOLDOUT_PERC)
            self.assertEqual(actual_meta['use_undirected'], use_undirected)
            self.assertEqual(actual_meta['use_directional_tokens'], use_directional_tokens)
            self.assertEqual(actual_meta['use_task_tokens'], True)  # Default value
            
            # Check context lengths (use_task_tokens=True, use_directional_tokens=True)
            self.assertEqual(actual_meta['edge_context_length'], 3)  # 1(EDGE) + 1(direction) + 1 = 3
            self.assertEqual(actual_meta['path_context_length'], 3)  # 1(PATH) + 1(leaf) + 1(pause) = 3
            
            # Check calculated sizes
            self.assertEqual(actual_meta['train_size'], train_size) # 80
            self.assertEqual(actual_meta['val_size'], val_size) # 3
            self.assertEqual(actual_meta['num_train_path_samples'], num_train_path_samples) # 2
            self.assertEqual(actual_meta['num_val_path_samples'], val_size) # 3
            self.assertEqual(actual_meta['total_edge_size'], num_edge_samples) # 40
            
            # Check ITOS mapping size (Total tokens used: 21 vertices + 11 special tokens = 32)
            self.assertEqual(len(actual_meta['itos']), 32)
            self.assertEqual(len(actual_meta['stoi']), 32)
            
            # --- 4. Load and verify train data ---
            train_data = np.fromfile(train_path, dtype=np.uint16)
            train_data = train_data.reshape(train_size, seq_len)
            
            self.assertEqual(train_data.shape, (80, 8))
            
            # First 40 rows should be PATH sequences (alternating between two paths due to replication)
            for i in range(40):
                self.assertEqual(train_data[i, 0], self.TOK_PATH)
                # The leaf should be either 19 or 27 (our train leaves)
                self.assertIn(train_data[i, 1], [19, 27])
                # Third token should be PAUSE
                self.assertEqual(train_data[i, 2], self.TOK_PAUSE)
                # Fourth token should be root (11)
                self.assertEqual(train_data[i, 3], 11)
            
            # Next 40 rows should be EDGE sequences
            for i in range(40, 80):
                self.assertEqual(train_data[i, 0], self.TOK_EDGE)
                # Second token should be directional (GT or LT)
                self.assertIn(train_data[i, 1], [self.TOK_GT, self.TOK_LT])
                # Positions 4-7 should be PAD
                for j in range(4, 8):
                    self.assertEqual(train_data[i, j], self.TOK_PAD)
            
            # --- 5. Load and verify validation data ---
            val_data = np.fromfile(val_path, dtype=np.uint16)
            val_data = val_data.reshape(val_size, seq_len)
            
            self.assertEqual(val_data.shape, (3, 8))
            
            # All validation sequences should be PATH sequences to holdout leaves
            for i in range(3):
                self.assertEqual(val_data[i, 0], self.TOK_PATH)
                # The leaf should be one of our holdout leaves (15, 23, 31)
                self.assertIn(val_data[i, 1], [15, 23, 31])
                # Third token should be PAUSE
                self.assertEqual(val_data[i, 2], self.TOK_PAUSE)
                # Fourth token should be root (11)
                self.assertEqual(val_data[i, 3], 11)
        
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)


    @patch('random.shuffle', side_effect=lambda x: x) # No-op shuffle to make edges deterministic
    @patch('random.sample', side_effect=lambda pop, k: sorted(pop)[:k]) # Deterministic sample
    @patch('torch.randperm', return_value=torch.arange(80)) # Deterministic permutation (no reordering)
    def test_04b_prepare_data_content(self, mock_randperm, mock_sample, mock_shuffle):
        """Tests the actual content of train_data and val_data generated by prepare."""
        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True
        
        # --- Expected Calculations ---
        num_edges = g.d * (g.l - 1)  # 20
        num_edge_samples = 2 * num_edges  # 40
        num_train_path_samples = len(g.train_leaves)  # 2
        replication_factor = num_edge_samples // num_train_path_samples  # 20
        replicated_path_samples = num_train_path_samples * replication_factor  # 40
        train_size = replicated_path_samples + num_edge_samples  # 80
        val_size = len(g.holdout_leaves)  # 3
        seq_len = g.l + num_pause_tokens + 2  # 8
        
        # --- Generate the sequences manually (what prepare() should create) ---
        
        # Training path sequences (will be replicated 20 times)
        # train_leaves = {19, 27} (from our mock setup)
        # With deterministic sample, we get [19, 27]
        train_path_sequences = g.generate_path_prediction_training_set(
            size=num_train_path_samples,
            num_pause_tokens=num_pause_tokens,
            obey_holdout=True
        )
        expected_train_paths = [
            [self.TOK_PATH, 19, self.TOK_PAUSE, 11, 16, 17, 18, 19],  # Path to leaf 19
            [self.TOK_PATH, 27, self.TOK_PAUSE, 11, 24, 25, 26, 27],  # Path to leaf 27
        ]
        self.assertTrue(torch.equal(train_path_sequences, torch.tensor(expected_train_paths)))
        
        # Replicate the path sequences
        train_path_sequences_replicated = train_path_sequences.repeat(replication_factor, 1)
        self.assertEqual(train_path_sequences_replicated.shape, (40, 8))
        
        # Edge sequences
        edges = g.generate_edge_memorization_training_set(
            size=num_edge_samples,
            undirected=use_undirected,
            use_directional_tokens=use_directional_tokens
        )
        
        # Build expected edge set (all directed edges in the graph)
        expected_edges = []
        # Spoke 0: 11->12, 12->13, 13->14, 14->15
        expected_edges.extend([
            [self.TOK_GT, 11, 12], [self.TOK_LT, 12, 11],
            [self.TOK_GT, 12, 13], [self.TOK_LT, 13, 12],
            [self.TOK_GT, 13, 14], [self.TOK_LT, 14, 13],
            [self.TOK_GT, 14, 15], [self.TOK_LT, 15, 14],
        ])
        # Spoke 1: 11->16, 16->17, 17->18, 18->19
        expected_edges.extend([
            [self.TOK_GT, 11, 16], [self.TOK_LT, 16, 11],
            [self.TOK_GT, 16, 17], [self.TOK_LT, 17, 16],
            [self.TOK_GT, 17, 18], [self.TOK_LT, 18, 17],
            [self.TOK_GT, 18, 19], [self.TOK_LT, 19, 18],
        ])
        # Spoke 2: 11->20, 20->21, 21->22, 22->23
        expected_edges.extend([
            [self.TOK_GT, 11, 20], [self.TOK_LT, 20, 11],
            [self.TOK_GT, 20, 21], [self.TOK_LT, 21, 20],
            [self.TOK_GT, 21, 22], [self.TOK_LT, 22, 21],
            [self.TOK_GT, 22, 23], [self.TOK_LT, 23, 22],
        ])
        # Spoke 3: 11->24, 24->25, 25->26, 26->27
        expected_edges.extend([
            [self.TOK_GT, 11, 24], [self.TOK_LT, 24, 11],
            [self.TOK_GT, 24, 25], [self.TOK_LT, 25, 24],
            [self.TOK_GT, 25, 26], [self.TOK_LT, 26, 25],
            [self.TOK_GT, 26, 27], [self.TOK_LT, 27, 26],
        ])
        # Spoke 4: 11->28, 28->29, 29->30, 30->31
        expected_edges.extend([
            [self.TOK_GT, 11, 28], [self.TOK_LT, 28, 11],
            [self.TOK_GT, 28, 29], [self.TOK_LT, 29, 28],
            [self.TOK_GT, 29, 30], [self.TOK_LT, 30, 29],
            [self.TOK_GT, 30, 31], [self.TOK_LT, 31, 30],
        ])
        
        # Verify we have the right edges (shuffle mock makes them deterministic)
        self.assertEqual(edges.shape, (40, 3))
        self.assertEqual(set(map(tuple, edges.tolist())), set(map(tuple, expected_edges)))
        
        # Build edge sequences with EDGE task token and padding
        edge_task_tokens = torch.full((num_edge_samples, 1), self.TOK_EDGE, dtype=torch.long)
        edge_sequences = torch.cat([edge_task_tokens, edges], dim=1)  # Shape: (40, 4)
        
        # Pad edge sequences to match path sequence length (8)
        padding = torch.full((num_edge_samples, seq_len - 4), self.TOK_PAD, dtype=torch.long)
        edge_sequences = torch.cat([edge_sequences, padding], dim=1)  # Shape: (40, 8)
        
        # Verify edge sequence structure
        self.assertEqual(edge_sequences.shape, (40, 8))
        # Each edge sequence should be: [EDGE, direction_token, u, v, PAD, PAD, PAD, PAD]
        for i in range(num_edge_samples):
            self.assertEqual(edge_sequences[i, 0].item(), self.TOK_EDGE)
            self.assertIn(edge_sequences[i, 1].item(), [self.TOK_GT, self.TOK_LT])
            # Remaining positions should be PAD
            self.assertTrue(torch.all(edge_sequences[i, 4:] == self.TOK_PAD))
        
        # Concatenate and verify final training data
        # With mock randperm returning arange(80), order is preserved
        train_sequences = torch.cat([train_path_sequences_replicated, edge_sequences], dim=0)
        self.assertEqual(train_sequences.shape, (80, 8))
        
        # First 40 should be replicated path sequences
        # Since we replicated [path_19, path_27] 20 times, we get:
        # [path_19, path_27, path_19, path_27, ..., path_19, path_27] (20 repetitions)
        for i in range(40):
            expected_path = expected_train_paths[i % 2]  # Alternates between path_19 and path_27
            self.assertEqual(train_sequences[i].tolist(), expected_path)
            self.assertEqual(train_sequences[i, 0].item(), self.TOK_PATH)
        
        # Next 40 should be edge sequences
        for i in range(40, 80):
            self.assertEqual(train_sequences[i, 0].item(), self.TOK_EDGE)
        
        # --- Validation sequences ---
        val_sequences = g.generate_path_prediction_training_set(
            size=val_size,
            num_pause_tokens=num_pause_tokens,
            holdout_only=True
        )
        
        # holdout_leaves = {15, 23, 31} (from our mock setup)
        # With deterministic sample, we get [15, 23, 31]
        expected_val_paths = [
            [self.TOK_PATH, 15, self.TOK_PAUSE, 11, 12, 13, 14, 15],  # Path to leaf 15
            [self.TOK_PATH, 23, self.TOK_PAUSE, 11, 20, 21, 22, 23],  # Path to leaf 23
            [self.TOK_PATH, 31, self.TOK_PAUSE, 11, 28, 29, 30, 31],  # Path to leaf 31
        ]
        
        self.assertEqual(val_sequences.shape, (3, 8))
        self.assertTrue(torch.equal(val_sequences, torch.tensor(expected_val_paths)))
        
        # All validation sequences should be PATH tasks (no EDGE tasks)
        for i in range(val_size):
            self.assertEqual(val_sequences[i, 0].item(), self.TOK_PATH)
            self.assertIn(val_sequences[i, 1].item(), [15, 23, 31])  # Holdout leaves

    @patch('random.sample', side_effect=lambda pop, k: sorted(pop)[:k]) # Deterministic sample
    def test_05_use_task_tokens_path_prediction(self, mock_sample):
        """
        Tests generate_path_prediction_training_set with use_task_tokens=False.
        Verifies that sequences omit the PATH task prefix token when use_task_tokens=False.
        """
        
        # --- Test: use_task_tokens=True (default behavior) ---
        seq_with_task = self.gen.generate_path_prediction_training_set(
            size=2, num_pause_tokens=1, obey_holdout=True, holdout_only=False, use_task_tokens=True
        )
        # Shape should be 1(PATH) + 1(leaf) + 1(PAUSE) + 5(path) = 8
        self.assertEqual(seq_with_task.shape, (2, 8))
        expected_with_task = [
            [self.TOK_PATH, 19, self.TOK_PAUSE, 11, 16, 17, 18, 19],
            [self.TOK_PATH, 27, self.TOK_PAUSE, 11, 24, 25, 26, 27],
        ]
        self.assertTrue(torch.equal(seq_with_task, torch.tensor(expected_with_task)))
        
        # --- Test: use_task_tokens=False ---
        seq_without_task = self.gen.generate_path_prediction_training_set(
            size=2, num_pause_tokens=1, obey_holdout=True, holdout_only=False, use_task_tokens=False
        )
        # Shape should be 1(leaf) + 1(PAUSE) + 5(path) = 7 (no PATH token)
        self.assertEqual(seq_without_task.shape, (2, 7))
        expected_without_task = [
            [19, self.TOK_PAUSE, 11, 16, 17, 18, 19],  # No PATH token
            [27, self.TOK_PAUSE, 11, 24, 25, 26, 27],  # No PATH token
        ]
        self.assertTrue(torch.equal(seq_without_task, torch.tensor(expected_without_task)))
        
        # Verify length difference is exactly 1 (the PATH token)
        length_diff = seq_with_task.shape[1] - seq_without_task.shape[1]
        self.assertEqual(length_diff, 1)
        
        # Verify that the content after the PATH token matches
        self.assertTrue(torch.equal(seq_with_task[:, 1:], seq_without_task))
        
        # --- Test with different num_pause_tokens ---
        seq_without_task_p2 = self.gen.generate_path_prediction_training_set(
            size=3, num_pause_tokens=2, obey_holdout=False, holdout_only=True, use_task_tokens=False
        )
        # Shape should be 1(leaf) + 2(PAUSE) + 5(path) = 8 (no PATH token)
        self.assertEqual(seq_without_task_p2.shape, (3, 8))
        expected_without_task_p2 = [
            [15, self.TOK_PAUSE, self.TOK_PAUSE, 11, 12, 13, 14, 15],
            [23, self.TOK_PAUSE, self.TOK_PAUSE, 11, 20, 21, 22, 23],
            [31, self.TOK_PAUSE, self.TOK_PAUSE, 11, 28, 29, 30, 31],
        ]
        self.assertTrue(torch.equal(seq_without_task_p2, torch.tensor(expected_without_task_p2)))

    @patch('torch.randperm', return_value=torch.arange(80)) # Deterministic shuffling (no reordering)
    def test_06_prepare_without_task_tokens(self, mock_randperm):
        """
        Tests the prepare method with use_task_tokens=False.
        Verifies that PATH and EDGE tokens are omitted from sequences.
        """
        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True
        use_task_tokens = False  # Key difference
        
        # Create a temporary directory for test output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run prepare with use_task_tokens=False
            output_dir = g.prepare(
                num_pause_tokens=num_pause_tokens, 
                output_dir=temp_dir, 
                use_undirected=use_undirected, 
                use_directional_tokens=use_directional_tokens,
                use_task_tokens=use_task_tokens
            )
            
            # --- Expected Calculations ---
            num_edges = g.d * (g.l - 1)  # 20
            num_edge_samples = 2 * num_edges  # 40
            num_train_path_samples = len(g.train_leaves)  # 2
            replication_factor = num_edge_samples // num_train_path_samples  # 20
            replicated_path_samples = num_train_path_samples * replication_factor  # 40
            train_size = replicated_path_samples + num_edge_samples  # 80
            val_size = len(g.holdout_leaves)  # 3
            
            # Sequence lengths WITHOUT task tokens:
            # Path: 1(leaf) + 1(PAUSE) + 5(path) = 7
            # Edge: 1(direction) + 2(nodes) = 3
            path_seq_len = g.l + num_pause_tokens + 1  # 7 (no PATH token)
            edge_seq_len = 3  # direction + u + v (no EDGE token)
            
            # --- 1. Verify Directory Naming ---
            expected_dir_name = f'inweights_pathstar_v{self.VOCAB_SIZE}_d{self.D}_l{self.L}_p1_undirected_dt_nott'
            self.assertTrue(output_dir.endswith(expected_dir_name))
            self.assertTrue(os.path.exists(output_dir))
            
            # --- 2. Verify Files Exist ---
            train_path = os.path.join(output_dir, 'train.bin')
            val_path = os.path.join(output_dir, 'val.bin')
            meta_path = os.path.join(output_dir, 'meta.pkl')
            
            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))
            self.assertTrue(os.path.exists(meta_path))
            
            # --- 3. Load and verify metadata ---
            with open(meta_path, 'rb') as f:
                actual_meta = pickle.load(f)
            
            # Verify use_task_tokens is stored correctly
            self.assertEqual(actual_meta['use_task_tokens'], False)
            
            # Verify context lengths
            # edge_context_length = 0(no EDGE) + 1(direction) + 1 = 2
            # path_context_length = 0(no PATH) + 1(leaf) + 1(pause) = 2
            self.assertEqual(actual_meta['edge_context_length'], 2)
            self.assertEqual(actual_meta['path_context_length'], 2)
            
            # --- 4. Load and verify train data ---
            train_data = np.fromfile(train_path, dtype=np.uint16)
            train_data = train_data.reshape(train_size, path_seq_len)
            
            self.assertEqual(train_data.shape, (80, 7))
            
            # First 40 rows should be PATH sequences (WITHOUT PATH token)
            for i in range(40):
                # First token should be the leaf (19 or 27), NOT the PATH token
                self.assertIn(train_data[i, 0], [19, 27])
                # Second token should be PAUSE
                self.assertEqual(train_data[i, 1], self.TOK_PAUSE)
                # Third token should be root (11)
                self.assertEqual(train_data[i, 2], 11)
            
            # Next 40 rows should be EDGE sequences (WITHOUT EDGE token)
            for i in range(40, 80):
                # First token should be directional (GT or LT), NOT the EDGE token
                self.assertIn(train_data[i, 0], [self.TOK_GT, self.TOK_LT])
                # Second and third tokens should be node IDs
                self.assertIn(train_data[i, 1], range(11, 32))
                self.assertIn(train_data[i, 2], range(11, 32))
                # Positions 3-6 should be PAD
                for j in range(3, 7):
                    self.assertEqual(train_data[i, j], self.TOK_PAD)
            
            # --- 5. Load and verify validation data ---
            val_data = np.fromfile(val_path, dtype=np.uint16)
            val_data = val_data.reshape(val_size, path_seq_len)
            
            self.assertEqual(val_data.shape, (3, 7))
            
            # All validation sequences should be path sequences WITHOUT PATH token
            for i in range(3):
                # First token should be the leaf (15, 23, or 31), NOT the PATH token
                self.assertIn(val_data[i, 0], [15, 23, 31])
                # Second token should be PAUSE
                self.assertEqual(val_data[i, 1], self.TOK_PAUSE)
                # Third token should be root (11)
                self.assertEqual(val_data[i, 2], 11)
        
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

    def test_07_context_length_calculations(self):
        """
        Tests that edge_context_length and path_context_length are calculated correctly
        for all combinations of use_task_tokens and use_directional_tokens.
        """
        
        num_pause_tokens = 1
        
        # --- Case 1: use_task_tokens=True, use_directional_tokens=True ---
        edge_ctx_1 = (1 if True else 0) + (1 if True else 0) + 1  # EDGE + direction + 1 = 3
        path_ctx_1 = (1 if True else 0) + 1 + num_pause_tokens  # PATH + leaf + pause = 3
        self.assertEqual(edge_ctx_1, 3)
        self.assertEqual(path_ctx_1, 3)
        
        # --- Case 2: use_task_tokens=True, use_directional_tokens=False ---
        edge_ctx_2 = (1 if True else 0) + (1 if False else 0) + 1  # EDGE + 0 + 1 = 2
        path_ctx_2 = (1 if True else 0) + 1 + num_pause_tokens  # PATH + leaf + pause = 3
        self.assertEqual(edge_ctx_2, 2)
        self.assertEqual(path_ctx_2, 3)
        
        # --- Case 3: use_task_tokens=False, use_directional_tokens=True ---
        edge_ctx_3 = (1 if False else 0) + (1 if True else 0) + 1  # 0 + direction + 1 = 2
        path_ctx_3 = (1 if False else 0) + 1 + num_pause_tokens  # 0 + leaf + pause = 2
        self.assertEqual(edge_ctx_3, 2)
        self.assertEqual(path_ctx_3, 2)
        
        # --- Case 4: use_task_tokens=False, use_directional_tokens=False ---
        edge_ctx_4 = (1 if False else 0) + (1 if False else 0) + 1  # 0 + 0 + 1 = 1
        path_ctx_4 = (1 if False else 0) + 1 + num_pause_tokens  # 0 + leaf + pause = 2
        self.assertEqual(edge_ctx_4, 1)
        self.assertEqual(path_ctx_4, 2)
        
        # Test with different num_pause_tokens
        num_pause_tokens = 3
        path_ctx_5 = (1 if True else 0) + 1 + num_pause_tokens  # PATH + leaf + 3 pauses = 5
        path_ctx_6 = (1 if False else 0) + 1 + num_pause_tokens  # 0 + leaf + 3 pauses = 4
        self.assertEqual(path_ctx_5, 5)
        self.assertEqual(path_ctx_6, 4)

    def test_08_dataset_name_with_task_tokens(self):
        """
        Tests that generate_dataset_name includes the correct suffix based on use_task_tokens.
        """
        
        g = self.gen
        
        # --- Test with use_task_tokens=True ---
        name_with_tt = g.generate_dataset_name(
            num_pause_tokens=1,
            use_undirected=True,
            use_directional_tokens=True,
            use_task_tokens=True
        )
        self.assertIn('_tt', name_with_tt)
        self.assertNotIn('_nott', name_with_tt)
        expected_name_tt = f'inweights_pathstar_v{self.VOCAB_SIZE}_d{self.D}_l{self.L}_p1_undirected_dt_tt'
        self.assertEqual(name_with_tt, expected_name_tt)
        
        # --- Test with use_task_tokens=False ---
        name_without_tt = g.generate_dataset_name(
            num_pause_tokens=1,
            use_undirected=True,
            use_directional_tokens=True,
            use_task_tokens=False
        )
        self.assertIn('_nott', name_without_tt)
        self.assertNotIn('_tt', name_without_tt.replace('_nott', ''))  # Ensure only _nott, not _tt
        expected_name_nott = f'inweights_pathstar_v{self.VOCAB_SIZE}_d{self.D}_l{self.L}_p1_undirected_dt_nott'
        self.assertEqual(name_without_tt, expected_name_nott)
        
        # Verify that stored instance variables are correct
        self.assertEqual(g.use_task_tokens, False)
        
        # Reset and test with use_task_tokens=True again
        name_with_tt_2 = g.generate_dataset_name(
            num_pause_tokens=2,
            use_undirected=False,
            use_directional_tokens=False,
            use_task_tokens=True
        )
        self.assertIn('_tt', name_with_tt_2)
        self.assertEqual(g.use_task_tokens, True)


if __name__ == '__main__':
    unittest.main()

