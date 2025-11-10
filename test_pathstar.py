import unittest
import torch
import numpy as np
import os
import pickle
from unittest.mock import patch, mock_open, call, Mock
from pathstar import InWeightsPathStar
import random

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
    
        
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=os.path.join) # Use real path join logic
    @patch('torch.randperm', return_value=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])) # Deterministic shuffling (no change)
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.Tensor.numpy') # <--- Mocks the problematic NumPy I/O operation
    def test_04_prepare(self, mock_numpy_method, mock_file_open, mock_pickle_dump, mock_randperm, mock_join, mock_makedirs):
        """Tests the prepare method: directory naming, size calculations, and data shape/content."""

        mock_tofile = Mock()
        mock_astype = mock_numpy_method.return_value.astype
        mock_astype.return_value.tofile = mock_tofile

        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True

        # Run prepare (the actual train.bin/val.bin saving is mocked via mock_tofile)
        output_dir = g.prepare(
            num_pause_tokens=num_pause_tokens, 
            output_dir='./data', 
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
        # Directory name should use the initial vocab_size (32) as per the f-string in prepare
        expected_dir = f'./data/inweights_pathstar_v{self.VOCAB_SIZE}_d{self.D}_l{self.L}_p1_undirected_dt'
        self.assertEqual(output_dir, expected_dir)
        mock_makedirs.assert_called_with(expected_dir, exist_ok=True)
        
        # --- 2. Verify train.bin/val.bin data ggshapes and saving ---
        
        # Since we mocked numpy.ndarray.tofile, we check if it was called twice (for train and val)
        self.assertEqual(mock_tofile.call_count, 2)

        # We check the arguments passed to pickle.dump (metadata)
        meta_call_args, meta_call_kwargs = mock_pickle_dump.call_args
        actual_meta = meta_call_args[0]
        
        # --- 3. Verify Metadata (meta.pkl) ---
        
        self.assertEqual(actual_meta['vocab_size'], final_vocab_size)
        self.assertEqual(actual_meta['d'], self.D)
        self.assertEqual(actual_meta['l'], self.L)
        self.assertEqual(actual_meta['holdout_percentage'], self.HOLDOUT_PERC)
        self.assertEqual(actual_meta['use_undirected'], use_undirected)
        self.assertEqual(actual_meta['use_directional_tokens'], use_directional_tokens)
        
        # Check calculated sizes
        self.assertEqual(actual_meta['train_size'], train_size) # 62
        self.assertEqual(actual_meta['val_size'], val_size) # 1
        self.assertEqual(actual_meta['num_train_path_samples'], num_train_path_samples) # 3
        self.assertEqual(actual_meta['num_val_path_samples'], val_size) # 1
        self.assertEqual(actual_meta['num_edge_samples'], num_edge_samples) # 32
        
        # Check ITOS mapping size (Total tokens: 21 vertices + 11 special tokens = 32)
        self.assertEqual(len(actual_meta['itos']), 32)
        self.assertEqual(len(actual_meta['stoi']), 32)
        
        # Verify the file paths used for saving (open() and pickle() calls)
        train_path = os.path.join(expected_dir, 'train.bin')
        val_path = os.path.join(expected_dir, 'val.bin')
        meta_path = os.path.join(expected_dir, 'meta.pkl')
        
