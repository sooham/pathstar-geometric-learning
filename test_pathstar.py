import unittest
import torch
import numpy as np
import os
import pickle
from unittest.mock import patch, mock_open, call, Mock
from pathstar import InWeightsPathStar
import tempfile
import shutil
import math

class TestInWeightsPathStar(unittest.TestCase):
    @classmethod
    @patch('random.sample')
    def setUpClass(cls, mock_random_sample):
        """
        Set up a deterministic generator instance for all tests.
        Mocks random.sample to control:
        1. The vocabulary mapping (to be canonical: node <i> maps to i + 6).
        2. The holdout leaf selection (to be deterministic).
        """
        
        # --- Test Parameters ---
        cls.D = 5
        cls.L = 5
        cls.RANDOMIZE_VOCAB_SIZE = 2000
        cls.HOLDOUT_PERC = 0.4  # Will result in 2 holdout leaves (ceil(5 * 0.4) = 2)
        cls.NUM_VERTICES = cls.D * (cls.L - 1) + 1  # 5 * 4 + 1 = 21

        # --- Mock Configuration ---
        
        # 1. Mock the __init__ mapping generation:
        #    random.sample(range(randomize_vocab_size), num_vertices)
        #    We return [0, 1, ..., 20] so node <i> maps to <i>.
        #    The class logic then adds 6 (for special tokens).
        #    Final mapping: 0 -> 6, 1 -> 7, ..., 20 -> 26.
        init_mapping_call = list(range(cls.NUM_VERTICES))

        # 2. Mock the _setup_holdout_paths holdout selection:
        #    Canonical leaves are [4, 8, 12, 16, 20].
        #    Mapped leaves are [10, 14, 18, 22, 26].
        #    num_holdout = ceil(5 * 0.4) = 2.
        #    We will deterministically select [10, 18] as holdout.
        #    The call will be random.sample([10, 14, 18, 22, 26], 2)
        holdout_selection_call = [10, 18]

        # Configure the mock to return values based on the call signature
        def sample_side_effect(*args, **kwargs):
            if args[0] == range(cls.RANDOMIZE_VOCAB_SIZE) and args[1] == cls.NUM_VERTICES:
                return init_mapping_call
            elif set(args[0]) == {10, 14, 18, 22, 26} and args[1] == 2:
                return holdout_selection_call
            else:
                raise ValueError(f'Unexpected random.sample call: args={args}, kwargs={kwargs}')

        mock_random_sample.side_effect = sample_side_effect

        # --- Instantiate the Class ---
        cls.gen = InWeightsPathStar(
            d=cls.D,
            l=cls.L,
            randomize_vocab_size=cls.RANDOMIZE_VOCAB_SIZE,
            holdout_percentage=cls.HOLDOUT_PERC
        )
        
        # --- Define Expected Values ---
        
        # 6 special tokens [0-5] + 21 nodes [6-26]
        cls.MAPPED_NODES = list(range(6, 27))  # 6 ... 26
        
        # Canonical 0 -> Mapped 6
        cls.ROOT = 6
        
        # Canonical [4, 8, 12, 16, 20] -> Mapped [10, 14, 18, 22, 26]
        cls.LEAVES = [10, 14, 18, 22, 26]
        
        # From our mock:
        cls.HOLDOUT_LEAVES = {10, 18}
        cls.TRAIN_LEAVES = {14, 22, 26}
        
        # Special Tokens
        cls.TOK_PAD = 0
        cls.TOK_PAUSE = 1
        cls.TOK_GT = 2
        cls.TOK_LT = 3
        cls.TOK_PATH = 4
        cls.TOK_EDGE = 5

    def test_01_init_and_graph_structure(self):
        """
        Tests __init__, graph properties, mapping, holdout, and adjacency list.
        """
        # Test basic properties
        self.assertEqual(self.gen.d, self.D)
        self.assertEqual(self.gen.l, self.L)
        self.assertEqual(self.gen.randomize_vocab_size, self.RANDOMIZE_VOCAB_SIZE)
        self.assertEqual(self.gen.num_vertices, self.NUM_VERTICES)
        
        # Test root and leaf nodes (post-mapping)
        self.assertEqual(self.gen.v_root, self.ROOT)
        self.assertEqual(set(self.gen.v_leaf), set(self.LEAVES))
        
        # Test holdout/train split (post-mapping)
        self.assertEqual(set(self.gen.holdout_leaves), self.HOLDOUT_LEAVES)
        self.assertEqual(set(self.gen.train_leaves), self.TRAIN_LEAVES)

        # Test adjacency list (post-mapping)
        # Node <i> maps to i+6
        expected_adj_list = {
            6: [7, 11, 15, 19, 23],  # Root -> spoke starts
            7: [8], 8: [9], 9: [10], 10: [],  # Spoke 0
            11: [12], 12: [13], 13: [14], 14: [],  # Spoke 1
            15: [16], 16: [17], 17: [18], 18: [],  # Spoke 2
            19: [20], 20: [21], 21: [22], 22: [],  # Spoke 3
            23: [24], 24: [25], 25: [26], 26: [],  # Spoke 4
        }
        self.assertEqual(self.gen.adj_list, expected_adj_list)
        
        # Test paths_by_leaf (post-mapping)
        expected_paths = {
            10: [6, 7, 8, 9, 10],
            14: [6, 11, 12, 13, 14],
            18: [6, 15, 16, 17, 18],
            22: [6, 19, 20, 21, 22],
            26: [6, 23, 24, 25, 26],
        }
        self.assertEqual(self.gen.paths_by_leaf, expected_paths)

    @patch('random.shuffle', side_effect=lambda x: x)  # No-op shuffle
    def test_02_generate_edge_memorization(self, mock_shuffle):
        """
        Tests _generate_edge_memorization_training_set for all 4 combinations
        of undirected and use_directional_tokens.
        """
        # Total directed edges = D * (L-1) = 5 * 4 = 20
        
        # --- Case 1: undirected=False, use_directional_tokens=False, use_task_tokens=False ---
        edges_case_1 = self.gen._generate_edge_memorization_training_set(
            size=20, undirected=False, use_directional_tokens=False, use_task_tokens=False
        )
        self.assertEqual(edges_case_1.shape, (20, 2))
        expected_set_1 = {
            (6, 7), (7, 8), (8, 9), (9, 10),
            (6, 11), (11, 12), (12, 13), (13, 14),
            (6, 15), (15, 16), (16, 17), (17, 18),
            (6, 19), (19, 20), (20, 21), (21, 22),
            (6, 23), (23, 24), (24, 25), (25, 26),
        }
        self.assertEqual(set(map(tuple, edges_case_1.tolist())), expected_set_1)

        # --- Case 2: undirected=False, use_directional_tokens=True, use_task_tokens=False ---
        edges_case_2 = self.gen._generate_edge_memorization_training_set(
            size=20, undirected=False, use_directional_tokens=True, use_task_tokens=False
        )
        self.assertEqual(edges_case_2.shape, (20, 3))
        expected_set_2 = {
            # Edge format (no task token, predict endpoint): [u, GT/LT, v]
            (u, self.TOK_GT, v) for (u, v) in expected_set_1
        }
        self.assertEqual(set(map(tuple, edges_case_2.tolist())), expected_set_2)

        # --- Case 3: undirected=True, use_directional_tokens=False, use_task_tokens=False ---
        edges_case_3 = self.gen._generate_edge_memorization_training_set(
            size=40, undirected=True, use_directional_tokens=False, use_task_tokens=False
        )
        self.assertEqual(edges_case_3.shape, (40, 2))
        expected_set_3 = set()
        for (u, v) in expected_set_1:
            expected_set_3.add((u, v))
            expected_set_3.add((v, u))
        self.assertEqual(set(map(tuple, edges_case_3.tolist())), expected_set_3)

        # --- Case 4: undirected=True, use_directional_tokens=True, use_task_tokens=True ---
        edges_case_4 = self.gen._generate_edge_memorization_training_set(
            size=40, undirected=True, use_directional_tokens=True, use_task_tokens=True
        )
        self.assertEqual(edges_case_4.shape, (40, 4))  # EDGE + u + direction + v
        expected_set_4 = set()
        for (u, v) in expected_set_1:
            # Edge format (task token, predict endpoint): [EDGE, u, GT/LT, v]
            expected_set_4.add((self.TOK_EDGE, u, self.TOK_GT, v))
            expected_set_4.add((self.TOK_EDGE, v, self.TOK_LT, u))
        self.assertEqual(set(map(tuple, edges_case_4.tolist())), expected_set_4)

    @patch('random.sample', side_effect=lambda pop, k: sorted(pop)[:k])  # Deterministic sample
    def test_03_generate_path_prediction(self, mock_sample):
        """
        Tests _generate_path_prediction_training_set for:
        - num_pause_tokens = 1, 2, 3
        - split = 'train', 'val', 'all'
        - use_task_tokens = True, False
        """
        
        # --- Paths ---
        # Path 14 (train): [6, 11, 12, 13, 14]
        # Path 22 (train): [6, 19, 20, 21, 22]
        # Path 26 (train): [6, 23, 24, 25, 26]
        # Path 10 (holdout): [6, 7, 8, 9, 10]
        # Path 18 (holdout): [6, 15, 16, 17, 18]

        # --- Test: num_pause_tokens=1, split='train', use_task_tokens=True ---
        seq_p1 = self.gen._generate_path_prediction_training_set(
            size=3, split='train', num_pause_tokens=1, use_task_tokens=True
        )
        # Mock samples [14, 22, 26] from self.TRAIN_LEAVES
        self.assertEqual(seq_p1.shape, (3, 8))  # 1(PATH)+1(leaf)+1(PAUSE)+5(path)
        expected_p1 = [
            [self.TOK_PATH, 14, self.TOK_PAUSE, 6, 11, 12, 13, 14],
            [self.TOK_PATH, 22, self.TOK_PAUSE, 6, 19, 20, 21, 22],
            [self.TOK_PATH, 26, self.TOK_PAUSE, 6, 23, 24, 25, 26],
        ]
        self.assertTrue(torch.equal(seq_p1, torch.tensor(expected_p1)))

        # --- Test: num_pause_tokens=2, split='val', use_task_tokens=True ---
        seq_p2 = self.gen._generate_path_prediction_training_set(
            size=2, split='val', num_pause_tokens=2, use_task_tokens=True
        )
        # Mock samples [10, 18] from self.HOLDOUT_LEAVES
        self.assertEqual(seq_p2.shape, (2, 9))  # 1+1+2+5
        expected_p2 = [
            [self.TOK_PATH, 10, self.TOK_PAUSE, self.TOK_PAUSE, 6, 7, 8, 9, 10],
            [self.TOK_PATH, 18, self.TOK_PAUSE, self.TOK_PAUSE, 6, 15, 16, 17, 18],
        ]
        self.assertTrue(torch.equal(seq_p2, torch.tensor(expected_p2)))

        # --- Test: num_pause_tokens=1, split='train', use_task_tokens=False ---
        seq_p3 = self.gen._generate_path_prediction_training_set(
            size=3, split='train', num_pause_tokens=1, use_task_tokens=False
        )
        self.assertEqual(seq_p3.shape, (3, 7))  # 1(leaf)+1(PAUSE)+5(path), no PATH token
        expected_p3 = [
            [14, self.TOK_PAUSE, 6, 11, 12, 13, 14],
            [22, self.TOK_PAUSE, 6, 19, 20, 21, 22],
            [26, self.TOK_PAUSE, 6, 23, 24, 25, 26],
        ]
        self.assertTrue(torch.equal(seq_p3, torch.tensor(expected_p3)))

    def test_04_prepare_basic(self):
        """Tests the prepare method: directory naming, file creation, and metadata."""
        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True
        use_task_tokens = True

        # Create a temporary directory for test output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run prepare
            output_dir = g.prepare(
                num_pause_tokens=num_pause_tokens, 
                output_dir=temp_dir, 
                use_undirected=use_undirected, 
                use_directional_tokens=use_directional_tokens,
                use_task_tokens=use_task_tokens
            )
            
            # --- Expected Calculations ---
            num_edges = g.d * (g.l - 1)  # 20
            num_edge_samples = 2 * num_edges if use_undirected else num_edges  # 40
            num_train_path_samples = len(g.train_leaves)  # 3
            num_val_path_samples = len(g.holdout_leaves)  # 2
            
            # Final vocab_size: randomize_vocab_size + special tokens = 2000 + 6 = 2006
            final_vocab_size = self.RANDOMIZE_VOCAB_SIZE + self.gen.num_special_tokens
            
            # Sequence Length: l + num_pause_tokens + 1(leaf) + 1(task_token) = 5 + 1 + 1 + 1 = 8
            path_seq_len = g.l + num_pause_tokens + 1 + (1 if use_task_tokens else 0)
            
            # --- 1. Verify Directory Naming ---
            expected_dir_name = f'inweights_pathstar_v{self.RANDOMIZE_VOCAB_SIZE}_ped_elv2_plplain_d{self.D}_l{self.L}_p1_undirected_dt_tt'
            self.assertTrue(output_dir.endswith(expected_dir_name))
            self.assertTrue(os.path.exists(output_dir))
            
            # --- 2. Verify Files Exist ---
            paths_path = os.path.join(output_dir, 'paths.bin')
            edges_path = os.path.join(output_dir, 'edges.bin')
            val_path = os.path.join(output_dir, 'val.bin')
            meta_path = os.path.join(output_dir, 'meta.pkl')
            
            self.assertTrue(os.path.exists(paths_path))
            self.assertTrue(os.path.exists(edges_path))
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
            self.assertEqual(actual_meta['use_task_tokens'], use_task_tokens)
            
            # Check context lengths
            self.assertEqual(actual_meta['edge_context_length'], 3)  # 1(EDGE) + 1(direction) + 1
            self.assertEqual(actual_meta['path_context_length'], 3)  # 1(PATH) + 1(leaf) + 1(pause)
            
            # Check dataset sizes
            self.assertEqual(actual_meta['num_train_path_samples'], num_train_path_samples)
            self.assertEqual(actual_meta['num_val_path_samples'], num_val_path_samples)
            self.assertEqual(actual_meta['total_edge_size'], num_edge_samples)
            self.assertEqual(actual_meta['PATHS_DATASET_SIZE'], num_train_path_samples)
            self.assertEqual(actual_meta['EDGES_DATASET_SIZE'], num_edge_samples)
            self.assertEqual(actual_meta['VAL_DATASET_SIZE'], num_val_path_samples)
            
            # --- 4. Verify data shapes ---
            paths_data = np.memmap(paths_path, dtype=np.uint16, mode='r')
            edges_data = np.memmap(edges_path, dtype=np.uint16, mode='r')
            val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
            
            # Paths: 3 sequences of length 8
            self.assertEqual(paths_data.shape[0], num_train_path_samples * path_seq_len)
            
            # Edges: 40 sequences of length 8 (padded to match path length)
            self.assertEqual(edges_data.shape[0], num_edge_samples * path_seq_len)
            
            # Val: 2 sequences of length 8
            self.assertEqual(val_data.shape[0], num_val_path_samples * path_seq_len)
        
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

    def test_05_prepare_without_task_tokens(self):
        """Tests prepare with use_task_tokens=False."""
        
        g = self.gen
        num_pause_tokens = 1
        use_undirected = True
        use_directional_tokens = True
        use_task_tokens = False

        temp_dir = tempfile.mkdtemp()
        
        try:
            output_dir = g.prepare(
                num_pause_tokens=num_pause_tokens, 
                output_dir=temp_dir, 
                use_undirected=use_undirected, 
                use_directional_tokens=use_directional_tokens,
                use_task_tokens=use_task_tokens
            )
            
            # --- Expected Calculations ---
            path_seq_len = g.l + num_pause_tokens + 1  # 7 (no task token)
            
            # --- 1. Verify Directory Naming ---
            expected_dir_name = f'inweights_pathstar_v{self.RANDOMIZE_VOCAB_SIZE}_ped_elv2_plplain_d{self.D}_l{self.L}_p1_undirected_dt_nott'
            self.assertTrue(output_dir.endswith(expected_dir_name))
            
            # --- 2. Load and verify metadata ---
            meta_path = os.path.join(output_dir, 'meta.pkl')
            with open(meta_path, 'rb') as f:
                actual_meta = pickle.load(f)
            
            self.assertEqual(actual_meta['use_task_tokens'], False)
            self.assertEqual(actual_meta['edge_context_length'], 2)  # direction + 1
            self.assertEqual(actual_meta['path_context_length'], 2)  # leaf + pause
            self.assertEqual(actual_meta['block_size'], path_seq_len)
            
            # --- 3. Verify data shapes ---
            paths_path = os.path.join(output_dir, 'paths.bin')
            paths_data = np.memmap(paths_path, dtype=np.uint16, mode='r')
            paths_data = paths_data.reshape(-1, path_seq_len)
            
            # First token should be leaf (not PATH token)
            self.assertIn(paths_data[0, 0], [14, 22, 26])
            self.assertEqual(paths_data[0, 1], self.TOK_PAUSE)
        
        finally:
            shutil.rmtree(temp_dir)

    def test_06_edge_cases_holdout_percentage(self):
        """Tests edge cases for holdout_percentage."""
        
        # Test 0% holdout
        gen_0 = InWeightsPathStar(d=5, l=5, holdout_percentage=0.0)
        self.assertEqual(len(gen_0.holdout_leaves), 0)
        self.assertEqual(len(gen_0.train_leaves), 5)
        
        # Test 100% holdout
        gen_100 = InWeightsPathStar(d=5, l=5, holdout_percentage=1.0)
        self.assertEqual(len(gen_100.holdout_leaves), 5)
        self.assertEqual(len(gen_100.train_leaves), 0)
        
        # Test invalid holdout percentage
        with self.assertRaises(ValueError):
            InWeightsPathStar(d=5, l=5, holdout_percentage=1.5)
        
        with self.assertRaises(ValueError):
            InWeightsPathStar(d=5, l=5, holdout_percentage=-0.1)

    def test_07_edge_cases_size_validation(self):
        """Tests size validation in generation methods."""
        
        g = self.gen
        
        # Test requesting more paths than available (train)
        with self.assertRaises(ValueError):
            g._generate_path_prediction_training_set(
                size=10, split='train', num_pause_tokens=1
            )
        
        # Test requesting more paths than available (val)
        with self.assertRaises(ValueError):
            g._generate_path_prediction_training_set(
                size=10, split='val', num_pause_tokens=1
            )
        
        # Test requesting more edges than available
        max_edges = 2 * g.d * (g.l - 1)  # 40 for undirected
        with self.assertRaises(ValueError):
            g._generate_edge_memorization_training_set(
                size=max_edges + 1, undirected=True, use_directional_tokens=False
            )

    def test_08_load_dataset(self):
        """Tests load_dataset method."""
        
        g = self.gen
        temp_dir = tempfile.mkdtemp()
        
        try:
            # First prepare a dataset
            g.prepare(
                num_pause_tokens=1,
                output_dir=temp_dir,
                use_undirected=True,
                use_directional_tokens=True,
                use_task_tokens=True
            )
            
            # Update dir_name to match what was created
            g.dir_name = f'inweights_pathstar_v{self.RANDOMIZE_VOCAB_SIZE}_ped_elv2_plplain_d{self.D}_l{self.L}_p1_undirected_dt_tt'
            
            # Now load it
            meta, paths_data, edges_data, val_data = g.load_dataset()
            
            # Verify metadata
            self.assertEqual(meta['d'], self.D)
            self.assertEqual(meta['l'], self.L)
            
            # Verify data is loaded as memmap
            self.assertIsInstance(paths_data, np.memmap)
            self.assertIsInstance(edges_data, np.memmap)
            self.assertIsInstance(val_data, np.memmap)
            
            # Verify sizes
            self.assertEqual(paths_data.shape[0], 3 * 8)  # 3 paths * 8 tokens
            self.assertEqual(edges_data.shape[0], 40 * 8)  # 40 edges * 8 tokens
            self.assertEqual(val_data.shape[0], 2 * 8)  # 2 val paths * 8 tokens
        
        finally:
            shutil.rmtree(temp_dir)

    def test_09_check_dataset_exists(self):
        """Tests _check_dataset_exists method."""
        
        g = self.gen
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate dataset name
            g._generate_dataset_name(
                num_pause_tokens=1,
                use_undirected=True,
                use_directional_tokens=True,
                use_task_tokens=True
            )
            
            # Should not exist initially
            self.assertFalse(g._check_dataset_exists())
            
            # Create the dataset
            g.prepare(
                num_pause_tokens=1,
                output_dir=temp_dir,
                use_undirected=True,
                use_directional_tokens=True,
                use_task_tokens=True
            )
            
            # Now should exist
            self.assertTrue(g._check_dataset_exists())
            
            # Change parameters - should not match
            g._generate_dataset_name(
                num_pause_tokens=2,  # Different!
                use_undirected=True,
                use_directional_tokens=True,
                use_task_tokens=True
            )
            self.assertFalse(g._check_dataset_exists())
        
        finally:
            shutil.rmtree(temp_dir)

    def test_10_vocab_size_validation(self):
        """Tests vocabulary size validation."""
        
        # Test insufficient vocab_size
        with self.assertRaises(ValueError):
            gen = InWeightsPathStar(d=100, l=10, randomize_vocab_size=50)
        
        # Test valid vocab_size
        gen = InWeightsPathStar(d=5, l=5, randomize_vocab_size=100)
        self.assertEqual(gen.randomize_vocab_size, 100)
        
        # Test auto vocab_size
        gen_auto = InWeightsPathStar(d=5, l=5, randomize_vocab_size='auto')
        self.assertEqual(gen_auto.randomize_vocab_size, gen_auto.num_vertices)

    def test_11_special_tokens(self):
        """Tests special token definitions."""
        
        g = self.gen
        
        # Verify all special tokens are defined
        self.assertEqual(g.SPECIAL_TOKENS['PAD'], 0)
        self.assertEqual(g.SPECIAL_TOKENS['PAUSE'], 1)
        self.assertEqual(g.SPECIAL_TOKENS['GT'], 2)
        self.assertEqual(g.SPECIAL_TOKENS['LT'], 3)
        self.assertEqual(g.SPECIAL_TOKENS['PATH'], 4)
        self.assertEqual(g.SPECIAL_TOKENS['EDGE'], 5)
        
        # Verify num_special_tokens
        self.assertEqual(g.num_special_tokens, 6)
        
        # Verify convenience attributes
        self.assertEqual(g.pause_token, 1)
        self.assertEqual(g.pad_token, 0)


if __name__ == '__main__':
    unittest.main()
