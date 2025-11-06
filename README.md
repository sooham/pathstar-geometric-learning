# Path-Star Geometric Learning

Implementation of path-star graph learning tasks for studying in-context learning and geometric representations in neural networks.

## Overview

This repository provides two complementary approaches to path-star graph learning tasks:

1. **InContextPathStar**: In-context learning task (B&N'24 style) where each training example has a fresh, randomly-labeled graph
2. **InWeightsPathStar**: In-weights memorization task where a single fixed graph is memorized during training

## Path-Star Graph Structure

A path-star graph is a tree structure where:
- A single root node branches into `d` paths
- Each path has length `ℓ` (from root to leaf)
- Total nodes: `d * (ℓ - 1) + 1`
- Total edges: `d * (ℓ - 1)`

```
        Leaf₁
         |
      n₂ - n₃
         |
       Root --- n₄ - n₅ - Leaf₂
         |
      n₆ - n₇
         |
        Leaf₃
```

## Installation

```bash
# Install requirements
pip install -r requirements.txt
```

## Usage

### InContextPathStar

Each training example contains a randomized adjacency list followed by a query:

```python
from generate_pathstar import InContextPathStar

# Create generator: 5 paths, each of length 5
generator = InContextPathStar(d=5, l=5, vocab_size=200)

# Generate a single example
example = generator.generate_training_example(
    use_directional_tokens=False,
    num_pause_tokens=3
)

print(f"Root: {example['root']}")
print(f"Goal: {example['goal']}")
print(f"Prefix: {example['prefix']}")  # [adjacency_list, PAUSE, PAUSE, PAUSE, root, goal]
print(f"Target: {example['target']}")  # [root, n₂, n₃, ..., goal]

# Generate a training batch
batch = generator.generate_training_set(
    size=100,
    use_directional_tokens=False,
    num_pause_tokens=3,
    return_tensors=True
)

print(f"Prefixes shape: {batch['prefixes'].shape}")
print(f"Targets shape: {batch['targets'].shape}")
```

**Key Features:**
- Each example has a **fresh random graph** with different node labels
- Tests model's ability to parse adjacency lists in-context
- Prefix format: `[edge₁_u, edge₁_v, edge₂_u, edge₂_v, ..., PAUSE, ..., root, goal]`
- Target: Full path from root to goal
- Optional directional tokens: `[u, >, v]` format

### InWeightsPathStar

Single fixed graph where the model memorizes structure in weights:

```python
from generate_pathstar import InWeightsPathStar

# Create generator: 5 paths, each of length 5
generator = InWeightsPathStar(d=5, l=5)

# Get the fixed graph structure
adj_list = generator.generate_adjacency_list()
print(f"Graph edges: {adj_list}")

# Generate path prediction training set
sequences = generator.generate_path_prediction_training_set(
    size=100,
    pause_token=999,
    num_pause_tokens=3
)

print(f"Sequences shape: {sequences.shape}")
# Each sequence: [leaf, PAUSE, PAUSE, PAUSE, root, n₂, ..., leaf]

# Generate edge memorization training set
x, y = generator.generate_edge_memorization_training_set(
    size=100,
    undirected=True
)
# x: predecessor nodes, y: successor nodes
```

**Key Features:**
- Single **fixed graph** throughout training
- Tests model's ability to memorize graph structure
- Sequence format: `[leaf, PAUSE, ..., PAUSE, root, n₂, ..., leaf]`
- Can generate edge-level training data for memorization

### Custom Node Mapping

Both classes support custom node label mappings:

```python
import random

# Create custom mapping
d, l = 3, 4
total_nodes = d * (l - 1) + 1
mapping = {i: random.randint(100, 200) for i in range(total_nodes)}

# Use with InWeightsPathStar
generator = InWeightsPathStar(d=d, l=l, mapping=mapping)
```

## Special Tokens

InContextPathStar reserves 9 special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| PAUSE | vocab_size | Compute/pause token |
| PAD | vocab_size + 1 | Padding token |
| GT (>) | vocab_size + 2 | Directional token (edge direction) |
| LT (<) | vocab_size + 3 | Directional token (reverse) |
| SEP | vocab_size + 4 | Separator token |
| START | vocab_size + 5 | Start marker |
| GOAL | vocab_size + 6 | Goal marker |
| PATH_START | vocab_size + 7 | Path start marker |
| EOS | vocab_size + 8 | End of sequence |

Effective vocabulary size: `|V| = vocab_size + 9`

## Examples

Run the comprehensive examples:

```bash
python example_usage.py
```

This demonstrates:
- Single example generation
- Batch generation
- Special token usage
- Different graph configurations
- Comparison between in-context and in-weights approaches

## API Reference

### InContextPathStar

#### Constructor
```python
InContextPathStar(d=5, l=5, vocab_size=2000)
```

**Parameters:**
- `d` (int): Number of paths/spokes in the path-star
- `l` (int): Length of each path (from root to leaf)
- `vocab_size` (int): Size of vocabulary for node labels (excludes special tokens)

#### Methods

##### `generate_training_example(use_directional_tokens=False, num_pause_tokens=1)`

Generate a single training example.

**Parameters:**
- `use_directional_tokens` (bool): Whether to use `>` tokens in adjacency list
- `num_pause_tokens` (int): Number of PAUSE tokens between adjacency list and query

**Returns:**
- dict with keys:
  - `'prefix'`: List of tokens forming the input prefix
  - `'target'`: List of tokens forming the target path
  - `'full_sequence'`: Concatenated prefix + target
  - `'mapping'`: The random node mapping used
  - `'root'`: The mapped root node
  - `'goal'`: The mapped goal node

##### `generate_training_set(size, use_directional_tokens=False, num_pause_tokens=1, return_tensors=True, pad_to_length=None)`

Generate a batch of training examples.

**Parameters:**
- `size` (int): Number of examples to generate
- `use_directional_tokens` (bool): Whether to use directional tokens
- `num_pause_tokens` (int): Number of PAUSE tokens
- `return_tensors` (bool): Whether to return PyTorch tensors
- `pad_to_length` (int, optional): Pad sequences to this length

**Returns:**
- dict with keys:
  - `'prefixes'`: Tensor/list of prefix sequences
  - `'targets'`: Tensor/list of target paths
  - `'full_sequences'`: Tensor/list of full sequences
  - `'prefix_lengths'`: Tensor/list of prefix lengths
  - `'target_lengths'`: Tensor/list of target lengths

### InWeightsPathStar

#### Constructor
```python
InWeightsPathStar(d=5, l=5, vocab=None, mapping=None)
```

**Parameters:**
- `d` (int): Number of paths/spokes in the path-star
- `l` (int): Length of each path (from root to leaf)
- `vocab` (list, optional): Custom vocabulary list
- `mapping` (dict, optional): Custom node ID to token mapping

#### Methods

##### `generate_adjacency_list()`

Generate the adjacency list as a shuffled list of edge pairs.

**Returns:**
- List of tuples: `[(u₁, v₁), (u₂, v₂), ...]`

##### `generate_paths_by_leaf()`

Generate paths indexed by leaf nodes.

**Returns:**
- dict mapping `leaf_node -> path_from_root`

##### `generate_edge_memorization_training_set(size, undirected=True)`

Generate edge-level training data for memorization.

**Parameters:**
- `size` (int): Number of samples
- `undirected` (bool): Whether to include reverse edges

**Returns:**
- Tuple of tensors: `(x, y)` where x are predecessors and y are successors

##### `generate_path_prediction_training_set(size, pause_token, num_pause_tokens=1)`

Generate path prediction training data.

**Parameters:**
- `size` (int): Number of samples
- `pause_token` (int): Token ID for PAUSE
- `num_pause_tokens` (int): Number of PAUSE tokens

**Returns:**
- Tensor of shape `[size, l+1+num_pause_tokens]` containing sequences

## Comparison: In-Context vs In-Weights

| Aspect | InContext | InWeights |
|--------|-----------|-----------|
| Graph | Fresh random per example | Single fixed graph |
| Node labels | Random each time | Fixed or custom mapping |
| Input format | Adjacency list + query | Leaf + pause + path |
| Tests | Generalization | Memorization |
| Sequence length | Long (includes adj list) | Short (just path) |
| Use case | In-context learning | Weight-based learning |

## Theory and Background

This implementation is based on research into:
- **In-context learning**: How models learn to solve new tasks from examples in the prompt
- **Geometric representations**: How neural networks represent graph structures
- **Path finding**: Reasoning about connectivity and shortest paths

The path-star structure is particularly useful because:
1. It has a simple, controlled structure
2. Path-finding has a unique solution
3. It tests both memorization and reasoning capabilities
4. It can scale in complexity (by varying `d` and `ℓ`)

## References

Based on the path-star graph learning tasks described in:
- Bietti & Nichani, 2024 (B&N'24): In-context path-star learning
- Research on geometric representations in neural networks

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

