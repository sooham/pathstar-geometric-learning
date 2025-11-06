# Quick Start Guide

Get up and running with path-star graph learning tasks in 5 minutes!

## Installation

```bash
git clone <repository-url>
cd pathstar-geometric-learning
pip install -r requirements.txt
```

## Run Examples

### 1. Basic Examples
```bash
python example_usage.py
```

This demonstrates both InContext and InWeights approaches with various configurations.

### 2. Visualizations
```bash
python visualize_pathstar.py
```

Shows ASCII art visualizations of graph structures and example sequences.

### 3. Run Tests
```bash
python test_basic.py
```

Verifies that everything is working correctly.

## Minimal Code Examples

### InContextPathStar (Fresh graphs each time)

```python
from generate_pathstar import InContextPathStar

# Create generator
gen = InContextPathStar(d=5, l=5, vocab_size=200)

# Generate single example
example = gen.generate_training_example()
print(f"Input: {example['prefix']}")
print(f"Output: {example['target']}")

# Generate training batch
batch = gen.generate_training_set(size=100)
print(f"Batch shape: {batch['full_sequences'].shape}")
```

### InWeightsPathStar (Fixed graph, memorization)

```python
from generate_pathstar import InWeightsPathStar

# Create generator
gen = InWeightsPathStar(d=5, l=5)

# Get graph structure
adj_list = gen.generate_adjacency_list()
print(f"Graph has {len(adj_list)} edges")

# Generate training sequences
sequences = gen.generate_path_prediction_training_set(
    size=100,
    pause_token=999,
    num_pause_tokens=3
)
print(f"Sequences shape: {sequences.shape}")
```

## Key Parameters

### InContextPathStar
- `d`: Number of paths (default: 5)
- `l`: Length of each path (default: 5)
- `vocab_size`: Size of node vocabulary (default: 2000)
- `use_directional_tokens`: Add `>` tokens between edges (default: False)
- `num_pause_tokens`: Number of PAUSE tokens (default: 1)

### InWeightsPathStar
- `d`: Number of paths (default: 5)
- `l`: Length of each path (default: 5)
- `vocab`: Custom vocabulary list (optional)
- `mapping`: Custom node ID mapping (optional)

## Understanding the Output

### InContextPathStar Output
```python
{
    'prefix': [adjacency_list_tokens, PAUSE, ..., root, goal],
    'target': [root, n₂, n₃, ..., goal],
    'full_sequence': prefix + target,
    'mapping': {canonical_id -> random_token},
    'root': mapped_root_token,
    'goal': mapped_goal_token
}
```

### InWeightsPathStar Output
```python
# Path prediction:
sequences: [leaf, PAUSE, ..., PAUSE, root, n₂, ..., leaf]

# Edge memorization:
x: [predecessor_nodes]
y: [successor_nodes]
```

## Graph Structure

Path-star graph with `d=3` paths and `l=4` length:

```
        [0] (root)
         |
         +-- [1] -- [2] -- [3] (leaf)
         |
         +-- [4] -- [5] -- [6] (leaf)
         |
         +-- [7] -- [8] -- [9] (leaf)
```

- Total nodes: `d * (l-1) + 1 = 10`
- Total edges: `d * (l-1) = 9`
- Number of paths: `d = 3`
- Path length: `l = 4` nodes

## Typical Use Cases

### 1. Training a Transformer for In-Context Learning
```python
gen = InContextPathStar(d=5, l=5, vocab_size=200)

# Each batch has different graphs
for epoch in range(num_epochs):
    batch = gen.generate_training_set(size=batch_size)
    
    # Train model to predict target given prefix
    outputs = model(batch['prefixes'])
    loss = criterion(outputs, batch['targets'])
```

### 2. Training a Model to Memorize Graph Structure
```python
gen = InWeightsPathStar(d=5, l=5)

# Same graph throughout training
for epoch in range(num_epochs):
    sequences = gen.generate_path_prediction_training_set(
        size=batch_size,
        pause_token=PAUSE_TOKEN
    )
    
    # Train model to complete sequences
    # Input: sequence[:-(l-1)]
    # Target: sequence[-(l-1):]
```

### 3. Custom Node Labeling
```python
# Use custom token range
mapping = {i: 1000 + i for i in range(21)}  # Nodes 1000-1020
gen = InWeightsPathStar(d=5, l=5, mapping=mapping)
```

## Special Tokens Reference

| Token | Purpose | ID |
|-------|---------|-----|
| PAUSE | Compute/pause token | vocab_size |
| PAD | Padding | vocab_size + 1 |
| GT (>) | Directional token | vocab_size + 2 |
| LT (<) | Reverse direction | vocab_size + 3 |
| SEP | Separator | vocab_size + 4 |
| START | Start marker | vocab_size + 5 |
| GOAL | Goal marker | vocab_size + 6 |
| PATH_START | Path start | vocab_size + 7 |
| EOS | End of sequence | vocab_size + 8 |

## Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- Check out [example_usage.py](example_usage.py) for more complex examples
- Run [visualize_pathstar.py](visualize_pathstar.py) to understand graph structures
- See [test_basic.py](test_basic.py) for usage patterns and edge cases

## Troubleshooting

**Q: KeyError when using custom mapping**
- Ensure your mapping covers all nodes: `{0, 1, ..., d*(l-1)}`

**Q: Unexpected sequence lengths**
- InContext: length = `2*edges + num_pause + 2 + l`
- InWeights: length = `1 + num_pause + l`

**Q: Need different vocabulary for each example?**
- Use `InContextPathStar` - it generates fresh random mappings automatically

## Performance Tips

- Use `return_tensors=True` for PyTorch integration
- Set `pad_to_length` for fixed-size batches (faster on GPU)
- Use `undirected=True` for edge memorization to double training data
- Larger `vocab_size` ensures better separation between graphs

## Citation

If you use this code in your research, please cite the original path-star learning work:
- Bietti & Nichani, 2024 (B&N'24): In-context path-star learning task

