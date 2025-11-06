# Interleave Feature for InWeightsPathStar

## Overview

The `InWeightsPathStar` class now supports creating **interleaved datasets** that mix path prediction sequences with edge memorization sequences. This feature allows you to train models on both tasks simultaneously with a configurable ratio.

## Key Concept

Instead of training on pure path prediction sequences:
```
[<PATH>, leaf, PAUSE, root, n_2, n_3, ..., n_ℓ]
[<PATH>, leaf, PAUSE, root, n_2, n_3, ..., n_ℓ]
...
```

Or pure edge memorization sequences:
```
[<EDGE>, x, y]
[<EDGE>, x, y]
...
```

You can create a **mixed dataset** with both types interleaved:
```
[<PATH>, leaf, PAUSE, root, n_2, n_3, ..., n_ℓ]  # Path sequence
[<EDGE>, x, y, PAUSE, PAUSE, ...]                 # Edge sequence (padded)
[<PATH>, leaf, PAUSE, root, n_2, n_3, ..., n_ℓ]  # Path sequence
[<EDGE>, x, y, PAUSE, PAUSE, ...]                 # Edge sequence (padded)
[<PATH>, leaf, PAUSE, root, n_2, n_3, ..., n_ℓ]  # Path sequence
...
```

**Task Prefixes:** Each sequence starts with a task prefix token (`<PATH>` or `<EDGE>`) that allows the model to distinguish between the two tasks.

The ratio between path and edge sequences is controlled by the `interleave_ratio` parameter.

## Usage

### Python API

```python
from pathstar import InWeightsPathStar

# Create generator
generator = InWeightsPathStar(d=10, l=5)

# Prepare dataset with 3:1 ratio (75% paths, 25% edges)
generator.prepare(
    train_size=100000,
    val_size=10000,
    num_pause_tokens=1,
    output_dir='./data',
    interleave_ratio=(3, 1)  # A:B format
)
```

### Command Line

```bash
# Generate dataset with 3:1 interleave ratio
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --train_size 100000 \
    --val_size 10000 \
    --interleave 3:1 \
    --output_dir ./data
```

## Interleave Ratio Format

The `interleave_ratio` is specified as a tuple `(A, B)` or string `"A:B"` where:
- **A**: Number of parts for path prediction sequences
- **B**: Number of parts for edge memorization sequences

### Examples

| Ratio | Format | Path % | Edge % | Description |
|-------|--------|--------|--------|-------------|
| 1:1 | `(1, 1)` or `"1:1"` | 50% | 50% | Equal mix |
| 3:1 | `(3, 1)` or `"3:1"` | 75% | 25% | Mostly paths |
| 1:3 | `(1, 3)` or `"1:3"` | 25% | 75% | Mostly edges |
| 9:1 | `(9, 1)` or `"9:1"` | 90% | 10% | Sparse edge training |
| 1:9 | `(1, 9)` or `"1:9"` | 10% | 90% | Sparse path training |

### Calculation

For a dataset of size `N` with ratio `A:B`:
- Number of path sequences: `N * A / (A + B)`
- Number of edge sequences: `N * B / (A + B)`

Example with `N=1000` and ratio `3:1`:
- Path sequences: `1000 * 3 / 4 = 750`
- Edge sequences: `1000 * 1 / 4 = 250`

## Implementation Details

### Sequence Format

**Path Prediction Sequences:**
```
[<PATH>, leaf, PAUSE, ..., PAUSE, root, n_2, n_3, ..., n_ℓ]
Length: l + 2 + num_pause_tokens
```

**Edge Memorization Sequences:**
```
[<EDGE>, x, y, PAUSE, PAUSE, ..., PAUSE]
Length: 3 (padded to match path sequence length)
```

**Task Prefix Tokens:**
- `<PATH>`: Indicates a path prediction sequence
- `<EDGE>`: Indicates an edge memorization sequence

Edge sequences are **padded with PAUSE tokens** to match the length of path sequences, ensuring uniform tensor dimensions. The task prefix allows the model to condition its predictions on the task type.

### Shuffling

After generating both types of sequences, they are:
1. Concatenated into a single tensor
2. Randomly shuffled using `torch.randperm()`

This ensures that path and edge sequences are randomly interleaved throughout the dataset.

### Holdout Compatibility

The interleave feature works seamlessly with the holdout feature:

```python
# Create generator with 20% holdout
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)

# Generate interleaved dataset (path sequences only use training paths)
generator.prepare(
    train_size=100000,
    val_size=10000,
    interleave_ratio=(3, 1)  # 75% paths (from training set), 25% edges (all edges)
)
```

**Important:** 
- Path sequences respect the holdout split (only training paths are used)
- Edge sequences include **all edges** (including those in holdout paths)

## Use Cases

### 1. Multi-Task Learning

Train a model to simultaneously learn edge relationships and path composition:

```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --interleave 1:1 \
    --train_size 100000
```

**Research Question:** Does joint training on edges and paths improve performance on both tasks?

### 2. Curriculum Learning

Start with more edge training, gradually shift to path training:

**Phase 1: Edge-heavy (1:9 ratio)**
```python
generator.prepare(train_size=50000, interleave_ratio=(1, 9))
```

**Phase 2: Balanced (1:1 ratio)**
```python
generator.prepare(train_size=50000, interleave_ratio=(1, 1))
```

**Phase 3: Path-heavy (9:1 ratio)**
```python
generator.prepare(train_size=50000, interleave_ratio=(9, 1))
```

### 3. Data Efficiency Studies

Compare models trained on different ratios with the same total dataset size:

```python
ratios = [(10, 0), (9, 1), (7, 3), (5, 5), (3, 7), (1, 9), (0, 10)]
for ratio in ratios:
    generator.prepare(train_size=100000, interleave_ratio=ratio)
    # Train and evaluate model
```

**Research Question:** What's the optimal ratio for learning path prediction?

### 4. Auxiliary Task Learning

Use edge prediction as an auxiliary task to improve path prediction:

```bash
# 90% paths, 10% edges (edge prediction as auxiliary task)
python pathstar.py \
    --mode inweights \
    --interleave 9:1 \
    --train_size 100000
```

### 5. Compositional Generalization with Interleaving

Combine interleaving with holdout to study compositional generalization:

```python
# Hold out 20% of paths
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)

# Train on edges + training paths
generator.prepare(
    train_size=100000,
    interleave_ratio=(3, 1)  # 75% training paths, 25% all edges
)

# Evaluate on holdout paths
holdout_data = generator.generate_path_prediction_training_set(
    size=1000,
    holdout_only=True
)
```

**Research Question:** Can a model compose learned edges into unseen paths when trained with interleaved data?

## Command-Line Examples

### Example 1: Basic Interleaving

```bash
python pathstar.py \
    --mode inweights \
    --d 5 \
    --l 5 \
    --interleave 3:1 \
    --train_size 10000 \
    --val_size 1000
```

Output:
```
Preparing InWeightsPathStar dataset...
  Parameters: d=5, l=5
  Train size: 10000, Val size: 1000
  Interleave ratio: 3:1 (path:edge)
  Path percentage: 75.0%
  Edge percentage: 25.0%

Generating interleaved training set...
Generating interleaved validation set...
...
```

### Example 2: Interleaving with Holdout

```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --holdout_percentage 0.2 \
    --interleave 4:1 \
    --train_size 50000 \
    --val_size 5000
```

### Example 3: Edge-Heavy Training

```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --interleave 1:4 \
    --train_size 100000
```

### Example 4: No Interleaving (Path Only)

```bash
# Omit --interleave to get pure path prediction dataset
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --train_size 100000
```

## Metadata

When using interleaving, the metadata file includes:

```python
import pickle

with open('./data/inweights_pathstar_d10_l5/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print(meta['interleave_ratio'])  # e.g., (3, 1) or None
```

The `interleave_ratio` field in metadata:
- Is `None` if no interleaving was used
- Is a tuple `(A, B)` if interleaving was used

This allows you to:
1. Verify the ratio used during dataset generation
2. Reproduce the exact dataset configuration
3. Track experimental conditions

## Technical Notes

### Task Prefix Tokens

To enable the model to distinguish between path prediction and edge memorization tasks, each sequence is prefixed with a task-specific token:

**Token Definitions:**
- `<PATH>` token: Added at the start of every path prediction sequence
- `<EDGE>` token: Added at the start of every edge memorization sequence

**Token Values:**
- `<PATH>` = `pause_token + 1`
- `<EDGE>` = `pause_token + 2`

**Benefits:**
1. **Task Identification**: Model can immediately identify the task type from the first token
2. **Conditional Generation**: Model can condition its predictions on the task prefix
3. **Multi-Task Learning**: Enables effective multi-task learning with shared representations
4. **Clear Separation**: Provides explicit signal for task switching

**Example:**
```python
generator = InWeightsPathStar(d=5, l=5)
print(f"Pause token: {generator.pause_token}")
print(f"PATH token: {generator.TASK_TOKENS['PATH']}")
print(f"EDGE token: {generator.TASK_TOKENS['EDGE']}")

# Output:
# Pause token: 21
# PATH token: 22
# EDGE token: 23
```

**Metadata:**
The task tokens are saved in the metadata file:
```python
import pickle
with open('./data/inweights_pathstar_d5_l5/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print(meta['task_tokens'])  # {'PATH': 22, 'EDGE': 23}
print(meta['itos'][22])     # '<PATH>'
print(meta['itos'][23])     # '<EDGE>'
```

### Padding Strategy

Edge sequences `[<EDGE>, x, y]` are much shorter than path sequences. To create uniform tensors, edge sequences are padded with `PAUSE` tokens:

```python
# Path sequence (length: l + 2 + num_pause_tokens)
[<PATH>, leaf, PAUSE, root, n_2, n_3, n_4, n_5]  # Length: 8 (for l=5, num_pause_tokens=1)

# Edge sequence (length: 3, padded to 8)
[<EDGE>, x, y, PAUSE, PAUSE, PAUSE, PAUSE, PAUSE]
```

This ensures:
- Uniform tensor shapes for efficient batching
- Clear separation between meaningful tokens and padding
- Task prefix tokens allow models to distinguish between sequence types
- Models can condition their behavior on the task type

### Undirected Edges

By default, edge memorization uses **undirected edges** (both `x→y` and `y→x`):

```python
# In _generate_interleaved_dataset:
edge_x, edge_y = self.generate_edge_memorization_training_set(
    size=num_edge,
    undirected=True  # Default
)
```

This doubles the effective edge training data and ensures bidirectional edge learning.

### Random Shuffling

Sequences are shuffled **after** generation to ensure random interleaving:

```python
# Concatenate path and edge sequences
all_sequences = torch.cat([path_sequences, edge_sequences], dim=0)

# Shuffle
indices = torch.randperm(all_sequences.shape[0])
all_sequences = all_sequences[indices]
```

This prevents any ordering bias in the training data.

## Limitations

1. **Fixed Padding Length**: Edge sequences are padded to match path sequence length, which may be inefficient for very long paths.

2. **No Separate Validation Ratios**: Both training and validation sets use the same interleave ratio. If you need different ratios, generate them separately.

3. **Undirected Edges Only**: The interleaving currently uses undirected edges. For directed edges, you'd need to modify `_generate_interleaved_dataset()`.

4. **Integer Ratios Only**: The ratio must be specified as positive integers (e.g., `3:1`), not decimals (e.g., `0.75:0.25`).

## Error Handling

### Invalid Ratio Format

```bash
python pathstar.py --mode inweights --interleave 3.5:1
# Error: Invalid interleave ratio '3.5:1': invalid literal for int()
```

### Missing Colon

```bash
python pathstar.py --mode inweights --interleave 3-1
# Error: Invalid interleave ratio '3-1': Interleave ratio must be in format A:B
```

### Non-Positive Values

```bash
python pathstar.py --mode inweights --interleave 0:1
# Error: Invalid interleave ratio '0:1': Interleave ratio values must be positive integers
```

### Too Few Paths/Edges

If the requested size exceeds available paths or edges, you'll get an error:

```python
generator = InWeightsPathStar(d=2, l=5)  # Only 2 paths

# Requesting 1000 path sequences with ratio 3:1 means 750 paths needed
# But only 2 unique paths available
generator.prepare(train_size=1000, interleave_ratio=(3, 1))
# ValueError: Requested size (750) exceeds the number of available training paths (2)
```

**Note:** This uses sampling **with replacement**, so you can request more sequences than unique paths/edges, but the validation still checks to prevent obvious errors.

## Comparison with Pure Datasets

| Feature | Pure Path | Pure Edge | Interleaved |
|---------|-----------|-----------|-------------|
| Sequence Types | Path only | Edge only | Both |
| Sequence Length | `l + 1 + num_pause_tokens` | 2 | Uniform (padded) |
| Training Signal | Path composition | Edge relationships | Both |
| Holdout Respect | Yes | No | Path: Yes, Edge: No |
| Use Case | Path prediction | Edge memorization | Multi-task learning |

## Future Enhancements

Potential improvements to the interleave feature:

1. **Separate Train/Val Ratios**: Allow different interleave ratios for training and validation sets
2. **Dynamic Padding**: Use variable-length sequences instead of padding
3. **Directed Edge Option**: Add parameter to use directed edges only
4. **Weighted Sampling**: Sample paths/edges with different probabilities
5. **Batch-Level Interleaving**: Interleave at batch level instead of sequence level

