# InWeightsPathStar Dataset Structure

## Overview

The `InWeightsPathStar` dataset has a specific structure designed to train models on both edge memorization and path prediction tasks:

- **Training Set**: All training paths + All edges (mixed and shuffled)
- **Validation Set**: Only validation paths (no edges)

This structure allows models to learn edge relationships during training while evaluating path prediction capability on held-out path samples.

## Dataset Composition

### Graph Structure

For a path-star graph with `d` spokes and path length `l`:
- **Total vertices**: `d * (l - 1) + 1`
- **Total edges**: `d * (l - 1)`
- **Total paths**: `d` (one per spoke)

### Holdout Split

Paths are first split based on `holdout_percentage`:
- **Training paths**: `d * (1 - holdout_percentage)` paths available for training/validation
- **Holdout paths**: `d * holdout_percentage` paths completely held out for final evaluation

### Train/Validation Split

The training paths are then split into train and validation:
- **Train path samples**: `num_training_paths * train_val_split`
- **Val path samples**: `num_training_paths * (1 - train_val_split)`

### Edge Samples

All edges are included in the training set:
- **Edge samples**: `(2 if undirected else 1) * num_edges`
- Edges include those from ALL paths (including holdout paths)

## Final Dataset Sizes

```
Training set size = train_path_samples + edge_samples
Validation set size = val_path_samples
```

**Example:** d=10, l=5, holdout=20%, train_val_split=0.9, undirected
- Total edges: 40
- Edge samples: 80 (undirected)
- Training paths after holdout: 8
- Train path samples: 7 (90% of 8)
- Val path samples: 1 (10% of 8)
- **Training set: 7 + 80 = 87 sequences**
- **Validation set: 1 sequence**

## Sequence Formats

### Path Sequences

```
[<PATH>, leaf, <PAUSE>, root, n_2, n_3, ..., n_ℓ]
```

- Starts with `<PATH>` task token
- Followed by leaf node
- One or more `<PAUSE>` tokens
- Then the path from root to leaf

**Length**: `l + 2 + num_pause_tokens`

### Edge Sequences

```
[<EDGE>, x, y, <PAUSE>, <PAUSE>, ...]
```

- Starts with `<EDGE>` task token
- Followed by source node `x`
- Then target node `y`
- Padded with `<PAUSE>` tokens to match path sequence length

**Length**: 3 (padded to match path sequences)

## Training Set Structure

The training set contains a **shuffled mixture** of:

1. **Path sequences** from training split
2. **Edge sequences** from all edges

**Example training set (d=10, l=5, 90/10 split, undirected):**
```
Sequence 1: [<EDGE>, 5, 9, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>]
Sequence 2: [<PATH>, 16, <PAUSE>, 0, 13, 14, 15, 16]
Sequence 3: [<EDGE>, 3, 7, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>]
Sequence 4: [<PATH>, 8, <PAUSE>, 0, 5, 6, 7, 8]
...
Sequence 87: [<EDGE>, 12, 16, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>, <PAUSE>]
```

Total: 7 path sequences + 80 edge sequences = 87 sequences (shuffled)

## Validation Set Structure

The validation set contains **only path sequences** from the validation split:

**Example validation set:**
```
Sequence 1: [<PATH>, 12, <PAUSE>, 0, 9, 10, 11, 12]
```

Total: 1 path sequence (no edges)

## Rationale

### Why Edges in Training Only?

1. **Edge Learning**: Models learn edge relationships from explicit edge examples
2. **Path Composition**: Models must compose learned edges into paths
3. **Generalization Testing**: Validation tests if models can predict paths using learned edges
4. **No Data Leakage**: Validation paths are different from training paths

### Why Include Holdout Path Edges?

Edges from holdout paths are included in training because:
1. **Edge Coverage**: Ensures all edges in the graph are seen during training
2. **Compositional Generalization**: Tests if models can compose known edges into unseen paths
3. **Realistic Scenario**: In real graphs, edges are often observable even if full paths aren't

## Usage Examples

### Example 1: Basic Usage

```python
from pathstar import InWeightsPathStar

# Create generator
generator = InWeightsPathStar(d=10, l=5)

# Prepare dataset
generator.prepare()

# Output:
# Training set: 9 paths + 80 edges = 89 sequences
# Validation set: 1 path
```

### Example 2: With Holdout

```python
# Hold out 20% of paths
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)
generator.prepare()

# Output:
# Training paths after holdout: 8
# Training set: 7 paths + 80 edges = 87 sequences
# Validation set: 1 path
# (Holdout paths: 2 - not in train or val)
```

### Example 3: Directed Edges

```python
# Use directed edges instead of undirected
generator = InWeightsPathStar(d=10, l=5)
generator.prepare(use_undirected=False)

# Output:
# Training set: 9 paths + 40 edges = 49 sequences
# Validation set: 1 path
```

### Example 4: Custom Split

```python
# 80/20 train/val split
generator = InWeightsPathStar(d=10, l=5)
generator.prepare(train_val_split=0.8)

# Output:
# Training set: 8 paths + 80 edges = 88 sequences
# Validation set: 2 paths
```

## Command Line

```bash
# Basic usage
python pathstar.py --mode inweights --d 10 --l 5

# With holdout
python pathstar.py --mode inweights --d 10 --l 5 --holdout_percentage 0.2

# Directed edges
python pathstar.py --mode inweights --d 10 --l 5 --use_directed

# Custom split
python pathstar.py --mode inweights --d 10 --l 5 --train_val_split 0.8

# Full example
python pathstar.py \
    --mode inweights \
    --d 20 \
    --l 100 \
    --holdout_percentage 0.2 \
    --train_val_split 0.9 \
    --use_directed \
    --output_dir ./data
```

## Metadata

The dataset metadata includes detailed information about the composition:

```python
import pickle

with open('./data/inweights_pathstar_d10_l5/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print(meta['train_size'])              # e.g., 87
print(meta['val_size'])                # e.g., 1
print(meta['num_train_path_samples'])  # e.g., 7
print(meta['num_val_path_samples'])    # e.g., 1
print(meta['num_edge_samples'])        # e.g., 80
print(meta['use_undirected'])          # e.g., True
print(meta['train_val_split'])         # e.g., 0.9
```

## Training Strategy

### Recommended Approach

1. **Train on mixed dataset**: Model learns both edges and paths
2. **Validate on paths only**: Test path prediction without edge hints
3. **Evaluate on holdout paths**: Final test on completely unseen paths

### Loss Computation

You can compute separate losses for each task:

```python
def compute_loss(model, sequences, targets):
    # Separate by task token
    task_tokens = sequences[:, 0]
    path_mask = (task_tokens == PATH_TOKEN)
    edge_mask = (task_tokens == EDGE_TOKEN)
    
    # Compute task-specific losses
    if path_mask.any():
        path_loss = F.cross_entropy(
            model(sequences[path_mask]),
            targets[path_mask]
        )
    
    if edge_mask.any():
        edge_loss = F.cross_entropy(
            model(sequences[edge_mask]),
            targets[edge_mask]
        )
    
    # Combined loss (can weight differently)
    total_loss = path_loss + edge_loss
    return total_loss
```

## Comparison with Previous Design

### Old Design (Interleaved with Ratio)

- Training: Paths and edges mixed according to ratio (e.g., 3:1)
- Validation: Paths and edges mixed with same ratio
- Problem: Validation included edges, making it easier

### New Design (Edges in Training Only)

- Training: All training paths + All edges
- Validation: Only validation paths (no edges)
- Benefit: True test of path prediction capability

## Benefits

### 1. Clear Separation

Training and validation have different purposes:
- **Training**: Learn edges AND practice paths
- **Validation**: Test path prediction only

### 2. Realistic Evaluation

Validation tests the model's ability to:
- Predict paths without edge hints
- Compose learned edges into paths
- Generalize to unseen path samples

### 3. Efficient Use of Data

- All edges are used for training
- Path samples are split for train/val
- No wasted data

### 4. Compositional Generalization

By including holdout path edges in training:
- Model learns all edge relationships
- Can test if model composes edges into unseen paths
- Realistic scenario for graph learning

## Edge Cases

### Very Small Graphs

For very small graphs, validation might have 0 or 1 sample:

```python
generator = InWeightsPathStar(d=2, l=3)
generator.prepare()

# Output:
# Training set: 1 path + 4 edges = 5 sequences
# Validation set: 1 path
```

### Large Graphs

For large graphs, training set is dominated by edges:

```python
generator = InWeightsPathStar(d=100, l=10)
generator.prepare()

# Output:
# Training set: 90 paths + 1800 edges = 1890 sequences
# Validation set: 10 paths
```

### All Paths Held Out

If all paths are held out, only edges remain:

```python
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=1.0)
generator.prepare()

# Output:
# Training set: 0 paths + 80 edges = 80 sequences
# Validation set: 0 paths
# (All 10 paths are in holdout)
```

## Summary

The InWeightsPathStar dataset structure:

✅ **Training**: Paths + Edges (mixed)  
✅ **Validation**: Paths only (no edges)  
✅ **Automatic sizing**: Based on graph structure  
✅ **Task prefixes**: `<PATH>` and `<EDGE>` tokens  
✅ **Holdout support**: Separate test set  
✅ **Flexible**: Adjustable splits and edge directionality  

This structure enables effective training on edge relationships while properly evaluating path prediction capability.

