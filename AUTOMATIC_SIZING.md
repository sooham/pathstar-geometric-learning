# Automatic Dataset Sizing for InWeightsPathStar

## Overview

The `InWeightsPathStar.prepare()` method now **automatically calculates** training and validation dataset sizes based on the graph structure. This ensures that dataset sizes are always appropriate for the graph and eliminates the need to manually specify `train_size` and `val_size`.

## Motivation

Previously, users had to manually specify `train_size` and `val_size`, which could lead to:
- **Undersized datasets**: Not using all available unique paths/edges
- **Oversized requests**: Requesting more unique samples than available
- **Inconsistent sizing**: Different graphs with different optimal sizes

With automatic sizing, the dataset size is always **optimal** for the given graph structure.

## How It Works

### Graph Structure

For a path-star graph with `d` spokes and path length `l`:
- **Number of vertices**: `d * (l - 1) + 1`
- **Number of edges**: `d * (l - 1)`
- **Number of paths**: `d` (one per spoke)

### Dataset Size Calculation

#### Pure Path Prediction (No Interleaving)

```
total_size = number_of_training_paths
```

Where `number_of_training_paths = d * (1 - holdout_percentage)`

**Example:** d=10, l=5, holdout=20%
- Total paths: 10
- Training paths: 8 (80%)
- Holdout paths: 2 (20%)
- **Dataset size: 8**

#### Interleaved Mode (Path + Edge)

```
total_size = full_edge_size + full_path_size

where:
  full_edge_size = (2 if undirected else 1) * num_edges
  full_path_size = number_of_training_paths
```

**Example:** d=10, l=5, holdout=20%, undirected edges
- Edges: 10 * (5-1) = 40
- Undirected edges: 40 * 2 = 80
- Training paths: 8
- **Dataset size: 80 + 8 = 88**

### Train/Validation Split

The total dataset is split into training and validation sets:

```
train_size = int(total_size * train_val_split)
val_size = total_size - train_size
```

Default split: 90% train, 10% validation

**Example:** total_size=88, split=0.9
- **Train size: 79**
- **Val size: 9**

## Usage

### Python API

```python
from pathstar import InWeightsPathStar

# Create generator
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)

# Prepare dataset (sizes calculated automatically)
generator.prepare(
    num_pause_tokens=1,
    output_dir='./data',
    interleave_ratio=(3, 1),  # Optional
    use_undirected=True,      # Default: True
    train_val_split=0.9       # Default: 0.9
)

# Sizes are printed during preparation:
# Train size: 79, Val size: 9
```

### Command Line

```bash
# Pure path prediction
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --holdout_percentage 0.2

# Interleaved with custom split
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --interleave 3:1 \
    --train_val_split 0.8

# Directed edges instead of undirected
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --interleave 1:1 \
    --use_directed
```

## Parameters

### New Parameters

1. **`use_undirected`** (default: `True`)
   - If `True`: Use both `x→y` and `y→x` edges (doubles edge dataset size)
   - If `False`: Use only `x→y` edges (directed)
   - Only affects interleaved mode

2. **`train_val_split`** (default: `0.9`)
   - Fraction of data for training
   - Range: 0.0 to 1.0
   - Example: 0.9 = 90% train, 10% validation

### Removed Parameters

- **`train_size`**: Now calculated automatically
- **`val_size`**: Now calculated automatically

These parameters are **removed** from `InWeightsPathStar.prepare()` but **remain** in `InContextPathStar.prepare()`.

## Examples

### Example 1: Small Graph, Pure Paths

```python
generator = InWeightsPathStar(d=5, l=4)
generator.prepare()

# Output:
# Graph structure:
#   Total vertices: 16
#   Total edges: 15
#   Total paths (spokes): 5
#   Training paths: 5
# Dataset calculation:
#   Mode: Pure path prediction
#   Total dataset size: 5
# Split: 90% train, 10% validation
# Train size: 4, Val size: 1
```

### Example 2: Medium Graph with Holdout

```python
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)
generator.prepare()

# Output:
# Graph structure:
#   Total vertices: 41
#   Total edges: 40
#   Total paths (spokes): 10
#   Training paths: 8
#   Holdout paths: 2
# Dataset calculation:
#   Mode: Pure path prediction
#   Total dataset size: 8
# Split: 90% train, 10% validation
# Train size: 7, Val size: 1
```

### Example 3: Interleaved with Undirected Edges

```python
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)
generator.prepare(
    interleave_ratio=(3, 1),
    use_undirected=True
)

# Output:
# Graph structure:
#   Total vertices: 41
#   Total edges: 40
#   Total paths (spokes): 10
#   Training paths: 8
#   Holdout paths: 2
# Dataset calculation:
#   Interleave ratio: 3:1 (path:edge)
#   Full edge dataset: 80 (undirected)
#   Full path dataset: 8
#   Total dataset size: 88
# Split: 90% train, 10% validation
# Train size: 79, Val size: 9
```

### Example 4: Interleaved with Directed Edges

```python
generator = InWeightsPathStar(d=10, l=5)
generator.prepare(
    interleave_ratio=(1, 1),
    use_undirected=False  # Use directed edges
)

# Output:
# Graph structure:
#   Total vertices: 41
#   Total edges: 40
#   Total paths (spokes): 10
#   Training paths: 10
# Dataset calculation:
#   Interleave ratio: 1:1 (path:edge)
#   Full edge dataset: 40 (directed)
#   Full path dataset: 10
#   Total dataset size: 50
# Split: 90% train, 10% validation
# Train size: 45, Val size: 5
```

### Example 5: Custom Train/Val Split

```python
generator = InWeightsPathStar(d=20, l=6)
generator.prepare(
    interleave_ratio=(2, 1),
    train_val_split=0.8  # 80% train, 20% val
)

# Output:
# Graph structure:
#   Total vertices: 101
#   Total edges: 100
#   Total paths (spokes): 20
#   Training paths: 20
# Dataset calculation:
#   Interleave ratio: 2:1 (path:edge)
#   Full edge dataset: 200 (undirected)
#   Full path dataset: 20
#   Total dataset size: 220
# Split: 80% train, 20% validation
# Train size: 176, Val size: 44
```

## Metadata

The calculated sizes are saved in the metadata file:

```python
import pickle

with open('./data/inweights_pathstar_d10_l5/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print(meta['train_size'])        # e.g., 79
print(meta['val_size'])          # e.g., 9
print(meta['total_edges'])       # e.g., 40
print(meta['use_undirected'])    # e.g., True
print(meta['train_val_split'])   # e.g., 0.9
```

## Comparison: InContext vs InWeights

| Feature | InContextPathStar | InWeightsPathStar |
|---------|-------------------|-------------------|
| Dataset Size | Manual (train_size, val_size) | **Automatic** (calculated from graph) |
| Size Parameters | Required | Not needed |
| Graph-Dependent | No (arbitrary size) | Yes (based on structure) |
| Optimal Sizing | User must calculate | **Always optimal** |

### InContextPathStar (Unchanged)

```python
# Still requires manual sizes
generator = InContextPathStar(d=5, l=5, vocab_size=2000)
generator.prepare(
    train_size=100000,  # Required
    val_size=10000      # Required
)
```

### InWeightsPathStar (New Behavior)

```python
# Sizes calculated automatically
generator = InWeightsPathStar(d=5, l=5)
generator.prepare()  # No train_size/val_size needed!
```

## Benefits

### 1. Optimal Dataset Size

The dataset always uses the full capacity of the graph:
- All available training paths are used
- All edges are included (if interleaving)
- No wasted capacity, no oversized requests

### 2. Simplified API

No need to manually calculate or specify sizes:

**Before:**
```python
# User had to calculate manually
d, l = 10, 5
num_edges = d * (l - 1)  # 40
num_paths = d             # 10
train_size = ???          # What should this be?
val_size = ???            # And this?

generator.prepare(train_size=train_size, val_size=val_size)
```

**After:**
```python
# Automatic calculation
generator.prepare()  # Done!
```

### 3. Consistent Behavior

Different graphs automatically get appropriate sizes:

```python
# Small graph
gen1 = InWeightsPathStar(d=5, l=3)
gen1.prepare()  # Small dataset

# Large graph
gen2 = InWeightsPathStar(d=20, l=10)
gen2.prepare()  # Large dataset

# Both are optimally sized!
```

### 4. Holdout-Aware

Dataset size automatically accounts for holdout:

```python
# Without holdout
gen1 = InWeightsPathStar(d=10, l=5, holdout_percentage=0.0)
gen1.prepare()  # Uses all 10 paths

# With holdout
gen2 = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)
gen2.prepare()  # Uses only 8 training paths

# Sizes are automatically adjusted!
```

## Edge Cases

### Very Small Graphs

For very small graphs, validation set might be 0:

```python
generator = InWeightsPathStar(d=2, l=3)
generator.prepare()

# Output:
# Total dataset size: 2
# Train size: 1, Val size: 1
```

If you need more validation data, adjust the split:

```python
generator.prepare(train_val_split=0.5)  # 50/50 split
# Train size: 1, Val size: 1
```

### Large Graphs

For large graphs, dataset sizes scale automatically:

```python
generator = InWeightsPathStar(d=100, l=10)
generator.prepare(interleave_ratio=(1, 1))

# Output:
# Total edges: 900
# Undirected edges: 1800
# Total paths: 100
# Total dataset size: 1900
# Train size: 1710, Val size: 190
```

### Holdout = 100%

If all paths are held out, only edge data is available:

```python
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=1.0)
generator.prepare(interleave_ratio=(1, 1))

# Output:
# Training paths: 0
# Holdout paths: 10
# Full edge dataset: 80
# Full path dataset: 0
# Total dataset size: 80
```

## Migration Guide

### From Old API to New API

**Old Code (will break):**
```python
generator = InWeightsPathStar(d=10, l=5)
generator.prepare(
    train_size=100000,  # No longer accepted
    val_size=10000      # No longer accepted
)
```

**New Code:**
```python
generator = InWeightsPathStar(d=10, l=5)
generator.prepare(
    # Sizes calculated automatically
    train_val_split=0.9  # Optional: adjust split
)
```

### Adjusting Dataset Size

If you want a different train/val split:

```python
# 80% train, 20% validation
generator.prepare(train_val_split=0.8)

# 95% train, 5% validation
generator.prepare(train_val_split=0.95)

# 50% train, 50% validation
generator.prepare(train_val_split=0.5)
```

### Controlling Edge Direction

```python
# Undirected edges (default) - larger dataset
generator.prepare(use_undirected=True)

# Directed edges - smaller dataset
generator.prepare(use_undirected=False)
```

## Command-Line Reference

### Old Command (will break)

```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --train_size 100000 \  # No longer valid
    --val_size 10000       # No longer valid
```

### New Commands

**Basic usage:**
```bash
python pathstar.py --mode inweights --d 10 --l 5
```

**With custom split:**
```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --train_val_split 0.8
```

**With directed edges:**
```bash
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --use_directed
```

**Full example:**
```bash
python pathstar.py \
    --mode inweights \
    --d 20 \
    --l 6 \
    --holdout_percentage 0.2 \
    --interleave 3:1 \
    --use_directed \
    --train_val_split 0.85 \
    --output_dir ./data
```

## Summary

Automatic dataset sizing for `InWeightsPathStar`:

✅ **Always optimal**: Uses full graph capacity  
✅ **Simpler API**: No manual size calculation  
✅ **Consistent**: Appropriate for any graph size  
✅ **Holdout-aware**: Accounts for held-out paths  
✅ **Flexible**: Adjustable train/val split  
✅ **Transparent**: Prints detailed calculation  

The `train_size` and `val_size` parameters are **removed** from `InWeightsPathStar.prepare()` and **calculated automatically** based on graph structure.

