# Holdout Feature for InWeightsPathStar

## Overview

The `InWeightsPathStar` class now supports holding out a percentage of paths for testing generalization. This feature allows you to:

1. **Train on a subset of paths** while holding out others for evaluation
2. **Test generalization** to unseen paths that share edges with training paths
3. **Control data leakage** by ensuring holdout paths don't appear in path prediction training data

## Key Features

### 1. Holdout Percentage

When creating an `InWeightsPathStar` instance, you can specify a `holdout_percentage` (0.0 to 1.0):

```python
from pathstar import InWeightsPathStar

# Hold out 20% of paths
generator = InWeightsPathStar(d=5, l=5, holdout_percentage=0.2)

# This will randomly select 1 path (20% of 5) as holdout
# The remaining 4 paths are available for training
```

### 2. Path Prediction Training Modes

The `generate_path_prediction_training_set()` method now supports three modes:

#### Mode 1: Training Only (default)
```python
# Only sample from training paths (respects holdout)
train_data = generator.generate_path_prediction_training_set(
    size=1000,
    obey_holdout=True  # Default
)
```

#### Mode 2: Holdout Only
```python
# Only sample from holdout paths (for evaluation)
holdout_data = generator.generate_path_prediction_training_set(
    size=100,
    holdout_only=True
)
```

#### Mode 3: All Paths (ignore holdout)
```python
# Sample from all paths (ignores holdout split)
all_data = generator.generate_path_prediction_training_set(
    size=1000,
    obey_holdout=False
)
```

### 3. Edge Memorization (No Restrictions)

**Important:** The `generate_edge_memorization_training_set()` method is **not affected** by holdout. All edges, including those in holdout paths, are available for edge memorization training.

```python
# This includes edges from both training and holdout paths
edge_data = generator.generate_edge_memorization_training_set(
    size=1000,
    undirected=True
)
```

This design allows you to:
- Train edge representations on the full graph
- Test path prediction generalization on unseen paths
- Study how well the model can compose learned edges into new paths

## Usage Examples

### Example 1: Basic Holdout Setup

```python
import random
from pathstar import InWeightsPathStar

# Set seed for reproducibility
random.seed(42)

# Create generator with 20% holdout
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)

print(f"Total paths: {generator.d}")
print(f"Training paths: {len(generator.train_leaves)}")
print(f"Holdout paths: {len(generator.holdout_leaves)}")

# Output:
# Total paths: 10
# Training paths: 8
# Holdout paths: 2
```

### Example 2: Training and Evaluation Split

```python
# Generate training data (only from training paths)
train_sequences = generator.generate_path_prediction_training_set(
    size=8000,  # Can sample with replacement from 8 training paths
    num_pause_tokens=1,
    obey_holdout=True
)

# Generate evaluation data (only from holdout paths)
eval_sequences = generator.generate_path_prediction_training_set(
    size=2000,  # Can sample with replacement from 2 holdout paths
    num_pause_tokens=1,
    holdout_only=True
)

# Train model on train_sequences
# Evaluate model on eval_sequences to test generalization
```

### Example 3: Edge Memorization + Path Generalization

```python
# Phase 1: Train on edges (includes all edges)
edge_x, edge_y = generator.generate_edge_memorization_training_set(
    size=10000,
    undirected=True
)
# Train model to predict edges: given x, predict y

# Phase 2: Train on paths (only training paths)
path_sequences = generator.generate_path_prediction_training_set(
    size=5000,
    obey_holdout=True
)
# Train model to predict full paths

# Phase 3: Evaluate on holdout paths
holdout_sequences = generator.generate_path_prediction_training_set(
    size=1000,
    holdout_only=True
)
# Evaluate: Can the model predict paths it hasn't seen?
```

### Example 4: Command Line Usage

```bash
# Generate dataset with 30% holdout
python pathstar.py \
    --mode inweights \
    --d 10 \
    --l 5 \
    --holdout_percentage 0.3 \
    --train_size 100000 \
    --val_size 10000 \
    --output_dir ./data
```

## Error Handling

The implementation includes validation to prevent invalid configurations:

### Error 1: Size Exceeds Available Paths

```python
generator = InWeightsPathStar(d=5, l=5, holdout_percentage=0.2)
# 4 training paths, 1 holdout path

# This will raise ValueError
try:
    data = generator.generate_path_prediction_training_set(
        size=10,  # Requesting 10 unique paths
        obey_holdout=True  # But only 4 training paths available
    )
except ValueError as e:
    print(e)
    # "Requested size (10) exceeds the number of available training paths (4)"
```

### Error 2: No Holdout Paths Available

```python
generator = InWeightsPathStar(d=5, l=5, holdout_percentage=0.0)
# No holdout paths

# This will raise ValueError
try:
    data = generator.generate_path_prediction_training_set(
        size=1,
        holdout_only=True  # But no holdout paths exist
    )
except ValueError as e:
    print(e)
    # "Cannot generate holdout_only data: no holdout paths available"
```

### Error 3: Invalid Holdout Percentage

```python
# This will raise ValueError
try:
    generator = InWeightsPathStar(d=5, l=5, holdout_percentage=1.5)
except ValueError as e:
    print(e)
    # "holdout_percentage must be between 0.0 and 1.0, got 1.5"
```

## Metadata Storage

When using the `prepare()` method, holdout information is saved in the metadata:

```python
generator = InWeightsPathStar(d=5, l=5, holdout_percentage=0.2)
output_dir = generator.prepare(
    train_size=100000,
    val_size=10000,
    output_dir='./data'
)

# The meta.pkl file will contain:
# - 'holdout_percentage': 0.2
# - 'train_leaves': [list of training leaf nodes]
# - 'holdout_leaves': [list of holdout leaf nodes]
```

You can load this metadata later to reconstruct the same train/holdout split:

```python
import pickle

with open('./data/inweights_pathstar_d5_l5/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print(f"Holdout percentage: {meta['holdout_percentage']}")
print(f"Training leaves: {meta['train_leaves']}")
print(f"Holdout leaves: {meta['holdout_leaves']}")
```

## Implementation Details

### Holdout Selection

- Holdout paths are selected **randomly** during initialization
- The number of holdout paths is `int(d * holdout_percentage)`
- Selection uses `random.sample()` without replacement
- The same random seed will produce the same holdout split

### Attributes Added

The `InWeightsPathStar` class now has these additional attributes:

- `holdout_percentage`: The holdout percentage (0.0 to 1.0)
- `train_leaves`: List of leaf nodes available for training
- `holdout_leaves`: List of leaf nodes held out for evaluation

### String Representation

The `__str__()` method now shows holdout information:

```python
generator = InWeightsPathStar(d=5, l=5, holdout_percentage=0.2)
print(generator)

# Output includes:
# InWeightsPathStar(d=5, l=5, holdout_percentage=0.2)
#   Train leaves: [0, 4, 8, 12] (4 paths)
#   Holdout leaves: [16] (1 paths)
#   ...
#   Paths by Leaf:
#     Leaf 0: [0, 1, 2, 3, 4]
#     Leaf 8: [0, 5, 6, 7, 8]
#     Leaf 12: [0, 9, 10, 11, 12]
#     Leaf 16: [0, 13, 14, 15, 16] [HOLDOUT]
```

## Use Cases

### 1. Compositional Generalization Testing

Test whether a model can compose learned edges into unseen paths:

```python
# Train on edges + subset of paths
# Evaluate on held-out paths
# Question: Can the model generalize to new path compositions?
```

### 2. Few-Shot Path Learning

Study how many path examples are needed when edges are known:

```python
# Train on all edges
# Train on k training paths (vary k)
# Evaluate on holdout paths
# Question: How does performance scale with number of path examples?
```

### 3. Transfer Learning

Pre-train on edges, fine-tune on paths:

```python
# Phase 1: Pre-train on edge prediction (all edges)
# Phase 2: Fine-tune on path prediction (training paths only)
# Phase 3: Evaluate on holdout paths
# Question: Does edge pre-training help path generalization?
```

## Notes

- **Sampling with replacement**: The `generate_path_prediction_training_set()` method samples with replacement, so you can request `size > len(train_leaves)`. However, the validation ensures you don't request more unique paths than available.
- **Edge memorization unchanged**: The holdout split does NOT affect edge memorization training. This is intentional to allow studying edge-to-path composition.
- **Deterministic splits**: Use `random.seed()` before creating the generator to ensure reproducible train/holdout splits.
- **Metadata preservation**: The holdout split is saved in metadata, allowing you to reconstruct the exact split later.

