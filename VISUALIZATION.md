# PathStar Graph Visualization

## Overview

The `InWeightsPathStar` class includes a `visualize()` method that creates beautiful visualizations of the path-star graph structure. This helps you understand the graph topology, see the train/holdout split, and verify the graph configuration before generating datasets.

## Features

- **Radial Layout**: Root node at center with spokes radiating outward
- **Color-Coded Nodes**: Different colors for root, training leaves, holdout leaves, and intermediate nodes
- **Statistics Display**: Shows graph metrics (vertices, edges, paths, train/holdout counts)
- **Legend**: Clear identification of node types
- **High-Quality Output**: Saves as PNG with 300 DPI

## Usage

### Command Line

Use the `--viz` flag to visualize the graph before dataset generation:

```bash
# Basic visualization
python pathstar.py --mode inweights --d 10 --l 5 --viz

# With holdout
python pathstar.py --mode inweights --d 10 --l 5 --train_val_split 0.8 --viz

# Specify output directory
python pathstar.py --mode inweights --d 10 --l 5 --viz --output_dir ./my_data
```

The visualization will be saved to: `{output_dir}/pathstar_d{d}_l{l}_viz.png`

### Python API

```python
from pathstar import InWeightsPathStar

# Create generator
generator = InWeightsPathStar(d=10, l=5, holdout_percentage=0.2)

# Visualize and save to file
generator.visualize(output_path='./pathstar_viz.png')

# Or display interactively (if running in Jupyter or with display)
generator.visualize()

# Customize figure size and labels
generator.visualize(
    output_path='./pathstar_viz.png',
    figsize=(16, 12),
    show_labels=True
)
```

## Visualization Elements

### Node Colors

- **Red (#FF6B6B)**: Root node
- **Green (#6BCB77)**: Training leaf nodes
- **Yellow (#FFD93D)**: Holdout leaf nodes
- **Light Teal (#95E1D3)**: Intermediate nodes

### Layout

The visualization uses a radial layout:
- Root node positioned at the center (0, 0)
- Each spoke extends outward at equal angular intervals
- Nodes positioned along their respective spokes at increasing radii
- Distance from center corresponds to depth in the path

### Information Displayed

**Title**: Shows graph parameters (d, l, holdout percentage)

**Legend**: Identifies node types and counts

**Statistics Box**: Shows:
- Total vertices
- Total edges
- Total paths
- Training paths count
- Holdout paths count

## Examples

### Example 1: Small Graph

```bash
python pathstar.py --mode inweights --d 5 --l 4 --viz
```

Creates a visualization showing:
- 5 spokes radiating from center
- Each spoke with 4 nodes (including root)
- All leaves in green (no holdout)

### Example 2: Graph with Holdout

```bash
python pathstar.py --mode inweights --d 10 --l 5 --train_val_split 0.8 --viz
```

Creates a visualization showing:
- 10 spokes radiating from center
- 8 training leaves (green)
- 2 holdout leaves (yellow)
- Clear visual separation of train/holdout

### Example 3: Large Graph

```bash
python pathstar.py --mode inweights --d 20 --l 10 --train_val_split 0.9 --viz
```

Creates a visualization showing:
- 20 spokes (more densely packed)
- Longer paths (10 nodes each)
- 18 training leaves, 2 holdout leaves

### Example 4: Custom Mapping

```python
import random
from pathstar import InWeightsPathStar

# Create custom mapping
num_vertices = 10 * (5 - 1) + 1  # 41 vertices
canonical_nodes = list(range(num_vertices))
vocab_tokens = random.sample(range(2000), num_vertices)
mapping = dict(zip(canonical_nodes, vocab_tokens))

# Create generator with mapping
generator = InWeightsPathStar(
    d=10,
    l=5,
    holdout_percentage=0.2,
    mapping=mapping
)

# Visualize (shows mapped node IDs)
generator.visualize(output_path='./pathstar_mapped_viz.png')
```

## Visualization Method Parameters

```python
def visualize(self, output_path=None, figsize=(12, 10), show_labels=True):
    """
    Visualize the path-star graph structure.
    
    Args:
        output_path: If provided, save the figure to this path. Otherwise, display it.
        figsize: Figure size as (width, height) tuple (default: (12, 10))
        show_labels: If True, show node labels (default: True)
    """
```

### Parameters

- **`output_path`** (str, optional): 
  - If provided, saves the visualization to this file path
  - If `None`, displays the visualization interactively
  - Recommended format: `.png` for best quality

- **`figsize`** (tuple, default: `(12, 10)`):
  - Figure size in inches as `(width, height)`
  - Larger sizes for graphs with many nodes
  - Examples: `(16, 12)` for large graphs, `(8, 8)` for small graphs

- **`show_labels`** (bool, default: `True`):
  - If `True`, displays node IDs on each node
  - Set to `False` for cleaner visualization of large graphs

## Use Cases

### 1. Verify Graph Structure

Before generating a large dataset, visualize to ensure the graph is constructed correctly:

```bash
python pathstar.py --mode inweights --d 10 --l 100 --viz
```

Check that:
- Correct number of spokes
- Correct path lengths
- Proper train/holdout split

### 2. Understand Holdout Split

Visualize to see which paths are held out:

```bash
python pathstar.py --mode inweights --d 10 --l 5 --train_val_split 0.7 --viz
```

Yellow nodes show which paths are held out for validation.

### 3. Compare Different Configurations

Generate visualizations for different parameters:

```bash
# Configuration 1
python pathstar.py --mode inweights --d 5 --l 5 --viz --output_dir ./viz1

# Configuration 2
python pathstar.py --mode inweights --d 10 --l 5 --viz --output_dir ./viz2

# Configuration 3
python pathstar.py --mode inweights --d 5 --l 10 --viz --output_dir ./viz3
```

Compare the visualizations side-by-side.

### 4. Documentation and Presentations

Create high-quality visualizations for papers or presentations:

```python
generator = InWeightsPathStar(d=8, l=6, holdout_percentage=0.25)
generator.visualize(
    output_path='./paper_figure.png',
    figsize=(16, 14),
    show_labels=False  # Cleaner for presentations
)
```

### 5. Debug Mapping Issues

If using custom node mappings, visualize to verify the mapping is correct:

```python
# Create generator with mapping
generator = InWeightsPathStar(d=5, l=4, mapping=my_mapping)

# Visualize to see mapped node IDs
generator.visualize(output_path='./debug_mapping.png')
```

## Tips

### For Large Graphs

When visualizing graphs with many nodes:

1. **Increase figure size**:
   ```python
   generator.visualize(figsize=(20, 18))
   ```

2. **Hide labels** for cleaner visualization:
   ```python
   generator.visualize(show_labels=False)
   ```

3. **Use higher DPI** (already set to 300):
   The visualization automatically saves at 300 DPI for high quality.

### For Small Graphs

When visualizing small graphs:

1. **Use smaller figure size**:
   ```python
   generator.visualize(figsize=(8, 8))
   ```

2. **Keep labels visible**:
   ```python
   generator.visualize(show_labels=True)
   ```

### Interactive Display

In Jupyter notebooks or interactive Python sessions:

```python
# Display interactively (no output_path)
generator.visualize()
```

This will open a matplotlib window showing the visualization.

## Output Format

The visualization is saved as a PNG file with:
- **Resolution**: 300 DPI (high quality for printing)
- **Format**: PNG (lossless compression)
- **Size**: Depends on `figsize` parameter
- **Naming**: `pathstar_d{d}_l{l}_viz.png` when using `--viz` flag

## Dependencies

The visualization feature requires:
- `matplotlib >= 3.9.0`
- `networkx >= 3.5`
- `numpy` (already required)

These are included in `requirements.txt`.

## Troubleshooting

### Issue: "No display found"

If running on a headless server without display:

```python
# Always save to file instead of displaying
generator.visualize(output_path='./output.png')
```

Or set matplotlib backend:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Issue: Overlapping labels

For graphs with many nodes, labels may overlap:

```python
# Disable labels
generator.visualize(show_labels=False)

# Or increase figure size
generator.visualize(figsize=(20, 20))
```

### Issue: Nodes too small/large

The node size is fixed at 500. To adjust, modify the `visualize()` method:

```python
# In the visualize() method, change:
nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                      node_size=500,  # Change this value
                      alpha=0.9, ax=ax)
```

## Example Output

A typical visualization shows:

```
┌─────────────────────────────────────────────────┐
│ PathStar Graph: d=10, l=5, holdout=20%         │
├─────────────────────────────────────────────────┤
│                                                 │
│              ●  ●  ●  ●                        │
│             /  /  |  \  \                      │
│            ●  ●   ●   ●  ●                     │
│             \ |   |   | /                      │
│              ●    ●    ●                       │
│               \   |   /                        │
│                \  |  /                         │
│                  ●●●  (root)                   │
│                /  |  \                         │
│               /   |   \                        │
│              ●    ●    ●                       │
│             / |   |   | \                      │
│            ●  ●   ●   ●  ●                     │
│             \  \  |  /  /                      │
│              ●  ●  ●  ●                        │
│                                                 │
│  Legend:                                        │
│  ● Red: Root                                    │
│  ● Green: Training Leaves (8)                   │
│  ● Yellow: Holdout Leaves (2)                   │
│  ● Teal: Intermediate Nodes                     │
│                                                 │
│  Stats:                                         │
│  Vertices: 41                                   │
│  Edges: 40                                      │
│  Paths: 10                                      │
│  Train: 8                                       │
│  Holdout: 2                                     │
└─────────────────────────────────────────────────┘
```

## Summary

The visualization feature:

✅ **Easy to use**: Single `--viz` flag or method call  
✅ **Informative**: Shows structure, colors, and statistics  
✅ **High quality**: 300 DPI PNG output  
✅ **Customizable**: Adjustable size and labels  
✅ **Integrated**: Works seamlessly with dataset generation  

Use it to verify your graph configuration before generating large datasets!

