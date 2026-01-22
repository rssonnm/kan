"""
Visualization utilities for KAN networks.

Implements:
- Fig 2.3: Pruning visualization
- Activation function plotting
- Network structure visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection


def plot_activation_function(layer, edge_idx, x_range=(-1, 1), n_points=100, ax=None):
    """
    Plot the learned activation function for a specific edge.
    
    Args:
        layer: KANLinear layer
        edge_idx: Tuple (in_idx, out_idx) specifying the edge
        x_range: Range of x values to plot
        n_points: Number of points to sample
        ax: Matplotlib axis (created if None)
    Returns:
        ax: Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    in_idx, out_idx = edge_idx
    
    # Sample x values
    x = torch.linspace(x_range[0], x_range[1], n_points)
    
    with torch.no_grad():
        # Create input with only the relevant dimension active
        x_input = torch.zeros(n_points, layer.in_features)
        x_input[:, in_idx] = x
        
        # Get base function contribution
        y_base = layer.base_fun(x_input[:, in_idx]) * layer.scale_base[out_idx, in_idx]
        
        # Get spline contribution (simplified)
        from utils.spline_utils import B_batch
        splines = B_batch(x_input, layer.grid, layer.spline_order)
        y_spline = torch.einsum('bik,k->b', splines[:, in_idx:in_idx+1, :], 
                                layer.spline_weight[out_idx, in_idx, :])
        y_spline = y_spline * layer.scale_spline[out_idx, in_idx]
        
        y_total = y_base + y_spline
    
    ax.plot(x.numpy(), y_base.numpy(), 'b--', alpha=0.5, label='Base (SiLU)')
    ax.plot(x.numpy(), y_spline.numpy(), 'g--', alpha=0.5, label='Spline')
    ax.plot(x.numpy(), y_total.numpy(), 'r-', linewidth=2, label='Total φ(x)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('φ(x)')
    ax.set_title(f'Edge ({in_idx}, {out_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_kan_structure(model, figsize=(12, 8), show_activations=True):
    """
    Visualize the KAN network structure with edge importance.
    
    Similar to Fig 0.1(d) in the paper.
    
    Args:
        model: KAN model
        figsize: Figure size
        show_activations: If True, show activation functions on edges
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect layer dimensions
    layer_dims = [model.layers[0].in_features]
    for layer in model.layers:
        layer_dims.append(layer.out_features)
    
    n_layers = len(layer_dims)
    max_neurons = max(layer_dims)
    
    # Node positions
    node_positions = {}
    for l, n_neurons in enumerate(layer_dims):
        x = l / (n_layers - 1) if n_layers > 1 else 0.5
        for i in range(n_neurons):
            y = (i + 0.5) / max_neurons if max_neurons > 0 else 0.5
            node_positions[(l, i)] = (x, y)
    
    # Draw edges with importance-based colors
    for l, layer in enumerate(model.layers):
        importance = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
        importance = importance / (importance.max() + 1e-8)  # Normalize
        
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                x1, y1 = node_positions[(l, i)]
                x2, y2 = node_positions[(l + 1, j)]
                
                # Color based on importance (masked edges are gray)
                if hasattr(layer, 'mask') and layer.mask[j, i] < 0.5:
                    color = 'lightgray'
                    alpha = 0.3
                else:
                    imp = importance[j, i].item()
                    color = plt.cm.viridis(imp)
                    alpha = 0.3 + 0.7 * imp
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=2)
    
    # Draw nodes
    for (l, i), (x, y) in node_positions.items():
        circle = plt.Circle((x, y), 0.02, color='white', ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'{i}', ha='center', va='center', fontsize=8, zorder=11)
    
    # Layer labels
    for l, n_neurons in enumerate(layer_dims):
        x = l / (n_layers - 1) if n_layers > 1 else 0.5
        ax.text(x, -0.05, f'Layer {l}\n({n_neurons})', ha='center', va='top', fontsize=10)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('KAN Network Structure (edge color = importance)')
    
    return fig, ax


def plot_pruning_comparison(model_before, model_after, x_samples, figsize=(14, 5)):
    """
    Fig 2.3: Before/after pruning comparison.
    
    Args:
        model_before: Model before pruning
        model_after: Model after pruning
        x_samples: Sample inputs for visualization
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Before pruning structure
    # (simplified - just show edge counts)
    axes[0].set_title('Before Pruning')
    axes[0].text(0.5, 0.5, f'Active edges: {sum(l.mask.sum().item() for l in model_before.layers):.0f}',
                 ha='center', va='center', fontsize=14)
    axes[0].axis('off')
    
    # After pruning structure
    axes[1].set_title('After Pruning')
    axes[1].text(0.5, 0.5, f'Active edges: {sum(l.mask.sum().item() for l in model_after.layers):.0f}',
                 ha='center', va='center', fontsize=14)
    axes[1].axis('off')
    
    # Prediction comparison
    with torch.no_grad():
        y_before = model_before(x_samples)
        y_after = model_after(x_samples)
    
    axes[2].scatter(y_before.numpy(), y_after.numpy(), alpha=0.5)
    axes[2].plot([y_before.min(), y_before.max()], [y_before.min(), y_before.max()], 'r--')
    axes[2].set_xlabel('Before Pruning')
    axes[2].set_ylabel('After Pruning')
    axes[2].set_title('Prediction Comparison')
    
    plt.tight_layout()
    return fig


def plot_grid_extension(layer, x_samples, figsize=(10, 4)):
    """
    Visualize grid refinement (Section 2.5).
    
    Shows old grid vs new grid positions.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot grid positions for first input dimension
    grid = layer.grid[0].numpy()
    axes[0].eventplot([grid], orientation='horizontal', colors='blue')
    axes[0].set_title(f'Current Grid ({len(grid)} knots)')
    axes[0].set_xlabel('x')
    
    # Histogram of sample distribution
    axes[1].hist(x_samples[:, 0].numpy(), bins=30, alpha=0.7, density=True)
    for g in grid:
        axes[1].axvline(g, color='red', linestyle='--', alpha=0.3)
    axes[1].set_title('Sample Distribution vs Grid')
    axes[1].set_xlabel('x')
    
    plt.tight_layout()
    return fig
