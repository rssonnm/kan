"""
KAN Network Visualization with Activation Functions on Edges.

Implements the visualization style from the paper (Figure 0.1d, 2.2)
showing nodes, edges, and learned activation functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.spline_utils import B_batch


def plot_kan_diagram(model, x_range=(-1, 1), n_points=50, figsize=(14, 10), 
                     save_path=None, title="KAN Network"):
    """
    Create a paper-style KAN network diagram with activation functions on edges.
    
    This replicates the visualization from Figure 0.1(d) and Figure 2.2 of the paper,
    showing:
    - Nodes arranged in layers (x_{l,i})
    - Edges connecting nodes with small plots of learned φ functions
    - Pre-activation values (x̃_{l,i,j})
    
    Args:
        model: KAN model
        x_range: Range for plotting activation functions
        n_points: Number of points for activation curves
        figsize: Figure size
        save_path: Optional path to save the figure
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
    
    # Layout parameters
    layer_spacing = 3.0
    neuron_spacing = 1.5
    node_radius = 0.15
    activation_box_size = 0.4
    
    # Calculate node positions (bottom-up: input at bottom, output at top)
    node_positions = {}
    for l in range(n_layers):
        n_neurons = layer_dims[l]
        y = l * layer_spacing
        # Center neurons horizontally
        start_x = -(n_neurons - 1) * neuron_spacing / 2
        for i in range(n_neurons):
            x = start_x + i * neuron_spacing
            node_positions[(l, i)] = (x, y)
    
    # Draw edges with activation function plots
    x_samples = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
    
    for l, layer in enumerate(model.layers):
        with torch.no_grad():
            # Get spline basis for this layer
            x_full = x_samples.repeat(1, layer.in_features)
            splines = B_batch(x_full, layer.grid, layer.spline_order)
        
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                x1, y1 = node_positions[(l, i)]
                x2, y2 = node_positions[(l + 1, j)]
                
                # Draw edge line
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5, alpha=0.3, zorder=1)
                
                # Calculate midpoint for activation box
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Get activation function values
                with torch.no_grad():
                    # Base component
                    y_base = layer.base_fun(x_samples.squeeze()) * layer.scale_base[j, i]
                    
                    # Spline component
                    spline_coef = layer.spline_weight[j, i, :]
                    y_spline = torch.einsum('bk,k->b', splines[:, i, :], spline_coef)
                    y_spline = y_spline * layer.scale_spline[j, i]
                    
                    y_total = y_base + y_spline
                
                # Draw activation function box
                box_x = mid_x - activation_box_size / 2
                box_y = mid_y - activation_box_size / 2
                
                # Create mini axes for activation plot
                # Convert data coordinates to figure coordinates
                trans = ax.transData.transform
                inv_trans = fig.transFigure.inverted().transform
                
                box_left, box_bottom = inv_trans(trans((box_x, box_y)))
                box_width = inv_trans(trans((box_x + activation_box_size, box_y)))[0] - box_left
                box_height = inv_trans(trans((box_x, box_y + activation_box_size)))[1] - box_bottom
                
                # Create inset axes for activation function
                inset_ax = fig.add_axes([box_left, box_bottom, box_width, box_height])
                inset_ax.plot(x_samples.numpy(), y_total.numpy(), 'r-', linewidth=0.8)
                inset_ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
                inset_ax.set_xlim(x_range)
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                inset_ax.patch.set_facecolor('white')
                inset_ax.patch.set_edgecolor('black')
                inset_ax.patch.set_linewidth(0.5)
                
                # Add label for activation function
                label = f'$\\phi_{{_{l},{i},{j}}}$'
                ax.text(mid_x, mid_y + activation_box_size/2 + 0.1, label, 
                       ha='center', va='bottom', fontsize=6, color='red')
    
    # Draw nodes
    for (l, i), (x, y) in node_positions.items():
        circle = Circle((x, y), node_radius, facecolor='white', edgecolor='black', 
                        linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        
        # Node label
        if l == 0:
            label = f'$x_{{0,{i+1}}}$'
        elif l == n_layers - 1:
            label = f'$x_{{{l},{i+1}}}$'
        else:
            label = f'$x_{{{l},{i+1}}}$'
        
        ax.text(x, y - node_radius - 0.2, label, ha='center', va='top', fontsize=10)
    
    # Set axis properties
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    return fig, ax


def plot_kan_simple(model, figsize=(12, 8), save_path=None, title="KAN Network Structure"):
    """
    Simplified KAN network visualization (without activation plots).
    
    Shows nodes and edges with color intensity based on edge importance.
    
    Args:
        model: KAN model
        figsize: Figure size
        save_path: Optional path to save
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect layer dimensions
    layer_dims = [model.layers[0].in_features]
    for layer in model.layers:
        layer_dims.append(layer.out_features)
    
    n_layers = len(layer_dims)
    
    # Layout
    layer_spacing = 2.0
    neuron_spacing = 1.0
    node_radius = 0.12
    
    # Node positions
    node_positions = {}
    for l in range(n_layers):
        n_neurons = layer_dims[l]
        y = l * layer_spacing
        start_x = -(n_neurons - 1) * neuron_spacing / 2
        for i in range(n_neurons):
            x = start_x + i * neuron_spacing
            node_positions[(l, i)] = (x, y)
    
    # Draw edges with importance-based colors
    for l, layer in enumerate(model.layers):
        with torch.no_grad():
            importance = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
            importance = importance / (importance.max() + 1e-8)
        
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                x1, y1 = node_positions[(l, i)]
                x2, y2 = node_positions[(l + 1, j)]
                
                imp = importance[j, i].item()
                
                # Check mask
                if hasattr(layer, 'mask') and layer.mask[j, i] < 0.5:
                    color = 'lightgray'
                    alpha = 0.2
                    linewidth = 0.5
                else:
                    color = plt.cm.Reds(0.3 + 0.7 * imp)
                    alpha = 0.4 + 0.6 * imp
                    linewidth = 0.5 + 2 * imp
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=linewidth, zorder=1)
    
    # Draw nodes
    for (l, i), (x, y) in node_positions.items():
        circle = Circle((x, y), node_radius, facecolor='white', edgecolor='black', 
                        linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=8, zorder=11)
    
    # Layer labels
    for l in range(n_layers):
        y = l * layer_spacing
        ax.text(min(node_positions.values(), key=lambda p: p[0])[0] - 0.8, y,
               f'Layer {l}', ha='right', va='center', fontsize=10, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    return fig, ax


def demo_visualization():
    """Demo the visualization with a trained KAN model."""
    from modules.kan_model import KAN
    import torch.optim as optim
    
    # Create and train a small KAN
    print("Creating and training a [2, 5, 1] KAN...")
    model = KAN(layers_hidden=[2, 5, 1], grid_size=5, spline_order=3)
    
    # Train on a simple function
    x = torch.rand(200, 2) * 2 - 1
    y = torch.sin(torch.pi * x[:, 0:1]) * torch.cos(torch.pi * x[:, 1:2])
    
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    print(f"Training complete. Final loss: {loss.item():.6f}")
    
    # Generate visualizations
    print("Generating simple visualization...")
    plot_kan_simple(model, save_path="/Users/sonn/Sonn/Workspace/Projects/kan/reports/kan_structure_simple.png")
    
    print("Generating detailed visualization with activation plots...")
    plot_kan_diagram(model, save_path="/Users/sonn/Sonn/Workspace/Projects/kan/reports/kan_structure_detailed.png")
    
    print("Done!")


if __name__ == "__main__":
    demo_visualization()
