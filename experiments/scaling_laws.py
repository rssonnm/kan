"""
Scaling Laws Experiments (Section 3 / Fig 3.2).

Tests how accuracy scales with model parameters for KAN vs MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.kan_model import KAN


class MLP(nn.Module):
    """Standard MLP for comparison."""
    def __init__(self, layers_hidden):
        super().__init__()
        layers = []
        for i in range(len(layers_hidden) - 1):
            layers.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            if i < len(layers_hidden) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, x_train, y_train, x_test, y_test, epochs=500, lr=0.01):
    """Train model and return test loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        test_loss = criterion(model(x_test), y_test).item()
    
    return test_loss


def scaling_experiment(target_func, n_inputs=1, n_train=1000, n_test=200, 
                       kan_configs=None, mlp_configs=None, epochs=500):
    """
    Fig 3.2: Test scaling laws.
    
    Args:
        target_func: Function to approximate
        n_inputs: Number of input dimensions
        kan_configs: List of (hidden_dims, grid_size) tuples for KAN
        mlp_configs: List of hidden_dims lists for MLP
    Returns:
        results: Dict with parameter counts and test losses
    """
    # Default configs if not provided
    if kan_configs is None:
        kan_configs = [
            ([n_inputs, 3, 1], 3),
            ([n_inputs, 5, 1], 5),
            ([n_inputs, 8, 1], 5),
            ([n_inputs, 5, 5, 1], 5),
            ([n_inputs, 8, 8, 1], 8),
        ]
    
    if mlp_configs is None:
        mlp_configs = [
            [n_inputs, 8, 1],
            [n_inputs, 16, 1],
            [n_inputs, 32, 1],
            [n_inputs, 16, 16, 1],
            [n_inputs, 32, 32, 1],
            [n_inputs, 64, 64, 1],
        ]
    
    # Generate data
    x_train = torch.rand(n_train, n_inputs) * 2 - 1
    y_train = target_func(x_train)
    x_test = torch.rand(n_test, n_inputs) * 2 - 1
    y_test = target_func(x_test)
    
    results = {
        "kan_params": [],
        "kan_losses": [],
        "mlp_params": [],
        "mlp_losses": []
    }
    
    # Test KAN configurations
    print("Testing KAN configurations...")
    for hidden_dims, grid_size in kan_configs:
        model = KAN(layers_hidden=hidden_dims, grid_size=grid_size, spline_order=3)
        n_params = count_params(model)
        test_loss = train_and_eval(model, x_train, y_train, x_test, y_test, epochs=epochs)
        results["kan_params"].append(n_params)
        results["kan_losses"].append(test_loss)
        print(f"  KAN {hidden_dims} G={grid_size}: {n_params} params, loss={test_loss:.6f}")
    
    # Test MLP configurations
    print("Testing MLP configurations...")
    for hidden_dims in mlp_configs:
        model = MLP(layers_hidden=hidden_dims)
        n_params = count_params(model)
        test_loss = train_and_eval(model, x_train, y_train, x_test, y_test, epochs=epochs)
        results["mlp_params"].append(n_params)
        results["mlp_losses"].append(test_loss)
        print(f"  MLP {hidden_dims}: {n_params} params, loss={test_loss:.6f}")
    
    return results


def plot_scaling_laws(results, save_path=None):
    """
    Plot accuracy vs parameters (Fig 3.2 style).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KAN results
    ax.loglog(results["kan_params"], results["kan_losses"], 'bo-', markersize=8, 
              linewidth=2, label='KAN')
    
    # Plot MLP results
    ax.loglog(results["mlp_params"], results["mlp_losses"], 'rs-', markersize=8, 
              linewidth=2, label='MLP')
    
    ax.set_xlabel("# Parameters", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title("Scaling Laws: KAN vs MLP", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    
    return fig


def run_scaling_experiment():
    """Run the main scaling experiment."""
    # Test function: sin(πx)
    target_func = lambda x: torch.sin(torch.pi * x[:, 0:1])
    
    print("=" * 50)
    print("Scaling Laws Experiment: sin(πx)")
    print("=" * 50)
    
    results = scaling_experiment(target_func, n_inputs=1, epochs=500)
    plot_scaling_laws(results, save_path="/Users/sonn/Sonn/Workspace/Projects/kan/reports/scaling_laws.png")
    
    return results


if __name__ == "__main__":
    run_scaling_experiment()
