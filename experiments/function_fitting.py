"""
Function Fitting Experiments (Section 3 / Fig 3.1).

Compares KAN vs MLP on various test functions.
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
    """Standard Multi-Layer Perceptron for comparison."""
    def __init__(self, layers_hidden, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(layers_hidden) - 1):
            layers.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            if i < len(layers_hidden) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Test Functions from Feynman Dataset (Table 3 in paper)
TEST_FUNCTIONS = {
    "sin_pi_x": lambda x: torch.sin(torch.pi * x[:, 0:1]),
    "x_squared": lambda x: x[:, 0:1] ** 2,
    "exp_sin": lambda x: torch.exp(torch.sin(torch.pi * x[:, 0:1])),
    "xy": lambda x: x[:, 0:1] * x[:, 1:2],
    "sin_xy": lambda x: torch.sin(torch.pi * x[:, 0:1] * x[:, 1:2]),
    "sum_squares": lambda x: (x ** 2).sum(dim=1, keepdim=True),
    "feynman_I_6_2a": lambda x: torch.exp(-x[:, 0:1]**2 / 2) / torch.sqrt(torch.tensor(2 * np.pi)),
    "feynman_I_9_18": lambda x: x[:, 0:1] * x[:, 1:2] / (4 * np.pi * 8.85e-12 * x[:, 2:3]**2 + 1e-8),
}


def train_model(model, x_train, y_train, epochs=500, lr=0.01, reg_lambda=0.0):
    """Train a model and return loss history."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Add regularization for KAN
        if reg_lambda > 0 and hasattr(model, 'get_reg'):
            loss = loss + reg_lambda * model.get_reg()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def compare_kan_mlp(func_name, n_samples=1000, n_inputs=1, epochs=500):
    """
    Fig 3.1: Compare KAN and MLP on a test function.
    
    Args:
        func_name: Name of function from TEST_FUNCTIONS
        n_samples: Number of training samples
        n_inputs: Number of input dimensions
        epochs: Training epochs
    Returns:
        results: Dict with training histories and final losses
    """
    # Generate data
    x_train = torch.rand(n_samples, n_inputs) * 2 - 1  # [-1, 1]
    func = TEST_FUNCTIONS[func_name]
    y_train = func(x_train)
    
    # Set up models with similar parameter counts
    # KAN: [n_inputs, 5, 1] with G=5, k=3 -> ~90 params for n=1
    # MLP: [n_inputs, 16, 16, 1] -> ~305 params for n=1
    
    kan_model = KAN(layers_hidden=[n_inputs, 5, 1], grid_size=5, spline_order=3)
    mlp_model = MLP(layers_hidden=[n_inputs, 16, 16, 1])
    
    kan_params = sum(p.numel() for p in kan_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    
    print(f"\n=== {func_name} ===")
    print(f"KAN params: {kan_params}, MLP params: {mlp_params}")
    
    # Train
    kan_losses = train_model(kan_model, x_train, y_train, epochs=epochs)
    mlp_losses = train_model(mlp_model, x_train, y_train, epochs=epochs)
    
    # Test
    x_test = torch.rand(200, n_inputs) * 2 - 1
    y_test = func(x_test)
    
    with torch.no_grad():
        kan_test_loss = nn.MSELoss()(kan_model(x_test), y_test).item()
        mlp_test_loss = nn.MSELoss()(mlp_model(x_test), y_test).item()
    
    print(f"KAN test loss: {kan_test_loss:.6f}")
    print(f"MLP test loss: {mlp_test_loss:.6f}")
    print(f"KAN/MLP ratio: {kan_test_loss / (mlp_test_loss + 1e-8):.2f}x")
    
    return {
        "function": func_name,
        "kan_losses": kan_losses,
        "mlp_losses": mlp_losses,
        "kan_test_loss": kan_test_loss,
        "mlp_test_loss": mlp_test_loss,
        "kan_params": kan_params,
        "mlp_params": mlp_params
    }


def plot_comparison(results, save_path=None):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training curves
    axes[0].semilogy(results["kan_losses"], label=f"KAN ({results['kan_params']} params)")
    axes[0].semilogy(results["mlp_losses"], label=f"MLP ({results['mlp_params']} params)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"Training: {results['function']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final comparison bar
    axes[1].bar(["KAN", "MLP"], [results["kan_test_loss"], results["mlp_test_loss"]], 
                color=["blue", "orange"])
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Final Test Loss")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    return fig


def run_all_experiments(save_dir="/Users/sonn/Sonn/Workspace/Projects/kan/reports"):
    """Run experiments on all test functions."""
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    for func_name in ["sin_pi_x", "x_squared", "exp_sin"]:
        n_inputs = 2 if "xy" in func_name else 1
        results = compare_kan_mlp(func_name, n_inputs=n_inputs)
        all_results.append(results)
        plot_comparison(results, save_path=f"{save_dir}/{func_name}_comparison.png")
    
    return all_results


if __name__ == "__main__":
    run_all_experiments()
