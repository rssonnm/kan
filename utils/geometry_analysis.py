"""
KAN Geometry Analysis
=====================

Explore and compare the geometry of KAN vs MLP architectures.

Features:
- Loss landscape visualization
- Gradient flow analysis
- Representation geometry
- Function approximation analysis
- Activation pattern visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
from copy import deepcopy


# =============================================================================
# MLP BASELINE FOR COMPARISON
# =============================================================================

class MLP(nn.Module):
    """Simple MLP for comparison with KAN."""
    
    def __init__(self, layers: List[int], activation: str = 'relu'):
        super().__init__()
        
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i + 1]))
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers_list[:-1]):
            x = self.activation(layer(x))
        return self.layers_list[-1](x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# GEOMETRY METRICS
# =============================================================================

def compute_gradient_norm(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute gradient norm at current point."""
    model.zero_grad()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    return np.sqrt(total_norm)


def compute_hessian_trace(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                           n_samples: int = 10) -> float:
    """
    Approximate Hessian trace using Hutchinson's estimator.
    Tr(H) ≈ E[v^T H v] where v is random ±1 vector.
    """
    model.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # First gradient
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])
    
    trace_est = 0.0
    for _ in range(n_samples):
        v = torch.randint(0, 2, flat_grad.shape).float() * 2 - 1
        Hv = torch.autograd.grad(flat_grad @ v, model.parameters(), retain_graph=True)
        flat_Hv = torch.cat([h.flatten() for h in Hv])
        trace_est += (v * flat_Hv).sum().item()
    
    return trace_est / n_samples


def compute_loss_sharpness(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                            radius: float = 0.01, n_directions: int = 20) -> Dict:
    """
    Compute loss sharpness in random directions.
    Sharp minima generalize worse than flat minima.
    """
    original_state = deepcopy(model.state_dict())
    
    with torch.no_grad():
        base_loss = nn.MSELoss()(model(x), y).item()
    
    perturbations = []
    for _ in range(n_directions):
        direction = {}
        for name, param in model.named_parameters():
            direction[name] = torch.randn_like(param)
            direction[name] /= direction[name].norm()
        
        # Apply perturbation
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(radius * direction[name])
        
        perturbed_loss = nn.MSELoss()(model(x), y).item()
        perturbations.append(perturbed_loss - base_loss)
        
        # Restore
        model.load_state_dict(original_state)
    
    return {
        'sharpness_mean': np.mean(perturbations),
        'sharpness_max': np.max(perturbations),
        'sharpness_std': np.std(perturbations),
        'base_loss': base_loss
    }


# =============================================================================
# REPRESENTATION GEOMETRY
# =============================================================================

def get_layer_representations(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Extract intermediate representations from each layer."""
    representations = [x]
    
    if hasattr(model, 'layers'):  # KAN
        h = x
        for layer in model.layers:
            h = layer(h)
            representations.append(h.detach())
    else:  # MLP
        h = x
        for layer in model.layers_list[:-1]:
            h = model.activation(layer(h))
            representations.append(h.detach())
        representations.append(model.layers_list[-1](h).detach())
    
    return representations


def compute_representation_rank(representations: List[torch.Tensor], 
                                  threshold: float = 0.01) -> List[int]:
    """Compute effective rank of each layer's representation."""
    ranks = []
    for rep in representations:
        if rep.dim() > 1 and rep.shape[0] > 1:
            # SVD
            U, S, V = torch.svd(rep)
            # Effective rank: count singular values above threshold
            eff_rank = (S > threshold * S.max()).sum().item()
            ranks.append(eff_rank)
        else:
            ranks.append(1)
    return ranks


def compute_representation_similarity(reps1: List[torch.Tensor], 
                                        reps2: List[torch.Tensor]) -> List[float]:
    """Compute CKA similarity between representations."""
    similarities = []
    for r1, r2 in zip(reps1, reps2):
        if r1.shape == r2.shape and r1.dim() > 1:
            # Linear CKA
            K1 = r1 @ r1.T
            K2 = r2 @ r2.T
            
            hsic = torch.trace(K1 @ K2)
            norm1 = torch.sqrt(torch.trace(K1 @ K1))
            norm2 = torch.sqrt(torch.trace(K2 @ K2))
            
            cka = (hsic / (norm1 * norm2 + 1e-10)).item()
            similarities.append(cka)
        else:
            similarities.append(0.0)
    return similarities


# =============================================================================
# FUNCTION APPROXIMATION GEOMETRY
# =============================================================================

def compute_function_complexity(model: nn.Module, x_range: Tuple[float, float] = (-1, 1),
                                  n_points: int = 1000) -> Dict:
    """
    Analyze learned function complexity.
    Measures: variation, frequency content, smoothness.
    """
    x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        y = model(x).squeeze()
    
    # Total variation
    diff = torch.abs(y[1:] - y[:-1])
    total_variation = diff.sum().item()
    
    # Smoothness (second derivative approximation)
    diff2 = torch.abs(y[2:] - 2*y[1:-1] + y[:-2])
    smoothness = 1 / (diff2.mean().item() + 1e-10)
    
    # Frequency content (FFT)
    y_fft = torch.fft.fft(y)
    power = torch.abs(y_fft) ** 2
    n = len(power) // 2
    freq_weights = torch.arange(n).float()
    mean_frequency = (power[:n] * freq_weights).sum() / (power[:n].sum() + 1e-10)
    
    return {
        'total_variation': total_variation,
        'smoothness': smoothness,
        'mean_frequency': mean_frequency.item(),
        'y_range': (y.min().item(), y.max().item())
    }


def compare_approximation_quality(kan_model, mlp_model, 
                                    target_func: Callable,
                                    x_range: Tuple[float, float] = (-1, 1),
                                    n_points: int = 500) -> Dict:
    """Compare how well KAN and MLP approximate a target function."""
    if isinstance(x_range[0], (int, float)):
        x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
    else:
        x = torch.rand(n_points, len(x_range)) * 2 - 1
    
    y_true = target_func(x)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    
    kan_model.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        y_kan = kan_model(x)
        y_mlp = mlp_model(x)
    
    kan_mse = nn.MSELoss()(y_kan, y_true).item()
    mlp_mse = nn.MSELoss()(y_mlp, y_true).item()
    
    return {
        'kan_mse': kan_mse,
        'mlp_mse': mlp_mse,
        'kan_better': kan_mse < mlp_mse,
        'improvement_ratio': mlp_mse / (kan_mse + 1e-10),
        'kan_params': sum(p.numel() for p in kan_model.parameters()),
        'mlp_params': sum(p.numel() for p in mlp_model.parameters())
    }


# =============================================================================
# LOSS LANDSCAPE VISUALIZATION
# =============================================================================

def compute_loss_landscape_2d(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                                range_val: float = 1.0, resolution: int = 20) -> Tuple:
    """
    Compute 2D loss landscape slice.
    Uses two random orthogonal directions.
    """
    original_state = deepcopy(model.state_dict())
    param_names = list(original_state.keys())
    
    # Get two random orthogonal directions
    params_flat = torch.cat([p.flatten() for p in model.parameters()])
    n_params = len(params_flat)
    
    d1 = torch.randn(n_params)
    d1 = d1 / d1.norm()
    
    d2 = torch.randn(n_params)
    d2 = d2 - (d2 @ d1) * d1  # Orthogonalize
    d2 = d2 / d2.norm()
    
    # Create grid
    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)
    losses = np.zeros((resolution, resolution))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Apply perturbation
            perturbation = alpha * d1 + beta * d2
            idx = 0
            with torch.no_grad():
                for k, (name, param) in enumerate(model.named_parameters()):
                    n = param.numel()
                    param.copy_(original_state[name] + perturbation[idx:idx+n].reshape(param.shape))
                    idx += n
                
                loss = nn.MSELoss()(model(x), y).item()
                losses[j, i] = loss
            
            model.load_state_dict(original_state)
    
    return alphas, betas, losses


def plot_loss_landscape(losses: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                         title: str = "Loss Landscape", save_path: str = None,
                         trajectory: np.ndarray = None):
    """Plot loss landscape contours with optional optimization trajectory."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour = ax.contourf(alphas, betas, losses, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Loss')
    
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title(title)
    ax.plot(0, 0, 'r*', markersize=15, label='Final')
    
    # Plot trajectory if provided
    if trajectory is not None and len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'w-', linewidth=1.5, alpha=0.8)
        ax.plot(traj[:, 0], traj[:, 1], 'wo', markersize=3, alpha=0.6)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def train_with_trajectory(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                           epochs: int = 500, lr: float = 0.01,
                           record_every: int = 10) -> Tuple[nn.Module, List[Dict]]:
    """
    Train model while recording parameter trajectory.
    
    Returns:
        (trained_model, trajectory_list)
        trajectory_list contains: {'params': flat_params, 'loss': loss}
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trajectory = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        if epoch % record_every == 0:
            with torch.no_grad():
                params_flat = torch.cat([p.flatten() for p in model.parameters()])
            trajectory.append({
                'params': params_flat.clone(),
                'loss': loss.item(),
                'epoch': epoch
            })
    
    return model, trajectory


def project_trajectory_to_2d(trajectory: List[Dict], 
                               d1: torch.Tensor, d2: torch.Tensor,
                               center: torch.Tensor) -> np.ndarray:
    """
    Project high-dimensional trajectory onto 2D plane defined by d1, d2.
    
    Args:
        trajectory: List of {'params': tensor, 'loss': float}
        d1, d2: Direction vectors (normalized)
        center: Center point (final parameters)
    Returns:
        Nx2 array of projected coordinates
    """
    projected = []
    for point in trajectory:
        diff = point['params'] - center
        alpha = (diff @ d1).item()
        beta = (diff @ d2).item()
        projected.append([alpha, beta])
    return np.array(projected)


def compute_loss_landscape_with_trajectory(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                                            trajectory: List[Dict],
                                            range_val: float = 1.0, resolution: int = 25) -> Tuple:
    """
    Compute loss landscape and project trajectory onto it.
    
    Returns:
        (alphas, betas, losses, projected_trajectory)
    """
    original_state = deepcopy(model.state_dict())
    final_params = torch.cat([p.flatten() for p in model.parameters()])
    
    # Get two directions - from first to last point, and orthogonal
    if trajectory:
        start_params = trajectory[0]['params']
        main_dir = final_params - start_params
        if main_dir.norm() > 1e-10:
            d1 = main_dir / main_dir.norm()
        else:
            d1 = torch.randn_like(final_params)
            d1 = d1 / d1.norm()
    else:
        d1 = torch.randn_like(final_params)
        d1 = d1 / d1.norm()
    
    d2 = torch.randn_like(final_params)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / d2.norm()
    
    # Compute landscape
    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)
    losses = np.zeros((resolution, resolution))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbation = alpha * d1 + beta * d2
            idx = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    n = param.numel()
                    param.copy_(original_state[name] + perturbation[idx:idx+n].reshape(param.shape))
                    idx += n
                
                loss = nn.MSELoss()(model(x), y).item()
                losses[j, i] = loss
            
            model.load_state_dict(original_state)
    
    # Project trajectory
    projected = project_trajectory_to_2d(trajectory, d1, d2, final_params)
    
    return alphas, betas, losses, projected, d1, d2


def plot_landscape_with_trajectory(alphas: np.ndarray, betas: np.ndarray, 
                                     losses: np.ndarray, trajectory: np.ndarray,
                                     title: str = "Loss Landscape with Trajectory",
                                     save_path: str = None):
    """Plot loss landscape with optimization trajectory overlay."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot
    contour = ax.contourf(alphas, betas, losses, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Loss')
    
    # Add contour lines
    ax.contour(alphas, betas, losses, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Plot trajectory
    if len(trajectory) > 1:
        # Path
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'w-', linewidth=2, alpha=0.9, label='Optimization Path')
        
        # Points with gradient coloring
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(trajectory)))
        for i, (pt, color) in enumerate(zip(trajectory, colors)):
            ax.plot(pt[0], pt[1], 'o', color=color, markersize=4)
        
        # Start and end markers
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, 
                markeredgecolor='white', markeredgewidth=2, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1, label='End')
    
    ax.set_xlabel('Direction 1 (optimization direction)')
    ax.set_ylabel('Direction 2 (orthogonal)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# GEOMETRY COMPARISON
# =============================================================================

def compare_geometry(kan_model, mlp_model, x: torch.Tensor, y: torch.Tensor) -> Dict:
    """
    Comprehensive geometry comparison between KAN and MLP.
    """
    results = {
        'kan': {},
        'mlp': {},
        'comparison': {}
    }
    
    # Parameters
    kan_params = sum(p.numel() for p in kan_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    results['kan']['parameters'] = kan_params
    results['mlp']['parameters'] = mlp_params
    results['comparison']['param_ratio'] = kan_params / mlp_params
    
    # Training loss
    kan_model.eval()
    mlp_model.eval()
    with torch.no_grad():
        kan_loss = nn.MSELoss()(kan_model(x), y).item()
        mlp_loss = nn.MSELoss()(mlp_model(x), y).item()
    results['kan']['loss'] = kan_loss
    results['mlp']['loss'] = mlp_loss
    results['comparison']['loss_ratio'] = kan_loss / (mlp_loss + 1e-10)
    
    # Gradient norm
    kan_grad = compute_gradient_norm(kan_model, x, y)
    mlp_grad = compute_gradient_norm(mlp_model, x, y)
    results['kan']['gradient_norm'] = kan_grad
    results['mlp']['gradient_norm'] = mlp_grad
    
    # Sharpness
    kan_sharp = compute_loss_sharpness(kan_model, x, y)
    mlp_sharp = compute_loss_sharpness(mlp_model, x, y)
    results['kan']['sharpness'] = kan_sharp['sharpness_mean']
    results['mlp']['sharpness'] = mlp_sharp['sharpness_mean']
    
    # Function complexity (1D only)
    if x.shape[1] == 1:
        kan_func = compute_function_complexity(kan_model)
        mlp_func = compute_function_complexity(mlp_model)
        results['kan']['smoothness'] = kan_func['smoothness']
        results['mlp']['smoothness'] = mlp_func['smoothness']
    
    # Representation rank
    kan_reps = get_layer_representations(kan_model, x)
    mlp_reps = get_layer_representations(mlp_model, x)
    results['kan']['rep_ranks'] = compute_representation_rank(kan_reps)
    results['mlp']['rep_ranks'] = compute_representation_rank(mlp_reps)
    
    return results


def print_geometry_comparison(results: Dict):
    """Pretty print geometry comparison results."""
    print("\n" + "=" * 60)
    print("GEOMETRY COMPARISON: KAN vs MLP")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'KAN':>15} {'MLP':>15}")
    print("-" * 55)
    
    print(f"{'Parameters':<25} {results['kan']['parameters']:>15,} {results['mlp']['parameters']:>15,}")
    print(f"{'Loss':<25} {results['kan']['loss']:>15.6f} {results['mlp']['loss']:>15.6f}")
    print(f"{'Gradient Norm':<25} {results['kan']['gradient_norm']:>15.4f} {results['mlp']['gradient_norm']:>15.4f}")
    print(f"{'Sharpness':<25} {results['kan']['sharpness']:>15.6f} {results['mlp']['sharpness']:>15.6f}")
    
    if 'smoothness' in results['kan']:
        print(f"{'Smoothness':<25} {results['kan']['smoothness']:>15.2f} {results['mlp']['smoothness']:>15.2f}")
    
    print(f"\n{'Representation Ranks:':<25}")
    print(f"  KAN: {results['kan']['rep_ranks']}")
    print(f"  MLP: {results['mlp']['rep_ranks']}")
    
    print("\n" + "=" * 60)


# =============================================================================
# DEMO
# =============================================================================

def demo_geometry_analysis(save_dir: str = "reports"):
    """Demonstrate geometry analysis with loss landscape visualization."""
    import sys
    import os
    sys.path.insert(0, '.')
    from modules import KAN
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("KAN vs MLP Geometry Analysis Demo")
    print("=" * 60)
    
    # Create models
    kan = KAN([1, 5, 1], grid_size=5)
    mlp = MLP([1, 10, 1], activation='silu')
    
    print(f"\nKAN parameters: {sum(p.numel() for p in kan.parameters())}")
    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters())}")
    
    # Training data
    x = torch.linspace(-1, 1, 200).unsqueeze(1)
    y = torch.sin(2 * np.pi * x)
    
    # Train both
    for model, name in [(kan, 'KAN'), (mlp, 'MLP')]:
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(500):
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
        print(f"{name} trained, loss: {loss.item():.6f}")
    
    # Compare geometry
    results = compare_geometry(kan, mlp, x, y)
    print_geometry_comparison(results)
    
    # Function complexity
    print("\nFunction Complexity:")
    kan_comp = compute_function_complexity(kan)
    mlp_comp = compute_function_complexity(mlp)
    print(f"  KAN smoothness: {kan_comp['smoothness']:.2f}")
    print(f"  MLP smoothness: {mlp_comp['smoothness']:.2f}")
    
    # Loss Landscape Visualization
    print("\nGenerating Loss Landscape...")
    
    # KAN loss landscape
    print("  Computing KAN loss landscape...")
    alphas_kan, betas_kan, losses_kan = compute_loss_landscape_2d(kan, x, y, range_val=0.5, resolution=25)
    fig_kan, _ = plot_loss_landscape(losses_kan, alphas_kan, betas_kan, 
                                      title="KAN Loss Landscape",
                                      save_path=f"{save_dir}/loss_landscape_kan.png")
    plt.close(fig_kan)
    print(f"  Saved: {save_dir}/loss_landscape_kan.png")
    
    # MLP loss landscape
    print("  Computing MLP loss landscape...")
    alphas_mlp, betas_mlp, losses_mlp = compute_loss_landscape_2d(mlp, x, y, range_val=0.5, resolution=25)
    fig_mlp, _ = plot_loss_landscape(losses_mlp, alphas_mlp, betas_mlp, 
                                      title="MLP Loss Landscape",
                                      save_path=f"{save_dir}/loss_landscape_mlp.png")
    plt.close(fig_mlp)
    print(f"  Saved: {save_dir}/loss_landscape_mlp.png")
    
    # Combined comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KAN
    c1 = axes[0].contourf(alphas_kan, betas_kan, losses_kan, levels=50, cmap='viridis')
    plt.colorbar(c1, ax=axes[0], label='Loss')
    axes[0].set_xlabel('Direction 1')
    axes[0].set_ylabel('Direction 2')
    axes[0].set_title(f'KAN Loss Landscape (min={losses_kan.min():.4f})')
    axes[0].plot(0, 0, 'r*', markersize=15)
    
    # MLP
    c2 = axes[1].contourf(alphas_mlp, betas_mlp, losses_mlp, levels=50, cmap='viridis')
    plt.colorbar(c2, ax=axes[1], label='Loss')
    axes[1].set_xlabel('Direction 1')
    axes[1].set_ylabel('Direction 2')
    axes[1].set_title(f'MLP Loss Landscape (min={losses_mlp.min():.4f})')
    axes[1].plot(0, 0, 'r*', markersize=15)
    
    plt.tight_layout()
    combined_path = f"{save_dir}/loss_landscape_comparison.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {combined_path}")
    
    # Function comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_plot = torch.linspace(-1, 1, 500).unsqueeze(1)
    
    with torch.no_grad():
        y_true = torch.sin(2 * np.pi * x_plot)
        y_kan = kan(x_plot)
        y_mlp = mlp(x_plot)
    
    ax.plot(x_plot.numpy(), y_true.numpy(), 'k-', linewidth=2, label='True: sin(2πx)')
    ax.plot(x_plot.numpy(), y_kan.numpy(), 'b--', linewidth=2, label=f'KAN (loss={results["kan"]["loss"]:.6f})')
    ax.plot(x_plot.numpy(), y_mlp.numpy(), 'r:', linewidth=2, label=f'MLP (loss={results["mlp"]["loss"]:.6f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Function Approximation: KAN vs MLP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    func_path = f"{save_dir}/function_comparison.png"
    plt.savefig(func_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {func_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete! All images saved to:", save_dir)
    print("=" * 60)


if __name__ == "__main__":
    demo_geometry_analysis()
