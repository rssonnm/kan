"""
Feynman Dataset for testing KAN on physics equations.

Contains symbolic physics equations from the Feynman Symbolic Regression Database.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Callable, List


# Feynman equations from the paper's experiments
FEYNMAN_EQUATIONS = {
    # Simple equations (1-2 variables)
    "I.6.2a": {
        "name": "Uniform Distribution",
        "formula": "exp(-θ²/2) / sqrt(2π)",
        "n_vars": 1,
        "func": lambda x: torch.exp(-x[:, 0:1]**2 / 2) / np.sqrt(2 * np.pi),
        "ranges": [(-3, 3)],
    },
    "I.8.14": {
        "name": "Relativistic Energy",
        "formula": "sqrt(E0² + (c*p)²)",
        "n_vars": 2,
        "func": lambda x: torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2),
        "ranges": [(0.1, 5), (0.1, 5)],
    },
    "I.9.18": {
        "name": "Gravitational Force",
        "formula": "G*m1*m2 / r²",
        "n_vars": 3,
        "func": lambda x: x[:, 0:1] * x[:, 1:2] / (x[:, 2:3]**2 + 0.01),
        "ranges": [(0.1, 2), (0.1, 2), (0.5, 3)],
    },
    "I.10.7": {
        "name": "Kinetic Energy",
        "formula": "0.5 * m * v²",
        "n_vars": 2,
        "func": lambda x: 0.5 * x[:, 0:1] * x[:, 1:2]**2,
        "ranges": [(0.1, 5), (-3, 3)],
    },
    "I.12.1": {
        "name": "Hooke's Law",
        "formula": "k * (x1 - x2)",
        "n_vars": 2,
        "func": lambda x: x[:, 0:1] - x[:, 1:2],
        "ranges": [(-2, 2), (-2, 2)],
    },
    "I.12.2": {
        "name": "Electric Charge Force",
        "formula": "q1*q2 / (4πε₀*r²)",
        "n_vars": 3,
        "func": lambda x: x[:, 0:1] * x[:, 1:2] / (x[:, 2:3]**2 + 0.01),
        "ranges": [(-2, 2), (-2, 2), (0.5, 3)],
    },
    "I.15.10": {
        "name": "Momentum",
        "formula": "m₀*v / sqrt(1 - v²/c²)",
        "n_vars": 2,
        "func": lambda x: x[:, 0:1] * x[:, 1:2] / torch.sqrt(1 - x[:, 1:2]**2 / 4 + 0.01),
        "ranges": [(0.1, 2), (-1.5, 1.5)],
    },
    "I.18.4": {
        "name": "2D Distance",
        "formula": "sqrt(x² + y²)",
        "n_vars": 2,
        "func": lambda x: torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + 0.01),
        "ranges": [(-2, 2), (-2, 2)],
    },
    "I.24.6": {
        "name": "Spring Energy",
        "formula": "0.5 * m * (ω² * x² + v²)",
        "n_vars": 3,
        "func": lambda x: 0.5 * x[:, 0:1] * (x[:, 1:2]**2 + x[:, 2:3]**2),
        "ranges": [(0.1, 2), (-2, 2), (-2, 2)],
    },
    "I.29.16": {
        "name": "Wave Amplitude",
        "formula": "A * sin(ωt + φ)",
        "n_vars": 3,
        "func": lambda x: x[:, 0:1] * torch.sin(x[:, 1:2] + x[:, 2:3]),
        "ranges": [(0.5, 2), (-np.pi, np.pi), (-np.pi, np.pi)],
    },
    "I.30.5": {
        "name": "Interference",
        "formula": "sin(θ/2)² / (θ/2)²",
        "n_vars": 1,
        "func": lambda x: torch.sinc(x[:, 0:1] / np.pi)**2,  # torch.sinc(x) = sin(πx)/(πx)
        "ranges": [(-5, 5)],
    },
    "II.6.11": {
        "name": "Electric Potential",
        "formula": "q / (4πε₀*r)",
        "n_vars": 2,
        "func": lambda x: x[:, 0:1] / (x[:, 1:2] + 0.1),
        "ranges": [(-2, 2), (0.5, 3)],
    },
    "II.11.3": {
        "name": "Magnetic Field",
        "formula": "q*v*B*sin(θ)",
        "n_vars": 4,
        "func": lambda x: x[:, 0:1] * x[:, 1:2] * x[:, 2:3] * torch.sin(x[:, 3:4]),
        "ranges": [(-1, 1), (-1, 1), (0.1, 2), (0, np.pi)],
    },
}


def generate_feynman_data(equation_id: str, n_samples: int = 1000, 
                          noise_level: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for a Feynman equation.
    
    Args:
        equation_id: ID of the equation (e.g., "I.6.2a")
        n_samples: Number of samples to generate
        noise_level: Standard deviation of Gaussian noise to add
    Returns:
        x, y: Input and output tensors
    """
    if equation_id not in FEYNMAN_EQUATIONS:
        raise ValueError(f"Unknown equation: {equation_id}. Available: {list(FEYNMAN_EQUATIONS.keys())}")
    
    eq = FEYNMAN_EQUATIONS[equation_id]
    n_vars = eq["n_vars"]
    ranges = eq["ranges"]
    func = eq["func"]
    
    # Generate random inputs within ranges
    x = torch.zeros(n_samples, n_vars)
    for i, (low, high) in enumerate(ranges):
        x[:, i] = torch.rand(n_samples) * (high - low) + low
    
    # Compute outputs
    y = func(x)
    
    # Add noise
    if noise_level > 0:
        y = y + torch.randn_like(y) * noise_level
    
    return x, y


def test_kan_on_feynman(model_class, equation_id: str, 
                        hidden_dims: List[int] = [10],
                        epochs: int = 500,
                        n_train: int = 1000,
                        n_test: int = 200) -> Dict:
    """
    Test a KAN model on a Feynman equation.
    
    Args:
        model_class: KAN model class
        equation_id: Feynman equation ID
        hidden_dims: Hidden layer dimensions
        epochs: Training epochs
        n_train, n_test: Sample sizes
    Returns:
        Results dict
    """
    eq = FEYNMAN_EQUATIONS[equation_id]
    n_vars = eq["n_vars"]
    
    print(f"\n=== Testing on {equation_id}: {eq['name']} ===")
    print(f"Formula: {eq['formula']}")
    print(f"Variables: {n_vars}")
    
    # Generate data
    x_train, y_train = generate_feynman_data(equation_id, n_train)
    x_test, y_test = generate_feynman_data(equation_id, n_test)
    
    # Create model
    layer_dims = [n_vars] + hidden_dims + [1]
    model = model_class(layers_hidden=layer_dims, grid_size=5, spline_order=3)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_loss = criterion(train_pred, y_train).item()
        test_loss = criterion(test_pred, y_test).item()
    
    # R² score
    ss_res = ((y_test - test_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    print(f"Train MSE: {train_loss:.6f}")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test R²: {r2:.4f}")
    
    return {
        "equation_id": equation_id,
        "name": eq["name"],
        "formula": eq["formula"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "r2": r2,
        "n_params": sum(p.numel() for p in model.parameters())
    }


def run_feynman_benchmark(model_class, equations: List[str] = None,
                          save_path: str = None) -> List[Dict]:
    """
    Run benchmark on multiple Feynman equations.
    
    Args:
        model_class: KAN model class
        equations: List of equation IDs (default: all)
        save_path: Optional path to save results as CSV
    Returns:
        List of result dicts
    """
    if equations is None:
        equations = list(FEYNMAN_EQUATIONS.keys())
    
    results = []
    for eq_id in equations:
        try:
            result = test_kan_on_feynman(model_class, eq_id)
            results.append(result)
        except Exception as e:
            print(f"Error on {eq_id}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("FEYNMAN BENCHMARK SUMMARY")
    print("="*60)
    avg_r2 = np.mean([r["r2"] for r in results])
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Equations tested: {len(results)}")
    
    if save_path:
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {save_path}")
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from modules.kan_model import KAN
    
    # Quick test on a few equations
    run_feynman_benchmark(KAN, equations=["I.6.2a", "I.10.7", "I.18.4"])
