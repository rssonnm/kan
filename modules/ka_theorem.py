"""
Kolmogorov-Arnold Representation Theorem Implementation.

Implements Eq 2.1 from the paper:
f(x) = Σ_{q=1}^{2n+1} Φ_q(Σ_{p=1}^{n} φ_{q,p}(x_p))

This is a shallow 2-layer KAN that can represent any continuous function.
"""

import torch
import torch.nn as nn
from .kan_layer import KANLinear


class KolmogorovArnold2Layer(nn.Module):
    """
    Eq 2.1: Original Kolmogorov-Arnold Representation.
    
    A 2-layer network that can represent any continuous multivariate function:
    f(x_1, ..., x_n) = Σ_{q=1}^{2n+1} Φ_q(Σ_{p=1}^{n} φ_{q,p}(x_p))
    
    Layer 1 (Inner): n inputs -> 2n+1 hidden neurons
    Layer 2 (Outer): 2n+1 -> 1 output
    """
    
    def __init__(self, n_inputs, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        self.n = n_inputs
        self.hidden_size = 2 * n_inputs + 1  # 2n+1 as per theorem
        
        # Inner functions φ_{q,p}: Each edge has its own learnable function
        # Shape: (n_inputs, 2n+1) edges
        self.inner_layer = KANLinear(
            in_features=n_inputs,
            out_features=self.hidden_size,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range
        )
        
        # Outer functions Φ_q: Each hidden neuron has one output function
        # Shape: (2n+1, 1) edges
        self.outer_layer = KANLinear(
            in_features=self.hidden_size,
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range
        )
    
    def forward(self, x):
        """
        Forward pass implementing Eq 2.1.
        
        Args:
            x: (batch, n_inputs) input tensor
        Returns:
            y: (batch, 1) output tensor
        """
        # Step 1: Inner sum: Σ_p φ_{q,p}(x_p) for each q
        # (batch, n_inputs) -> (batch, 2n+1)
        inner_sum = self.inner_layer(x)
        
        # Step 2: Outer sum: Σ_q Φ_q(inner_sum_q)
        # (batch, 2n+1) -> (batch, 1)
        output = self.outer_layer(inner_sum)
        
        return output
    
    def get_theorem_structure(self):
        """Return a dict describing the theorem structure."""
        return {
            "n_inputs": self.n,
            "hidden_neurons": self.hidden_size,
            "inner_edges": self.n * self.hidden_size,
            "outer_edges": self.hidden_size,
            "total_edges": self.n * self.hidden_size + self.hidden_size,
            "theorem_form": f"f(x) = Σ_{{q=1}}^{{{self.hidden_size}}} Φ_q(Σ_{{p=1}}^{{{self.n}}} φ_{{q,p}}(x_p))"
        }


class DeepKAN(nn.Module):
    """
    Eq 2.3: Deep KAN with L layers.
    
    KAN(x) = (Φ_{L-1} ∘ Φ_{L-2} ∘ ... ∘ Φ_1 ∘ Φ_0)(x)
    
    Generalizes the 2-layer theorem to arbitrary depth.
    """
    
    def __init__(self, layer_dims, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        """
        Args:
            layer_dims: List of layer dimensions [n_0, n_1, ..., n_L]
                        e.g., [2, 5, 5, 1] for 2 inputs, 2 hidden layers of 5, 1 output
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of KAN layers
        
        self.layers = nn.ModuleList()
        for l in range(self.L):
            self.layers.append(KANLinear(
                in_features=layer_dims[l],
                out_features=layer_dims[l + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range
            ))
    
    def forward(self, x):
        """
        Eq 2.3: Compose all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_layer_info(self):
        """Return layer-by-layer information."""
        info = []
        for l, layer in enumerate(self.layers):
            info.append({
                "layer": l,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "n_edges": layer.in_features * layer.out_features,
                "n_params_per_edge": layer.grid_size + layer.spline_order + 1
            })
        return info


# Alias for backward compatibility
KAN = DeepKAN
