import torch
import torch.nn as nn
from .kan_layer import KANLinear
from utils.spline_utils import suggest_symbolic

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_base=1.0, scale_spline=1.0, base_fun=nn.SiLU(), grid_range=[-1, 1]):
        """
        Kolmogorov-Arnold Network (KAN) model.
        
        Args:
            layers_hidden: List of integers [in_dim, h1, h2, ..., out_dim]
            grid_size: Number of grid intervals (G in paper)
            spline_order: B-spline degree (k in paper)
        """
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_h, out_h in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_h, 
                    out_h, 
                    grid_size=grid_size, 
                    spline_order=spline_order, 
                    scale_base=scale_base, 
                    scale_spline=scale_spline, 
                    base_fun=base_fun, 
                    grid_range=grid_range
                )
            )

    def forward(self, x):
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grid(self, x):
        """
        Grid refinement/extension (Sec 2.5).
        Update all layers' grids based on the input distribution.
        """
        with torch.no_grad():
            for layer in self.layers:
                layer.update_grid_from_samples(x)
                x = layer(x)

    def prune(self, threshold=1e-2):
        """Prune edges with low importance (average weight < threshold)."""
        for layer in self.layers:
            # Importance based on average activation strength
            importance = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
            layer.mask = (importance > threshold).float()

    def get_reg(self, lamb_l1=1.0, lamb_entropy=2.0):
        """Calculate regularization for sparsification (Sec 2.3)."""
        reg_l1 = 0
        reg_entropy = 0
        for layer in self.layers:
            # L1 sparsity
            reg_l1 += layer.get_l1()
            
            # Entropy-based sparsity for nodes
            importance = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
            p = importance / (torch.sum(importance, dim=0, keepdim=True) + 1e-6)
            reg_entropy += -torch.mean(torch.sum(p * torch.log(p + 1e-6), dim=0))
            
        return lamb_l1 * reg_l1 + lamb_entropy * reg_entropy

    @torch.no_grad()
    def symbolic_fit(self, x, threshold=0.9):
        """Try to convert activation functions to symbolic formulas (Sec 2.3)."""
        results = []
        for l, layer in enumerate(self.layers):
            layer_results = []
            
            # Forward pass to get activations for this layer
            # (batch, in_features)
            x_in = x
            
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    # Extract activation curve for edge (i, j)
                    # This is highly simplified
                    y_edge = layer.base_fun(x_in[:, i]) * layer.scale_base[j, i]
                    # Note: Simplified curve extraction
                    
                    symbol, score = suggest_symbolic(x_in[:, i], y_edge)
                    if score > threshold:
                        layer_results.append((i, j, symbol, score.item()))
            
            results.append(layer_results)
            x = layer(x)
            
        return results

    def get_parameter_count(self):
        """Total parameters formula from paper (Sec 2.1)."""
        total = 0
        for layer in self.layers:
            # Each edge has (grid_size + spline_order) spline weights + base scale + spline scale
            # Total parameters per edge = G + k + 1
            # Total edges = in * out
            total += layer.in_features * layer.out_features * (layer.grid_size + layer.spline_order + 1)
        return total
