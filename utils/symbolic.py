"""
Enhanced Symbolic Regression for KAN.

Provides advanced symbolic fitting using SymPy, including:
- Automatic function detection
- Symbolic simplification
- Formula extraction from trained networks
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import warnings

try:
    import sympy as sp
    from sympy import symbols, sin, cos, exp, log, sqrt, Abs, pi, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    warnings.warn("SymPy not installed. Symbolic regression features will be limited.")


# Standard symbolic function library
SYMBOLIC_LIBRARY = {
    'x': lambda x: x,
    'x^2': lambda x: x**2,
    'x^3': lambda x: x**3,
    'x^4': lambda x: x**4,
    '1/x': lambda x: 1 / (x + 1e-8),
    'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-8),
    'sin': torch.sin,
    'cos': torch.cos,
    'tan': torch.tan,
    'exp': torch.exp,
    'log': lambda x: torch.log(torch.abs(x) + 1e-8),
    'abs': torch.abs,
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'sinh': torch.sinh,
    'cosh': torch.cosh,
    'arcsin': torch.asin,
    'arccos': torch.acos,
    'arctan': torch.atan,
}

# SymPy equivalents
if SYMPY_AVAILABLE:
    SYMPY_LIBRARY = {
        'x': lambda x: x,
        'x^2': lambda x: x**2,
        'x^3': lambda x: x**3,
        'x^4': lambda x: x**4,
        '1/x': lambda x: 1/x,
        'sqrt': sp.sqrt,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log,
        'abs': sp.Abs,
        'tanh': sp.tanh,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'arcsin': sp.asin,
        'arccos': sp.acos,
        'arctan': sp.atan,
    }


def fit_symbolic(x: torch.Tensor, y: torch.Tensor, 
                 functions: Optional[Dict[str, Callable]] = None,
                 top_k: int = 3) -> List[Tuple[str, float, float]]:
    """
    Fit symbolic functions to data and return best matches.
    
    Args:
        x: (n_samples,) input values
        y: (n_samples,) target values
        functions: Dict of {name: func} to try
        top_k: Number of top matches to return
    Returns:
        List of (function_name, r2_score, scale_factor) tuples
    """
    if functions is None:
        functions = SYMBOLIC_LIBRARY
    
    results = []
    y_np = y.detach().numpy()
    y_mean = y_np.mean()
    ss_tot = np.sum((y_np - y_mean) ** 2) + 1e-8
    
    for name, func in functions.items():
        try:
            with torch.no_grad():
                phi_x = func(x)
                phi_np = phi_x.detach().numpy()
            
            # Fit scale: y ≈ a * phi(x) + b
            if np.std(phi_np) > 1e-8:
                # Linear regression
                A = np.vstack([phi_np, np.ones_like(phi_np)]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, y_np, rcond=None)
                a, b = coeffs
                
                y_pred = a * phi_np + b
                ss_res = np.sum((y_np - y_pred) ** 2)
                r2 = 1 - ss_res / ss_tot
                
                results.append((name, r2, a, b))
        except:
            continue
    
    # Sort by R² score
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def symbolic_formula(model, layer_idx: int, in_idx: int, out_idx: int,
                     x_range: Tuple[float, float] = (-1, 1),
                     n_samples: int = 100,
                     threshold: float = 0.95) -> Optional[str]:
    """
    Extract symbolic formula for a specific edge activation.
    
    Args:
        model: KAN model
        layer_idx: Layer index
        in_idx: Input neuron index
        out_idx: Output neuron index
        x_range: Range for sampling
        n_samples: Number of samples
        threshold: R² threshold for accepting a fit
    Returns:
        Symbolic formula string or None
    """
    layer = model.layers[layer_idx]
    x = torch.linspace(x_range[0], x_range[1], n_samples)
    
    with torch.no_grad():
        # Get activation values for this edge
        x_full = torch.zeros(n_samples, layer.in_features)
        x_full[:, in_idx] = x
        
        from utils.spline_utils import B_batch
        splines = B_batch(x_full, layer.grid, layer.spline_order)
        
        y_base = layer.base_fun(x) * layer.scale_base[out_idx, in_idx]
        y_spline = torch.einsum('bk,k->b', splines[:, in_idx, :], 
                                layer.spline_weight[out_idx, in_idx, :])
        y_spline = y_spline * layer.scale_spline[out_idx, in_idx]
        y = y_base + y_spline
    
    # Fit symbolic functions
    results = fit_symbolic(x, y, top_k=1)
    
    if results and results[0][1] >= threshold:
        name, r2, a, b = results[0]
        
        if SYMPY_AVAILABLE:
            sym_x = sp.Symbol('x')
            sym_func = SYMPY_LIBRARY.get(name, lambda x: x)
            formula = a * sym_func(sym_x) + b
            return str(simplify(formula))
        else:
            return f"{a:.4f} * {name}(x) + {b:.4f}"
    
    return None


def extract_network_formula(model, x_range: Tuple[float, float] = (-1, 1),
                            n_samples: int = 100,
                            threshold: float = 0.9) -> Dict:
    """
    Extract symbolic formulas for the entire network.
    
    Returns:
        Dict with layer -> edge -> formula mapping
    """
    formulas = {}
    
    for l, layer in enumerate(model.layers):
        layer_formulas = {}
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                formula = symbolic_formula(model, l, i, j, x_range, n_samples, threshold)
                if formula:
                    layer_formulas[(i, j)] = formula
        formulas[l] = layer_formulas
    
    return formulas


class SymbolicKAN:
    """
    Wrapper for KAN with enhanced symbolic capabilities.
    """
    
    def __init__(self, model):
        self.model = model
        self.symbolic_edges = {}  # (layer, in, out) -> sympy expression
        self.locked_edges = set()  # Set of (layer, in, out) that are locked
    
    def auto_symbolic(self, x_samples: torch.Tensor, threshold: float = 0.95):
        """
        Automatically detect and set symbolic formulas for edges.
        
        Args:
            x_samples: Representative input samples
            threshold: R² threshold for symbolic fit
        """
        print("Auto-detecting symbolic formulas...")
        
        for l, layer in enumerate(self.model.layers):
            # Get activations at this layer
            with torch.no_grad():
                if l == 0:
                    x_in = x_samples
                else:
                    x_in = x_samples
                    for prev_l in range(l):
                        x_in = self.model.layers[prev_l](x_in)
            
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    formula = symbolic_formula(self.model, l, i, j, threshold=threshold)
                    if formula:
                        self.symbolic_edges[(l, i, j)] = formula
                        print(f"  Edge ({l}, {i}, {j}): {formula}")
    
    def lock_edge(self, layer: int, in_idx: int, out_idx: int):
        """Lock an edge to prevent further training updates."""
        self.locked_edges.add((layer, in_idx, out_idx))
        # Zero out gradients for this edge
        self.model.layers[layer].spline_weight.grad = None
    
    def unlock_edge(self, layer: int, in_idx: int, out_idx: int):
        """Unlock a previously locked edge."""
        self.locked_edges.discard((layer, in_idx, out_idx))
    
    def set_symbolic(self, layer: int, in_idx: int, out_idx: int, func_name: str):
        """
        Set an edge to use a specific symbolic function.
        
        This fits the spline to approximate the symbolic function.
        """
        if func_name not in SYMBOLIC_LIBRARY:
            raise ValueError(f"Unknown function: {func_name}. Available: {list(SYMBOLIC_LIBRARY.keys())}")
        
        # Sample points
        x = torch.linspace(-1, 1, 100)
        y_target = SYMBOLIC_LIBRARY[func_name](x)
        
        # Fit spline coefficients to match target
        layer_obj = self.model.layers[layer]
        
        with torch.no_grad():
            x_full = torch.zeros(100, layer_obj.in_features)
            x_full[:, in_idx] = x
            
            from utils.spline_utils import B_batch, curve2coef
            splines = B_batch(x_full, layer_obj.grid, layer_obj.spline_order)
            
            # Solve for coefficients
            B = splines[:, in_idx, :]
            coeffs, _, _, _ = torch.linalg.lstsq(B, y_target.unsqueeze(1))
            
            layer_obj.spline_weight.data[out_idx, in_idx, :] = coeffs.squeeze()
            layer_obj.scale_spline.data[out_idx, in_idx] = 1.0
            layer_obj.scale_base.data[out_idx, in_idx] = 0.0
        
        self.symbolic_edges[(layer, in_idx, out_idx)] = func_name
        self.lock_edge(layer, in_idx, out_idx)
        
        print(f"Set edge ({layer}, {in_idx}, {out_idx}) to {func_name}")
    
    def get_formula(self) -> str:
        """Get the overall symbolic formula of the network."""
        if not SYMPY_AVAILABLE:
            return "SymPy not available"
        
        formulas = extract_network_formula(self.model)
        return str(formulas)
