"""
KAN Functions Module
====================

Core KAN functions for easy use in training and inference.
Import from this module for all KAN operations.

Usage:
    from modules.kan_functions import *
    
    # Create model
    model = create_kan([2, 5, 1])
    
    # Train
    history = train_kan(model, x, y, epochs=500)
    
    # Prune
    model = prune_kan(model, threshold=0.01)
    
    # Symbolic
    formulas = extract_symbolic(model, x)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union
from copy import deepcopy

# Import core modules
from .kan_layer import KANLinear
from .kan_model import KAN


__all__ = [
    # Creation
    'create_kan', 'create_kan_from_config',
    # Training
    'train_kan', 'train_step', 'evaluate',
    # Inference
    'predict', 'predict_batch',
    # Pruning
    'prune_kan', 'get_importance_scores', 'count_active_edges',
    # Grid
    'update_grid', 'extend_grid',
    # Regularization
    'get_l1_loss', 'get_entropy_loss', 'get_total_reg',
    # Symbolic
    'extract_symbolic', 'set_symbolic', 'freeze_edge',
    # Save/Load
    'save_kan', 'load_kan',
    # Utilities
    'count_parameters', 'summary', 'clone_kan',
]


# =============================================================================
# CREATION
# =============================================================================

def create_kan(layers: List[int], 
               grid_size: int = 5, 
               spline_order: int = 3,
               base_fun: Optional[nn.Module] = None) -> KAN:
    """
    Create a KAN model with specified architecture.
    
    Args:
        layers: Layer widths [input, hidden..., output]
        grid_size: B-spline grid intervals
        spline_order: Spline polynomial order (default: 3 = cubic)
        base_fun: Base activation (default: SiLU)
    Returns:
        KAN model
        
    Example:
        >>> model = create_kan([2, 5, 1])
        >>> model = create_kan([10, 20, 10, 1], grid_size=10)
    """
    return KAN(
        layers_hidden=layers,
        grid_size=grid_size,
        spline_order=spline_order,
        base_fun=base_fun or nn.SiLU()
    )


def create_kan_from_config(config: Dict) -> KAN:
    """
    Create KAN from configuration dict.
    
    Config keys: 'layers', 'grid_size', 'spline_order'
    """
    return create_kan(
        layers=config.get('layers', [2, 5, 1]),
        grid_size=config.get('grid_size', 5),
        spline_order=config.get('spline_order', 3)
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_step(model: KAN, x: torch.Tensor, y: torch.Tensor,
               optimizer: optim.Optimizer, 
               criterion: nn.Module = None,
               reg_lambda: float = 0.0) -> float:
    """
    Single training step.
    
    Returns:
        Loss value
    """
    criterion = criterion or nn.MSELoss()
    optimizer.zero_grad()
    
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    if reg_lambda > 0 and hasattr(model, 'get_reg'):
        loss = loss + reg_lambda * model.get_reg()
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_kan(model: KAN, x: torch.Tensor, y: torch.Tensor,
              epochs: int = 500,
              lr: float = 0.01,
              reg_lambda: float = 0.0,
              x_val: Optional[torch.Tensor] = None,
              y_val: Optional[torch.Tensor] = None,
              early_stopping: int = 0,
              verbose: bool = True,
              print_every: int = 100) -> Dict:
    """
    Train a KAN model.
    
    Args:
        model: KAN model
        x, y: Training data
        epochs: Training epochs
        lr: Learning rate
        reg_lambda: Regularization strength
        x_val, y_val: Optional validation data
        early_stopping: Stop after N epochs without improvement (0=disabled)
        verbose: Print progress
        print_every: Print frequency
    Returns:
        History dict with 'train_loss', 'val_loss'
        
    Example:
        >>> history = train_kan(model, x, y, epochs=500, lr=0.01)
        >>> history = train_kan(model, x, y, x_val=x_test, y_val=y_test)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # Train step
        model.train()
        loss = train_step(model, x, y, optimizer, criterion, reg_lambda)
        history['train_loss'].append(loss)
        
        # Validation
        val_loss = 0
        if x_val is not None:
            val_loss = evaluate(model, x_val, y_val, criterion)
            history['val_loss'].append(val_loss)
        
        # Early stopping
        current_loss = val_loss if x_val is not None else loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        
        if early_stopping > 0 and no_improve >= early_stopping:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            msg = f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}"
            if x_val is not None:
                msg += f", Val: {val_loss:.6f}"
            print(msg)
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def evaluate(model: KAN, x: torch.Tensor, y: torch.Tensor,
             criterion: nn.Module = None) -> float:
    """Evaluate model on data."""
    criterion = criterion or nn.MSELoss()
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return loss.item()


# =============================================================================
# INFERENCE
# =============================================================================

def predict(model: KAN, x: torch.Tensor) -> torch.Tensor:
    """Get predictions."""
    model.eval()
    with torch.no_grad():
        return model(x)


def predict_batch(model: KAN, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Predict in batches for large datasets."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            outputs.append(model(batch))
    return torch.cat(outputs, dim=0)


# =============================================================================
# PRUNING
# =============================================================================

def get_importance_scores(model: KAN) -> List[torch.Tensor]:
    """Get importance scores for all edges in each layer."""
    scores = []
    for layer in model.layers:
        score = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
        scores.append(score)
    return scores


def prune_kan(model: KAN, threshold: float = 0.01) -> KAN:
    """
    Prune unimportant edges.
    
    Args:
        model: KAN model
        threshold: Relative threshold (prune if < threshold * max)
    Returns:
        Pruned model
        
    Example:
        >>> pruned = prune_kan(model, threshold=0.05)
        >>> print(f"Active edges: {count_active_edges(pruned)}")
    """
    model.prune(threshold=threshold)
    return model


def count_active_edges(model: KAN) -> int:
    """Count number of active (non-pruned) edges."""
    total = 0
    for layer in model.layers:
        if hasattr(layer, 'mask'):
            total += layer.mask.sum().item()
        else:
            total += layer.in_features * layer.out_features
    return int(total)


# =============================================================================
# GRID OPERATIONS
# =============================================================================

def update_grid(model: KAN, x: torch.Tensor) -> KAN:
    """
    Update grid based on data distribution.
    
    Args:
        model: KAN model
        x: Training data for grid adaptation
    Returns:
        Model with updated grid
    """
    model.update_grid(x)
    return model


def extend_grid(model: KAN, new_grid_size: int, x: torch.Tensor) -> KAN:
    """
    Extend grid to higher resolution.
    
    Args:
        model: KAN model
        new_grid_size: New grid size (must be > current)
        x: Data for interpolation
    Returns:
        Model with extended grid
    """
    for layer in model.layers:
        layer.update_grid_from_samples(x)
    return model


# =============================================================================
# REGULARIZATION
# =============================================================================

def get_l1_loss(model: KAN) -> torch.Tensor:
    """Get total L1 regularization loss."""
    total = 0
    for layer in model.layers:
        if hasattr(layer, 'get_l1'):
            total = total + layer.get_l1()
    return total


def get_entropy_loss(model: KAN) -> torch.Tensor:
    """Get entropy regularization loss (for sparsity)."""
    if hasattr(model, 'get_reg'):
        return model.get_reg()
    return torch.tensor(0.0)


def get_total_reg(model: KAN, lambda_l1: float = 1.0, 
                   lambda_entropy: float = 1.0) -> torch.Tensor:
    """Get total regularization loss."""
    return lambda_l1 * get_l1_loss(model) + lambda_entropy * get_entropy_loss(model)


# =============================================================================
# SYMBOLIC REGRESSION
# =============================================================================

def extract_symbolic(model: KAN, x: torch.Tensor, 
                      threshold: float = 0.99) -> Dict[Tuple[int, int, int], str]:
    """
    Extract symbolic formulas from trained model.
    
    Args:
        model: Trained KAN
        x: Sample inputs for fitting
        threshold: R² threshold for accepting fit
    Returns:
        Dict mapping (layer, in_idx, out_idx) -> formula
        
    Example:
        >>> formulas = extract_symbolic(model, x_train)
        >>> for edge, formula in formulas.items():
        ...     print(f"{edge}: {formula}")
    """
    if hasattr(model, 'symbolic_fit'):
        return model.symbolic_fit(x, threshold=threshold)
    return {}


def set_symbolic(model: KAN, layer: int, in_idx: int, out_idx: int,
                  func_name: str) -> KAN:
    """
    Set an edge to use a specific symbolic function.
    
    Args:
        model: KAN model
        layer: Layer index
        in_idx: Input neuron index
        out_idx: Output neuron index
        func_name: Function name ('sin', 'cos', 'x²', etc.)
    Returns:
        Modified model
    """
    # This requires SymbolicKAN wrapper
    try:
        from utils.symbolic import SymbolicKAN
        sym = SymbolicKAN(model)
        sym.set_symbolic(layer, in_idx, out_idx, func_name)
    except ImportError:
        print("Warning: SymbolicKAN not available")
    return model


def freeze_edge(model: KAN, layer: int, in_idx: int, out_idx: int) -> KAN:
    """Freeze an edge to prevent further training."""
    layer_obj = model.layers[layer]
    with torch.no_grad():
        # Zero out gradients for this edge
        if layer_obj.spline_weight.grad is not None:
            layer_obj.spline_weight.grad[out_idx, in_idx, :] = 0
    return model


# =============================================================================
# SAVE/LOAD
# =============================================================================

def save_kan(model: KAN, path: str, include_optimizer: bool = False,
              optimizer: Optional[optim.Optimizer] = None) -> None:
    """
    Save KAN model.
    
    Args:
        model: KAN model
        path: Save path
        include_optimizer: Whether to save optimizer state
        optimizer: Optimizer (required if include_optimizer=True)
        
    Example:
        >>> save_kan(model, "my_model.pt")
    """
    state = {
        'model_state': model.state_dict(),
        'config': {
            'layers': [model.layers[0].in_features] + 
                      [l.out_features for l in model.layers],
            'grid_size': model.layers[0].grid.shape[-1] - 2 * 3 - 1,  # Approximate
            'spline_order': model.layers[0].spline_order,
        }
    }
    if include_optimizer and optimizer:
        state['optimizer_state'] = optimizer.state_dict()
    
    torch.save(state, path)
    print(f"Saved to {path}")


def load_kan(path: str, device: str = 'cpu') -> Tuple[KAN, Optional[Dict]]:
    """
    Load KAN model.
    
    Args:
        path: Load path
        device: Device to load to
    Returns:
        (model, optimizer_state) tuple
        
    Example:
        >>> model, _ = load_kan("my_model.pt")
    """
    state = torch.load(path, map_location=device)
    
    model = create_kan(**state['config'])
    model.load_state_dict(state['model_state'])
    
    opt_state = state.get('optimizer_state', None)
    print(f"Loaded from {path}")
    
    return model, opt_state


# =============================================================================
# UTILITIES
# =============================================================================

def count_parameters(model: KAN, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def summary(model: KAN) -> str:
    """Get model summary string."""
    lines = ["KAN Model Summary", "=" * 40]
    
    # Architecture
    arch = [model.layers[0].in_features]
    for layer in model.layers:
        arch.append(layer.out_features)
    lines.append(f"Architecture: {arch}")
    lines.append(f"Grid size: {model.layers[0].grid.shape[-1]}")
    lines.append(f"Spline order: {model.layers[0].spline_order}")
    lines.append(f"Parameters: {count_parameters(model):,}")
    lines.append(f"Active edges: {count_active_edges(model)}")
    
    return "\n".join(lines)


def clone_kan(model: KAN) -> KAN:
    """Create a deep copy of KAN model."""
    return deepcopy(model)
