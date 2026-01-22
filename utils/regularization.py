"""
Regularization utilities for KAN (Section 2.3 of paper).

Implements:
- Eq 2.9-2.12: L1 regularization on activations
- Eq 2.13-2.14: Entropy regularization for sparsification
"""

import torch
import torch.nn as nn


def compute_activation_l1(phi_func, x_samples):
    """
    Eq 2.9: L1 norm of activation function over samples.
    
    |φ|_1 = (1/n_p) * Σ |φ(x^(s))|
    
    Args:
        phi_func: Callable activation function
        x_samples: (n_samples,) input samples
    Returns:
        l1_norm: Scalar L1 norm
    """
    activations = phi_func(x_samples)
    return torch.abs(activations).mean()


def compute_layer_l1(layer, x_samples):
    """
    Eq 2.10-2.11: L1 norm of entire KAN layer.
    
    |Φ_l|_1 = Σ_{i,j} |φ_{l,i,j}|_1
    
    For incoming edges to node i:
    |Φ_l|_{1,i} = Σ_j |φ_{l,i,j}|_1
    
    Args:
        layer: KANLinear layer
        x_samples: (batch, in_features) input samples
    Returns:
        layer_l1: Scalar L1 norm for the layer
        node_l1: (out_features,) L1 norm per output node
    """
    with torch.no_grad():
        # Get base activations
        base_act = layer.base_fun(x_samples)  # (batch, in_features)
        
        # For each edge (i, j), compute |φ_{i,j}|_1
        # i = in_features, j = out_features
        edge_l1 = torch.zeros(layer.out_features, layer.in_features)
        
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                # Base component
                base_contribution = torch.abs(base_act[:, i] * layer.scale_base[j, i]).mean()
                
                # Spline component (simplified - uses average spline output)
                splines = layer.grid  # This is simplified
                spline_contribution = torch.abs(layer.spline_weight[j, i, :]).mean()
                
                edge_l1[j, i] = base_contribution + spline_contribution
        
        # Eq 2.11: Sum over input edges for each output node
        node_l1 = edge_l1.sum(dim=1)  # (out_features,)
        
        # Eq 2.10: Total layer L1
        layer_l1 = edge_l1.sum()
        
    return layer_l1, node_l1


def compute_entropy_regularization(edge_importance):
    """
    Eq 2.13-2.14: Entropy regularization for sparsification.
    
    S(Φ_l) = -Σ_{i,j} p_{i,j} * log(p_{i,j})
    where p_{i,j} = |φ_{i,j}|_1 / Σ_{i',j'} |φ_{i',j'}|_1
    
    Args:
        edge_importance: (out_features, in_features) importance matrix
    Returns:
        entropy: Scalar entropy value (lower = sparser)
    """
    # Normalize to get probabilities
    total = edge_importance.sum() + 1e-8
    p = edge_importance / total
    
    # Compute entropy
    log_p = torch.log(p + 1e-8)
    entropy = -torch.sum(p * log_p)
    
    return entropy


def total_regularization(model, x_samples, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.0):
    """
    Eq 2.12: Total training loss regularization.
    
    L_total = L_pred + λ * (μ_1 * |Φ|_1 + μ_2 * S(Φ))
    
    Args:
        model: KAN model
        x_samples: (batch, in_features) samples for computing L1
        lamb_l1: Weight for L1 regularization (μ_1)
        lamb_entropy: Weight for entropy regularization (μ_2)
        lamb_coef: Weight for coefficient smoothness (optional)
    Returns:
        reg_loss: Total regularization loss
    """
    total_l1 = 0
    total_entropy = 0
    total_coef = 0
    
    for layer in model.layers:
        # L1 on importance
        importance = torch.abs(layer.spline_weight).mean(dim=-1) + torch.abs(layer.scale_base)
        total_l1 += importance.sum()
        
        # Entropy
        total_entropy += compute_entropy_regularization(importance)
        
        # Optional: Coefficient smoothness (penalize large 2nd derivatives)
        if lamb_coef > 0:
            # Approximate 2nd derivative via finite differences of coefficients
            coef_diff = layer.spline_weight[:, :, 2:] - 2 * layer.spline_weight[:, :, 1:-1] + layer.spline_weight[:, :, :-2]
            total_coef += (coef_diff ** 2).mean()
    
    reg_loss = lamb_l1 * total_l1 + lamb_entropy * total_entropy + lamb_coef * total_coef
    return reg_loss
