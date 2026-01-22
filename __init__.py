"""
KAN - Kolmogorov-Arnold Networks

A PyTorch implementation of Kolmogorov-Arnold Networks from scratch.
Based on paper: https://arxiv.org/abs/2404.19756

Usage:
    from kan import KAN, KANTrainer
    
    # Create model
    model = KAN([input_dim, hidden1, hidden2, output_dim])
    
    # Train
    trainer = KANTrainer(model)
    trainer.train(x_train, y_train, epochs=500)
    
    # Save/Load
    trainer.save("model.pt")
    trainer.load("model.pt")
"""

# Core modules
from modules import KAN, KANLinear, KolmogorovArnold2Layer, DeepKAN

# Utilities
from utils import (
    KANTrainer, ContinualLearner,
    SymbolicKAN, fit_symbolic,
    B_batch, coef2curve, curve2coef
)

__version__ = "1.0.0"
__author__ = "KAN Implementation"

__all__ = [
    # Core
    'KAN', 'KANLinear', 'KolmogorovArnold2Layer', 'DeepKAN',
    # Training
    'KANTrainer', 'ContinualLearner',
    # Symbolic
    'SymbolicKAN', 'fit_symbolic',
    # Spline utilities
    'B_batch', 'coef2curve', 'curve2coef',
]
