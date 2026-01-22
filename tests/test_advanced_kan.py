import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.kan_model import KAN

def test_advanced_features():
    print("--- Testing Advanced KAN Features ---")
    
    # 1. Initialize KAN
    model = KAN(layers_hidden=[1, 5, 1], grid_size=5, spline_order=3)
    print(f"Initial parameter count: {model.get_parameter_count()}")
    
    # 2. Test Grid Refinement
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_true = torch.sin(torch.pi * x)
    
    print("Before grid update, loss on sin(x):")
    y_pred_before = model(x)
    loss_before = nn.MSELoss()(y_pred_before, y_true)
    print(f"Loss: {loss_before.item():.6f}")
    
    print("Performing grid refinement...")
    model.update_grid(x) 
    
    # 3. Test Sparsification & Pruning
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    print("Training with L1 + Entropy Sparsification...")
    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model(x)
        mse_loss = nn.MSELoss()(y_pred, y_true)
        reg_loss = model.get_reg(lamb_l1=0.01, lamb_entropy=0.01)
        loss = mse_loss + reg_loss
        loss.backward()
        optimizer.step()
        
    print(f"Final MSE Loss: {nn.MSELoss()(model(x), y_true).item():.6f}")
    
    # 4. Prune
    print("Pruning low-importance edges (threshold=0.05)...")
    model.prune(threshold=0.05)
    
    active_edges = 0
    total_edges = 0
    for layer in model.layers:
        active_edges += layer.mask.sum().item()
        total_edges += layer.mask.numel()
    
    print(f"Active edges after pruning: {active_edges}/{total_edges}")
    
    # 5. Symbolic Fit
    print("Suggesting symbolic forms...")
    symbolic_results = model.symbolic_fit(x, threshold=0.5)
    for l, res in enumerate(symbolic_results):
        print(f"Layer {l} symbolic suggestions: {res}")

if __name__ == "__main__":
    test_advanced_features()
