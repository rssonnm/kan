import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.kan_model import KAN

def test_kan_approximation():
    # 1. Generate synthetic data: f(x) = sin(2*pi*x) + 0.5 * cos(4*pi*x)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.5 * torch.cos(4 * torch.pi * x)
    
    # 2. Initialize KAN [1, 5, 1]
    model = KAN(layers_hidden=[1, 5, 1], grid_size=5, spline_order=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. Training loop
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
    # 4. Final verification
    with torch.no_grad():
        final_pred = model(x)
        final_loss = criterion(final_pred, y)
        print(f"Final Loss: {final_loss.item():.6f}")
    
    # Check if loss is reasonably small
    assert final_loss.item() < 0.1, f"Loss too high: {final_loss.item()}"
    print("Success: KAN successfully approximated the function!")

    # Plot results (saved to file)
    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), y.numpy(), label='True function', alpha=0.5)
    plt.plot(x.numpy(), final_pred.numpy(), '--', label='KAN Prediction')
    plt.legend()
    plt.title("KAN Function Approximation")
    plt.savefig("reports/kan_test_plot.png")
    print("Plot saved to reports/kan_test_plot.png")

if __name__ == "__main__":
    test_kan_approximation()
