import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.spline_utils import B_batch, curve2coef

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_base=1.0, scale_spline=1.0, base_fun=nn.SiLU(), grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.scale_base = nn.Parameter(torch.ones(out_features, in_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features, in_features) * scale_spline)
        
        # Base function (usually SiLU)
        self.base_fun = base_fun
        
        # Grid initialization
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        # Pad grid for spline_order
        grid = self._extend_grid(grid)
        self.register_buffer("grid", grid.unsqueeze(0).repeat(in_features, 1))
        
        # Spline coefficients
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # Mask for pruning
        self.register_buffer("mask", torch.ones(out_features, in_features))
        
        self._init_parameters()

    def _extend_grid(self, grid):
        # Extend grid by adding knots at both ends to support B-spline order
        step = grid[1] - grid[0]
        for _ in range(self.spline_order):
            grid = torch.cat([grid[0:1] - step, grid, grid[-1:] + step])
        return grid

    def _init_parameters(self):
        # Xavier/Kaiming-like initialization for spline weights
        nn.init.normal_(self.spline_weight, std=0.1 * (1 / np.sqrt(self.in_features + self.grid_size)))

    def forward(self, x):
        # x: (batch, in_features)
        
        # 1. Base function part: y_base = scale_base * base_fun(x)
        # (batch, in_features) -> (batch, 1, in_features)
        y_base = self.base_fun(x).unsqueeze(1)
        # scale_base: (out_features, in_features)
        # y_base: (batch, out_features, in_features)
        y_base = y_base * self.scale_base.unsqueeze(0)
        # Transpose to (batch, in_features, out_features)
        y_base = y_base.permute(0, 2, 1)
        
        # 2. Spline part
        # splines: (batch, in_features, grid_size + spline_order)
        splines = B_batch(x, self.grid, self.spline_order)
        # spline_weight: (out_features, in_features, grid_size + spline_order)
        # y_spline: (batch, in_features, out_features)
        y_spline = torch.einsum('bik,oik->bio', splines, self.spline_weight)
        # scale_spline: (out_features, in_features)
        y_spline = y_spline * self.scale_spline.T.unsqueeze(0)
        
        # 3. Combine and apply mask
        y_all = (y_base + y_spline) * self.mask.T.unsqueeze(0)
        
        # 4. Sum over in_features
        # (batch, in_features, out_features) -> (batch, out_features)
        y = y_all.sum(dim=1)
        
        return y

    def get_l1(self):
        """Calculate L1 norm of activation functions for sparsification."""
        return torch.abs(self.spline_weight).mean() + torch.abs(self.scale_base).mean()

    @torch.no_grad()
    def update_grid_from_samples(self, x):
        """
        Scale-up grid resolution while preserving the learned function.
        Sec 2.5: Grid Refinement.
        """
        batch = x.shape[0]
        # Current activation values
        splines = B_batch(x, self.grid, self.spline_order)
        y_spline = torch.einsum('bik,oik->bio', splines, self.spline_weight)
        
        # New grid (usually finer)
        # For simplicity, we keep same size but we could pass a new grid_size
        x_min, _ = x.min(dim=0)
        x_max, _ = x.max(dim=0)
        
        for i in range(self.in_features):
            new_grid = torch.linspace(x_min[i], x_max[i], self.grid_size + 1).to(x.device)
            self.grid[i] = self._extend_grid(new_grid)
            
        # Re-interpolate coefficients
        self.spline_weight.data = curve2coef(x, y_spline, self.grid, self.spline_order)
