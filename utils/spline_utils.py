import torch

def coef2curve(x_eval, grid, coef, k):
    """
    Convert spline coefficients to curve values.
    
    Args:
        x_eval: (batch, in_features)
        grid: (in_features, grid_size + 2k + 1)
        coef: (out_features, in_features, grid_size + k)
        k: spline degree
    Returns:
        y: (batch, in_features, out_features)
    """
    # x_eval: (batch, in_features)
    # b_splines: (batch, in_features, grid_size + k)
    b_splines = B_batch(x_eval, grid, k)
    # y: (batch, in_features, out_features)
    y = torch.einsum('bik,oik->bio', b_splines, coef)
    return y

def B_batch(x, grid, k=3):
    """
    Compute B-spline basis functions.
    
    Args:
        x: (batch, in_features)
        grid: (in_features, grid_size + 2k + 1)
        k: spline degree
    Returns:
        splines: (batch, in_features, grid_size + k)
    """
    # x: (batch, in_features) -> (batch, in_features, 1)
    x = x.unsqueeze(-1)
    
    # 0-th degree B-splines
    # (batch, in_features, grid_size + 2k)
    value = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    
    # Recursive Cox-de Boor formula
    for p in range(1, k + 1):
        # (batch, in_features, grid_size + 2k - p)
        v1 = (x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)]) * value[:, :, :-1]
        v2 = (grid[:, p + 1:] - x) / (grid[:, p + 1:] - grid[:, 1:-p]) * value[:, :, 1:]
        value = v1 + v2
        
    return value # (batch, in_features, grid_size + k)

def curve2coef(x_eval, y_eval, grid, k):
    """
    Interpolate coefficients from a curve.
    
    Args:
        x_eval: (batch, in_features)
        y_eval: (batch, in_features, out_features)
        grid: (in_features, grid_size + 2k + 1)
        k: spline degree
    Returns:
        coef: (in_features, out_features, grid_size + k)
    """
    # b_splines: (batch, in_features, grid_size + k)
    b_splines = B_batch(x_eval, grid, k)
    
    # Solve for coef in: b_splines * coef = y_eval
    # We use pseudo-inverse for robust solution
    # (batch, in_features, grid_size + k) -> (in_features, batch, grid_size + k)
    B = b_splines.permute(1, 0, 2)
    # (batch, in_features, out_features) -> (in_features, batch, out_features)
    Y = y_eval.permute(1, 0, 2)
    
    # Coefficient calculation: coef = (B^T B)^-1 B^T Y
    # (in_features, grid_size + k, out_features)
    coef = torch.linalg.lstsq(B, Y).solution
    
    # (out_features, in_features, grid_size + k)
    return coef.permute(2, 0, 1)

def suggest_symbolic(x, y, functions=None):
    """
    Very simple symbolic suggestion by trying to fit common functions.
    In a full implementation, this would use a library like SymbolicRegression.
    """
    if functions is None:
        functions = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'log': lambda x: torch.log(torch.abs(x) + 1e-6),
            'sqrt': lambda x: torch.sqrt(torch.abs(x)),
            'x^2': lambda x: x**2,
            'x^3': lambda x: x**3,
        }
    
    # For each function, calculate R^2 or correlation
    # This is highly simplified
    best_fit = None
    best_score = -1
    
    for name, func in functions.items():
        try:
            phi = func(x)
            # Simple correlation as score
            score = torch.abs(torch.corrcoef(torch.stack([phi.flatten(), y.flatten()]))[0, 1])
            if score > best_score:
                best_score = score
                best_fit = name
        except:
            continue
            
    return best_fit, best_score
