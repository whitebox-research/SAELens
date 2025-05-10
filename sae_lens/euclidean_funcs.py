import numpy as np
import torch


def euclidean_to_hyperbolic(x, curvature=-1.0, eps=1e-6, max_norm=0.999):
    """
    Convert points from Euclidean space to Poincaré ball (hyperbolic space).
    
    Parameters:
    -----------
    x : torch.Tensor or numpy.ndarray
        Points in Euclidean space
    curvature : float, default=-1.0
        Curvature of the hyperbolic space (c < 0)
    eps : float, default=1e-6
        Small constant for numerical stability
    max_norm : float, default=0.999
        Maximum allowed norm in the Poincaré ball (must be < 1)
    
    Returns:
    --------
    torch.Tensor or numpy.ndarray
        Corresponding points in the Poincaré ball model of hyperbolic space
    """
    using_torch = isinstance(x, torch.Tensor)
    
    if using_torch:
        norm = torch.norm(x, dim=-1, keepdim=True)
        c = torch.tensor(-curvature, device=x.device, dtype=x.dtype)
    else:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        c = np.array(-curvature, dtype=x.dtype)
    
    # Ensure curvature is negative for hyperbolic space
    c = abs(c)
    
    # Apply the mapping from Euclidean to Poincaré ball
    # Using the formula 2x / (1 + c||x||^2 + sqrt(1 + c||x||^2))
    denominator = 1 + c * (norm ** 2) + torch.sqrt(1 + c * (norm ** 2)) if using_torch else \
                  1 + c * (norm ** 2) + np.sqrt(1 + c * (norm ** 2))
    
    # Apply the mapping
    hyperbolic_x = (2 * x) / (denominator + eps)
    
    # Ensure points remain within the Poincaré ball
    if using_torch:
        norm_result = torch.norm(hyperbolic_x, dim=-1, keepdim=True)
        # Scale points that are outside the boundary
        mask = norm_result >= max_norm
        if mask.any():
            scale = (max_norm - eps) / (norm_result[mask] + eps)
            hyperbolic_x[mask.squeeze(-1)] = hyperbolic_x[mask.squeeze(-1)] * scale
    else:
        norm_result = np.linalg.norm(hyperbolic_x, axis=-1, keepdims=True)
        # Scale points that are outside the boundary
        mask = norm_result >= max_norm
        if np.any(mask):
            scale = (max_norm - eps) / (norm_result[mask] + eps)
            hyperbolic_x[mask.squeeze(-1)] = hyperbolic_x[mask.squeeze(-1)] * scale
    
    return hyperbolic_x



def hyperbolic_to_euclidean(x, curvature=-1.0, eps=1e-6, max_norm=1e6):
    """
    Convert points from Poincaré ball (hyperbolic space) to Euclidean space.
    
    Parameters:
    -----------
    x : torch.Tensor or numpy.ndarray
        Points in the Poincaré ball model of hyperbolic space
    curvature : float, default=-1.0
        Curvature of the hyperbolic space (c < 0)
    eps : float, default=1e-6
        Small constant for numerical stability
    max_norm : float, default=1e6
        Maximum norm allowed for points (to avoid numerical instability)
    
    Returns:
    --------
    torch.Tensor or numpy.ndarray
        Corresponding points in Euclidean space
    """
    using_torch = isinstance(x, torch.Tensor)
    
    if using_torch:
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps, max=max_norm)
        c = torch.tensor(-curvature, device=x.device, dtype=x.dtype)
    else:
        norm = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, max_norm)
        c = np.array(-curvature, dtype=x.dtype)
    
    # Ensure curvature is negative for hyperbolic space
    c = abs(c)
    
    # Scale factor from hyperbolic to Euclidean space
    # Using the formula for the Poincaré disk model
    factor = 2 / (1 - c * (norm ** 2) + eps)
    
    # Apply the mapping
    euclidean_x = factor * x
    
    return euclidean_x

