import numpy as np
import torch


import torch
import torch.nn.functional as F

def euclidean_to_hyperbolic_hook(activation, hook_name=None, **kwargs):
    """
    Transform euclidean activations to hyperbolic space using the Poincaré ball model.
    
    Args:
        activation: Input tensor of shape (..., d) where d is the feature dimension
        hook_name: Name of the hook (unused but required by hook interface)
        **kwargs: Additional keyword arguments from the hook system
    
    Returns:
        Transformed activations in hyperbolic space (Poincaré ball)
    """
    # Ensure we're working with float tensors
    if activation.dtype != torch.float32:
        activation = activation.float()
    
    # Compute the norm of each vector
    norms = torch.norm(activation, dim=-1, keepdim=True)
    
    # Handle zero vectors (map to origin in hyperbolic space)
    zero_mask = (norms < 1e-8)
    
    # For non-zero vectors, apply the exponential map to hyperbolic space
    # This maps R^d to the Poincaré ball B^d
    # Formula: x_h = tanh(||x_e||/2) * (x_e / ||x_e||)
    
    # Avoid division by zero
    safe_norms = torch.where(zero_mask.squeeze(-1, keepdim=True), 
                            torch.ones_like(norms), norms)
    
    # Normalize the vectors
    normalized = activation / safe_norms
    
    # Apply tanh scaling based on euclidean norm
    hyperbolic_radius = torch.tanh(norms / 2.0)
    
    # Construct hyperbolic coordinates
    hyperbolic_activation = hyperbolic_radius * normalized
    
    # Handle zero vectors explicitly
    hyperbolic_activation = torch.where(zero_mask, 
                                      torch.zeros_like(activation), 
                                      hyperbolic_activation)
    
    return hyperbolic_activation


def hyperbolic_to_euclidean_hook(activation, hook_name=None, **kwargs):
    """
    Transform hyperbolic activations back to euclidean space from Poincaré ball model.
    
    Args:
        activation: Input tensor in hyperbolic space (Poincaré ball) of shape (..., d)
        hook_name: Name of the hook (unused but required by hook interface)
        **kwargs: Additional keyword arguments from the hook system
    
    Returns:
        Transformed activations in euclidean space
    """
    # Ensure we're working with float tensors
    if activation.dtype != torch.float32:
        activation = activation.float()
    
    # Compute the hyperbolic norm (distance from origin in Poincaré ball)
    hyperbolic_norms = torch.norm(activation, dim=-1, keepdim=True)
    
    # Handle points at origin
    zero_mask = (hyperbolic_norms < 1e-8)
    
    # Clip norms to prevent numerical issues (points should be in unit ball)
    hyperbolic_norms = torch.clamp(hyperbolic_norms, max=1.0 - 1e-6)
    
    # For non-zero vectors, apply the logarithmic map back to euclidean space
    # This is the inverse of the exponential map
    # Formula: x_e = 2 * arctanh(||x_h||) * (x_h / ||x_h||)
    
    # Avoid division by zero
    safe_norms = torch.where(zero_mask.squeeze(-1, keepdim=True), 
                            torch.ones_like(hyperbolic_norms), hyperbolic_norms)
    
    # Normalize the hyperbolic vectors
    normalized = activation / safe_norms
    
    # Apply inverse tanh scaling
    euclidean_radius = 2.0 * torch.atanh(hyperbolic_norms)
    
    # Construct euclidean coordinates
    euclidean_activation = euclidean_radius * normalized
    
    # Handle zero vectors explicitly
    euclidean_activation = torch.where(zero_mask, 
                                     torch.zeros_like(activation), 
                                     euclidean_activation)
    
    return euclidean_activation