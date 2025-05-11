import torch
import numpy as np

def euclidean_to_hyperbolic_hook(x: torch.Tensor, hook_name: str,
                                 curvature: float = -1.0,
                                 eps: float = 1e-6,
                                 max_norm: float = 0.999) -> torch.Tensor:
    """
    Forward‐hook wrapper around your euclidean_to_hyperbolic,
    discarding the hook_name and always returning a Tensor.
    """
    # reuse your original logic, assuming x is a torch.Tensor
    norm = torch.norm(x, dim=-1, keepdim=True)
    c = torch.tensor(-curvature, device=x.device, dtype=x.dtype).abs()
    denom = 1 + c * (norm ** 2) + torch.sqrt(1 + c * (norm ** 2))
    hyp = (2 * x) / (denom + eps)
    # clamp back into ball
    norm_h = torch.norm(hyp, dim=-1, keepdim=True)
    mask = norm_h >= max_norm
    if mask.any():
        scale = (max_norm - eps) / (norm_h[mask] + eps)
        hyp[mask.squeeze(-1)] = hyp[mask.squeeze(-1)] * scale
    return hyp

def hyperbolic_to_euclidean_hook(x: torch.Tensor, hook_name: str,
                                 curvature: float = -1.0,
                                 eps: float = 1e-6,
                                 max_norm: float = 1e6) -> torch.Tensor:
    """
    Forward‐hook wrapper around your hyperbolic_to_euclidean.
    """
    norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps, max=max_norm)
    c = torch.tensor(-curvature, device=x.device, dtype=x.dtype).abs()
    factor = 2 / (1 - c * (norm ** 2) + eps)
    return factor * x
