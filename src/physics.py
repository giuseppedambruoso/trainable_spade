import torch
import logging

logger = logging.getLogger(__name__)

def compute_U_matrix(sigma: float, N: int = 28) -> torch.Tensor:
    """
    Computes the 36 x N^2 measurement matrix U corresponding to the discrete tensor 
    equation for the quantum optical image transformation.
    """
    logger.debug(f"Computing U matrix for sigma={sigma}, N={N}")
    
    # Generate physical grid centered at 0
    grid = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    
    # Compute the analytical f_nm basis functions
    # Based on Eq. 12-17
    f00 = torch.exp(-(x**2 + y**2) / (8 * sigma**2))
    f10 = x * f00
    f01 = y * f00
    f11 = x * y * f00
    f20 = (4 * sigma**2 + x**2 - 2) * f00
    f02 = (4 * sigma**2 + y**2 - 2) * f00
    
    # Shape: (6, N^2)
    f = torch.stack([f00, f10, f01, f11, f20, f02], dim=0)
    
    # Compute tensor product f_mu * f_nu^* (Since f is real, no conjugate needed)
    # Shape: (6, 6, N^2)
    U = torch.einsum('mk,nk->mnk', f, f)
    
    # Flatten to (36, N^2)
    U_flat = U.reshape(36, N**2)
    
    logger.debug("U matrix computation complete.")
    return U_flat
