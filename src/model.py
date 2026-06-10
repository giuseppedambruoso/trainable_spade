import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumOpticsClassifier(nn.Module):
    def __init__(self, U_matrix: torch.Tensor, num_classes: int = 10):
        super().__init__()
        # Precomputed measurement matrix U (requires no gradients)
        self.register_buffer('U_matrix', U_matrix.to(torch.complex64))
        
        # To guarantee V is unitary, we parameterize a skew-Hermitian generator S
        # V = exp(S) where S = A - A^dagger
        self.generator_real = nn.Parameter(torch.randn(6, 6) * 0.1)
        self.generator_imag = nn.Parameter(torch.randn(6, 6) * 0.1)
        
        # Single-output Feed-Forward Neural Network
        self.ffnn = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Step 2 & 3: Vectorize and normalize image to sum to 1 (P_k)
        x_flat = x.view(batch_size, -1)
        P = x_flat / (x_flat.sum(dim=1, keepdim=True) + 1e-8)
        P_complex = P.to(torch.complex64)
        
        # Step 4 & 5: Compute density matrix \hat{\rho}^{(out)}
        # rho_flat is (batch_size, 36)
        rho_flat = F.linear(P_complex, self.U_matrix) 
        rho = rho_flat.view(batch_size, 6, 6)
        
        # Step 6: Parameterized Unitary Transformation V(\theta)
        A = self.generator_real + 1j * self.generator_imag
        S = A - A.mH # Skew-Hermitian
        V = torch.matrix_exp(S) # Guaranteed Unitary
        
        # Transform: \rho' = V \rho V^\dagger
        rho_prime = V @ rho @ V.mH
        
        # Step 7: Extract diagonal elements v_\alpha
        v = torch.diagonal(rho_prime, dim1=-2, dim2=-1).real
        
        # Step 8: Feed to FFNN
        return self.ffnn(v)
