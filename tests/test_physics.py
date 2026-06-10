import torch
import pytest
from src.physics import compute_U_matrix
from src.model import QuantumOpticsClassifier

def test_u_matrix_shape():
    N = 28
    U = compute_U_matrix(sigma=1.0, N=N)
    assert U.shape == (36, N**2), "U matrix has incorrect dimensions."

def test_unitary_generator():
    """Verify the parameterization yields a truly unitary matrix"""
    N = 10
    U_dummy = compute_U_matrix(sigma=1.0, N=N)
    model = QuantumOpticsClassifier(U_dummy)
    
    A = model.generator_real + 1j * model.generator_imag
    S = A - A.mH 
    V = torch.matrix_exp(S)
    
    # V * V^dagger should be the Identity matrix
    identity_approx = V @ V.mH
    identity_true = torch.eye(6, dtype=torch.complex64)
    
    assert torch.allclose(identity_approx, identity_true, atol=1e-5), "Matrix V is not unitary."
