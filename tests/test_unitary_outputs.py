import numpy as np
import pytest
from pathlib import Path

# Dynamically find the project root and search for all saved matrices
PROJECT_ROOT = Path(__file__).parent.parent
MATRIX_FILES = list(PROJECT_ROOT.rglob("unitary_matrix.npy"))

@pytest.mark.skipif(not MATRIX_FILES, reason="No unitary matrices found. Run the pipeline first.")
@pytest.mark.parametrize(
    "matrix_path", 
    MATRIX_FILES, 
    ids=[str(p.relative_to(PROJECT_ROOT)) for p in MATRIX_FILES] # Makes terminal output readable
)
def test_saved_matrix_is_unitary(matrix_path):
    """
    Loads the trained complex matrix from disk and asserts that V @ V^dagger == Identity.
    """
    # 1. Load the complex matrix
    V = np.load(matrix_path)
    
    # 2. Verify expected shape
    assert V.shape == (6, 6), f"Expected a 6x6 matrix, got {V.shape}"
    
    # 3. Compute V @ V^dagger (conjugate transpose)
    V_dagger = V.conj().T
    identity_approx = V @ V_dagger
    
    # 4. Construct the true 6x6 identity matrix
    identity_true = np.eye(6, dtype=np.complex64)
    
    # 5. Assert equality within floating-point tolerance
    # atol=1e-5 is standard for complex64 matrix exponentials
    is_unitary = np.allclose(identity_approx, identity_true, atol=1e-5)
    
    # Calculate the maximum deviation for debugging if the test fails
    max_deviation = np.max(np.abs(identity_approx - identity_true))
    
    assert is_unitary, f"Matrix is NOT strictly unitary. Max numerical deviation: {max_deviation:.2e}"
