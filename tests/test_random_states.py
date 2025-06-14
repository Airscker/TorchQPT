import torch
import pytest
from torchqpt.states import COMPLEX_DTYPE, random_state_vector, random_density_matrix, random_pure_state

def test_random_state_vector():
    # Test basic functionality
    num_qubits = 2
    state = random_state_vector(num_qubits)
    assert state.num_qubits == num_qubits
    assert state.state_vector.shape == (2**num_qubits,)
    assert state.state_vector.dtype == COMPLEX_DTYPE
    
    # Test normalization
    norm = torch.sqrt(torch.sum(torch.abs(state.state_vector)**2))
    assert torch.isclose(norm, torch.tensor(1.0))
    
    # Test device placement
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = random_state_vector(num_qubits, device=device)
    assert state.device == torch.device(device)
    
    # Test invalid input
    with pytest.raises(ValueError):
        random_state_vector(0)
    with pytest.raises(ValueError):
        random_state_vector(-1)

def test_random_density_matrix():
    # Test basic functionality
    num_qubits = 2
    rho = random_density_matrix(num_qubits)
    assert rho.num_qubits == num_qubits
    assert rho.density_matrix.shape == (2**num_qubits, 2**num_qubits)
    assert rho.density_matrix.dtype == COMPLEX_DTYPE
    
    # Test Hermiticity
    assert torch.allclose(rho.density_matrix, rho.density_matrix.conj().T)
    
    # Test positive semidefiniteness
    eigenvalues = torch.linalg.eigvals(rho.density_matrix).real
    assert torch.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    
    # Test normalization
    trace = torch.trace(rho.density_matrix).real
    assert torch.isclose(trace, torch.tensor(1.0))
    
    # Test device placement
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = random_density_matrix(num_qubits, device=device)
    assert rho.device == torch.device(device)
    
    # Test invalid input
    with pytest.raises(ValueError):
        random_density_matrix(0)
    with pytest.raises(ValueError):
        random_density_matrix(-1)

def test_random_pure_state():
    # Test basic functionality
    num_qubits = 2
    rho = random_pure_state(num_qubits)
    assert rho.num_qubits == num_qubits
    assert rho.density_matrix.shape == (2**num_qubits, 2**num_qubits)
    assert rho.density_matrix.dtype == COMPLEX_DTYPE
    
    # Test purity (for pure states, Tr(rho^2) = 1)
    purity = torch.trace(rho.density_matrix @ rho.density_matrix).real
    assert torch.isclose(purity, torch.tensor(1.0))
    
    # Test device placement
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = random_pure_state(num_qubits, device=device)
    assert rho.device == torch.device(device)
    
    # Test invalid input
    with pytest.raises(ValueError):
        random_pure_state(0)
    with pytest.raises(ValueError):
        random_pure_state(-1) 

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])