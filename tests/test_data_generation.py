import pytest
import torch
import numpy as np
from torchqpt.data import (
    generate_pauli_basis_states,
    generate_measurement_operators,
    generate_training_data,
    calculate_process_fidelity,
    evaluate_channel_reconstruction
)
from torchqpt.tomography import get_basis_projectors_single_qubit
import itertools

SEMIDEFINITE_TOLERANCE = -1e-6
CLOSE_TOLERANCE = 1e-6

@pytest.fixture
def device():
    """Fixture to provide the device for tensor operations."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def dtype():
    """Fixture to provide the data type for tensors."""
    return torch.complex128

def test_pauli_basis_states_1q(device):
    """Test the generation of 1-qubit Pauli basis states."""
    states = generate_pauli_basis_states(num_qubits=1, device=device)

    # Check number of states
    assert len(states) == 6, "1-qubit should have 6 Pauli basis states"

    # Verify that states are valid density matrices
    for state in states:
        # Check shape
        assert state.shape == (2, 2), "1-qubit states should have shape (2, 2)"
        # Check Hermiticity
        assert torch.allclose(state, state.conj().T), "States should be Hermitian"
        # Check trace = 1
        assert torch.allclose(
            torch.trace(state).real, torch.tensor(1.0)
        ) and torch.allclose(
            torch.trace(state).imag, torch.tensor(0.0)
        ), "States should have trace 1"
        # Check positive semidefinite
        eigvals = torch.linalg.eigvals(state)
        assert torch.all(eigvals.real >= SEMIDEFINITE_TOLERANCE), "States should be positive semidefinite"

def test_pauli_basis_states_2q(device):
    """Test the generation of 2-qubit Pauli basis states."""
    states = generate_pauli_basis_states(num_qubits=2, device=device)

    # Check number of states
    assert len(states) == 36, "2-qubit should have 36 Pauli basis states"

    # Verify that states are valid density matrices
    for state in states:
        # Check shape
        assert state.shape == (4, 4), "2-qubit states should have shape (4, 4)"
        assert torch.allclose(state, state.conj().T), "States should be Hermitian"
        assert torch.allclose(
            torch.trace(state).real, torch.tensor(1.0)
        ) and torch.allclose(
            torch.trace(state).imag, torch.tensor(0.0)
        ), "States should have trace 1"
        eigvals = torch.linalg.eigvals(state)
        assert torch.all(eigvals.real >= SEMIDEFINITE_TOLERANCE), "States should be positive semidefinite"

def test_measurement_operators_1q(device):
    """Test the generation of 1-qubit measurement operators."""
    num_samples = 6  # Number of Pauli basis operators
    meas = generate_measurement_operators(num_qubits=1, num_samples=num_samples, device=device)
    
    # Check number of operators
    assert len(meas) == num_samples, f"Should have {num_samples} measurement operators"
    
    # Verify that operators are valid POVM elements
    for op in meas:
        # Check shape
        assert op.shape == (2, 2), "1-qubit operators should have shape (2, 2)"
        # Check Hermiticity
        assert torch.allclose(op, op.conj().T), "Operators should be Hermitian"
        # Check positive semidefinite
        eigvals = torch.linalg.eigvals(op)
        assert torch.all(eigvals.real >= SEMIDEFINITE_TOLERANCE), "Operators should be positive semidefinite"

def test_measurement_operators_2q(device):
    """Test the generation of 2-qubit measurement operators."""
    num_samples = 36  # Number of Pauli basis operators
    meas = generate_measurement_operators(num_qubits=2, num_samples=num_samples, device=device)
    
    # Check number of operators
    assert len(meas) == num_samples, f"Should have {num_samples} measurement operators"
    
    # Verify that operators are valid POVM elements
    for i, op in enumerate(meas):
        # Check shape
        assert op.shape == (4, 4), "2-qubit operators should have shape (4, 4)"
        assert torch.allclose(op, op.conj().T), "Operators should be Hermitian"
        eigvals = torch.linalg.eigvals(op)
        assert torch.all(eigvals.real >= SEMIDEFINITE_TOLERANCE), "Operators should be positive semidefinite"
        print(f"Operator {i} trace:", torch.trace(op).real.item())
    
    # Verify completeness: sum of operators should be identity / 4
    sum_ops = torch.sum(torch.stack(meas), dim=0)
    identity = torch.eye(4, dtype=sum_ops.dtype, device=device) / 4
    print("Sum of operators:", sum_ops)
    print("Expected identity / 4:", identity)
    assert torch.allclose(sum_ops, identity), "Sum of operators should be identity / 4"

@pytest.mark.parametrize("num_qubits", [1, 2])
def test_training_data(num_qubits, device):
    """Test the generation of training data for different numbers of qubits."""
    num_samples = 100
    train_data, val_data = generate_training_data(
        num_qubits=num_qubits,
        num_samples=num_samples,
        device=device
    )
    # Check data structure
    assert len(train_data) > 0, "Training data should not be empty"
    assert len(val_data) > 0, "Validation data should not be empty"
    # Allow overlap in train/val sets
    # Verify data format
    for rho, M in train_data + val_data:
        dim = 2**num_qubits
        assert rho.shape == (dim, dim), f"Input states should have shape ({dim}, {dim})"
        assert M.shape == (dim, dim), f"Measurement operators should have shape ({dim}, {dim})"

def test_channel_reconstruction(device):
    """Test the channel reconstruction evaluation."""
    # Create a simple test channel (identity channel)
    class IdentityChannel(torch.nn.Module):
        def forward(self, x):
            return x
    
    model = IdentityChannel()
    
    # Test with different numbers of qubits
    for num_qubits in [1, 2]:
        fidelity = evaluate_channel_reconstruction(
            model=model,
            num_qubits=num_qubits,
            num_test_states=10,
            device=device
        )
        
        # Verify fidelity
        assert 0 <= fidelity <= 1, "Fidelity should be between 0 and 1"
        assert abs(fidelity - 1.0) < CLOSE_TOLERANCE, "Identity channel should have fidelity 1"

@pytest.mark.parametrize("num_qubits", [1, 2])
def test_data_consistency(num_qubits, device):
    """Test consistency between generated states and measurements using only Z-basis POVM."""
    dev = device
    # Use only Z-basis projectors for a true POVM
    from torchqpt.tomography import get_basis_projectors_single_qubit
    z_projs = get_basis_projectors_single_qubit("Z", device=dev)
    single_projs = list(z_projs)
    meas = []
    for proj_combo in itertools.product(single_projs, repeat=num_qubits):
        op = proj_combo[0]
        for p in proj_combo[1:]:
            op = torch.kron(op, p)
        meas.append(op)
    states = generate_pauli_basis_states(num_qubits=num_qubits, device=device)
    # Compute probabilities for each state-measurement pair
    for state in states:
        probs = []
        for M in meas:
            prob = torch.trace(M @ state).real
            probs.append(prob)
        # Verify probabilities
        probs = torch.tensor(probs)
        print(f"Probabilities for state {state}: {probs}|{torch.sum(probs)}")
        assert torch.all(probs >= SEMIDEFINITE_TOLERANCE), "Probabilities should be non-negative"
        assert abs(torch.sum(probs) - 1.0) < CLOSE_TOLERANCE, "Probabilities should sum to 1"

def test_single_qubit_projectors(device):
    """Test that single-qubit projectors are properly normalized."""
    x_projectors = get_basis_projectors_single_qubit("X", device=device)
    y_projectors = get_basis_projectors_single_qubit("Y", device=device)
    z_projectors = get_basis_projectors_single_qubit("Z", device=device)
    
    # Check that each pair of projectors sums to identity
    for projectors in [x_projectors, y_projectors, z_projectors]:
        sum_proj = projectors[0] + projectors[1]
        identity = torch.eye(2, dtype=torch.complex64, device=device)
        print("Sum of projectors:", sum_proj)
        print("Expected identity:", identity)
        assert torch.allclose(sum_proj, identity), "Projectors should sum to identity"
        
        # Check individual traces
        for i, proj in enumerate(projectors):
            print(f"Projector {i} trace:", torch.trace(proj).real.item())
            assert torch.allclose(torch.trace(proj).real, torch.tensor(1.0)), "Each projector should have trace 1"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
