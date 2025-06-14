import torch
import pytest
import sys
import os
import itertools
import numpy as np
from typing import Dict, List

# Adjust sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from torchqpt.tomography import (
    qst_linear_inversion_single_qubit,
    qst_linear_inversion_multi_qubit,
    get_pauli_operator,
    get_basis_projectors_single_qubit,
    simulate_measurement_probabilities,
    COMPLEX_DTYPE,  # Use COMPLEX_DTYPE from tomography.py
    # For single-qubit tests, tomography.py defines I_MATRIX, X_MATRIX etc.
    I_MATRIX as I_1Q,
    X_MATRIX as X_1Q,
    Y_MATRIX as Y_1Q,
    Z_MATRIX as Z_1Q,
)
from torchqpt.states import DensityMatrix, QuantumStateVector  # For creating test states

# Helper Pauli map for get_pauli_operator tests (matches tomography.py internal)
_PAULI_MAP_SINGLE_QUBIT_TEST = {
    'I': torch.eye(2, dtype=COMPLEX_DTYPE),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE),
    'Z': torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE),
}


# --- Tests for get_pauli_operator() ---

def test_get_pauli_operator_single():
    torch.testing.assert_close(get_pauli_operator("I"), _PAULI_MAP_SINGLE_QUBIT_TEST['I'])
    torch.testing.assert_close(get_pauli_operator("X"), _PAULI_MAP_SINGLE_QUBIT_TEST['X'])
    torch.testing.assert_close(get_pauli_operator("Y"), _PAULI_MAP_SINGLE_QUBIT_TEST['Y'])
    torch.testing.assert_close(get_pauli_operator("Z"), _PAULI_MAP_SINGLE_QUBIT_TEST['Z'])
    assert get_pauli_operator("I").shape == (2,2)

def test_get_pauli_operator_multi():
    # Test IX
    expected_IX = torch.kron(_PAULI_MAP_SINGLE_QUBIT_TEST['I'], _PAULI_MAP_SINGLE_QUBIT_TEST['X'])
    torch.testing.assert_close(get_pauli_operator("IX"), expected_IX)
    assert get_pauli_operator("IX").shape == (4,4)

    # Test XYZ
    expected_XYZ = torch.kron(torch.kron(_PAULI_MAP_SINGLE_QUBIT_TEST['X'], _PAULI_MAP_SINGLE_QUBIT_TEST['Y']), _PAULI_MAP_SINGLE_QUBIT_TEST['Z'])
    torch.testing.assert_close(get_pauli_operator("XYZ"), expected_XYZ)
    assert get_pauli_operator("XYZ").shape == (8,8)

def test_get_pauli_operator_invalid_char():
    with pytest.raises(ValueError, match="Invalid character 'A' in Pauli string"):
        get_pauli_operator("IXA")

def test_get_pauli_operator_empty_string():
    with pytest.raises(ValueError, match="Pauli string cannot be empty"):
        get_pauli_operator("")

def test_get_pauli_operator_device_dtype():
    op_cpu = get_pauli_operator("X", device='cpu', dtype=COMPLEX_DTYPE)
    assert op_cpu.device == torch.device('cpu')
    assert op_cpu.dtype == COMPLEX_DTYPE

    if torch.cuda.is_available():
        op_cuda = get_pauli_operator("Y", device='cuda', dtype=COMPLEX_DTYPE)
        assert op_cuda.device.type == 'cuda'
        assert op_cuda.dtype == COMPLEX_DTYPE

    # Test with float32 (though COMPLEX_DTYPE is default)
    op_f32 = get_pauli_operator("I", dtype=torch.float32)
    assert op_f32.dtype == torch.float32


# --- Tests for get_basis_projectors_single_qubit() ---

def test_get_projectors_Z():
    P0, P1 = get_basis_projectors_single_qubit("Z")
    torch.testing.assert_close(P0, torch.tensor([[1,0],[0,0]], dtype=COMPLEX_DTYPE))
    torch.testing.assert_close(P1, torch.tensor([[0,0],[0,1]], dtype=COMPLEX_DTYPE))
    torch.testing.assert_close(P0 + P1, torch.eye(2, dtype=COMPLEX_DTYPE))

def test_get_projectors_X():
    P_plus, P_minus = get_basis_projectors_single_qubit("X")
    expected_P_plus = 0.5 * torch.tensor([[1,1],[1,1]], dtype=COMPLEX_DTYPE)
    expected_P_minus = 0.5 * torch.tensor([[1,-1],[-1,1]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(P_plus, expected_P_plus)
    torch.testing.assert_close(P_minus, expected_P_minus)
    torch.testing.assert_close(P_plus + P_minus, torch.eye(2, dtype=COMPLEX_DTYPE))

def test_get_projectors_Y():
    P_plus_i, P_minus_i = get_basis_projectors_single_qubit("Y")
    expected_P_plus_i = 0.5 * torch.tensor([[1,-1j],[1j,1]], dtype=COMPLEX_DTYPE)
    expected_P_minus_i = 0.5 * torch.tensor([[1,1j],[-1j,1]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(P_plus_i, expected_P_plus_i)
    torch.testing.assert_close(P_minus_i, expected_P_minus_i)
    torch.testing.assert_close(P_plus_i + P_minus_i, torch.eye(2, dtype=COMPLEX_DTYPE))

def test_get_projectors_invalid_basis():
    with pytest.raises(ValueError, match="Unknown basis 'W'"):
        get_basis_projectors_single_qubit("W")

# --- Tests for simulate_measurement_probabilities() ---

def test_sim_meas_pure_states():
    # |0><0| state
    rho0 = torch.tensor([[1,0],[0,0]], dtype=COMPLEX_DTYPE)
    P0_z, P1_z = get_basis_projectors_single_qubit("Z")
    probs_z = simulate_measurement_probabilities(rho0, (P0_z, P1_z))
    assert np.allclose(probs_z, [1.0, 0.0])

    # |+><+| state
    rho_plus = 0.5 * torch.tensor([[1,1],[1,1]], dtype=COMPLEX_DTYPE)
    P0_x, P1_x = get_basis_projectors_single_qubit("X")
    probs_x = simulate_measurement_probabilities(rho_plus, (P0_x, P1_x))
    assert np.allclose(probs_x, [1.0, 0.0])

    # Measure |+><+| in Z basis
    probs_plus_in_z = simulate_measurement_probabilities(rho_plus, (P0_z, P1_z))
    assert np.allclose(probs_plus_in_z, [0.5, 0.5])

def test_sim_meas_maximally_mixed():
    rho_mixed = 0.5 * torch.eye(2, dtype=COMPLEX_DTYPE)
    for basis in ["X", "Y", "Z"]:
        P0, P1 = get_basis_projectors_single_qubit(basis)
        probs = simulate_measurement_probabilities(rho_mixed, (P0, P1))
        assert np.allclose(probs, [0.5, 0.5])

# --- Tests for qst_linear_inversion_single_qubit() ---

def generate_probs_for_qst(rho_tensor: torch.Tensor, device='cpu') -> Dict[str, List[float]]:
    data = {}
    for basis in ["X", "Y", "Z"]:
        projectors = get_basis_projectors_single_qubit(basis, device=device)
        # Ensure rho_tensor is on the same device as projectors for simulate_measurement_probabilities
        probs = simulate_measurement_probabilities(rho_tensor.to(device), projectors)
        data[basis] = probs
    return data

@pytest.mark.parametrize("state_name, get_state_tensor", [
    ("psi0", lambda: torch.tensor([[1,0],[0,0]], dtype=COMPLEX_DTYPE)),
    ("psi1", lambda: torch.tensor([[0,0],[0,1]], dtype=COMPLEX_DTYPE)),
    ("psi_plus", lambda: 0.5 * torch.tensor([[1,1],[1,1]], dtype=COMPLEX_DTYPE)),
    ("psi_minus", lambda: 0.5 * torch.tensor([[1,-1],[-1,1]], dtype=COMPLEX_DTYPE)),
    ("psi_plus_i", lambda: 0.5 * torch.tensor([[1,-1j],[1j,1]], dtype=COMPLEX_DTYPE)),
    ("psi_minus_i", lambda: 0.5 * torch.tensor([[1,1j],[-1j,1]], dtype=COMPLEX_DTYPE)),
])
def test_qst_single_pure_states(state_name, get_state_tensor):
    original_dm_tensor = get_state_tensor()
    measurement_data = generate_probs_for_qst(original_dm_tensor, device='cpu')

    reconstructed_dm = qst_linear_inversion_single_qubit(measurement_data, device='cpu')
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_single_mixed_state():
    # Example: 0.7|0><0| + 0.3|1><1|
    original_dm_tensor = torch.tensor([[0.7,0],[0,0.3]], dtype=COMPLEX_DTYPE)
    measurement_data = generate_probs_for_qst(original_dm_tensor)
    reconstructed_dm = qst_linear_inversion_single_qubit(measurement_data)
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_single_maximally_mixed():
    original_dm_tensor = 0.5 * torch.eye(2, dtype=COMPLEX_DTYPE)
    measurement_data = generate_probs_for_qst(original_dm_tensor)
    reconstructed_dm = qst_linear_inversion_single_qubit(measurement_data)
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_single_invalid_data():
    valid_probs = [0.5, 0.5]
    with pytest.raises(ValueError, match="measurement_data must contain keys 'X', 'Y', and 'Z'"):
        qst_linear_inversion_single_qubit({"X": valid_probs, "Y": valid_probs}) # Missing Z

    with pytest.raises(ValueError, match="Probabilities for basis X do not sum to 1"):
        qst_linear_inversion_single_qubit({"X": [0.1,0.2], "Y": valid_probs, "Z": valid_probs})

    with pytest.raises(ValueError, match="Probabilities for basis Y must be floats approximately between 0 and 1"):
        qst_linear_inversion_single_qubit({"X": valid_probs, "Y": [-0.1, 1.1], "Z": valid_probs})

# --- Tests for qst_linear_inversion_multi_qubit() ---

def generate_pauli_expectations(rho_tensor: torch.Tensor, num_qubits: int, device='cpu') -> Dict[str, float]:
    data = {}
    pauli_chars = ['I', 'X', 'Y', 'Z']
    for pauli_tuple in itertools.product(pauli_chars, repeat=num_qubits):
        pauli_string = "".join(pauli_tuple)
        if pauli_string == "I" * num_qubits:
            continue # Skip identity string as per function spec

        pauli_op = get_pauli_operator(pauli_string, device=device, dtype=rho_tensor.dtype)
        # Ensure rho_tensor is on the same device as pauli_op
        exp_val = torch.trace(pauli_op @ rho_tensor.to(device)).real.item()
        data[pauli_string] = exp_val
    return data

def test_qst_multi_pure_product_state_00():
    num_qubits = 2
    psi0 = torch.tensor([1,0], dtype=COMPLEX_DTYPE)
    rho0_0 = torch.outer(psi0, psi0.conj()) # |0><0|
    original_dm_tensor = torch.kron(rho0_0, rho0_0) # |00><00|

    measurement_data = generate_pauli_expectations(original_dm_tensor, num_qubits)
    reconstructed_dm = qst_linear_inversion_multi_qubit(measurement_data, num_qubits)
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_multi_maximally_mixed():
    num_qubits = 2
    dim = 2**num_qubits
    original_dm_tensor = (1/dim) * torch.eye(dim, dtype=COMPLEX_DTYPE)
    measurement_data = generate_pauli_expectations(original_dm_tensor, num_qubits)
    # For maximally mixed state, all non-identity Pauli expectations should be 0
    for k, v in measurement_data.items():
        assert np.isclose(v, 0.0), f"Expectation for {k} should be 0 for maximally mixed state, got {v}"

    reconstructed_dm = qst_linear_inversion_multi_qubit(measurement_data, num_qubits)
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_multi_bell_state():
    num_qubits = 2
    # Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    psi_bell_tensor = (1/np.sqrt(2.0)) * torch.tensor([1,0,0,1], dtype=COMPLEX_DTYPE)
    original_dm_tensor = torch.outer(psi_bell_tensor, psi_bell_tensor.conj())

    measurement_data = generate_pauli_expectations(original_dm_tensor, num_qubits)
    # Expected expectations for |Phi+>: <XX>=1, <YY>=-1, <ZZ>=1. Other 2-qubit non-I are 0.
    # Single qubit expectations <IX>, <IY>, <ZI> etc are 0.
    assert np.isclose(measurement_data.get("XX", 0.0), 1.0)
    assert np.isclose(measurement_data.get("YY", 0.0), -1.0)
    assert np.isclose(measurement_data.get("ZZ", 0.0), 1.0)
    assert np.isclose(measurement_data.get("XY", 0.0), 0.0)
    assert np.isclose(measurement_data.get("IX", 0.0), 0.0)

    reconstructed_dm = qst_linear_inversion_multi_qubit(measurement_data, num_qubits)
    torch.testing.assert_close(reconstructed_dm.density_matrix, original_dm_tensor, atol=1e-6, rtol=1e-5)

def test_qst_multi_invalid_num_qubits():
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        qst_linear_inversion_multi_qubit({}, 0)

# Note: qst_linear_inversion_multi_qubit implicitly assumes 0 for missing non-Identity Paulis.
# A test for data key length mismatch (e.g. key "IX" for num_qubits=3) would require
# get_pauli_operator to raise an error or for qst_linear_inversion_multi_qubit to validate key lengths.
# Current get_pauli_operator doesn't check len(pauli_string) vs num_qubits.
# qst_linear_inversion_multi_qubit iterates through all possible pauli_strings for num_qubits,
# so it won't use a key like "IX" if num_qubits=3, it would look for "IXI", "IXX" etc.
# Thus, a specific "key length mismatch" test against qst_linear_inversion_multi_qubit is not straightforward
# unless we modify its logic to validate input keys beyond just using .get().
# The current design is robust to extra keys in measurement_data.
# It's more about *missing* keys, which are handled by defaulting to expectation 0.

def test_get_basis_projectors_single_qubit():
    # Test Z basis
    P_0, P_1 = get_basis_projectors_single_qubit("Z")
    assert P_0.shape == (2, 2)
    assert P_1.shape == (2, 2)
    assert torch.allclose(P_0, torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat))
    assert torch.allclose(P_1, torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat))

    # Test X basis
    P_0, P_1 = get_basis_projectors_single_qubit("X")
    assert P_0.shape == (2, 2)
    assert P_1.shape == (2, 2)
    assert torch.allclose(P_0 @ P_0, P_0)  # Projector property
    assert torch.allclose(P_1 @ P_1, P_1)  # Projector property
    assert torch.allclose(P_0 + P_1, torch.eye(2, dtype=torch.cfloat))  # Completeness

    # Test Y basis
    P_0, P_1 = get_basis_projectors_single_qubit("Y")
    assert P_0.shape == (2, 2)
    assert P_1.shape == (2, 2)
    assert torch.allclose(P_0 @ P_0, P_0)  # Projector property
    assert torch.allclose(P_1 @ P_1, P_1)  # Projector property
    assert torch.allclose(P_0 + P_1, torch.eye(2, dtype=torch.cfloat))  # Completeness

    # Test invalid basis
    with pytest.raises(ValueError):
        get_basis_projectors_single_qubit("invalid")

def test_simulate_measurement_probabilities():
    # Test with |0⟩ state
    state = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat)
    P_0, P_1 = get_basis_projectors_single_qubit("Z")
    probs = simulate_measurement_probabilities(state, (P_0, P_1))
    assert len(probs) == 2
    assert np.isclose(probs[0], 1.0)
    assert np.isclose(probs[1], 0.0)

    # Test with |+⟩ state
    state = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.cfloat)
    P_0, P_1 = get_basis_projectors_single_qubit("X")
    probs = simulate_measurement_probabilities(state, (P_0, P_1))
    assert len(probs) == 2
    assert np.isclose(probs[0], 1.0)
    assert np.isclose(probs[1], 0.0)

    # Test invalid input
    with pytest.raises(ValueError):
        simulate_measurement_probabilities(torch.eye(3, dtype=torch.cfloat), (P_0, P_1))

def test_qst_linear_inversion_single_qubit():
    # Test reconstruction of |0⟩ state
    measurement_data = {
        "X": [0.5, 0.5],
        "Y": [0.5, 0.5],
        "Z": [1.0, 0.0]
    }
    rho = qst_linear_inversion_single_qubit(measurement_data)
    assert isinstance(rho, DensityMatrix)
    assert rho.num_qubits == 1
    expected = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat)
    assert torch.allclose(rho.density_matrix, expected)

    # Test reconstruction of |+⟩ state
    measurement_data = {
        "X": [1.0, 0.0],
        "Y": [0.5, 0.5],
        "Z": [0.5, 0.5]
    }
    rho = qst_linear_inversion_single_qubit(measurement_data)
    assert isinstance(rho, DensityMatrix)
    assert rho.num_qubits == 1
    expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.cfloat)
    assert torch.allclose(rho.density_matrix, expected)

    # Test invalid input
    with pytest.raises(ValueError):
        qst_linear_inversion_single_qubit({"X": [0.5, 0.5]})  # Missing Y and Z

def test_get_pauli_operator():
    # Test single-qubit operators
    assert torch.allclose(get_pauli_operator("I"), torch.eye(2, dtype=torch.cfloat))
    assert torch.allclose(get_pauli_operator("X"), torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))
    assert torch.allclose(get_pauli_operator("Y"), torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat))
    assert torch.allclose(get_pauli_operator("Z"), torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))

    # Test two-qubit operators
    IX = get_pauli_operator("IX")
    assert IX.shape == (4, 4)
    assert torch.allclose(IX, torch.kron(torch.eye(2, dtype=torch.cfloat),
                                       torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)))

    # Test invalid input
    with pytest.raises(ValueError):
        get_pauli_operator("")

    with pytest.raises(ValueError):
        get_pauli_operator("A")  # Invalid Pauli character

def test_qst_linear_inversion_multi_qubit():
    # Test reconstruction of |00⟩ state
    measurement_data = {
        "IX": 0.0,
        "XI": 0.0,
        "XX": 0.0,
        "IY": 0.0,
        "YI": 0.0,
        "YY": 0.0,
        "IZ": 1.0,
        "ZI": 1.0,
        "ZZ": 1.0
    }
    rho = qst_linear_inversion_multi_qubit(measurement_data, num_qubits=2)
    assert isinstance(rho, DensityMatrix)
    assert rho.num_qubits == 2
    expected = torch.tensor([[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=torch.cfloat)
    assert torch.allclose(rho.density_matrix, expected)

    # Test invalid input
    with pytest.raises(ValueError):
        qst_linear_inversion_multi_qubit(measurement_data, num_qubits=0)

    with pytest.raises(TypeError):
        qst_linear_inversion_multi_qubit({"IX": "invalid"}, num_qubits=2)

if __name__ == "__main__":
    # Run all tests using pytest
    pytest.main([__file__, "-v"])
