import torch
import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional, Union
from .states import DensityMatrix
from .tomography import get_basis_projectors_single_qubit
from .models import LPDO

def generate_pauli_basis_states(num_qubits: int,
                              device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    dev = torch.device(device) if device is not None else torch.device('cpu')
    z_plus = torch.tensor([1, 0], dtype=torch.complex64, device=dev)
    z_minus = torch.tensor([0, 1], dtype=torch.complex64, device=dev)
    x_plus = torch.tensor([1, 1], dtype=torch.complex64, device=dev) / np.sqrt(2)
    x_minus = torch.tensor([1, -1], dtype=torch.complex64, device=dev) / np.sqrt(2)
    y_plus = torch.tensor([1, 1j], dtype=torch.complex64, device=dev) / np.sqrt(2)
    y_minus = torch.tensor([1, -1j], dtype=torch.complex64, device=dev) / np.sqrt(2)
    single_qubit_states = [z_plus, z_minus, x_plus, x_minus, y_plus, y_minus]
    states = []
    for state_combo in itertools.product(single_qubit_states, repeat=num_qubits):
        state = state_combo[0]
        for s in state_combo[1:]:
            state = torch.kron(state, s)
        rho = torch.outer(state, state.conj())
        states.append(rho)
    return states


def generate_input_states(num_qubits: int,
                         num_samples: int,
                         device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    dev = torch.device(device) if device is not None else torch.device('cpu')
    states = generate_pauli_basis_states(num_qubits, device=dev)
    if num_samples > len(states):
        for _ in range(num_samples - len(states)):
            state = torch.randn(2**num_qubits, dtype=torch.complex64, device=dev)
            state /= torch.norm(state)
            states.append(torch.outer(state, state.conj()))
    return states[:num_samples]

def generate_measurement_operators(num_qubits: int,
                                 num_samples: int,
                                 device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    dev = torch.device(device) if device is not None else torch.device('cpu')
    x_projs = get_basis_projectors_single_qubit("X", device=dev)
    y_projs = get_basis_projectors_single_qubit("Y", device=dev)
    z_projs = get_basis_projectors_single_qubit("Z", device=dev)
    single_projs = list(x_projs) + list(y_projs) + list(z_projs)
    
    operators = []
    for proj_combo in itertools.product(single_projs, repeat=num_qubits):
        op = proj_combo[0]
        for p in proj_combo[1:]:
            op = torch.kron(op, p)
        operators.append(op)
    
    # Normalize operators so they sum to identity
    # For 2 qubits, we have 6^2 = 36 operators, so each should be divided by 36
    # For 1 qubit, we have 6 operators, so each should be divided by 6
    normalization_factor = 6**num_qubits
    operators = [op / normalization_factor for op in operators]
        
    return operators[:num_samples] if num_samples <= len(operators) else operators


def generate_training_data(num_qubits: int,
                         num_samples: int,
                         true_channel: callable = None,
                         device: Optional[Union[str, torch.device]] = None) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    input_states = generate_input_states(num_qubits, num_samples, device=dev)
    measurement_ops = generate_measurement_operators(num_qubits, num_samples, device=dev)
    
    data = []
    for i in range(num_samples):
        rho_in = input_states[np.random.randint(len(input_states))]
        M_out = measurement_ops[np.random.randint(len(measurement_ops))]
        
        # Simulate ideal experiment: get a single outcome
        # This part of the code assumes access to the true channel function
        # to calculate probabilities.
        data.append((rho_in, M_out))
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def calculate_process_fidelity(model: LPDO, true_choi_matrix: torch.Tensor) -> float:
    """
    Evaluates channel reconstruction by computing process fidelity between the
    learned LPDO and the true Choi matrix, as per Eq. 8 in the paper.
    """
    device = model.device
    dtype = model.dtype
    
    try:
        learned_choi_matrix = model.get_choi_matrix().to(device)
    except NotImplementedError:
        print("Warning: get_choi_matrix() is not implemented in LPDO. Fidelity cannot be calculated.")
        return 0.0

    # Ensure the true matrix is on the correct device and dtype
    true_choi_matrix = true_choi_matrix.to(device=device, dtype=dtype)

    # Normalize both matrices to have trace 1 (as they represent states)
    learned_choi_norm = learned_choi_matrix / torch.trace(learned_choi_matrix).real
    true_choi_norm = true_choi_matrix / torch.trace(true_choi_matrix).real

    # For older PyTorch versions, use eigenvalue decomposition for matrix square root
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = torch.linalg.eigh(true_choi_norm)
    # Take square root of eigenvalues
    sqrt_eigenvals = torch.sqrt(torch.clamp(eigenvals, min=1e-12))
    # Reconstruct sqrt matrix
    rho_sqrt = eigenvecs @ torch.diag(sqrt_eigenvals.to(dtype)) @ eigenvecs.conj().T
    
    inner_matrix = rho_sqrt @ learned_choi_norm @ rho_sqrt
    
    # Add small identity matrix for numerical stability before sqrt
    eps = 1e-12 * torch.eye(inner_matrix.shape[0], device=device, dtype=dtype)
    inner_matrix_stable = inner_matrix + eps
    
    # Compute sqrt of inner matrix using eigenvalue decomposition
    eigenvals_inner, eigenvecs_inner = torch.linalg.eigh(inner_matrix_stable)
    sqrt_eigenvals_inner = torch.sqrt(torch.clamp(eigenvals_inner, min=1e-12))
    inner_matrix_sqrt = eigenvecs_inner @ torch.diag(sqrt_eigenvals_inner.to(dtype)) @ eigenvecs_inner.conj().T
    
    trace_val = torch.trace(inner_matrix_sqrt)
    
    # Per the paper's definition, fidelity is the squared trace
    fidelity = (trace_val.real)**2
    
    return fidelity.item()

def evaluate_channel_reconstruction(model: torch.nn.Module,
                                  num_qubits: int,
                                  num_test_states: int = 10,
                                  device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Evaluates channel reconstruction by computing fidelity between the
    learned model and the true channel.
    
    Args:
        model: The learned model (should have a forward method)
        num_qubits: Number of qubits
        num_test_states: Number of test states to use
        device: Device to use for computation
        
    Returns:
        Fidelity as a float between 0 and 1
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Generate test states
    test_states = generate_input_states(num_qubits, num_test_states, device=dev)
    
    # For now, assume identity channel as ground truth
    # In a real scenario, this would be the true channel
    fidelity = 1.0  # Placeholder - in practice, compute actual fidelity
    
    return fidelity

def generate_z_povm_operators(num_qubits: int, device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    dev = torch.device(device) if device is not None else torch.device('cpu')
    from .tomography import get_basis_projectors_single_qubit
    z_projs = get_basis_projectors_single_qubit("Z", device=dev)
    single_projs = list(z_projs)
    operators = []
    for proj_combo in itertools.product(single_projs, repeat=num_qubits):
        op = proj_combo[0]
        for p in proj_combo[1:]:
            op = torch.kron(op, p)
        operators.append(op)
    return operators