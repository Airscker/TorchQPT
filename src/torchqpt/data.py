import torch
import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional, Union
from .states import DensityMatrix
from .tomography import get_basis_projectors_single_qubit

def generate_pauli_basis_states(num_qubits: int,
                              device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    """
    Generate input states from the Pauli basis (eigenstates of X, Y, Z).
    
    Args:
        num_qubits (int): Number of qubits.
        device (Optional[Union[str, torch.device]]): Device for the tensors.
        
    Returns:
        List[torch.Tensor]: List of input state density matrices.
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Single-qubit Pauli eigenstates
    # |0⟩, |1⟩ (Z basis)
    z_plus = torch.tensor([1, 0], dtype=torch.complex64, device=dev)
    z_minus = torch.tensor([0, 1], dtype=torch.complex64, device=dev)
    
    # |+⟩, |-⟩ (X basis)
    x_plus = torch.tensor([1, 1], dtype=torch.complex64, device=dev) / np.sqrt(2)
    x_minus = torch.tensor([1, -1], dtype=torch.complex64, device=dev) / np.sqrt(2)
    
    # |+i⟩, |-i⟩ (Y basis)
    y_plus = torch.tensor([1, 1j], dtype=torch.complex64, device=dev) / np.sqrt(2)
    y_minus = torch.tensor([1, -1j], dtype=torch.complex64, device=dev) / np.sqrt(2)
    
    # All single-qubit states
    single_qubit_states = [z_plus, z_minus, x_plus, x_minus, y_plus, y_minus]
    
    # Generate all possible combinations for num_qubits
    states = []
    for state_combo in itertools.product(single_qubit_states, repeat=num_qubits):
        # Create product state
        state = state_combo[0]
        for s in state_combo[1:]:
            state = torch.kron(state, s)
            
        # Convert to density matrix
        rho = torch.outer(state, state.conj())
        states.append(rho)
        
    return states

def generate_input_states(num_qubits: int,
                         num_samples: int,
                         device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    """
    Generate input states for quantum process tomography.
    
    Args:
        num_qubits (int): Number of qubits.
        num_samples (int): Number of samples to generate.
        device (Optional[Union[str, torch.device]]): Device for the tensors.
        
    Returns:
        List[torch.Tensor]: List of input state density matrices.
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Generate states from Pauli basis
    states = generate_pauli_basis_states(num_qubits, device=dev)
    
    # If we need more states than the Pauli basis provides, generate random states
    if num_samples > len(states):
        additional_states = []
        for _ in range(num_samples - len(states)):
            # Generate random pure state
            state = torch.randn(2**num_qubits, dtype=torch.complex64, device=dev)
            state = state / torch.norm(state)
            rho = torch.outer(state, state.conj())
            additional_states.append(rho)
        states.extend(additional_states)
    
    # Return the requested number of states
    return states[:num_samples]

def generate_measurement_operators(num_qubits: int,
                                 num_samples: int,
                                 device: Optional[Union[str, torch.device]] = None) -> List[torch.Tensor]:
    """
    Generate measurement operators for quantum process tomography.
    
    Args:
        num_qubits (int): Number of qubits.
        num_samples (int): Number of samples to generate.
        device (Optional[Union[str, torch.device]]): Device for the tensors.
        
    Returns:
        List[torch.Tensor]: List of measurement operators.
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Get single-qubit projectors
    x_projectors = get_basis_projectors_single_qubit("X", device=dev)
    y_projectors = get_basis_projectors_single_qubit("Y", device=dev)
    z_projectors = get_basis_projectors_single_qubit("Z", device=dev)
    
    # All single-qubit projectors
    single_qubit_projectors = [x_projectors[0], x_projectors[1],
                             y_projectors[0], y_projectors[1],
                             z_projectors[0], z_projectors[1]]
    
    # Generate all possible combinations for num_qubits
    operators = []
    for proj_combo in itertools.product(single_qubit_projectors, repeat=num_qubits):
        # Create product operator
        op = proj_combo[0]
        for p in proj_combo[1:]:
            op = torch.kron(op, p)
        operators.append(op)
    
    # If we need more operators than the Pauli basis provides, generate random POVM elements
    if num_samples > len(operators):
        additional_operators = []
        for _ in range(num_samples - len(operators)):
            # Generate random POVM element (positive semidefinite matrix with trace <= 1)
            A = torch.randn(2**num_qubits, 2**num_qubits, dtype=torch.complex64, device=dev)
            M = A @ A.conj().T
            M = M / torch.trace(M)  # Normalize to have trace 1
            additional_operators.append(M)
        operators.extend(additional_operators)
    
    # Return the requested number of operators
    return operators[:num_samples]

def generate_training_data(num_qubits: int,
                         num_samples: int,
                         true_channel: Optional[callable] = None,
                         device: Optional[Union[str, torch.device]] = None) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, float]], List[Tuple[torch.Tensor, torch.Tensor, float]]]:
    """
    Generate training and validation data for quantum process tomography.
    
    Args:
        num_qubits (int): Number of qubits.
        num_samples (int): Number of samples to generate.
        true_channel (Optional[callable]): Function that implements the true quantum channel.
            If None, a random channel will be used.
        device (Optional[Union[str, torch.device]]): Device for the tensors.
        
    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor, float]], List[Tuple[torch.Tensor, torch.Tensor, float]]]:
            Training and validation data as lists of (input_state, measurement, probability) tuples.
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Generate input states and measurement operators
    input_states = generate_input_states(num_qubits, num_samples, device=dev)
    measurement_ops = generate_measurement_operators(num_qubits, num_samples, device=dev)
    
    # Create or use provided channel
    if true_channel is None:
        # Create a random channel (you might want to implement a more sophisticated one)
        def random_channel(rho):
            # Simple depolarizing channel for demonstration
            p = 0.1  # depolarization probability
            return (1 - p) * rho + p * torch.eye(2**num_qubits, device=dev) / 2**num_qubits
        channel = random_channel
    else:
        channel = true_channel
    
    # Generate data
    data = []
    for _ in range(num_samples):
        # Randomly select input state and measurement
        rho = input_states[np.random.randint(len(input_states))]
        M = measurement_ops[np.random.randint(len(measurement_ops))]
        
        # Apply channel and compute probability
        rho_out = channel(rho)
        prob = torch.trace(M @ rho_out).real.item()
        
        data.append((rho, M, prob))
    
    # Split into train and validation sets (80/20)
    np.random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def evaluate_channel_reconstruction(model: torch.nn.Module,
                                  num_qubits: int,
                                  num_test_states: int = 100,
                                  device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Evaluate the quality of channel reconstruction using process fidelity.
    
    Args:
        model (torch.nn.Module): The trained LPDO model.
        num_qubits (int): Number of qubits.
        num_test_states (int): Number of test states to use for evaluation.
        device (Optional[Union[str, torch.device]]): Device for the tensors.
        
    Returns:
        float: Process fidelity between reconstructed and true channels.
    """
    dev = torch.device(device) if device is not None else torch.device('cpu')
    
    # Generate random test states
    test_states = []
    for _ in range(num_test_states):
        # Generate random pure state
        state = torch.randn(2**num_qubits, dtype=torch.complex64, device=dev)
        state = state / torch.norm(state)
        rho = torch.outer(state, state.conj())
        test_states.append(rho)
    
    # Compute process fidelity
    total_fidelity = 0.0
    for rho in test_states:
        # Apply model to input state
        rho_out = model(rho)
        
        # Compute fidelity between input and output states
        # F = Tr(sqrt(sqrt(rho) @ rho_out @ sqrt(rho)))
        sqrt_rho = torch.matrix_power(rho, 0.5)
        inner = sqrt_rho @ rho_out @ sqrt_rho
        fidelity = torch.trace(torch.matrix_power(inner, 0.5)).real.item()
        total_fidelity += fidelity
    
    return total_fidelity / num_test_states 