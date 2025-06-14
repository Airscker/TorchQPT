import torch
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np # For np.sqrt, though torch.sqrt(torch.tensor(x)) is used
from .gates import COMPLEX_DTYPE
from .circuits import QuantumCircuit

REAL_DTYPE = torch.tensor(0., dtype=COMPLEX_DTYPE).real.dtype # Derived real dtype

# Helper Pauli matrices (internal use, simple docstrings or comments suffice)
def _I() -> torch.Tensor:
    """Single-qubit Identity matrix."""
    return torch.tensor([[1, 0], [0, 1]], dtype=COMPLEX_DTYPE)

def _X() -> torch.Tensor:
    """Single-qubit Pauli-X matrix."""
    return torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)

def _Y() -> torch.Tensor:
    """Single-qubit Pauli-Y matrix."""
    return torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)

def _Z() -> torch.Tensor:
    """Single-qubit Pauli-Z matrix."""
    return torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)


def pauli_channel(px: float, py: float, pz: float) -> List[torch.Tensor]:
    """
    Generates Kraus operators for a Pauli channel.

    The Pauli channel applies a Pauli-X, Pauli-Y, or Pauli-Z error with
    probabilities px, py, pz respectively. With probability 1-px-py-pz,
    no error (Identity) occurs.

    Args:
        px (float): Probability of a Pauli-X error.
        py (float): Probability of a Pauli-Y error.
        pz (float): Probability of a Pauli-Z error.
            Constraint: 0 <= px, py, pz and px + py + pz <= 1.

    Returns:
        List[torch.Tensor]: A list of four Kraus operators [K_I, K_X, K_Y, K_Z],
                            where:
                            K_I = sqrt(1-px-py-pz) * I
                            K_X = sqrt(px) * X
                            K_Y = sqrt(py) * Y
                            K_Z = sqrt(pz) * Z
                            All operators are 2x2 torch.Tensors with COMPLEX_DTYPE.

    Raises:
        ValueError: If probabilities are not in the range [0,1] or if their sum exceeds 1.
    """
    if not (0 <= px <= 1 and 0 <= py <= 1 and 0 <= pz <= 1):
        raise ValueError("Probabilities px, py, pz must be between 0 and 1.")

    sum_p = px + py + pz
    if not (0 <= sum_p <= 1.000001): # Allow for small floating point inaccuracies if sum_p is very close to 1
        raise ValueError(f"Sum of probabilities px + py + pz ({sum_p}) must be between 0 and 1.")

    p_i = 1.0 - sum_p
    # Handle potential negative p_i due to floating point arithmetic if sum_p slightly > 1
    p_i = max(0.0, p_i)

    K_I = torch.sqrt(torch.tensor(p_i, dtype=REAL_DTYPE)) * _I()
    K_X = torch.sqrt(torch.tensor(px, dtype=REAL_DTYPE)) * _X()
    K_Y = torch.sqrt(torch.tensor(py, dtype=REAL_DTYPE)) * _Y()
    K_Z = torch.sqrt(torch.tensor(pz, dtype=REAL_DTYPE)) * _Z() # Corrected: was missing * _Z()

    return [K_I, K_X, K_Y, K_Z]


def depolarizing_channel(p: float, device: Union[str, torch.device] = 'cpu') -> List[torch.Tensor]:
    """
    Generate Kraus operators for the depolarizing channel.
    
    Args:
        p (float): Error probability
        device (Union[str, torch.device]): Device to place tensors on
        
    Returns:
        List[torch.Tensor]: List of Kraus operators
    """
    dev = torch.device(device)
    sqrt_p = torch.sqrt(torch.tensor(p/3, device=dev))
    
    # Identity operator
    K0 = torch.sqrt(torch.tensor(1 - p, device=dev)) * torch.eye(2, device=dev)
    
    # Pauli operators
    K1 = sqrt_p * torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE, device=dev)  # X
    K2 = sqrt_p * torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE, device=dev)  # Y
    K3 = sqrt_p * torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE, device=dev)  # Z
    
    return [K0, K1, K2, K3]


def amplitude_damping_channel(gamma: float, device: Union[str, torch.device] = 'cpu') -> List[torch.Tensor]:
    """
    Generate Kraus operators for the amplitude damping channel.
    
    Args:
        gamma (float): Damping rate
        device (Union[str, torch.device]): Device to place tensors on
        
    Returns:
        List[torch.Tensor]: List of Kraus operators
    """
    dev = torch.device(device)
    
    # Main operator
    K0 = torch.tensor([[1, 0], [0,np.sqrt(1 - gamma)]], dtype=COMPLEX_DTYPE, device=dev)
    
    # Damping operator
    K1 = torch.tensor([[0, np.sqrt(gamma)], [0, 0]], dtype=COMPLEX_DTYPE, device=dev)
    
    return [K0, K1]


def phase_damping_channel(gamma: float, device: Union[str, torch.device] = 'cpu') -> List[torch.Tensor]:
    """
    Generate Kraus operators for the phase damping channel.
    
    Args:
        gamma (float): Damping rate
        device (Union[str, torch.device]): Device to place tensors on
        
    Returns:
        List[torch.Tensor]: List of Kraus operators
    """
    dev = torch.device(device)
    
    # Main operator
    K0 = torch.tensor([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=COMPLEX_DTYPE, device=dev)
    
    # Damping operator
    K1 = torch.tensor([[0, 0], [0, np.sqrt(gamma)]], dtype=COMPLEX_DTYPE, device=dev)
    
    return [K0, K1]

def add_noise_to_circuit(circuit: QuantumCircuit, noise_model: Dict[int, Tuple[str, Dict[str, float]]]) -> QuantumCircuit:
    """
    Add noise to a quantum circuit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to add noise to
        noise_model (Dict[int, Tuple[str, Dict[str, float]]]): Noise model specification
            Format: {n_qubits: (noise_type, params)}
            Example: {1: ('depolarizing', {'p': 0.1}), 2: ('depolarizing', {'p': 0.2})}
            
    Returns:
        QuantumCircuit: A new circuit with noise gates inserted after each operation
    """
    from .circuits import QuantumCircuit
    
    # Create a new circuit with the same number of qubits
    noisy_circuit = QuantumCircuit(circuit.num_qubits)
    
    # Copy all operations from the original circuit
    for op in circuit.operations:
        # Add the original operation
        noisy_circuit.add_operation(op[0], op[1], op[2])
        
        # Get number of qubits the operation acts on
        if isinstance(op[1], tuple):
            n_qubits = len(op[1])
        else:
            n_qubits = 1
            
        # Check if we have a noise model for this gate size
        if n_qubits in noise_model:
            noise_type, params = noise_model[n_qubits]
            
            # Generate Kraus operators based on noise type
            if noise_type == 'depolarizing':
                kraus_ops = depolarizing_channel(params['p'], device=circuit.device)
            elif noise_type == 'amplitude_damping':
                kraus_ops = amplitude_damping_channel(params['gamma'], device=circuit.device)
            elif noise_type == 'phase_damping':
                kraus_ops = phase_damping_channel(params['gamma'], device=circuit.device)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
                
            # Add noise channel after the operation
            if n_qubits > 1:
                # For multi-qubit gates, apply noise to each qubit individually
                for qubit in op[1]:
                    noisy_circuit.add_kraus_channel(kraus_ops, qubit)
            else:
                noisy_circuit.add_kraus_channel(kraus_ops, op[1])
                
    return noisy_circuit
