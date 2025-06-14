import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchqpt.tomography import (
    get_basis_projectors_single_qubit,
    simulate_measurement_probabilities,
    qst_linear_inversion_single_qubit,
    get_pauli_operator,
    qst_linear_inversion_multi_qubit
)
from torchqpt.states import QuantumStateVector, DensityMatrix

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n1. Single-qubit tomography example:")
    print("-" * 50)
    
    # Create a pure state |+⟩
    plus_state = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.cfloat).to(device)
    print("Original state (density matrix):")
    print(plus_state)

    # Simulate measurements in different bases
    bases = ["X", "Y", "Z"]
    measurement_data = {}
    
    for basis in bases:
        P_0, P_1 = get_basis_projectors_single_qubit(basis, device=device)
        probs = simulate_measurement_probabilities(plus_state, (P_0, P_1))
        measurement_data[basis] = probs
        print(f"\nMeasurement probabilities in {basis} basis:")
        print(f"P(+{basis}) = {probs[0]:.4f}")
        print(f"P(-{basis}) = {probs[1]:.4f}")

    # Reconstruct the state using linear inversion
    reconstructed_state = qst_linear_inversion_single_qubit(measurement_data, device=device)
    print("\nReconstructed state (density matrix):")
    print(reconstructed_state.density_matrix)

    # Verify reconstruction
    print("\nVerification:")
    print(f"Max difference between original and reconstructed: {torch.max(torch.abs(plus_state - reconstructed_state.density_matrix)):.6f}")

    print("\n2. Two-qubit tomography example:")
    print("-" * 50)
    
    # Create a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    bell_state = torch.zeros((4, 4), dtype=torch.cfloat, device=device)
    bell_state[0, 0] = 0.5
    bell_state[0, 3] = 0.5
    bell_state[3, 0] = 0.5
    bell_state[3, 3] = 0.5
    print("Original Bell state (density matrix):")
    print(bell_state)

    # Generate measurement data for the Bell state
    measurement_data = {
        "IX": 0.0,
        "XI": 0.0,
        "XX": 1.0,
        "IY": 0.0,
        "YI": 0.0,
        "YY": -1.0,
        "IZ": 0.0,
        "ZI": 0.0,
        "ZZ": 1.0
    }

    # Reconstruct the state using linear inversion
    reconstructed_state = qst_linear_inversion_multi_qubit(measurement_data, num_qubits=2, device=device)
    print("\nReconstructed Bell state (density matrix):")
    print(reconstructed_state.density_matrix)

    # Verify reconstruction
    print("\nVerification:")
    print(f"Max difference between original and reconstructed: {torch.max(torch.abs(bell_state - reconstructed_state.density_matrix)):.6f}")

    # Demonstrate Pauli operator construction
    print("\n3. Pauli operator examples:")
    print("-" * 50)
    
    # Single-qubit Pauli operators
    for pauli in ["I", "X", "Y", "Z"]:
        op = get_pauli_operator(pauli, device=device)
        print(f"\nPauli {pauli}:")
        print(op)

    # Two-qubit Pauli operators
    for pauli in ["IX", "XI", "XX"]:
        op = get_pauli_operator(pauli, device=device)
        print(f"\nPauli {pauli}:")
        print(op)

if __name__ == "__main__":
    main() 