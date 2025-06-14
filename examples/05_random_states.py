import torch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchqpt.states import (
    COMPLEX_DTYPE,
    random_state_vector,
    random_density_matrix,
    random_pure_state,
)

def main():
    """Demonstrates the generation of random quantum states."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Random Quantum States Example ---")
    print(f"Using device: {device}\n")

    # 1. Generate a random state vector
    num_qubits = 2
    print("1. Random State Vector:")
    state = random_state_vector(num_qubits, device=device)
    print(f"State vector:\n{state.state_vector}")
    print(f"Norm: {torch.sqrt(torch.sum(torch.abs(state.state_vector)**2))}\n")

    # 2. Generate a random density matrix (mixed state)
    print("2. Random Density Matrix (Mixed State):")
    rho = random_density_matrix(num_qubits, device=device)
    print(f"Density matrix:\n{rho.density_matrix}")
    print(f"Trace: {torch.trace(rho.density_matrix).real}")
    print(f"Purity: {torch.trace(rho.density_matrix @ rho.density_matrix).real}\n")

    # 3. Generate a random pure state as a density matrix
    print("3. Random Pure State (as Density Matrix):")
    rho_pure = random_pure_state(num_qubits, device=device)
    print(f"Density matrix:\n{rho_pure.density_matrix}")
    print(f"Trace: {torch.trace(rho_pure.density_matrix).real}")
    print(f"Purity: {torch.trace(rho_pure.density_matrix @ rho_pure.density_matrix).real}\n")

    # 4. Compare properties of mixed vs pure states
    print("4. Comparison of Mixed vs Pure States:")
    print("Mixed state properties:")
    print(f"- Trace: {torch.trace(rho.density_matrix).real}")
    print(f"- Purity: {torch.trace(rho.density_matrix @ rho.density_matrix).real}")
    print(f"- Eigenvalues: {torch.linalg.eigvals(rho.density_matrix).real}\n")

    print("Pure state properties:")
    print(f"- Trace: {torch.trace(rho_pure.density_matrix).real}")
    print(f"- Purity: {torch.trace(rho_pure.density_matrix @ rho_pure.density_matrix).real}")
    print(f"- Eigenvalues: {torch.linalg.eigvals(rho_pure.density_matrix).real}")

if __name__ == "__main__":
    main() 
