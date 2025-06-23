import torch
import numpy as np
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchqpt.models import LPDO
from torchqpt.training import QPTTrainer
from torchqpt.data import generate_input_states, generate_measurement_operators, calculate_process_fidelity
from torchqpt.noise import amplitude_damping_channel

def get_true_noisy_channel(num_qubits: int, noise_gamma: float, device: torch.device):
    """
    Creates a true channel and its Choi matrix for an identity gate followed by
    amplitude damping on the first qubit.
    """
    kraus_ops_ad = amplitude_damping_channel(gamma=noise_gamma, device=device)
    dim = 2**num_qubits

    # Define the channel operation E(rho)
    def noisy_channel(rho_in: torch.Tensor) -> torch.Tensor:
        rho_out = torch.zeros_like(rho_in)
        for K in kraus_ops_ad:
            # Apply noise to the first qubit only
            full_k_op = torch.kron(K, torch.eye(dim // 2, device=device))
            rho_out += full_k_op @ rho_in @ full_k_op.conj().T
        return rho_out

    # Compute the Choi matrix for this channel: Λ = Σ_ij (E(|i><j|) ⊗ |i><j|)
    choi_matrix = torch.zeros((dim**2, dim**2), dtype=torch.complex64, device=device)
    for i in range(dim):
        for j in range(dim):
            E_ij = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
            E_ij[i, j] = 1.0
            E_ij_out = noisy_channel(E_ij)
            choi_matrix += torch.kron(E_ij_out, E_ij)
            
    return noisy_channel, choi_matrix

def main():
    print("--- Tomography of a Noisy Quantum Channel ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    num_qubits = 1
    bond_dim = 2
    kraus_dim = 2
    noise_gamma = 0.1
    
    num_train = 2000
    num_val = 400
    batch_size = 500
    learning_rate = 0.01
    num_epochs = 100
    patience = 10

    # 1. Get the true channel and its Choi matrix
    true_channel, true_choi = get_true_noisy_channel(num_qubits, noise_gamma, device)
    
    # 2. Generate data with probabilities
    print("Generating data from the true noisy channel...")
    inputs = generate_input_states(num_qubits, num_train + num_val, device)
    measurements = generate_measurement_operators(num_qubits, num_train + num_val, device)
    
    # Generate probabilities using the true channel
    all_data = []
    for rho_in, M_out in zip(inputs, measurements):
        # Apply the true channel to get the output state
        rho_out = true_channel(rho_in)
        # Calculate the probability: Tr[M_out * rho_out]
        prob = torch.trace(M_out @ rho_out).real.item()
        # Add small epsilon to avoid log(0)
        prob = max(prob, 1e-9)
        all_data.append((rho_in, M_out, prob))
    
    train_data = all_data[:num_train]
    val_data = all_data[num_train:]

    # 3. Initialize and train the model
    model = LPDO.random_initialization(
        num_sites=num_qubits, physical_dim=2, bond_dim=bond_dim, kraus_dim=kraus_dim, device=device
    )
    
    trainer = QPTTrainer(model, learning_rate=learning_rate, regularization_weight=0.0)
    
    print("Starting training...")
    trainer.train(train_data, val_data, num_epochs, batch_size, patience)

    # 4. Evaluate the final model
    print("\nEvaluating final model...")
    final_fidelity = calculate_process_fidelity(model, true_choi)
    
    print(f"\n--- Results ---")
    print(f"Reconstruction of a 1-qubit amplitude damping channel (gamma={noise_gamma})")
    print(f"Final Process Fidelity: {final_fidelity:.4f}")
    print("-----------------")

if __name__ == "__main__":
    main()
    