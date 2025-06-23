import torch
import numpy as np
from torchqpt.models import LPDO
from torchqpt.training import QPTTrainer
from torchqpt.data import generate_input_states, generate_measurement_operators

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    num_qubits = 1
    bond_dim = 2
    kraus_dim = 2 # Use > 1 to have capacity for noisy channels
    
    num_train_samples = 1000
    num_val_samples = 200
    batch_size = 200
    learning_rate = 0.01
    num_epochs = 50
    patience = 5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Generate dummy data (in a real scenario, this comes from a true channel)
    # This example only tests if the training loop runs.
    train_inputs = generate_input_states(num_qubits, num_train_samples, device)
    train_measurements = generate_measurement_operators(num_qubits, num_train_samples, device)
    train_data = list(zip(train_inputs, train_measurements))

    val_inputs = generate_input_states(num_qubits, num_val_samples, device)
    val_measurements = generate_measurement_operators(num_qubits, num_val_samples, device)
    val_data = list(zip(val_inputs, val_measurements))

    # 2. Initialize model
    model = LPDO.random_initialization(
        num_sites=num_qubits,
        physical_dim=2,
        bond_dim=bond_dim,
        kraus_dim=kraus_dim,
        device=device
    )
    
    # 3. Create trainer
    trainer = QPTTrainer(
        model=model,
        learning_rate=learning_rate,
        regularization_weight=0.1 
    )
    
    # 4. Train model
    print("Starting training loop test...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience
    )
    print("Training finished.")
    
    # 5. Save model
    trainer.save_model("qpt_model_test.pt")
    print("Model saved to qpt_model_test.pt")
    
if __name__ == "__main__":
    main()