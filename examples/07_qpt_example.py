import torch
import numpy as np
from torchqpt.models import LPDO
from torchqpt.training import QPTTrainer
from torchqpt.data import (
    generate_input_states,
    generate_measurement_operators,
    evaluate_channel_reconstruction
)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    num_qubits = 2
    bond_dim = 2
    kraus_dim = 2
    
    # Training parameters
    num_train_samples = 1000
    num_val_samples = 200
    batch_size = 800  # From paper
    learning_rate = 0.005  # From paper
    num_epochs = 100
    patience = 10
    
    # Initialize model using random initialization
    model = LPDO.random_initialization(
        num_sites=num_qubits,
        physical_dim=2,  # qubits
        bond_dim=bond_dim,
        kraus_dim=kraus_dim
    )
    
    # Generate training data
    print("Generating training data...")
    train_input_states = generate_input_states(num_qubits, num_train_samples)
    train_measurements = generate_measurement_operators(num_qubits, num_train_samples)
    
    # Generate validation data
    print("Generating validation data...")
    val_input_states = generate_input_states(num_qubits, num_val_samples)
    val_measurements = generate_measurement_operators(num_qubits, num_val_samples)
    
    # Create trainer
    trainer = QPTTrainer(
        model=model,
        learning_rate=learning_rate,
        regularization_weight=1.0
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_data=list(zip(train_input_states, train_measurements, [1.0] * num_train_samples)),
        val_data=list(zip(val_input_states, val_measurements, [1.0] * num_val_samples)),
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience
    )
    
    # Save model
    trainer.save_model("qpt_model.pt")
    
    # Evaluate reconstruction
    print("\nEvaluating channel reconstruction...")
    fidelity = evaluate_channel_reconstruction(model, num_qubits)
    print(f"Channel reconstruction fidelity: {fidelity:.4f}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main() 