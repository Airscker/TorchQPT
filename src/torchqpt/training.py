import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .models import LPDO
from .states import DensityMatrix

class QPTTrainer:
    """
    Trainer for Quantum Process Tomography using LPDO model.
    
    This class implements the training procedure described in the paper, including:
    1. Negative log-likelihood loss
    2. Trace-preserving regularization
    3. Training and validation loops
    4. Model selection based on validation loss
    """
    def __init__(self,
                 model: LPDO,
                 learning_rate: float = 0.005,  # Default from paper
                 regularization_weight: float = 1.0,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the QPT trainer.
        
        Args:
            model (LPDO): The LPDO model to train.
            learning_rate (float): Learning rate for the Adam optimizer (default: 0.005 from paper).
            regularization_weight (float): Weight for the trace-preserving regularization term.
            device (Optional[Union[str, torch.device]]): Device to use for training.
        """
        self.model = model
        self.device = torch.device(device) if device is not None else model.device
        self.regularization_weight = regularization_weight
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer with parameters from paper
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # beta1, beta2 from paper
            eps=1e-7  # epsilon from paper
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def compute_loss(self,
                    input_states: List[torch.Tensor],
                    measurements: List[torch.Tensor],
                    probabilities: List[float]) -> torch.Tensor:
        """
        Compute the total loss including negative log-likelihood and regularization.
        
        Args:
            input_states (List[torch.Tensor]): List of input state tensors.
            measurements (List[torch.Tensor]): List of measurement operator tensors.
            probabilities (List[float]): List of observed probabilities.
            
        Returns:
            torch.Tensor: Total loss value.
        """
        # Compute negative log-likelihood
        nll_loss = 0.0
        for rho, M, p in zip(input_states, measurements, probabilities):
            pred_prob = self.model.compute_probability(rho, M)
            nll_loss -= torch.log(pred_prob) * p
            
        # Add trace-preserving regularization
        reg_loss = self.model.trace_preserving_regularizer()
        
        # Total loss
        total_loss = nll_loss + self.regularization_weight * reg_loss
        
        return total_loss
    
    def train_epoch(self,
                   train_data: List[Tuple[torch.Tensor, torch.Tensor, float]],
                   batch_size: int = 800) -> float:  # Default batch size from paper
        """
        Train for one epoch.
        
        Args:
            train_data (List[Tuple[torch.Tensor, torch.Tensor, float]]): List of (input_state, measurement, probability) tuples.
            batch_size (int): Size of batches for training (default: 800 from paper).
            
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [train_data[idx] for idx in batch_indices]
            
            # Unpack batch
            input_states = [x[0] for x in batch_data]
            measurements = [x[1] for x in batch_data]
            probabilities = [x[2] for x in batch_data]
            
            # Compute loss
            loss = self.compute_loss(input_states, measurements, probabilities)
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self,
                val_data: List[Tuple[torch.Tensor, torch.Tensor, float]]) -> float:
        """
        Compute validation loss.
        
        Args:
            val_data (List[Tuple[torch.Tensor, torch.Tensor, float]]): List of (input_state, measurement, probability) tuples.
            
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            input_states = [x[0] for x in val_data]
            measurements = [x[1] for x in val_data]
            probabilities = [x[2] for x in val_data]
            
            loss = self.compute_loss(input_states, measurements, probabilities)
            
        return loss.item()
    
    def train(self,
             train_data: List[Tuple[torch.Tensor, torch.Tensor, float]],
             val_data: List[Tuple[torch.Tensor, torch.Tensor, float]],
             num_epochs: int = 100,
             batch_size: int = 800,  # Default batch size from paper
             patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_data (List[Tuple[torch.Tensor, torch.Tensor, float]]): Training data.
            val_data (List[Tuple[torch.Tensor, torch.Tensor, float]]): Validation data.
            num_epochs (int): Maximum number of epochs to train.
            batch_size (int): Size of batches for training (default: 800 from paper).
            patience (int): Number of epochs to wait for improvement before early stopping.
            
        Returns:
            Dict[str, List[float]]: Training history containing train and validation losses.
        """
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_data, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_data)
            self.val_losses.append(val_loss)
            
            # Model selection
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: str):
        """Save the best model state."""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }, path)
            
    def load_model(self, path: str):
        """Load a saved model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss'] 