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
    """
    def __init__(self,
                 model: LPDO,
                 learning_rate: float = 0.005,
                 regularization_weight: float = 1.0,
                 device: Optional[Union[str, torch.device]] = None):
        self.model = model
        self.device = torch.device(device) if device is not None else model.device
        self.regularization_weight = regularization_weight
        
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def compute_loss(self,
                    input_states: List[torch.Tensor],
                    measurements: List[torch.Tensor],
                    probabilities: List[float]) -> torch.Tensor:
        if len(input_states) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        nll_loss = 0.0
        for rho, M, true_prob in zip(input_states, measurements, probabilities):
            rho, M = rho.to(self.device), M.to(self.device)
            pred_prob = self.model.compute_probability(rho, M)
            # Use KL divergence: -true_prob * log(pred_prob)
            nll_loss -= true_prob * torch.log(pred_prob + 1e-9)
            
        nll_loss /= len(input_states)
        reg_loss = self.model.trace_preserving_regularizer()
        total_loss = nll_loss + self.regularization_weight * reg_loss
        return total_loss
    
    def train_epoch(self,
                   train_data: List[Tuple[torch.Tensor, torch.Tensor, float]],
                   batch_size: int = 800) -> float:
        self.model.train()
        total_epoch_loss = 0.0
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [train_data[idx] for idx in batch_indices]
            
            input_states = [x[0] for x in batch_data]
            measurements = [x[1] for x in batch_data]
            probabilities = [x[2] for x in batch_data]
            
            self.optimizer.zero_grad()
            loss = self.compute_loss(input_states, measurements, probabilities)
            loss.backward()
            self.optimizer.step()
            
            total_epoch_loss += loss.item()
            
        return total_epoch_loss / max(1, (len(train_data) / batch_size))
    
    def validate(self,
                val_data: List[Tuple[torch.Tensor, torch.Tensor, float]]) -> float:
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
             batch_size: int = 800,
             patience: int = 10) -> Dict[str, List[float]]:
        self.train_losses, self.val_losses = [], []
        self.best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_data, batch_size)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(val_data)
            self.val_losses.append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def save_model(self, path: str):
        if self.best_model_state:
            torch.save({'model_state_dict': self.best_model_state}, path)
            
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])