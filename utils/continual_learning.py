"""
Continual Learning Support for KAN.

Implements:
- EWC (Elastic Weight Consolidation) regularization
- Memory replay buffer
- Task-specific learning rate scheduling
- Forgetting prevention metrics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


class ContinualLearner:
    """
    Continual learning wrapper for KAN models.
    
    Supports learning multiple tasks sequentially without catastrophic forgetting.
    """
    
    def __init__(self, model, ewc_lambda: float = 1000.0):
        """
        Args:
            model: KAN model
            ewc_lambda: Importance weight for EWC regularization
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # For EWC
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # For memory replay
        self.memory_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.memory_size = 1000
        
        # Track task performance
        self.task_history: List[Dict] = []
    
    def compute_fisher(self, data_loader, n_samples: int = 200):
        """
        Compute Fisher Information Matrix for EWC.
        
        This estimates the importance of each parameter for the current task.
        
        Args:
            data_loader: DataLoader or list of (x, y) tuples
            n_samples: Number of samples to use
        """
        self.model.train()
        
        # Initialize Fisher information
        fisher = {name: torch.zeros_like(param) 
                  for name, param in self.model.named_parameters() if param.requires_grad}
        
        count = 0
        for x, y in data_loader:
            if count >= n_samples:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad ** 2
            
            count += x.shape[0]
        
        # Normalize
        for name in fisher:
            fisher[name] /= count
        
        # Update stored Fisher information (accumulate across tasks)
        for name in fisher:
            if name in self.fisher_information:
                self.fisher_information[name] += fisher[name]
            else:
                self.fisher_information[name] = fisher[name]
        
        # Store optimal parameters
        self.optimal_params = {name: param.clone().detach() 
                               for name, param in self.model.named_parameters() 
                               if param.requires_grad}
    
    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        L_ewc = Σ_i (λ/2) * F_i * (θ_i - θ*_i)²
        """
        if not self.fisher_information:
            return torch.tensor(0.0)
        
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and param.requires_grad:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return (self.ewc_lambda / 2) * loss
    
    def add_to_memory(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 100):
        """Add samples to memory buffer for replay."""
        # Randomly sample
        indices = torch.randperm(x.shape[0])[:n_samples]
        x_sample = x[indices].clone().detach()
        y_sample = y[indices].clone().detach()
        
        self.memory_buffer.append((x_sample, y_sample))
        
        # Limit memory size
        total_samples = sum(buf[0].shape[0] for buf in self.memory_buffer)
        while total_samples > self.memory_size and len(self.memory_buffer) > 1:
            self.memory_buffer.pop(0)
            total_samples = sum(buf[0].shape[0] for buf in self.memory_buffer)
    
    def replay_loss(self) -> Optional[torch.Tensor]:
        """Compute loss on memory replay buffer."""
        if not self.memory_buffer:
            return None
        
        total_loss = 0
        for x, y in self.memory_buffer:
            output = self.model(x)
            total_loss += nn.MSELoss()(output, y)
        
        return total_loss / len(self.memory_buffer)
    
    def train_task(self, x: torch.Tensor, y: torch.Tensor, 
                   task_name: str = "task",
                   epochs: int = 200, 
                   lr: float = 0.01,
                   use_ewc: bool = True,
                   use_replay: bool = True) -> Dict:
        """
        Train on a new task with continual learning regularization.
        
        Args:
            x, y: Training data
            task_name: Name for this task
            epochs: Number of training epochs
            lr: Learning rate
            use_ewc: Whether to use EWC regularization
            use_replay: Whether to use memory replay
        Returns:
            Dict with training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        history = {'loss': [], 'ewc_loss': [], 'replay_loss': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Main task loss
            output = self.model(x)
            task_loss = criterion(output, y)
            
            # EWC regularization
            ewc = self.ewc_loss() if use_ewc else torch.tensor(0.0)
            
            # Memory replay
            replay = self.replay_loss() if use_replay else None
            
            # Total loss
            total_loss = task_loss + ewc
            if replay is not None:
                total_loss += 0.5 * replay
            
            total_loss.backward()
            optimizer.step()
            
            history['loss'].append(task_loss.item())
            history['ewc_loss'].append(ewc.item())
            history['replay_loss'].append(replay.item() if replay else 0)
        
        # After training, compute Fisher for this task
        data = [(x, y)]
        self.compute_fisher(data)
        self.add_to_memory(x, y)
        
        # Record task
        self.task_history.append({
            'name': task_name,
            'final_loss': history['loss'][-1],
            'samples': x.shape[0]
        })
        
        print(f"Task '{task_name}' trained. Final loss: {history['loss'][-1]:.6f}")
        return history
    
    def evaluate_all_tasks(self, task_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        Evaluate model on all previous tasks.
        
        Returns:
            Dict mapping task index to loss
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for i, (x, y) in enumerate(task_data):
                output = self.model(x)
                loss = nn.MSELoss()(output, y).item()
                results[f'task_{i}'] = loss
        
        return results


class KANTrainer:
    """
    Comprehensive trainer for KAN models with callbacks and utilities.
    """
    
    def __init__(self, model, optimizer=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.01)
        self.scheduler = scheduler
        
        self.callbacks: List[callable] = []
        self.history = {'train_loss': [], 'val_loss': [], 'reg_loss': []}
        self.best_loss = float('inf')
        self.best_state = None
    
    def add_callback(self, callback: callable):
        """Add a callback function called after each epoch."""
        self.callbacks.append(callback)
    
    def train(self, x_train: torch.Tensor, y_train: torch.Tensor,
              x_val: Optional[torch.Tensor] = None,
              y_val: Optional[torch.Tensor] = None,
              epochs: int = 500,
              reg_lambda: float = 0.0,
              early_stopping: int = 50,
              verbose: bool = True) -> Dict:
        """
        Train the KAN model.
        
        Args:
            x_train, y_train: Training data
            x_val, y_val: Optional validation data
            epochs: Number of epochs
            reg_lambda: Regularization strength
            early_stopping: Stop after this many epochs without improvement
            verbose: Print progress
        Returns:
            Training history dict
        """
        criterion = nn.MSELoss()
        no_improve = 0
        
        for epoch in range(epochs):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(x_train)
            train_loss = criterion(output, y_train)
            
            # Regularization
            reg_loss = torch.tensor(0.0)
            if reg_lambda > 0 and hasattr(self.model, 'get_reg'):
                reg_loss = reg_lambda * self.model.get_reg()
            
            total_loss = train_loss + reg_loss
            total_loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Validation
            val_loss = 0
            if x_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(x_val)
                    val_loss = criterion(val_output, y_val).item()
            
            # Record history
            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss)
            self.history['reg_loss'].append(reg_loss.item())
            
            # Best model tracking
            current_loss = val_loss if x_val is not None else train_loss.item()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_state = deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if early_stopping and no_improve >= early_stopping:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Callbacks
            for callback in self.callbacks:
                callback(epoch, self.history, self.model)
            
            # Verbose output
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.6f}, Val: {val_loss:.6f}")
        
        return self.history
    
    def restore_best(self):
        """Restore the best model state."""
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            print(f"Restored best model (loss: {self.best_loss:.6f})")
    
    def save(self, path: str):
        """Save model state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss
        }, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        self.best_loss = checkpoint['best_loss']
        print(f"Loaded from {path}")
