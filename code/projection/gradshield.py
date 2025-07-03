#!/usr/bin/env python
# Gradient shielding implementation

import torch
import numpy as np


class GradShield:
    """Gradient shielding for Shielded RecRL.
    
    This class implements the gradient projection technique that prevents
    the explainer model from affecting the ranking model's performance.
    """
    
    def __init__(self, basis_matrix, device='cuda'):
        """Initialize the gradient shield.
        
        Args:
            basis_matrix: The orthogonal basis matrix (Q) for the embedding space
            device: The device to use for computation
        """
        self.Q = basis_matrix.to(device)
        self.device = device
        print(f"Initialized GradShield with basis matrix of shape {self.Q.shape}")
    
    def project_gradient(self, grad):
        """Project a gradient to remove components in the embedding space.
        
        Args:
            grad: The gradient tensor to project
            
        Returns:
            The projected gradient tensor
        """
        # Project the gradient onto the basis
        proj = self.Q @ (self.Q.T @ grad.view(-1)).view(-1, 1)
        
        # Subtract the projection to get the orthogonal component
        return grad - proj.view(grad.shape)
    
    def apply_to_model(self, model, parameter_names=None):
        """Apply gradient shielding to a model's parameters.
        
        Args:
            model: The PyTorch model to shield
            parameter_names: Optional list of parameter names to shield.
                If None, shield all parameters.
        """
        if parameter_names is None:
            # Shield all parameters
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = self.project_gradient(param.grad.data)
        else:
            # Shield only specified parameters
            for name, param in model.named_parameters():
                if name in parameter_names and param.grad is not None:
                    param.grad.data = self.project_gradient(param.grad.data)
    
    @staticmethod
    def load_from_file(basis_path, device='cuda'):
        """Load a GradShield instance from a saved basis file.
        
        Args:
            basis_path: Path to the saved basis matrix
            device: The device to use for computation
            
        Returns:
            A GradShield instance
        """
        basis = torch.load(basis_path, map_location=device)
        return GradShield(basis, device=device)


class ShieldedOptimizer(torch.optim.Optimizer):
    """Optimizer wrapper that applies gradient shielding before updates."""
    
    def __init__(self, optimizer, shield):
        """Initialize the shielded optimizer.
        
        Args:
            optimizer: The base optimizer to wrap
            shield: The GradShield instance to use
        """
        self.optimizer = optimizer
        self.shield = shield
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
    
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
    
    def step(self, closure=None):
        """Perform a shielded optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            The loss from the closure if provided
        """
        # Apply gradient shielding to all parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data = self.shield.project_gradient(p.grad.data)
        
        # Perform the optimization step
        return self.optimizer.step(closure)
