import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from src.logger import logging
from src.exception import CustomException
import sys

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)

        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Actor, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()
        # logging.info("Params Dictionary:", self.params)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        action_dim = int(action_dim)
        state_dim = int(state_dim)

        # logging.info("state_dim:", state_dim)
        # logging.info("action_dim:", action_dim)
        # logging.info("hidden_dim:", hidden_dim)
        # logging.info("num_layers:", num_layers)
        # logging.info("act_fn:", act_fn)
        # logging.info("dr:", dr)
        # logging.info(f"state_dim: {state_dim}, type: {type(state_dim)}")
        # logging.info(f"action_dim: {action_dim}, type: {type(action_dim)}")
        # logging.info(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")

        # Add input layer

        layers.append(nn.Flatten())
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        # layers.append(nn.Dropout(p=dr))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state):

        x = self.model(state)
        x = torch.tanh(x)
        # logging.info(" actor  Network forward (((((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))))")
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Critic, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        action_dim = int(action_dim)
        state_dim = int(state_dim)

        # logging.info("state_dim:", state_dim)
        # logging.info("action_dim:", action_dim)
        # logging.info("hidden_dim:", hidden_dim)
        # logging.info("num_layers:", num_layers)
        # logging.info("act_fn:", act_fn)
        # logging.info("dr:", dr)
        # logging.info(f"state_dim: {state_dim}, type: {type(state_dim)}")
        # logging.info(f"action_dim: {action_dim}, type: {type(action_dim)}")
        # logging.info(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")


        # Add input layer
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        # layers.append(nn.Dropout(p=dr))
        layers.append(nn.Linear(hidden_dim, 1))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        Forward pass of the Critic network.

        Args:
        - state (torch.Tensor): State tensor.
        - action (torch.Tensor): Action tensor.

        Returns:
        - Q-value estimation.
        """

        # 🔍 logging.info debug info
        # logging.info("Critic Network forward (((((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))))")
        # logging.info(f"State shape before reshape: {state.shape}, Action shape before reshape: {action.shape}")

        # 🔄 Flatten state if it has more than 2 dimensions (CNN case)
        if state.dim() > 2:
            state = state.view(state.shape[0], -1)  # Convert to (batch_size, features)

        # 🔄 Ensure action is 2D
        if action.dim() > 2:
            action = action.view(action.shape[0], -1)  # Convert to (batch_size, action_dim)

        # 🔍 logging.info final shapes
        # logging.info(f"State shape after reshape: {state.shape}, Action shape after reshape: {action.shape}")

        # ✅ Now both state and action are 2D → Safe to concatenate
        x = torch.cat([state, action], dim=1)

        # Forward pass through Critic layers
        x = self.model(x)

        return x

class CostNetwork(nn.Module):
    """
    Neural network for estimating portfolio risk (cost).
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CostNetwork, self).__init__()

        state_dim=int(state_dim)
        action_dim=int(action_dim)
        hidden_dim=int(hidden_dim)
        # logging.info("state_dim:", state_dim)
        # logging.info("action_dim:", action_dim)
        # logging.info("hidden_dim:", hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs cost estimate
        )

    def forward(self, state, action):
        """
        Forward pass for the cost network.

        Computes:
        c_wv(s, a) = E[VaR(s, a)]  (Eq. 19 in the paper)

        Args:
        - state (torch.Tensor): State tensor with shape [batch_size, *]
        - action (torch.Tensor): Action tensor with shape [batch_size, action_dim]

        Returns:
        - Cost estimation (torch.Tensor)
        """

        # logging.info(" cost network forward ((((((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))))))")
        # 🔍 logging.info debug info to check tensor shapes
        # logging.info("state :: ", type(state) , state.shape)
        # logging.info("action :: ", type(action) , action.shape)
        # 🔄 Flatten state if it has more than 2 dimensions
        if state.dim() > 2:
            state = state.view(state.shape[0], -1)  # Reshape to [batch_size, flattened_features]

        # 🔄 Ensure action is 2D
        if action.dim() > 2:
            action = action.view(action.shape[0], -1)  # Reshape to [batch_size, action_dim]

        # 🔍 logging.info final shapes
        # logging.info(f"State shape after reshape: {state.shape}, Action shape after reshape: {action.shape}")

        # ✅ Now both state and action are 2D → Safe to concatenate
        x = torch.cat([state, action], dim=1)
        # Forward pass through the Cost network
        return self.model(x)

def Noise(action, action_space, kappa=10):
    """
    Apply Dirichlet noise for exploration in DDPG according to the paper.

    Args:
    - action (torch.Tensor): Original action values from the actor network.
    - action_space (gym.spaces.Box): Action space defining valid ranges.
    - kappa (float): Controls exploration variance. Higher kappa = less noise.

    Returns:
    - np.array: Modified action values with Dirichlet noise, ensuring sum = 1.
    """

    try:
        # Ensure actions are non-negative before applying Dirichlet noise
        action = torch.clamp(action, min=0.0)

        # Convert actions to numpy array for Dirichlet sampling
        action_np = action.detach().cpu().numpy()

        # Compute shape parameter: υ = κ * a
        upsilon = kappa * action_np

        # Ensure upsilon is positive and correctly shaped
        upsilon = np.maximum(upsilon, 1e-6)  # Prevent zero or negative values
        upsilon = upsilon.flatten()  # Ensure it's a 1D array

        # Debugging: Check upsilon values
        if np.any(upsilon <= 0):
            raise ValueError(f"Dirichlet parameters must be positive. Found: {upsilon}")

        # Sample ϵ from Dirichlet distribution
        epsilon = np.random.dirichlet(upsilon)

        # Compute final action: a' = a + sg(ϵ - a)
        noisy_action = action_np + (epsilon - action_np)

        # Apply StopGradient (detach the noise term)
        noisy_action = action_np + torch.tensor(noisy_action - action_np, requires_grad=False).numpy()

        # Clip extreme values to prevent instability
        noisy_action = np.clip(noisy_action, 0.0, 1.0)

        # Ensure sum = 1 for valid portfolio allocation
        noisy_action = noisy_action / noisy_action.sum()

        return noisy_action

    except Exception as e:
    # Return the original action if an error occurs
        return action.detach().cpu().numpy()





