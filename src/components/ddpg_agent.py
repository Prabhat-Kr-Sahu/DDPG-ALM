import torch
import torch.nn as nn
import torch.optim as optim

from src.components.agent_utils import Actor, Critic, CostNetwork, Memory, Noise
from src.logger import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGagent:
    def __init__(self, env, params, max_memory_size=50000):
        """
        Initialize the DDPG agent with:
        - Actor-Critic Networks
        - Cost Network for risk constraints
        - Target Networks for stability
        - Lagrange multiplier for enforcing constraints
        """

        # logging.info(params)
        logging.info(" DDPG AGEnt Class- ++++++++++++++++++++++++++++++++++++++++++")

        # 1️⃣ Define State & Action Space Dimensions
        self.data = env.envs[0].df
        curr_state= env.envs[0].state
        # logging.info("states_ ddpg init ::", curr_state.shape)
        actions = env.action_space.shape[0]

        # logging.info("actions ::", actions)

        self.num_states = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.num_actions = env.action_space.shape[0]
        self.gamma = params['gamma']  # Discount factor (γ)
        self.tau = params['tau']  # Soft update factor (τ)
        self.batch_size = int(params['batch_size'])
        self.env = env
        self.eta = params['eta']

        # 2️⃣ Initialize Networks
        self.actor = Actor(self.num_states, self.num_actions, params['Ahidden_dim'],
                           params['Anum_layers'], params['Aact_fn'], params['Adr']).to(device)
        self.actor_target = Actor(self.num_states, self.num_actions, params['Ahidden_dim'],
                                  params['Anum_layers'], params['Aact_fn'], params['Adr']).to(device)

        self.critic = Critic(self.num_states, self.num_actions, params['Chidden_dim'],
                             params['Cnum_layers'], params['Cact_fn'], params['Cdr']).to(device)
        self.critic_target = Critic(self.num_states, self.num_actions, params['Chidden_dim'],
                                    params['Cnum_layers'], params['Cact_fn'], params['Cdr']).to(device)

        # 3️⃣ Initialize Cost Network for Constrained Reinforcement Learning
        self.cost_network = CostNetwork(self.num_states, self.num_actions, params['Chidden_dim']).to(device)
        self.cost_target = CostNetwork(self.num_states, self.num_actions, params['Chidden_dim']).to(device)

        # Copy weights to target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.cost_target.parameters(), self.cost_network.parameters()):
            target_param.data.copy_(param.data)

        # 4️⃣ Training Setup
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.cost_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['alr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['clr'])
        self.cost_optimizer = optim.Adam(self.cost_network.parameters(), lr=params['clr'])

        # 5️⃣ Initialize Lagrange Multiplier for Constraint Enforcement
        self.lambda_ = 0.01
        self.rho = 0.01  # Step size for updating lambda
        self.violations= 0
        self.zeta= env.envs[0].turbulence_threshold


    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        action = self.actor.forward(state_tensor).detach().cpu()

        #action = action.detach().numpy()
        return action



    def VaR(self, states, actions, confidence_level=0.95):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actions = actions.to(device)
        states = states.to(device)  # assume actions is already on the correct device

        batch_size = states.shape[0]  # ✅ Do NOT use `.to(device)` here
        num_assets = 30

        states = states.squeeze(1).to(device)  # [batch_size, 38, 30]
        states_n = states  # already squeezed

        cov_matrix = states[:, :num_assets, :].to(device)  # [batch_size, 30, 30]
        hist_volatility = states_n[:, -1, :].to(device)  # [batch_size, 30]

        z_score = torch.tensor(1.645, device=device)  # ✅ place tensor on the same device
        individual_VaR = z_score * hist_volatility  # [batch_size, 30]

        VaR_portfolio = torch.zeros(batch_size, device=device)  # ✅ directly initialize on device

        for i in range(num_assets):
            for j in range(num_assets):
                VaR_portfolio = VaR_portfolio + (
                    actions[:, i] * individual_VaR[:, i] *
                    actions[:, j] * individual_VaR[:, j] * cov_matrix[:, i, j]
                )

        return VaR_portfolio

    def compute_cost_target(self, states, actions, next_states, dones):
        """
        Compute the target cost using the Bellman equation.

        Equation (20):
        c_{w_v}(s, a) = VaR(s, a) + \eta (1 - d) c'_{w_v'}(s', a')
        """
        next_actions = self.actor_target.forward(next_states)  # π'(s')
        next_cost = self.cost_target.forward(next_states, next_actions.detach())  # c'_wv'(s', a')
        cost_target = self.VaR(next_states, next_actions) + self.eta * next_cost
        return cost_target

    def update(self):
        """
        Perform one update step for the Actor, Critic, and Cost networks.
        """

        # 1️⃣ Sample a batch from the Replay Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)


        # logging.info(" update states : ", type(states),  states.shape,  " action ", action.shape, type(action) )

        # Remove the singleton dimension at dim=1
        # states_n = states.squeeze(1)  # shape: (224, 38, 30)

        # # Now slicing makes sense
        # cov_mat = states_n[:, :30, :]                  # Shape: (224, 30, 30)
        # histrical_volatility = states_n[:, -1, :]


        # logging.info("ddpg update - states_n ::", states_n.shape)
        # logging.info("ddpg update - cov_mat_n ::", cov_mat.shape)
        # logging.info("ddpg update - histrical_volatility_n ::", histrical_volatility.shape)

        # logging.info("ddpg update - states_n ::", states_n)
        # logging.info("ddpg update - cov_mat_n ::", cov_mat)
        # logging.info("ddpg update - histrical_volatility_n ::", histrical_volatility)


        # next_states_n = next_states.squeeze(1)  # Shape: [batch_size, 38, 30]
        # next_cov_mat = next_states_n[:, :30, :]  # Shape: [batch_size, 30, 30]
        # next_hist_volatility = next_states_n[:, -1, :]
        # logging.info("ddpg update - next_states_n ::", next_states_n.shape)
        # logging.info("ddpg update - next_cov_mat ::", next_cov_mat.shape)
        # logging.info("ddpg update - histor vol :: " , next_hist_volatility.shape)
        # logging.info("ddpg update - next_states_n ::", next_states_n)
        # logging.info("ddpg update - next_cov_mat ::", next_cov_mat)
        # logging.info("ddpg update - histor vol :: " , next_hist_volatility)


        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 4️⃣ Compute Target Q-Value using Bellman Equation (Eq. 5)
        # Q(s, a) = r + γQ'(s', π'(s'))
        Q_target = rewards + self.gamma  * self.critic_target.forward(next_states, self.actor_target.forward(next_states).detach())

        # 6️⃣ Compute Critic Loss (Eq. 6)
        # L = 1/N \sum (Q(s, a) - Q_target)^2
        # logging.info(" critic loss calculation -start")
        critic_loss = self.critic_criterion(self.critic.forward(states, actions), Q_target.detach())

        # logging.info("critic loss calculation end ")
        # 8️⃣ Compute Cost Network Loss (Eq. 21)
        # L_C = 1/N \sum (c_{w_v}(s, a) - VaR(s, a) - η (1 - d) c'_{w_v'}(s', a'))^2
        # logging.info("cost loss calculation started ")
        cost_pred = self.cost_network.forward(states, actions)
        cost_target = self.compute_cost_target(states, actions, next_states, dones).detach()
        # logging.info(" cost_ target :: " , cost_target)

        cost_loss = self.cost_criterion(cost_pred, cost_target)

        # logging.info("cost loss calculation end ")
        # 🔟 Compute Actor Loss using Lagrangian method (Eq. 13)
        # L(w_π, λ) = -J_{w_π} + \sum \lambda_j C_{w_π, j} + \frac{\rho}{2} \sum (C_{w_π, j})^2

        # logging.info("actor loss calculation started ")
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        constraint_penalty =  cost_target

        # logging.info(" constraint penalty before :: ", constraint_penalty)
        # logging.info(" constraint_penalty :::: " , constraint_penalty.shape, type(constraint_penalty))

        violations_count = (constraint_penalty > self.zeta).sum().item()  # Count how many elements violate the constraint
        # logging.info(" violations ::: " , violations_count)
        # Update the number of violations
        self.violations  =  self.violations + violations_count


        constraint_penalty = torch.where(
            constraint_penalty <= self.zeta,
            torch.tensor(0.0, device=constraint_penalty.device, dtype=constraint_penalty.dtype),
            constraint_penalty - self.zeta
        )
        # logging.info(" constraint penalty after :: ", constraint_penalty)

        quadratic_penalty = (self.rho / 2) * (constraint_penalty ** 2).mean().clone()
        constraint_penalty=(self.lambda_ * constraint_penalty).mean().clone()
        actor_loss = policy_loss + constraint_penalty + quadratic_penalty

        self.actor_optimizer.zero_grad()
        # logging.info(" actor_ loss ",  actor_loss.shape)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()


        # logging.info("actor update end ")



        # 1️⃣3️⃣ Soft Update of Target Networks (Eq. 14)
        # logging.info("soft update - critic -")
        with torch.no_grad():
          for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
              target_param.data= param.data * self.tau + target_param.data * (1.0 - self.tau)

          for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
              target_param.data = param.data * self.tau + target_param.data * (1.0 - self.tau)

          for target_param, param in zip(self.cost_target.parameters(), self.cost_network.parameters()):
              target_param.data = param.data * self.tau + target_param.data * (1.0 - self.tau)




        # logging.info(" soft updates end")

    def buffer_fill(self, buffer_size):
      state = self.env.reset()

      # logging.info(" buffer fill ------ ")
      # logging.info(" buffer fil state   --- ",  state.shape)
      # logging.info("  buffer fill state  --- ", state)
      # logging.info(" buffer --- fill -- cov mat", state[:, :30, :].shape)

      # logging.info(" buffer fill -----hist vol", state[:, -1, :].shape)
      # logging.info(" buffer fill -----hist vol", state[:, -1, :])

      for _ in range(buffer_size):
        action = self.get_action(state)
        action = Noise(action, self.env.action_space)
        new_state, reward, done, _ = self.env.step(action)
        self.memory.push(state, action, reward, new_state, done)

    def trade(self, val_env, e_val_gym):
      Reward = []
      state = val_env.reset()

      for i in range(len(e_val_gym.df.index.unique())):
        action = self.get_action(state)
        next_obs, reward, done, _ = val_env.step(action.detach().numpy())
        Reward.append(reward)

        if i == (len(e_val_gym.df.index.unique()) - 2):
          account_memory = val_env.env_method(method_name="save_asset_memory")
          actions_memory = val_env.env_method(method_name="save_action_memory")

        if done[0]:
          logging.info("hit end!")
          break
        state = next_obs

      return account_memory, actions_memory, sum(Reward)

