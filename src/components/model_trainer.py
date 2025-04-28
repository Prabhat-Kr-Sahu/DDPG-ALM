from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch
from hyperopt import hp, fmin, tpe, Trials

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.ddpg_agent import DDPGagent, Noise
from src.components.stock_portfolio_env import StockPortfolioEnv
from src.utils import calculate_sharpe

@dataclass
class ModelTrainerConfig:
    INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
    TURBULENCE_THRESHOLD= 0.0020
    agent_file_path=os.path.join('artifacts','ddpg_agent')
    space = {
        'Ahidden_dim': hp.quniform('Ahidden_dim', 2, 512, 1),
        'Anum_layers': hp.quniform('Anum_layers', 1, 8, 1),
        'Chidden_dim': hp.quniform('Chidden_dim', 2, 512, 1),
        'Cnum_layers': hp.quniform('Cnum_layers', 1, 8, 1),

        'alr': hp.loguniform('alr', -8, -1),  # Actor learning rate
        'clr': hp.loguniform('clr', -8, -1),  # Critic learning rate
        'gamma': hp.uniform('gamma', 0.9, 0.99),  # Discount factor
        'tau': hp.uniform('tau', 0.08, 0.2),  # Soft target update rate
        'batch_size': hp.quniform('batch_size', 32, 256, 32),  # Mini-batch size

        'Aact_fn': hp.choice('Aact_fn', ['relu', 'tanh', 'sigmoid']),  # Actor activation
        'Adr': hp.uniform('Adr', 0, 0.5),  # Actor dropout
        'Cact_fn': hp.choice('Cact_fn', ['relu', 'tanh', 'sigmoid']),  # Critic activation
        'Cdr': hp.uniform('Cdr', 0, 0.5),  # Critic dropout


        'eta' : hp.uniform('eta', 0.01, 1),  # Exploration noise level
        # 🚀 **Newly Added Missing Hyperparameters**:
        # 'rho': hp.uniform('rho', 0.001, 0.1),  # Lagrange multiplier update step size
        # 'lambda_init': hp.uniform('lambda_init', 0.01, 1.0),  # Initial value of λ
        'buffer_size': hp.quniform('buffer_size', 10000, 1000000, 10000),  # Replay buffer size
        'noise_std': hp.uniform('noise_std', 0.01, 0.3),  # Exploration noise level
        'grad_clip': hp.uniform('grad_clip', 0.1, 10.0),  # Gradient clipping threshold
        'warmup_steps': hp.quniform('warmup_steps', 1000, 50000, 1000),  # Steps before training starts
        'reward_scaling': hp.uniform('reward_scaling', 0.1, 10.0)  # Reward scaling factor
    }

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig=ModelTrainerConfig()

    def design_environment(self,train, hist_vol_train, val, hist_vol_val, full_train, hist_vol_full_train):
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension
        env_kwargs_train = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": self.modelTrainerConfig.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "hist_vol":hist_vol_train,
            'turbulence_threshold': self.modelTrainerConfig.TURBULENCE_THRESHOLD

        }
        # print(hist_vol_val,"  ddddd ")
        env_kwargs_val = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": self.modelTrainerConfig.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "hist_vol":hist_vol_val,
            "turbulence_threshold": self.modelTrainerConfig.TURBULENCE_THRESHOLD
        }

        env_kwargs_full = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": self.modelTrainerConfig.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "hist_vol":hist_vol_full_train,
            "turbulence_threshold": self.modelTrainerConfig.TURBULENCE_THRESHOLD
        }
        
        self.e_train_gym = StockPortfolioEnv(df = train, **env_kwargs_train)
        self.env_train, _ = self.e_train_gym.get_sb_env()

        self.e_val_gym = StockPortfolioEnv(df = val, **env_kwargs_val)
        self.env_val, _ = self.e_val_gym.get_sb_env()

        self.e_train_full_gym = StockPortfolioEnv(df = full_train, **env_kwargs_full)
        self.env_full_train, _ = self.e_train_full_gym.get_sb_env()

        logging.info("design environment done")
    

    def objective(self,params):
        print(params)
        # Convert hyperparameters to integers where necessary
        params['Ahidden_dim'] = int(params['Ahidden_dim'])
        params['Anum_layers'] = int(params['Anum_layers'])
        params['Chidden_dim'] = int(params['Chidden_dim'])
        params['Cnum_layers'] = int(params['Cnum_layers'])
        params['batch_size'] = int(params['batch_size'])
        params['buffer_size'] = int(params['buffer_size'])
        params['warmup_steps'] = int(params['warmup_steps'])

        model = DDPGagent(self.env_train, params)
        model.buffer_fill(500)
        model.update()

        account_memory, actions_memory, rewardd = model.trade(self.env_val, self.e_val_gym)
        print( f" the reward is :::::::    {rewardd}  " )

        sharpe = calculate_sharpe(account_memory[0])
        return -sharpe
        # return -reward[0]

    def find_best_hyperparameters(self):
        best = fmin(fn=self.objective, space=self.modelTrainerConfig.space, algo=tpe.suggest, max_evals= 10 , trials=Trials())
        best['Aact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Aact_fn']]
        best['Cact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Cact_fn']]
        return best
    
    def save_params(self, params):
        with open("artifacts/params.json", 'w') as f:
            json.dump(params, f)
            
    def initiate_model_trainer(self,train, hist_vol_train, val, hist_vol_val, full_train, hist_vol_full_train):
        logging.info("Model training initiated")
        logging.info("Finding the best hyperparameters")
        self.design_environment(train, hist_vol_train, val, hist_vol_val, full_train, hist_vol_full_train)
        best = self.find_best_hyperparameters()
        logging.info(f"Best hyperparameters: {best}")
        agent = DDPGagent(self.env_full_train, best)

        batch_size = agent.batch_size

        rewards = []
        avg_rewards = []
        num_episodes = 1 #1000

        torch.autograd.set_detect_anomaly(True)
        for episode in range(num_episodes):

            state = self.env_full_train.reset()
            episode_reward = 0
            done = False
            # print(state.shape, type(state))
            # state = state.reshape(1, 1, 39, 30)
            # print((torch.tensor( np.expand_dims(state, axis=1))).dim)

            print(f"Episode: {episode+1}")
            while not done:

                # print(i)
                # print(f"done  : {done} ")
                action = agent.get_action(state)
                action = Noise(action, self.env_full_train.action_space)
                new_state, reward, done ,info = self.env_full_train.step(action)
                # done= terminated or truncated
                agent.memory.push(state, action, reward, new_state, done)

                if len(agent.memory) > batch_size:
                    agent.update()

                state = new_state
                episode_reward  = episode_reward + reward

                if done:
                #  sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                    break

            # agent.lambda_ = agent.lambda_ + agent.rho * agent.cost_network.forward(torch.tensor( np.expand_dims(state, axis=1)), agent.get_action(state)).mean()
            # agent.lambda_ = agent.lambda_ + agent.rho * agent.cost_network.forward(
                        #     torch.tensor(np.expand_dims(state, axis=1), dtype=torch.float32),
                        #     agent.get_action(state)
                        # ).mean().detach()

            device = next(agent.cost_network.parameters()).device  # Get the device of the cost network

            state_tensor = torch.tensor(np.expand_dims(state, axis=1), dtype=torch.float32, device=device)
            action_tensor = agent.get_action(state).to(device)  # Ensure action is also on same device

            agent.lambda_ = agent.lambda_ + agent.rho * agent.cost_network.forward(
                state_tensor,
                action_tensor
            ).mean().detach().to(device)

            agent.rho= agent.rho * 1.008

            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))
            print(f"Episode: {episode+1}, Total Reward: {episode_reward}")
            print(" violations : " ,  agent.violations)

            # save_object(self.modelTrainerConfig.agent_file_path, agent)
            agent.save(self.modelTrainerConfig.agent_file_path)
            self.save_params(best)

