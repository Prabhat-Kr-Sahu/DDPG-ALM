from dataclasses import dataclass
import json
import sys

from hyperopt import fmin,hp,tpe,Trials
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.ddpg_agent import DDPGagent
from src.components.stock_portfolio_env import StockPortfolioEnv
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from src.utils import calculate_sharpe
import torch
import os

@dataclass
class PredictPipelineConfig:
    daily_return_path:str=os.path.join('artifacts','df_daily_return.csv')
    actions_path:str=os.path.join('artifacts','df_actions.csv')
    INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
    TURBULENCE_THRESHOLD= 0.0020
    agent_path=os.path.join('artifacts','ddpg_agent')
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

    
class PredictPipeline:
    def __init__(self):
        self.pipeline_config=PredictPipelineConfig()
     
    def design_environment(self, trade, hist_vol_trade):
        stock_dimension = len(trade.tic.unique())
        state_space = stock_dimension
        
        env_kwargs_trade = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": self.pipeline_config.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "hist_vol":hist_vol_trade,
            "turbulence_threshold": self.pipeline_config.TURBULENCE_THRESHOLD
        }       
        self.e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs_trade)
        self.env_trade, _ = self.e_trade_gym.get_sb_env()

        logging.info("Environment design complete")
        return self.env_trade, self.e_trade_gym
    
    def load_agent(self,agent, checkpoint_path, device=None):
        """
        Load DDPG agent's components into an existing agent instance.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)

        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.cost_network.load_state_dict(checkpoint['cost_network_state_dict'])
        agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.cost_target.load_state_dict(checkpoint['cost_target_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        agent.cost_optimizer.load_state_dict(checkpoint['cost_optimizer_state_dict'])

        agent.lambda_ = checkpoint['lambda_']
        agent.rho = checkpoint['rho']
        agent.violations = checkpoint['violations']

        logging.info(f"✅ Agent successfully loaded from {checkpoint_path}")
    
    def load_params(self, path):
        with open(path, 'r') as f:
            params = json.load(f)
        return params
    
    def predict(self,trade, hist_vol_trade):
        try:
            logging.info("Before Loading")
            env_trade, e_trade_gym = self.design_environment(trade, hist_vol_trade)
            best=self.load_params('artifacts/params.json')
            agent = DDPGagent(self.env_trade, best)
            self.load_agent(agent , self.pipeline_config.agent_path)
            logging.info("Agent Loaded")
            logging.info("After Loading")
            account_memory, actions_memory, rewardd = agent.trade(env_trade, e_trade_gym)
            violations= agent.violations

            logging.info(f"violations : {violations}" ) 
            logging.info(f" reward :: {rewardd}" )
            logging.info(f"sharpe: {calculate_sharpe(account_memory[0])}")

            account_memory[0].to_csv(self.pipeline_config.daily_return_path, index=False,header=True)
            actions_memory[0].to_csv(self.pipeline_config.actions_path,index=False,header=True)

            return actions_memory[0]
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    full_train, hist_vol_full_train, train, hist_vol_train, val, hist_vol_val, trade, hist_vol_trade=data_transformation.initiate_data_transformation(data_path)

    obj=PredictPipeline()
    obj.predict(trade, hist_vol_trade)