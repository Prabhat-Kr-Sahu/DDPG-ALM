import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from src.logger import logging

class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day = 0, hist_vol= None):

        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.hist_vol=hist_vol
        self.DSR_A = 0.0
        self.DSR_B = 0.0

         # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+1 + len(self.tech_indicator_list), self.state_space))

        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]

        self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # logging.info(" state  :: " , self.day ,self.state.shape, self.state)
        # logging.info(" hist_ vol  :: " , self.day , type(self.hist_vol), self.hist_vol)

        hist_volll = self.hist_vol.values[self.day,:]
        # Concatenate along axis=0

        self.state = np.concatenate([self.state, hist_volll.reshape(1,-1) ], axis=0)



        # logging.info("states - " , self.state.shape)

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]



    def step(self, actions):
      logging.info(f" the len of the df is  {len(self.df.index.unique())}  and the current day is :  {self.day } and  if  terminal is  : { self.day >= len(self.df.index.unique()) - 1 }")
      self.terminal = self.day >= len(self.df.index.unique()) - 1

      if self.terminal:
          # logging.info("=================================")
          # logging.info("begin_total_asset:{}".format(self.asset_memory[0]))
          # logging.info("end_total_asset:{}".format(self.portfolio_value))
          # return self.state, self.reward, self.terminal, {}


          df = pd.DataFrame(self.portfolio_return_memory)
          df.columns = ['daily_return']
          # plt.plot(df.daily_return.cumsum(),'r')
          # plt.savefig('results/cumulative_reward.png')
          # plt.close()

          # plt.plot(self.portfolio_return_memory,'r')
          # plt.savefig('results/rewards.png')
          # plt.close()

          logging.info("=================================")
          logging.info("begin_total_asset:{}".format(self.asset_memory[0]))
          logging.info("end_total_asset:{}".format(self.portfolio_value))

          df_daily_return = pd.DataFrame(self.portfolio_return_memory)
          df_daily_return.columns = ['daily_return']
          if df_daily_return['daily_return'].std() !=0:
            sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                    df_daily_return['daily_return'].std()
            logging.info("Sharpe: {}".format(sharpe))
          logging.info("=================================")


          return self.state, self.reward, self.terminal,{}
      else:
          last_day_memory = self.data
          weights = self.softmax_normalization(actions)  # Ensure valid portfolio weights
          self.actions_memory.append(weights)

          # Load next state
          self.day = self.day+ 1
          self.data = self.df.loc[self.day, :]
          self.covs = self.data['cov_list'].values[0]
          self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
          hist_voll= self.hist_vol.values[self.day,:]
          self.state = np.concatenate([self.state, hist_voll.reshape(1,-1) ], axis=0)

          # Portfolio Value Update
          portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
          new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

          # Calculate Transaction Fee
          phi = 0.0025  # 0.25% transaction cost
          # Reshape portfolio_value to match dimensions of other arrays
          portfolio_value_reshaped = np.repeat(self.portfolio_value, len(weights))
          transaction_fee = phi * sum(
              abs(weights * new_portfolio_value * last_day_memory.close.values / self.data.close.values
                  - self.actions_memory[-2] * portfolio_value_reshaped)  # Use portfolio_value_reshaped
          )

          # Reward Calculation
          self.reward = (new_portfolio_value - self.portfolio_value) - transaction_fee  # r_t = u_t - u_t-1 - fee_t

          # Update portfolio value
          self.portfolio_value = new_portfolio_value

          # Save to memory
          self.portfolio_return_memory.append(portfolio_return)
          self.asset_memory.append(new_portfolio_value)
          self.date_memory.append(self.data.date.unique()[0])

          return self.state, self.reward, self.terminal, {}
    ##############################################

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0

        # returns = self.df['return_list'].values[0]
        # hist_vol = returns.rolling(window=30).std()
        # hist_vol.fillna(0, inplace=True)
        # hist_vol = hist_vol.iloc[self.day,:]


        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # logging.info(self.hist_vol)
        # self.hist_vol= self.hist_vol[self.day,]
        # Concatenate along axis=0

        hist_voll= self.hist_vol.values[self.day,:]
        self.state = np.concatenate([self.state, hist_voll.reshape(1,-1)], axis=0)
        # Concatenate along axis=0

        # logging.info(" reset -- ev  --state -", self.state.shape)
        # logging.info(" reset -- ev -- state - ", self.state)
        # logging.info(" reset -- ev-- cov - ", self.state[:30, :].shape)
        # logging.info(" reset -- ev-- his vol- ", self.state[:-1, :].shape)
        # logging.info(" reset -- ev-- his vol- ", self.state[-1:, :])
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.DSR_A = 0.0
        self.DSR_B = 0.0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def apply_dirichlet_noise(self, actions, alpha=0.1):
      """
      Apply Dirichlet noise to actions to encourage exploration.

      Args:
      - actions (np.array): Original action values from the RL model.
      - alpha (float): Dirichlet concentration parameter. Lower values = more noise.

      Returns:
      - np.array: Modified action values with noise, ensuring sum = 1.
      """
      noise = np.random.dirichlet([alpha] * len(actions))  # Sample from Dirichlet distribution
      noisy_actions = 0.75 * actions + 0.25 * noise  # Blend original actions with noise
      return noisy_actions / noisy_actions.sum()  # Normalize to ensure sum = 1

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #logging.info(len(date_list))
        #logging.info(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def calculate_DSR(self, R):
      eta = 0.004
      delta_A = R - self.DSR_A
      delta_B = R**2 - self.DSR_B
      Dt = (self.DSR_B*delta_A - 0.5*self.DSR_A*delta_B) / ((self.DSR_B-self.DSR_A**2)**(3/2) + 1e-6)
      self.DSR_A = self.DSR_A + eta*delta_A
      self.DSR_B = self.DSR_B + eta*delta_B
      return(Dt)