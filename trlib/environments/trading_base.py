import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os
import copy
import itertools

class TradingMain(gym.Env):
    """
        Abstract class which implements the trading actions. Must be extended for
        different types of observations and rewards.
    """

    metadata = {'render.modes'}
    def __init__(self, csv_path=None,
                 data_name1= 'USDEUR',data_name2 = 'None',
                 fees = 2/100000, horizon=1230, time_lag=60, window='fix'):
        from pathlib import Path
        trading_path = os.path.join(str(Path(__file__).parent), "trading_data")

        # Check data (prices CSV)
        assert csv_path is not None, "Need price data to create environment."
        csv_path = os.path.join(trading_path, csv_path)
        if data_name2 == "None":
            self.data_name = [data_name1]
        else:
            self.data_name=[data_name1, data_name2]
        self.num_inst = len(self.data_name)
        self.time_lag = time_lag
        self.data = pd.read_csv(csv_path, sep=None)
        self.horizon = horizon  # self.data.shape[0] // 365 #self.n_days circa 28
        self.window = window
        # Initialize parameters
        self.fees = fees
        # Internal variables
        self.done = True
        self.prices = [None]*self.num_inst
        self.current_timestep = 0
        # Prices
        # self.current_price = [None]*self.num_inst
        self.previous_price = [None]*self.num_inst
        # Portfolio
        self.current_portfolio = [None]*self.num_inst
        self.previous_portfolio = [None]*self.num_inst

    def seed(self, seed = None):
        np.random.seed(seed)
        random.seed(seed)

    def _observation(self):
        raise Exception('Not implemented in abstract class.')

    def _reward(self):
        raise Exception('Not implemented in abstract class.')

    def step(self, action):
        """
            Act on the environment. Target can be either a float from -1 to +1
            of an integer in {-1, 0, 1}. Float represent partial investments.
        """
        # Check if the environment has terminated.
        if self.done:
            return self._observation(), 0.0, True, {}

        # Transpose action if action space is discrete [0, 2] => [-1, +1]
        if isinstance(self.action_space, spaces.Discrete):
            action_map = {i: t for i, t
                          in enumerate([np.array(t) for t in itertools.product([-1, 0, 1], 
                                                                           repeat=self.num_inst)])}
            action = action_map[action]
        ########################### fixed action ###############################à
        # action = np.array([1])
        # Check the action is in the range
        for x in action:
            assert -1 <= x <= +1, "Action not in range!"

        # Check if day has ended
        self.done = self.current_timestep >= (len(self.prices) - self.time_lag - 2)

        # Perform action
        self.previous_portfolio, self.current_portfolio = self.current_portfolio, action
        # ########################## printing portfolio #####################################
        # print("previous", "current", self.previous_portfolio, self.current_portfolio)
        # Compute the reward and update the profit
        reward = self._reward()
        self.profit += reward
        self.current_timestep += 1

        return self._observation(), reward, self.done, {}

    def reset(self):
        # print("################RESET#################")

        # Extract day from data and set price
        assert divmod(self.data.shape[0], self.horizon)[1] == 0
        num_paths = divmod(self.data.shape[0], self.horizon)[0]

        if self.window == 'fix':
            selected_day = np.random.randint(0, num_paths - 2) * self.horizon
        else:
            selected_day = np.random.choice(self.data.shape[0] - self.horizon -1)
        # selected_day = 0

        self.selected_data = copy.deepcopy(self.data[int(selected_day):int(selected_day)+self.horizon])
        self.prices = self.selected_data[self.data_name].values

        # Init internals
        self.current_timestep = 0
        self.previous_price = [None]*self.num_inst
        self.current_portfolio = [0]*self.num_inst
        self.previous_portfolio = [None]*self.num_inst
        self.done = False
        self.profit = 0
        return None
