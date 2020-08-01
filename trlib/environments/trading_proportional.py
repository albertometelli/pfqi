'''
    Subclass of trading main environment, observations are prices derivatives
    in the previous N days.
'''

from .trading_base import TradingMain
from gym import spaces
from gym.spaces import Tuple, Discrete
import numpy as np

MAX_DERIVATIVE = 3

class Trading(TradingMain):
# fees = 7/100000
    def __init__(self, time_lag=60, fees=0.0000001, **kwargs):
        # Calling superclass init
        super().__init__(fees=fees, **kwargs )
        # Observation space and action space, appending PORTFOLIO and TIME
        # observation space  = historical prices, portfolio, time
        observation_low = np.concatenate([np.full(time_lag*self.num_inst, -MAX_DERIVATIVE), [-1.0]*self.num_inst, [0.0]])
        observation_high = np.concatenate([np.full(time_lag*self.num_inst, +MAX_DERIVATIVE), [+1.0]*self.num_inst, [1.0]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.action_space = spaces.Discrete(3**self.num_inst)

        # Internals
        self.derivatives = [None]*self.num_inst
        self.current_derivative = [0]*self.num_inst
        self.time_lag = time_lag

        # Required to normalize
        calculate = self.data[self.data_name].values
        deltas=(calculate[1:]-calculate[:-1])/calculate[:-1]
        self.mean_delta = np.mean(deltas, axis=0)
        self.std_delta = np.sqrt(np.var(deltas))

        # Required for FQI
        self.action_dim = self.num_inst
        self.state_dim = self.observation_space.shape[0]
        self.gamma = 1

    def _normalize(self, X):
        return (X-self.mean_delta)/self.std_delta

    def _denormalize(self, X):
        return X*self.std_delta+self.mean_delta

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        histprice_window = self.derivatives[self.current_timestep:self.current_timestep+self.time_lag]
        histprice_window = self._normalize(histprice_window).flatten('F')
        obs=np.concatenate([histprice_window, self.current_portfolio, [self.current_timestep / self.horizon]], axis=0)
        return obs
    def _reward(self):
        pl = self.current_portfolio * self.current_derivative - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees
        ################################## printing reward ################################
        # print("reward is: ", self.current_portfolio[0] * self.current_derivative[0])

        return np.sum(pl)

    def reset(self, **kwargs):
        super().reset()
        # Compute derivatives once for all, pad with time_lag zeros
        self.derivatives = np.concatenate(([[0.0]*self.num_inst], (self.prices[1:] - self.prices[:-1])/self.prices[:-1]), axis=0)
        self.current_derivative = self.derivatives[self.current_timestep + self.time_lag]
        return self._observation()

    def step(self, action):
        ##################### printing current price #####################################
        # print('current price is: ', self.current_derivative[0])
        # print('current action is: ', action)
        step_observation, step_reward, step_done, _ = super().step(action)
        #
        self.current_derivative = self.derivatives[self.current_timestep + self.time_lag]
        # print('current reward is', step_reward)
        return step_observation, step_reward,  step_done, {}





# UNIT TESTING
if __name__ == '__main__':
    import pandas as pd
    # sample_dataset = pd.DataFrame.from_dict({
    #     'USDEUR': [0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3],
    #     'USDJPY': [0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3],
    #     'day': [1, 1, 1, 1, 2, 2, 2, 2],
    #     'count': [1, 1, 1, 1, 3, 3, 3, 3],
    # })
    env = Trading(csv_path="sample_data.csv")
    ob, done, reward, t = env.reset(), False, 0.0, 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        reward += rew
        t += 1
    print('Final reward:', reward)
    print('Final length:', t)

