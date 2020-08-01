'''
    Subclass of trading main environment, observations are derivatives of prices
    in the previous N minutes.
'''

from gym import spaces
import numpy as np
from .ags_base import TradingMain

MAX_DERIVATIVE = 5.0

class TradingDerivatives(TradingMain):

    def __init__(self, time_lag=60, **kwargs):
        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        observation_low = np.concatenate([np.full((time_lag), -MAX_DERIVATIVE), [-1.0, 0.0]])
        observation_high = np.concatenate([np.full((time_lag), +MAX_DERIVATIVE), [+1.0, +1.0]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.action_space = spaces.Discrete(3)
        # Internals
        self.derivatives = None
        self.time_lag = time_lag
        # Required for FQI
        self.action_dim = 1
        self.state_dim = self.observation_space.shape[0]
        self.gamma = 1

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        lagged_derivatives = self.derivatives[self.current_timestep:self.current_timestep+self.time_lag][::-1]
        return np.concatenate([lagged_derivatives, [self.current_portfolio, self.current_timestep / self.horizon]])

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        return self.current_portfolio * (self.current_price - self.previous_price) - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees

    def reset(self, **kwargs):
        super().reset(**kwargs)
        # Compute derivatives once for all, pad with time_lag zeros
        self.derivatives = np.concatenate([np.zeros(self.time_lag), (self.prices[1:] - self.prices[:-1])])
        return self._observation()

# UNIT TESTING
if __name__ == '__main__':
    import pandas as pd
    sample_dataset = pd.DataFrame.from_dict({
        'open': [0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3],
        'day': [1, 1, 1, 1, 2, 2, 2, 2],
        'count': [1, 1, 1, 1, 3, 3, 3, 3],
    })
    env = TradingDerivatives(data=sample_dataset)
    ob, done, reward, t = env.reset(), False, 0.0, 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        reward += rew
        t += 1
    print('Final reward:', reward)
    print('Final length:', t)
