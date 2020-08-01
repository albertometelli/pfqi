'''
    Subclass of trading main environment, observations are derivatives of prices
    in the previous N minutes.
'''

from .vec_base import VecTradingMain
from gym import spaces
import numpy as np

MAX_DERIVATIVE = 5.0

class VecTradingDerivatives(VecTradingMain):

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
        lagged_derivatives = self.derivatives[:, self.current_timestep:self.current_timestep+self.time_lag][:,::-1]
        _portfolios = np.reshape(self.current_portfolios, (self.n_selected_days, 1))
        _time = np.full((self.n_selected_days, 1), self.current_timestep / self.horizon)
        return np.concatenate([lagged_derivatives, _portfolios, _time], axis=1)

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        return self.current_portfolios * (self.current_prices - self.previous_prices) - \
                abs(self.current_portfolios - self.previous_portfolios) * self.fees

    def reset(self, **kwargs):
        super().reset(**kwargs)
        # Compute derivatives once for all, pad with time_lag zeros
        self.derivatives = np.concatenate([np.zeros((self.n_selected_days, self.time_lag)), (self.prices[:,1:] - self.prices[:,:-1])], axis=1)
        return self._observation()

# UNIT TESTING
if __name__ == '__main__':
    import pandas as pd
    sample_dataset = pd.DataFrame.from_dict({
        'open': [0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3],
        'day': [1, 1, 1, 1, 2, 2, 2, 2],
        'count': [1, 1, 1, 1, 3, 3, 3, 3],
    })
    env = VecTradingDerivatives(data=sample_dataset, n_envs=2)
    ob, done, reward, t = env.reset(day_indexes=[0]), [False], np.zeros(1), 0
    while not all(done):
        actions = [env.action_space.sample() for _ in range(2)]
        obs, rew, done, _ = env.step(actions)
        reward += rew
        t += 1
    print('Final reward:', reward)
    print('Final length:', t)
