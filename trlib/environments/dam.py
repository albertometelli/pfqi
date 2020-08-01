import gym
import numpy as np
from gym import spaces
import random

"""
Cyclostationary Dam Control
Info
----
  - State space: 2D Box (storage,day)
  - Action space: 1D Box (release decision)
  - Parameters: capacity, demand, flooding threshold, inflow mean per day, inflow std, demand weight, flooding weigt=ht
References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)

  - A. Castelletti, S. Galelli, M. Restelli, R. Soncini-Sessa
    Tree-based reinforcement learning for optimal water reservoir operation
    Water Resources Research 46.9 (2010)

  - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
    Importance Weighted Transfer of Samples in Reinforcement Learning.
    International Conference on Machine Learning. 2018.
"""


class Dam(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, inflow_profile=1, alpha=0.5, beta=0.5, penalty_on=False):

        self.horizon = 360
        self.gamma = 0.999
        self.state_dim = 2
        self.action_dim = 1

        self.DEMAND = 10.0  # Water demand -> At least DEMAND/day must be supplied or a cost is incurred
        self.FLOODING = 300.0  # Flooding threshold -> No more than FLOODING can be stored or a cost is incurred
        self.MIN_STORAGE = 50.0  # Minimum storage capacity -> At most max{S - MIN_STORAGE, 0} must be released
        self.MAX_STORAGE = 500.0  # Maximum storage capacity -> At least max{S - MAX_STORAGE, 0} must be released

        self.INFLOW_MEAN = self._get_inflow_profile(
            inflow_profile)  # Random inflow (e.g. rain) mean for each day (360-dimensional vector)
        self.INFLOW_STD = 2.0  # Random inflow std

        assert alpha + beta == 1.0  # Check correctness
        self.ALPHA = alpha  # Weight for the flooding cost
        self.BETA = beta  # Weight for the demand cost

        self.penalty_on = penalty_on  # Whether to penalize illegal actions or not

        # Gym attributes
        self.viewer = None

        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([np.inf]), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0, 1]),
                                            high=np.array([np.inf, 360]))

        # Initialization
        self.seed()
        self.starting_day = None
        self.generative_setting = False
        self.states_levels = 12  # number of level of storage used in generative setting
        self._redistribution_gap = 330
        self.starting_factor = 2.5
        self.reset()

    '''
    Used in generative setting. The larger it is, the more the storage levels will be concentrated near 
    MIN_STORAGE level. Is must be a value between 0 and MAX_STORAGE - MIN_STORAGE
    '''
    @property
    def redistribution_gap(self):
        return self._redistribution_gap

    @redistribution_gap.setter
    def redistribution_gap(self, value):
        if value > self.MAX_STORAGE - self.MIN_STORAGE:
            value = self.MAX_STORAGE - self.MIN_STORAGE
        if value < 0:
            value = 0
        self._redistribution_gap = value

    def _get_inflow_profile(self, n):

        assert n >= 1 and n <= 7

        if n == 1:
            return self._get_inflow_1()
        elif n == 2:
            return self._get_inflow_2()
        elif n == 3:
            return self._get_inflow_3()
        elif n == 4:
            return self._get_inflow_4()
        elif n == 5:
            return self._get_inflow_5()
        elif n == 6:
            return self._get_inflow_6()
        elif n == 7:
            return self._get_inflow_7()

    def _get_inflow_1(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
        return y * 8 + 4

    def _get_inflow_2(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) / 2 + 0.25
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi) * 3 + 0.25
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi) / 4 + 0.25
        return y * 8 + 4

    def _get_inflow_3(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) * 3 + 0.25
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 4 + 0.25
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359) / 2 + 0.25
        return y * 8 + 4

    def _get_inflow_4(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2.5 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
        return y * 7 + 4

    def _get_inflow_5(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
        return y * 8 + 5

    def _get_inflow_6(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
        return y * 8 + 4

    def _get_inflow_7(self):

        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 3 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359) * 2 + 0.5
        return y * 8 + 5

    def step(self, action):

        action = float(action)
        state = self.get_state()
        storage = state[0]
        day = state[1]
        inflow = self._get_inflow(day)
        nextstorage, reward = self._get_nextstorage_and_reward(storage, action, inflow)
        nextday = day + 1 if day < 360 else 1
        self.state = [nextstorage, nextday]

        return self.get_state(), reward, False, {}

    def reset(self, state=None):

        if state is None:
            if self.generative_setting:
                starting_storage_levels = self._select_starting_storages(self._storages_generative_setting(),
                                                                         self.starting_factor)
                index = random.randrange(0, len(starting_storage_levels))
                self.state = [starting_storage_levels[index], self._generative_starting_day()]
            else:
                if self.starting_day is None:
                    # init_days = np.array([1, 120, 240])
                    self.state = [np.random.uniform(self.MIN_STORAGE, self.MAX_STORAGE),
                                  np.random.randint(low=1, high=360)]
                else:
                    self.state = [np.random.uniform(self.MIN_STORAGE, self.MAX_STORAGE),
                                  self.starting_day]
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def get_transition_mean(self, s, a):

        current_state = self.get_state()
        self.reset(s)
        ns, _, _, _ = self.step(a)
        self.reset(current_state)
        return ns

    def get_reward_mean(self, s, a):

        current_state = self.get_state()
        self.reset(s)
        _, r, _, _ = self.step(a)
        self.reset(current_state)
        return r

    def _get_inflow(self, day):
        return self.INFLOW_MEAN[int(day - 1)] + np.random.randn() * self.INFLOW_STD

    def _get_nextstorage_and_reward(self, storage, action, inflow):

        # Bound the action
        actionLB = np.maximum(storage - self.MAX_STORAGE, 0.0)
        actionUB = np.maximum(storage - self.MIN_STORAGE, 0.0)

        # Penalty proportional to the violation
        bounded_action = np.minimum(np.maximum(action, actionLB), actionUB)
        penalty = -abs(bounded_action - action) * self.penalty_on

        # Transition dynamics
        action = bounded_action

        nextstorage = np.maximum(storage + inflow - action, 0.0)
        # Cost due to the excess level wrt the flooding threshold
        reward_flooding = -np.maximum(storage - self.FLOODING, 0.0) / 4

        # Deficit in the water supply wrt the water demand
        reward_demand = -np.maximum(self.DEMAND - action, 0.0) ** 2

        # The final reward is a weighted average of the two costs
        reward = self.ALPHA * reward_flooding + self.BETA * reward_demand + penalty

        return nextstorage, reward

    def get_generative_episode(self, actions):

        sa = self._get_generative_sa(actions)
        rs_prime = self._calculate_generative_r_s_prime(sa, self.states_levels, len(actions))
        t = (np.array(range(len(sa)))[np.newaxis]).T
        absorbing = (np.zeros(len(t))[np.newaxis]).T
        return np.column_stack((t, sa, rs_prime, absorbing))

    def _get_inflow_history(self):

        days = self._generative_days()
        return np.array([self._get_inflow(i) for i in days])

    def _get_generative_sa(self, actions):

        days = self._generative_days()

        storage_levels = self._storages_generative_setting()
        storage_levels = np.tile(storage_levels, len(days) * len(actions))

        days = np.repeat(days, self.states_levels)
        days = np.tile(days, len(actions))

        actions = np.repeat(actions, self.states_levels * self.horizon)

        return np.column_stack((storage_levels, days, actions))

    def _calculate_generative_r_s_prime(self, sa, states_levels, n_actions):

        inflow_history = self._get_inflow_history()
        inflow_history = np.repeat(inflow_history, states_levels)
        inflow_history = np.tile(inflow_history, n_actions)
        nextstorage, reward = self._get_nextstorage_and_reward(sa[:, 0], sa[:, 2], inflow_history[:])
        nextdays = sa[:, 1] + 1
        nextdays[nextdays > 360] -= 360

        return np.column_stack((reward, nextstorage, nextdays))

    def select_starting_sa(self, sa):
        starting_storage_generative_setting = self._select_starting_storages(self._storages_generative_setting(),
                                                                             self.starting_factor)
        storage = sa[:, 0]
        days = sa[:, 1]
        return sa[np.bitwise_and.reduce((starting_storage_generative_setting[0] <= storage,
                                         storage <= starting_storage_generative_setting[-1],
                                         days == self._generative_starting_day()), 0)]

    def _storages_generative_setting(self):
        redistribution_levels = (self.states_levels - 2) * (self.states_levels - 1) / 2
        redistribution_mini_gap = int(self.redistribution_gap / redistribution_levels)
        storage_gap = int((self.MAX_STORAGE - self.MIN_STORAGE - self.redistribution_gap) / self.states_levels)
        storage_levels = [self.MIN_STORAGE]
        for i in range(self.states_levels - 1):
            storage_levels.append(storage_levels[-1] + storage_gap + redistribution_mini_gap * i)
        return storage_levels

    def _select_starting_storages(self, storage_levels, starting_factor):
        assert 2 < starting_factor <= self.states_levels
        starting_level = int(self.states_levels / starting_factor)
        stopping_level = self.states_levels - starting_level
        return [storage_levels[i] for i in range(starting_level, stopping_level)]

    def _generative_starting_day(self):
        return 1 if self.starting_day is None else self.starting_day

    def _generative_days(self):
        return (np.array(range(self.horizon)) + self._generative_starting_day() - 1) % 360 + 1

