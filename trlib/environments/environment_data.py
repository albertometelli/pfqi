import gym
import numpy as np
from trlib.utilities.create_grid import create_grid
from trlib.algorithms.callbacks import get_callback_list_entry
from gym.utils import seeding


class EnvironmentData(object):
    def __init__(self, max_episode_steps, name='', gamma=0.99, iterations_divisor=1, horizon_factor=1):
        self._max_episode_steps = max_episode_steps * horizon_factor
        self._name = name
        self._gamma = gamma ** (1 / horizon_factor)
        self._iterations_divisor = iterations_divisor
        self._max_iterations = int(self._max_episode_steps / self._iterations_divisor)

    def get_env_data(self, max_episode_steps, name='', repetitions=1):
        raise NotImplementedError


class CartPoleData(EnvironmentData):

    def __init__(self, max_episode_steps=200, discretization=1, name='', tau=0.02, gamma=0.99, iterations_divisor=4,
                 horizon_factor=1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._tau = tau / discretization
        self._max_episode_steps = self._max_episode_steps * discretization
        self._gamma = self._gamma ** (1 / discretization)
        self._max_iterations = int(self._max_episode_steps / self._iterations_divisor)
        self._discretization = discretization

    def get_env_data(self):
        max_q = (1 - self._gamma ** self._max_episode_steps) / (1 - self._gamma)
        print("max q: ", max_q)
        mdp = gym.make('CartPole-v1')
        mdp.env.tau = self._tau
        actions = [0, 1]
        file_name = 'CartPole_{}_D{}'.format(self._name, self._discretization)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class MountainCarData(EnvironmentData):

    def __init__(self, max_episode_steps=200, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        min_q = -(1 - self._gamma ** self._max_episode_steps) / (1 - self._gamma)
        print("min q: ", min_q)
        mdp = gym.make('MountainCar-v0')
        actions = [0, 1, 2]
        file_name = 'MountainCar_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class CarOnHillData(EnvironmentData):

    def __init__(self, max_episode_steps=200, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1, dt=.1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._dt = dt
        dt_base = .1
        factor = int(dt_base / dt)
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('CarOnHill-v0')
        actions = [0, 4, 8]
        file_name = 'CarOnHill_{}'.format(self._name)
        mdp._dt = self._dt

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class SwimmerData(EnvironmentData):

    def __init__(self, frame_skip=4, max_episode_steps=200, horizon_factor=1, name='', gamma=0.99, action_per_dim=2,
                 iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._action_per_dim = action_per_dim

        frame_skip_base = 4
        factor = int(frame_skip_base / frame_skip)
        self._frame_skip = frame_skip
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Swimmer-v3')
        mdp.env.frame_skip = self._frame_skip
        high_vals = mdp.env.action_space.high
        low_vals = mdp.env.action_space.low
        actions = create_grid(high_vals, low_vals, self._action_per_dim)
        file_name = 'Swimmer_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class DamData(EnvironmentData):

    def __init__(self, max_episode_steps=360, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1,
                 penalty_on=False, starting_day=None, states_levels=12, redistribution_gap=0, starting_factor=2.5):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._penalty_on = penalty_on
        self._starting_day = starting_day
        self._states_levels = states_levels
        self._redistribution_gap = redistribution_gap
        self._starting_factor = starting_factor

    def get_env_data(self):
        mdp = gym.make('Dam-v0', alpha=0.3, beta=0.7, penalty_on=self._penalty_on)
        actions = [0, 3, 5, 7, 10, 15, 20, 30]
        file_name = 'Dam_{}_sl{}_rg_{}'.format(self._name, self._states_levels, self._redistribution_gap)
        mdp.starting_day = self._starting_day
        mdp.states_levels = self._states_levels  # number of level of storage used in generative setting
        mdp.redistribution_gap = self._redistribution_gap
        mdp.starting_factor = self._starting_factor
        mdp.horizon = self._max_episode_steps
        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name

    def get_cb_eval(self):
        initial_states = self._initial_state_eval
        return get_callback_list_entry("eval_greedy_policy_callback", field_name="perf_disc_greedy",
                                       criterion='discounted', initial_states=initial_states)

    def get_initial_states(self):
        return np.reshape(self._initial_state_eval, (10, 2))

    @property
    def _initial_state_eval(self):
        return [np.array([100 + i*20, 1]) for i in range(10)]


class PendulumData(EnvironmentData):

    def __init__(self, max_episode_steps=360, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1, n_actions=3,
                 dt=.05):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        assert n_actions >= 2
        self._n_actions = n_actions
        self._dt = dt
        dt_base = .05
        factor = int(dt_base / dt)
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Pendulum-v1')
        actions = self._make_actions()
        file_name = 'Pendulum_{}'.format(self._name)
        mdp.dt = self._dt

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name

    def _make_actions(self):
        action_distace = 4 / (self._n_actions - 1)
        actions = list(np.arange(-2, 2 + action_distace, action_distace))
        return [[i] for i in actions]

    def get_cb_eval(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        high = np.array([np.pi/8, .1])
        initial_states = []
        for i in range(10):
            theta, thetadot = self.np_random.uniform(low=-high, high=high)
            theta += np.pi
            initial_state = np.array([np.cos(theta), np.sin(theta), thetadot])
            initial_states.append(initial_state)
        return get_callback_list_entry("eval_greedy_policy_callback", field_name="perf_disc_greedy",
                                       criterion='discounted', initial_states=initial_states)


class AcrobotData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1, dt=.2):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._dt = dt
        dt_base = .2
        factor = int(dt_base / dt)
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Acrobot-v1')
        actions = [0, 1, 2]
        file_name = 'Acrobot_{}'.format(self._name)
        mdp.dt = self._dt
        min_q = -(1 - self._gamma ** self._max_episode_steps) / (1 - self._gamma)
        print("min q: ", min_q)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class AcrobotMultiTaskData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1, dt=.2):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._dt = dt
        dt_base = .2
        factor = int(dt_base / dt)
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('AcrobotMultiTask-v0')
        actions = [0, 1]
        file_name = 'AcrobotMultiTask{}'.format(self._name)
        mdp.dt = self._dt

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class LunarLanderData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('LunarLander-v2')
        actions = [0, 1, 2, 3]
        file_name = 'LunarLander_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class CarRacingData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('CarRacing-v0')
        high_vals = mdp.env.action_space.high
        low_vals = mdp.env.action_space.low
        actions = create_grid(high_vals, low_vals, 2)
        file_name = 'CarRacing{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class BipedalWalkerData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=1):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('BipedalWalker-v2')
        high_vals = mdp.env.action_space.high
        low_vals = mdp.env.action_space.low
        actions = create_grid(high_vals, low_vals, 2)
        file_name = 'BipedalWalker{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class ReacherData(EnvironmentData):

    def __init__(self, frame_skip=4, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, action_per_dim=2,
                 iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._action_per_dim = action_per_dim
        frame_skip_base = 4
        factor = int(frame_skip_base / frame_skip)
        self._frame_skip = frame_skip
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Reacher-v2')
        mdp.env.frame_skip = self._frame_skip
        high_vals = mdp.env.action_space.high
        low_vals = mdp.env.action_space.low
        actions = create_grid(high_vals, low_vals, self._action_per_dim)
        file_name = 'Reacher' \
                    '_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class HopperData(EnvironmentData):

    def __init__(self, frame_skip=4, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, action_per_dim=2,
                 iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._action_per_dim = action_per_dim
        frame_skip_base = 4
        factor = int(frame_skip_base / frame_skip)
        self._frame_skip = frame_skip
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Hopper-v2')
        mdp.env.frame_skip = self._frame_skip
        high_vals = mdp.env.action_space.high
        low_vals = mdp.env.action_space.low
        actions = create_grid(high_vals, low_vals, self._action_per_dim)
        file_name = 'Hopper' \
                    '_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class AntData(EnvironmentData):

    def __init__(self, frame_skip=4, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, action_per_dim=2,
                 iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)
        self._action_per_dim = action_per_dim
        frame_skip_base = 4
        factor = int(frame_skip_base / frame_skip)
        self._frame_skip = frame_skip
        self._max_episode_steps = self._max_episode_steps * factor
        self._max_iterations = self._max_iterations * factor
        self._gamma = self._gamma ** (1 / factor)

    def get_env_data(self):
        mdp = gym.make('Ant-v2')
        mdp.env.frame_skip = self._frame_skip
        # set ankle actions to 0
        high_vals = np.array([1, 1, 1, 1])
        low_vals = np.array([-1, -1, -1, -1])
        actions = create_grid(high_vals, low_vals, self._action_per_dim)
        # set hip actuators to 0
        for i in range(4):
            for j in range(len(actions)):
                actions[j].insert(2 * i, 0)
        file_name = 'Ant_{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name


class TradingData(EnvironmentData):

    def __init__(self, name='', gamma=0.9999, max_iterations=32, for_curs=1,
                 eval_episodes=100, env_ver='v1', mul_env=False, use_train_limit=False):
        self._name = name
        self._gamma = gamma
        self._max_iterations = max_iterations
        self.for_curs = for_curs
        self.eval_episodes = eval_episodes
        self.env_ver = env_ver
        self.mul_env = mul_env
        self.initial_states = None
        self.use_train_limit = use_train_limit
        self.mdp = None

    def get_env_data(self):
        if self.for_curs == 1:
            mdp = gym.make('Old-Trading2017_EURUSD-' + self.env_ver)
            self.mdp = mdp
            if self.use_train_limit:
                self.train_limit = len(mdp.days) - self.eval_episodes
                mdp.train_limit = self.train_limit
                assert self.train_limit > 0
            actions = list(range(3**(self.for_curs)))
        else:
            mdp = gym.make('Trading2017-EURUSDJPY-v1')
            self.mdp = mdp
            actions = list(range(3**(self.for_curs)))
        file_name = 'Trading_{}'.format(self._name)

        self.initial_states = list(range(-self.eval_episodes, 0))

        return mdp, actions, mdp.horizon, self._gamma, self._max_iterations, file_name

    def get_cb_eval(self):
        return get_callback_list_entry("eval_greedy_policy_callback", 
                                       field_name="perf_disc_greedy",
                                       criterion='discounted', 
                                       n_episodes=self.eval_episodes, 
                                       mul_env=self.mul_env,
                                       initial_states=self.initial_states)
    
    def get_initial_states(self):
        if self.use_train_limit:
            return [self.mdp.reset(day_indexes=np.array([i])) for i in range(self.train_limit)]
        else:
            return None
        
class GridWorldData(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('Gridworld-v1')
        actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        file_name = 'Gridworld{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name

class GridWorld2Data(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('Gridworld-v2')
        actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        file_name = 'Gridworld_2{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name

class GridWorld3Data(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('Gridworld-v3')
        actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        file_name = 'Gridworld_3{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name

class GridWorld4Data(EnvironmentData):

    def __init__(self, max_episode_steps=128, horizon_factor=1, name='', gamma=0.99, iterations_divisor=8):
        super().__init__(max_episode_steps, name, gamma, iterations_divisor, horizon_factor)

    def get_env_data(self):
        mdp = gym.make('Gridworld-v4')
        actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        file_name = 'Gridworld_4{}'.format(self._name)

        return mdp, actions, self._max_episode_steps, self._gamma, self._max_iterations, file_name
