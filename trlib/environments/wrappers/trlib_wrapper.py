from gym import spaces
import gym
import numpy as np

class TrlibWrapper(gym.Wrapper):

    def __init__(self, env, gamma=0.99, max_episode_steps=100):

        super(TrlibWrapper, self).__init__(env)

        #Obtain discount factor
        if hasattr(env, 'gamma'):
            self.gamma = env.gamma
        else:
            self.gamma = gamma

        #Obtain horizon
        if hasattr(env, 'max_episode_steps') and env.max_episode_steps is not None:
            self.max_episode_steps = env.max_episode_steps
        elif hasattr(env, '_max_episode_steps') and env._max_episode_steps is not None:
            self.max_episode_steps = env._max_episode_steps
        elif hasattr(env, 'horizon') and env.horizon is not None:
            self.max_episode_steps = env.horizon
        elif hasattr(env, 'spec') and env.spec is not None:
            self.max_episode_steps = env.spec.max_episode_steps
        elif hasattr(env, 'env') and env.env.spec is not None:
            self.max_episode_steps = env.env.spec.max_episode_steps
        else:
            self.max_episode_steps = max_episode_steps

        self.state_to_reshape = isinstance(self.observation_space, spaces.Box) and len(self.observation_space.shape) > 1
        self.scalarize_action = isinstance(self.action_space, spaces.Box) and np.prod(self.action_space.shape) == 1 or isinstance(self.action_space, spaces.Discrete)


    @property
    def horizon(self):
        return self.max_episode_steps

    @property
    def state_dim(self):
        if isinstance(self.observation_space, spaces.Box):
            return np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, spaces.Discrete):
            return 1
        else:
            raise NotImplementedError

    @property
    def action_dim(self):
        if isinstance(self.action_space, spaces.Box):
            return np.prod(self.action_space.shape)
        elif isinstance(self.action_space, spaces.Discrete):
            return 1
        else:
            raise NotImplementedError()

    def reset(self, state=None, **kwargs):
        new_state = self.env.reset(**kwargs)
        if state is not None:
            new_state = self.unwrapped.state = self._flat2complex_state(state)

        return self._complex2flat_state(new_state)

    def step(self, action):

        if self.scalarize_action:
            action = np.asscalar(action)

        state, reward, done, info = self.env.step(action)
        return self._complex2flat_state(state), reward, done, info

    def _complex2flat_state(self, state):
        if self.state_to_reshape:
            return np.ravel(state)
        else:
            return state

    def _flat2complex_state(self, state):
        if self.state_to_reshape:
            return np.reshape(state, self.observation_space.shape)
        else:
            return state