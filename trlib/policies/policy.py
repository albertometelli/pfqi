from abc import ABC

import numpy as np
from random import *
from trlib.algorithms.optimization.quasi_newton import quasi_newton
from gym.envs.box2d.lunar_lander import heuristic

class Policy(object):
    """
    Base class for all policies.
    """
    
    def sample_actions(self, states):
        """
        Samples actions in the given states.
        
        Parameters
        ----------
        states: an NxS matrix, where N is the number of states and S is the state-space dimension
          
        Returns
        -------
        An NxA matrix, where N is the number of states and A is the action-space dimension
        """
        raise NotImplementedError
    
    def sample_action(self, state):
        """
        Samples an action in the given state.
        
        Parameters
        ----------
        state: an S-dimensional vector, where S is the state-space dimension
          
        Returns
        -------
        An A-dimensional vector, where A is the action-space dimension
        """
        raise NotImplementedError
    
class Uniform(Policy):
    """
    A uniform policy over a finite set of actions.
    """
    
    def __init__(self, actions):
        self._actions = actions
        self._n_actions = len(actions)
        
    def sample_action(self, state):
        return np.array(self._actions[np.random.choice(self._n_actions)])


class UniformContinuous(Policy):
    """
    A uniform policy over a vector of actions.

    low_bounds are the minimum value of the action vector
    high_bounds are the maximum value of the action vector
    """

    def __init__(self, low_bounds, high_bounds):
        assert len(low_bounds) == len(high_bounds), 'low_bounds and high_bounds must have the same dimension'
        self._actions = [low_bounds, high_bounds]

    def sample_action(self, state):
        low_bounds = self._actions[0]
        high_bounds = self._actions[1]
        action = np.random.rand(len(self._actions[0]))
        action = action*(high_bounds-low_bounds) + low_bounds
        return action


class EpsilonGreedyQN(Policy):

    def __init__(self, Q, epsilon, low_bounds, high_bounds):

        self.epsilon = epsilon
        self._low_bounds = low_bounds
        self._high_bounds = high_bounds
        self._Q = Q

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value < 0 or value > 1:
            raise AttributeError("Epsilon must be in [0,1]")
        self._epsilon = value

    def sample_action(self, state):

        random_a = UniformContinuous(self._low_bounds, self._high_bounds).sample_action(state)
        if np.random.uniform() < self._epsilon:
            return random_a
        else:
            return quasi_newton(self._Q, state, random_a, 0.001, 50)


class LunarLanderExplorativePolicy(Policy):
    """
    Call a custom Lunar Lander gym policy and modify it by adding some noise
    """

    def __init__(self, env, actions, epsilon=0):
        self._env = env
        self._actions = actions
        self._n_actions = len(actions)
        self._epsilon = epsilon
        assert 0 <= epsilon <= 1

    def sample_action(self, state):
        if random() < self._epsilon:
            a = np.array(self._actions[np.random.choice(self._n_actions)])
        else:
            a = np.array(heuristic(self._env, state))
        return a
