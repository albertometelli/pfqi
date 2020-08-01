from trlib.policies.policy import Policy
import numpy as np
import copy


class EpsilonGreedyDoubleQ(Policy):
    """
    The epsilon-greedy policy.
    The parameter epsilon defines the probability of taking a random action.
    Set epsilon to zero to have a greedy policy.
    """

    def __init__(self, actions, epsilon, Q1, Q2=None):

        self._actions = np.array(actions)
        self._n_actions = len(actions)
        self.Q1 = Q1
        if Q2 is None:
            self.Q2 = copy.deepcopy(Q1)
        else:
            self.Q2 = Q2
        self.epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value < 0 or value > 1:
            raise AttributeError("Epsilon must be in [0,1]")
        self._epsilon = value

    @property
    def actions(self):
        return self._actions

    def __call__(self, state):
        probs = np.ones(self._n_actions) * self._epsilon / self._n_actions
        probs[np.argmax(self._q_values(state))] += 1 - self._epsilon
        return probs

    def _q_values(self, state):
        if len(self._actions.shape) > 1:
            action_vec = self._actions
        else:
            action_vec = self._actions[:, np.newaxis]
        q1_vals = self.Q1.values(np.concatenate((np.matlib.repmat(state, self._n_actions, 1), action_vec), 1))
        q2_vals = self.Q2.values(np.concatenate((np.matlib.repmat(state, self._n_actions, 1), action_vec), 1))
        return q1_vals + q2_vals

    def sample_action(self, state):
        if np.random.uniform() < self._epsilon:
            return np.array([self._actions[np.random.choice(self._n_actions)]])
        else:
            return np.array([self._actions[np.argmax(self._q_values(state))]])
