import numpy as np
from numpy import matlib
from trlib.policies.policy import Policy
from trlib.policies.qfunction import QFunction

class ValueBased(Policy):
    """
    A value-based policy is a policy that chooses actions based on their value.
    The action-space is always discrete for this kind of policy.
    """
    
    def __init__(self,actions,Q):
        
        self._actions = np.array(actions)
        self._n_actions = len(actions)
        self.Q = Q
        
    @property
    def actions(self):
        return self._actions
    
    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self,value):
        if not isinstance(value, QFunction):
            raise TypeError("The argument must be a QFunction")
        self._Q = value
        
    def __call__(self, state):
        """
        Computes the policy value in the given state
        
        Parameters
        ----------
        state: S-dimensional vector
        
        Returns
        -------
        An A-dimensional vector containing the probabilities pi(.|s)
        """
        raise NotImplementedError
    
    def _q_values(self, state):

        if len(self._actions.shape) > 1:
            action_vec = self._actions
        else:
            action_vec = self._actions[:,np.newaxis]
        return self._Q.values(np.concatenate((matlib.repmat(state, self._n_actions, 1), action_vec), 1))

    def _multiple_q_values(self, state):

        if len(self._actions.shape) > 1:
            action_vec = self._actions
        else:
            action_vec = self._actions[:,np.newaxis]

        sa_arrays =  np.concatenate([np.concatenate((matlib.repmat(s, self._n_actions, 1), action_vec), 1)
                                     for s in state])
        q_values = self._Q.values(sa_arrays)
        q_values = q_values.reshape(state.shape[0], self._n_actions)
        return q_values # hardcoded for discrete actions!
    
    
class EpsilonGreedy(ValueBased):
    """
    The epsilon-greedy policy.
    The parameter epsilon defines the probability of taking a random action.
    Set epsilon to zero to have a greedy policy.
    """
    
    def __init__(self,actions,Q,epsilon):
        
        super().__init__(actions, Q)
        self.epsilon = epsilon
        
    @property
    def epsilon(self):
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self,value):
        if value < 0 or value > 1:
            raise AttributeError("Epsilon must be in [0,1]")
        self._epsilon = value
        
    def __call__(self, state):
        
        probs = np.ones(self._n_actions) * self._epsilon / self._n_actions
        probs[np.argmax(self._q_values(state))] += 1 - self._epsilon
        return probs
        
    def sample_action(self, state):
        if np.random.uniform() < self._epsilon:
            a = np.array([self._actions[np.random.choice(self._n_actions)]])
        else:
            a = np.array([self._actions[np.argmax(self._q_values(state))]])
        if a.shape.__len__() > 1:
            return a[0]
        else:
            return a
    
    def sample_multiple_actions(self, state):
        randoms = np.random.uniform(size=len(state)) < self._epsilon
        rand_states_ind = [i for i, u in enumerate(randoms) if u]
        not_rand_states_ind =   [i for i, u in enumerate(randoms) if not u]
        
        # select random actions
        rand_choices = np.random.choice(self._n_actions, len(rand_states_ind))
        rand_actions = [self._actions[c] for c in rand_choices]
        
        # select deterministic actions
        not_rand_states = np.array([s for i, s in enumerate(state) if i in not_rand_states_ind])
        q_values_matrix = self._multiple_q_values(not_rand_states)
        det_actions = [self._actions[np.argmax(qs)] for i, qs in enumerate(q_values_matrix)] 
        
        # put actions together
        a = []     
        for i in range(len(state)):
            if i in rand_states_ind:
                a.append(rand_actions[rand_states_ind[i].index(i)])
            else:
                a.append(det_actions[not_rand_states_ind.index(i)])
        return np.array(a)
        
class Softmax(ValueBased):
    """
    The softmax (or Boltzmann) policy.
    The parameter tau controls exploration (for tau close to zero the policy is almost greedy)
    """
    
    def __init__(self,actions,Q,tau):
        
        super().__init__(actions, Q)
        self.tau = tau
        
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self,value):
        if value <= 0:
            raise AttributeError("Tau must be strictly greater than zero")
        self._tau = value
        
    def __call__(self, state):
        
        exps = np.exp(self._q_values(state) / self._tau)
        return exps / np.sum(exps)
        
    def sample_action(self, state):
        
        return np.array([self._actions[np.random.choice(self._n_actions, p = self(state))]])
