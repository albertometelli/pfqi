import numpy as np
from trlib.algorithms.algorithm import Algorithm
from gym import spaces
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes, split_data, generate_episodes_imposed_actions
from trlib.policies.policy import Policy
from trlib.environments.wrappers.persistent_action_wrapper_ss import PersistentActionWrapperSharedSamples
from trlib.environments.wrappers.persistent_action_wrapper import PersistentActionWrapper
from trlib.policies.doubleq_policies import EpsilonGreedyDoubleQ
from trlib.utilities.save_trajectories import save_trajectories
import copy
import gym
import json

from contextlib import contextmanager
import time
@contextmanager
def timed(msg):
        print(msg)
        tstart = time.time()
        yield
        print("done in %.3f seconds"%(time.time() - tstart))

class FQI(Algorithm):
    """
    Fitted Q-Iteration
    
    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, get_q_differences=False,
                 init_policy=None, verbose=False, generative_setting=False, **regressor_params):
        
        super().__init__("FQI", mdp, policy, verbose)
        
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._regressor_type = regressor_type
        self._n_states = mdp.state_dim
        self._get_q_differences = get_q_differences
        self._generative_setting = generative_setting

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy
        
        if isinstance(mdp.action_space, spaces.Discrete):
            self._policy.Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
        else:
            self._policy.Q = FittedQ(regressor_type, self._n_states, mdp.action_dim, **regressor_params)
        
        self.reset()
        
    def _iter(self, sa, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))
        
        if self._iteration == 0:
            y = r
        else:
            if isinstance(self._policy.Q, FittedQ) and self._actions is None:
                actions = sa[:, self._n_states:]
            else:
                actions = self._actions
            maxq, _ = self._policy.Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self._policy.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1
        
    def _step_core(self, callbacks, **kwargs):
        
        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size,
                                            use_generative_setting=self._generative_setting))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0

        ### debugging check: get the absobing samples
        final_samples_index = data[:, -1] == 1
        final_samples = data[final_samples_index]
        ###

        _,_,_,r,s_prime,absorbing,sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)

        if (self._get_q_differences):
            q_k = np.zeros(len(sa))
            self._q_differences = []

        for _ in range(self._max_iterations):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            ### callbacks called to check if performance is improving
            # for cb in callbacks:
            #     cb(self)
            if self._get_q_differences:
                q_k_next = self._policy.Q.values(sa)
                difference = np.average(np.abs(q_k - q_k_next))
                q_k = q_k_next
                self._q_differences.append(difference)

        self._result.update_step(n_episodes = self.n_episodes, n_samples = data.shape[0])

    def step(self, n_run, callbacks = [], **kwargs):
        self.display("Step {0}".format(self._step))

        self._result.add_step(step=self._step)

        self._step_core(callbacks, **kwargs)

        if self._get_q_differences:
            if isinstance(self._mdp.env, PersistentActionWrapper):
                p = self._mdp.env.persistence
            else:
                p = 1
            with open('Q_differences_' + self._name + '_P' + str(p) + '_Step' + str(self._step) + 'Run' + str(n_run) +
                      '.json', 'w') as f:
                f.write(json.dumps(self._q_differences))

        for cb in callbacks:
            cb(self)

        self._step += 1

        return self._result

    def reset(self):
        
        super().reset()
        
        self._data = []
        self._iteration = 0
        
        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__), policy = str(self._policy.__class__.__name__))


class FQI_TrainTest(Algorithm):

    def __init__(self, mdp_train, mdp_test_list, policy, actions, batch_size, max_iterations, regressor_type, init_policy=None,
                 verbose=False, **regressor_params):

        super().__init__("FQI", mdp_train, policy, verbose)

        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._regressor_type = regressor_type
        self._n_states = mdp_train.state_dim
        self._mdp_train = mdp_train
        self.mdp_test_list = mdp_test_list

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        if isinstance(mdp_train.action_space, spaces.Discrete):
            self._policy.Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
        else:
            self._policy.Q = FittedQ(regressor_type, self._n_states, mdp_train.action_dim, **regressor_params)

        self.reset()

    def _iter(self, sa, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if isinstance(self._policy.Q, FittedQ) and self._actions is None:
                actions = sa[:, self._n_states:]
            else:
                actions = self._actions
            maxq, _ = self._policy.Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self._policy.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _step_core(self, **kwargs):

        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0

        _, _, _, r, s_prime, absorbing, sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)

        for _ in range(self._max_iterations):
            self._iter(sa, r, s_prime, absorbing, **kwargs)

        self._result.update_step(n_episodes=self.n_episodes, n_samples=data.shape[0])

    def step(self, n_run, callbacks=[], **kwargs):
        """
        Performs a training step. This varies based on the algorithm.
        Tipically, one or more episodes are collected and the internal structures are accordingly updated.

        Parameters
        ----------
        callbacks: a list of functions to be called with the algorithm as an input after this step
        kwargs: any other algorithm-dependent parameter

        Returns
        -------
        A Result object
        """
        self.display("Step {0}".format(self._step))

        self._result.add_step(step=self._step)

        self._mdp = self._mdp_train
        self._step_core(**kwargs)

        for i in range(len(self.mdp_test_list)):
            self._mdp = self.mdp_test_list[i]
            for cb in callbacks:
                cb(self)

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policy.__class__.__name__))


class FQI_SS(Algorithm):

    def __init__(self, mdps, policy, actions, batch_size, max_iterations, regressor_type, mdp_sampler, persistences,
                 name, init_policy=None, verbose=False, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, Policy)

        self._name = "FQI_SS"
        self._mdps = mdps
        self._policies = []

        for mdp in mdps:
            self._policies.append(copy.deepcopy(policy))
            if mdp.persistence == mdp_sampler.sampling_persistence:
                self._sampling_policy = self._policies[-1]

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._n_states = mdp_sampler.state_dim
        self._mdp_sampler = mdp_sampler
        self._sampling_persistence = mdp_sampler.sampling_persistence
        self._name = name

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
            else:
                self._policies[i].Q = FittedQ(regressor_type, self._n_states, mdps[i].action_dim, **regressor_params)

        self.reset()

    def _iter(self, sa, r, s_prime, absorbing, i, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if isinstance(self._policies[i].Q, FittedQ) and self._actions is None:
                actions = sa[:, self._n_states:]
            else:
                actions = self._actions
            maxq, _ = self._policies[i].Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdps[i].gamma * maxq

        self._policies[i].Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _step_core(self, **kwargs):

        policy = self._sampling_policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)

        i = 0
        for mdp in self._mdps:
            _, _, _, r, s_prime, absorbing, sa = split_data(data, mdp.state_dim, mdp.action_dim, mdp.persistence,
                                                            self._mdps[0].gamma)
            self.display("Mdp with persistence {0}".format(mdp.persistence))
            self._iteration = 0
            for _ in range(self._max_iterations[i]):
                self._iter(sa, r, s_prime, absorbing, i, **kwargs)
            i += 1
            self._result.update_step(n_episodes=self.n_episodes)
            self._result.update_step_with_persistence(persistence=mdp.persistence, n_samples=r.shape[0])

    def step(self, n_run, callbacks=[], **kwargs):
        """
        Performs a training step. This varies based on the algorithm.
        Tipically, one or more episodes are collected and the internal structures are accordingly updated.

        Parameters
        ----------
        callbacks: a list of functions to be called with the algorithm as an input after this step
        kwargs: any other algorithm-dependent parameter

        Returns
        -------
        A Result object
        """
        self.display("Step {0}".format(self._step))

        self._result.add_step(step=self._step)

        self._step_core(**kwargs)
        temp_result = []

        i = 0
        for mdp in self._mdps:
            self._mdp = mdp
            self._policy = self._policies[i]
            for cb in callbacks:
                cb(self)
            temp_result.append(self._result.steps[self._step-1]['perf_disc_greedy_mean'+'_P'+str(mdp.persistence)])
            i += 1

        import json
        with open('TempResults' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
            f.write(json.dumps(temp_result))

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, sampling_persistence=self._sampling_persistence,
                                regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))


class FQI_SS_EP_old(Algorithm):
    '''

    mdp: is one mdp of the environment
    policy: is the policy that will be cloned for each persistence value
    max_iterations: is the value referred to the 1-step case
    persistences: is the number of different values of persistence we want to test, from 1 to persistences

    Questioni da migliorare:
    - non fittare tutti i valori di persistenza ma solo una lista di valori passati
    - far decidere all'utente una funzione di distanza fra stati
    - pensare meglio l'errore massimo che si può commettere nella distanza fra stati

    '''

    def __init__(self, mdps, policy, actions, batch_size, max_iterations, regressor_type, persistences,
                 name, distance_fun=None, max_error=0.1, init_policy=None, verbose=False, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, Policy)
        persistences.sort()

        self._name = "FQI_SS_EP"
        self._mdps = mdps
        self._policies = []

        for _ in range(len(persistences)):
            self._policies.append(copy.deepcopy(policy))

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._n_states = mdps[0].state_dim
        self._name = name
        self._max_error = max_error

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        if distance_fun is None:
            self._distance_fun = self.sqr_euclidian_distance
        else:
            self._distance_fun = distance_fun

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
            else:
                self._policies[i].Q = FittedQ(regressor_type, self._n_states, mdps[i].action_dim, **regressor_params)

        self.reset()

    def sqr_euclidian_distance(self, s1, s2):
        return np.sum((s1 - s2) ** 2)

    def update_data(self, s, a, r, s_prime, absorbing, next_s_prime_old, next_r_old, next_absorbing_old,
                    old_valid_samples_indexes, old_next_step_indexes, gamma_powered):
        i_to_save = []  # indexes of lists with suffix _old that will be preserved in the new _next lists
        next_step_indexes = []  # indexes of the original data samples that will be the next states
        new_valid_samples_indexes = []  # indexes of the original data samples of action state couple saved
        persisted_a = a[old_valid_samples_indexes]

        for i in range(len(next_s_prime_old)):
            if next_absorbing_old[i]:
                i_to_save.append(i)
                next_step_indexes.append(old_next_step_indexes[i])
                new_valid_samples_indexes.append(old_valid_samples_indexes[i])
                continue
            min_dist = self._max_error
            min_idx = 0
            for j in range(len(s)):
                if persisted_a[i] == a[j]:
                    distance = self._distance_fun(next_s_prime_old[i], s[j])
                    if distance < min_dist and distance < self._max_error:
                        min_dist = distance
                        min_idx = j
            if min_dist < self._max_error:
                i_to_save.append(i)
                next_step_indexes.append(min_idx)
                new_valid_samples_indexes.append(old_valid_samples_indexes[i])

        inst_r = r[next_step_indexes]
        inst_r[next_absorbing_old[i_to_save] == 1] = 0  # don't add any more if the episode is already terminated
        r_next = next_r_old[i_to_save] + inst_r * gamma_powered
        s_prime_next = s_prime[next_step_indexes]
        absorbing_next = absorbing[next_step_indexes]

        return r_next, s_prime_next, absorbing_next, new_valid_samples_indexes, next_step_indexes

    def all_actions_explored(self, a, valid_samples_indexes, p):
        """
        method created to avoid crash in case a DiscreteFittedQ has no sample for one kind of action.
        In this case the iter function will crash after fitting the Q function
        """
        if isinstance(self._policies[p].Q, DiscreteFittedQ):
            actions = self._actions
            all_actions_explored = True
            for action in actions:
                if action not in a[valid_samples_indexes]:
                    all_actions_explored = False
        else:
            all_actions_explored = True
        return all_actions_explored

    def _iter(self, sa, r, s_prime, absorbing, p, **fit_params):
        """
        In this algorithm the argument must be preprocessed:
        r is the cumulative estimated reward starting from state s and persisting action i+1 steps
        s_prime is the next state after having persisted the action
        absorbing indicate if the s_prime previously defined is absorbing
        i is the index of the policy to fit, corresponding to the i+1 persistence
        """
        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if isinstance(self._policies[p].Q, FittedQ) and self._actions is None:
                actions = sa[:, self._n_states:]
            else:
                actions = self._actions
            maxq, _ = self._policies[p].Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdps[p].gamma * maxq

        self._policies[p].Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _step_core(self, **kwargs):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdps[0], policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        _, s, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        s_prime_next = s_prime
        r_next = r
        absorbing_next = absorbing
        valid_samples_indexes = np.arange(len(s))
        next_step_indexes = np.arange(len(s))
        gamma = self._mdps[0].gamma
        self._p = 0  # index of current persistence considered
        for i in range(self._persistences[-1]):
            persistence = i+1
            self.display("Mdp with persistence {0}".format(persistence))
            if self._max_iterations // persistence < 1:
                self.display(
                    "Persistence {0} is exeeding the number of max iterations which is: {1}. Breaking the step".format(
                        persistence, self._max_iterations))
                break;
            if i != 0:
                r_next, s_prime_next, absorbing_next, valid_samples_indexes, next_step_indexes = \
                    self.update_data(s, a, r, s_prime, absorbing, s_prime_next, r_next, absorbing_next,
                                     valid_samples_indexes, next_step_indexes, gamma ** i)
            if persistence in self._persistences:
                if not self.all_actions_explored(a, valid_samples_indexes, self._p):
                    self.display("Mdp with persistence {0} has not enough samples. Stopping the step.".format(persistence))
                    break;
                self._iteration = 0
                sa_selected = sa[valid_samples_indexes]
                for _ in range(self._max_iterations // persistence):
                    self._iter(sa_selected, r_next, s_prime_next, absorbing_next, self._p, **kwargs)
                self._result.update_step(n_episodes=self.n_episodes)
                self._result.update_step_with_persistence(persistence=persistence, n_samples=r.shape[0])
                self._p += 1

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)
        self._step_core(**kwargs)
        temp_result = []

        for i in range(self._p):
            self._policy = self._policies[i]
            self._mdp = self._mdps[i]
            for cb in callbacks:
                cb(self)
            if 'perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence) in self._result.steps[0]:
                temp_result.append(self._result.steps[self._step-1]['perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence)])

        import json
        with open('TempResults' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
            f.write(json.dumps(temp_result))

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        from itertools import repeat
        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))


class FQI_SS_EP(Algorithm):

    def __init__(self, mdps, mdp_sampler, policy, actions, batch_size, max_iterations, regressor_type, persistences,
                 name, init_policy=None, verbose=False, save_perfs_and_q=False, save_step=1, generative_setting=False,
                 alternative_policy=None, initial_states=None, n_jobs_regressors=1, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, Policy)
        persistences.sort()

        self._name = "FQI_SS_EP"
        self._mdps = mdps
        self._mdp_sampler = mdp_sampler
        self._policies = []

        for _ in range(len(persistences)):
            self._policies.append(copy.deepcopy(policy))

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._state_dim = mdps[0].state_dim
        self._action_dim = mdps[0].action_dim
        self._name = name
        self._save_perfs_and_q = save_perfs_and_q
        self._save_step = save_step
        self._generative_setting = generative_setting
        self._gamma1P = self._mdp_sampler.gamma ** (1 / self._mdp_sampler.sampling_persistence)
        self._alternative_policy = alternative_policy
        self._initial_states = initial_states

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        # add n_jobs_regressors to regressor params
        regressor_params['n_jobs'] = n_jobs_regressors

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q = DiscreteFittedQ(regressor_type, self._state_dim, actions, **regressor_params)
            else:
                self._policies[i].Q = FittedQ(regressor_type, self._state_dim, self._action_dim, **regressor_params)

        if save_perfs_and_q:
            self._perfs = []
            self._stds = []
            self._q_k = []
            self._q_diff = []
            self._q_diff_abs = []
            self._err_proj = []
            self._err_proj_abs = []

        self.reset()

    def _iter(self, sa, s_prime_a, r, s_prime, absorbing, mdp_index, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self._iteration % self._mdps[mdp_index].env.persistence == 0:
                actions = get_actions(self._policies[mdp_index].Q, self._actions, self._state_dim, sa)
                maxq, _ = self._policies[mdp_index].Q.max(s_prime, actions, absorbing)
                y = r.ravel() + self._gamma1P * maxq
            else:
                q_persisted = self._policies[mdp_index].Q.persisted_values(s_prime_a, absorbing)
                y = r.ravel() + self._gamma1P * q_persisted
        with timed("fitting iteration " + str(self._iteration)):
            self._policies[mdp_index].Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

        if self._save_perfs_and_q:
            self._y = y

    def _step_core(self, n_run, callbacks, **kwargs):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size,
                                            use_generative_setting=self._generative_setting,
                                            alternative_policy=self._alternative_policy))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        mdp_index = 0
        t, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        ### temporary code for car on hill - begin###
        # uni_r, cnt = np.unique(absorbing, return_counts=True)
        terminals = data[data[:, -1]==1, :]
        # if 1 not in uni_r:
        #    self.display("Samples not good, terminating")
        #    exit()
        ### temporary code for car on hill - end###
        s_prime_a = create_sa(s_prime, a)
        for p in self._persistences:
            self.display("Mdp with persistence {0}".format(p))
            self._iteration = 0
            if self._save_perfs_and_q:
                self._init_perfs_and_q(t, sa)
            for i in range(self._max_iterations):
                self._iter(sa, s_prime_a, r, s_prime, absorbing, mdp_index, **kwargs)
                if self._save_perfs_and_q:
                    self._update_perfs_and_q(callbacks, mdp_index, sa, t, p)
            if self._save_perfs_and_q:
                self._append_perfs_and_q()
            self._result.update_step(n_episodes=self.n_episodes)
            self._result.update_step_with_persistence(persistence=p, n_samples=r.shape[0])
            mdp_index += 1

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)
        self._step_core(n_run, callbacks, **kwargs)
        temp_result = []
        perf_means = []
        perf_std = []

        if self._save_perfs_and_q:
            with open('Perfs_and_qs_' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
                f.write(json.dumps([self._perfs, self._stds, self._q_k, self._q_diff, self._q_diff_abs,
                                    self._err_proj, self._err_proj_abs, self._save_step, self._gamma1P]))
        else:
            for i in range(len(self._persistences)):
                self._policy = self._policies[i]
                self._mdp = self._mdps[i]
                for cb in callbacks:
                    cb(self)
                if 'perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence) in self._result.steps[0]:
                    perf_means.append(self._result.steps[self._step-1]['perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence)])
                if 'perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence) in self._result.steps[0]:
                    perf_std.append(self._result.steps[self._step-1]['perf_disc_greedy_std'+'_P'+str(self._mdp.env.persistence)])
            with open('TempResults' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
                temp_result.append(perf_means)
                temp_result.append(perf_std)
                f.write(json.dumps(temp_result))

            #policy_data.append(generate_episodes(self._mdp, self._policy, 10))
            #save_trajectories(policy_data, self._mdps[0].state_dim, self._mdps[0].action_dim)

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        from itertools import repeat
        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))

    def _init_perfs_and_q(self, t, sa):

        if self._generative_setting:
            len_sa0 = len(self._mdps[0].unwrapped.select_starting_sa(sa))
        elif self._initial_states is None:
            unique, counts = np.unique(t, return_counts=True)
            assert unique[0] == 0
            len_sa0 = counts[0]
        else:
            len_sa0 = len(self._initial_states)
        self._q_k_prec = [np.zeros(len(sa)), np.zeros(len(sa)), np.zeros(len_sa0)]
        self._p_perfs = []
        self._p_stds = []
        self._p_q_k = [[0], [0], [0]]
        self._p_q_diff = [[], [], []]
        self._p_q_diff_abs = [[], [], []]
        self._p_err_proj = []
        self._p_err_proj_abs = []
        self._err_proj_curr = 0
        self._err_abs_proj_curr = 0

    def _update_perfs_and_q(self, callbacks, mdp_index, sa, t, p):
        self._policy = self._policies[mdp_index]
        self._mdp = self._mdps[mdp_index]

        # get 3 variants of sa
        s = sa[:, :self._mdps[0].state_dim]
        actions = get_actions(self._policies[mdp_index].Q, self._actions, self._state_dim, sa)
        _, best_a = self._policies[mdp_index].Q.max(s, actions)
        sa_greedy = create_sa(s, np.array(best_a))
        if self._generative_setting:
            s0a_greedy = self._mdp.unwrapped.select_starting_sa(sa_greedy)
        elif self._initial_states is None:
            s0a_greedy = sa_greedy[t == 0, :]
        else:
            s0 = self._initial_states
            _, best_a = self._policies[mdp_index].Q.max(s0, actions)
            s0a_greedy = create_sa(s0, np.array(best_a))
        # self._err_proj_curr = np.average((self._y - self._policies[mdp_index].Q.values(sa))) \
        #                        + gamma * self._err_proj_curr
        # self._err_abs_proj_curr = np.average(np.abs((self._y - self._policies[mdp_index].Q.values(sa)))) \
        #                         + gamma * self._err_abs_proj_curr

        if self._iteration % p == 0 and \
                (self._iteration % self._save_step == 0 or (self._iteration - p) % self._save_step == 0):

            # get 3 variants of q_k
            q_k = [self._policies[mdp_index].Q.values(sa),
                   self._policies[mdp_index].Q.values(sa_greedy),
                   self._policies[mdp_index].Q.values(s0a_greedy)]

            if (self._iteration - p) % self._save_step == 0:

                for i in range(len(q_k)):
                    self._p_q_diff_abs[i].append(np.average(np.abs(q_k[i] - self._q_k_prec[i])))

            if self._iteration % self._save_step == 0:
                for cb in callbacks:
                    cb(self)
                self._p_perfs.append(
                    self._result.steps[self._step - 1]['perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence)])
                self._p_stds.append(
                    self._result.steps[self._step - 1]['perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence)])

                for i in range(len(q_k)):
                    self._p_q_k[i].append(np.average(q_k[i]))
                    self._q_k_prec[i] = q_k[i]

            # self._p_err_proj.append(self._err_proj_curr)
            # self._p_err_proj_abs.append(self._err_abs_proj_curr)
            # self._err_proj_curr = 0
            # self._err_abs_proj_curr = 0

    def _append_perfs_and_q(self):
        self._perfs.append(self._p_perfs)
        self._stds.append(self._p_stds)

        for i in range(len(self._p_q_k)):
            # the last q cannot be used for the bound, final q_k+1 missing
            del self._p_q_k[i][-1]

        self._q_k.append(self._p_q_k)
        self._q_diff.append(self._p_q_diff)
        self._q_diff_abs.append(self._p_q_diff_abs)
        self._err_proj.append(self._p_err_proj)
        self._err_proj_abs.append(self._p_err_proj_abs)

    def get_q_for_bound(self, policy_idx, **fit_params):
        assert policy_idx <= len(self._policies)
        persistence = self._persistences[policy_idx]
        q_functions = []
        targets = []
        data = np.concatenate(self._data)
        gamma = self._mdp_sampler.gamma
        t, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        if len(a.shape) == 1:
            a_column = np.reshape(a, (len(a), 1))
        else:
            a_column = a
        s_prime_a = np.concatenate((s_prime, a_column), axis=1)
        self.display("Getting Q functions of mdp with persistence {0}".format(persistence))
        q_functions.append(copy.deepcopy(self._policies[policy_idx].Q))
        self._iteration = self._max_iterations
        for i in range(persistence):
            y = self._iter(sa, s_prime_a, r, s_prime, absorbing, policy_idx, gamma, targetNeeded=True, **fit_params)
            q_functions.append(copy.deepcopy(self._policies[policy_idx].Q))
            targets.append(y)
        if isinstance(self._policies[policy_idx].Q, FittedQ) and self._actions is None:
            actions = sa[:, self._n_states:]
        else:
            actions = self._actions
        return t, sa, q_functions, targets, actions

    def get_q_for_bound_2(self):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        data = []
        data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        data = np.concatenate(data)
        gamma = self._mdp_sampler.gamma
        _, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        q_functions = []
        targets = []
        for i in range(len(self._persistences)):
            if isinstance(self._policies[i].Q, FittedQ) and self._actions is None:
                actions = sa[:, self._n_states:]
            else:
                actions = self._actions
            maxq, _ = self._policies[i].Q.max(s_prime, actions, absorbing)
            y = r.ravel() + gamma * maxq
            targets.append(y)
            q_functions.append(self._policies[i].Q)
        return sa, q_functions, targets


class FQI_SS_EP_1P_Control(Algorithm):

    def __init__(self, mdps, mdp_sampler, policy, actions, batch_size, max_iterations, regressor_type, persistences,
                 name, init_policy=None, verbose=False, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, Policy)
        persistences.sort()

        self._name = "FQI_SS_EP_1P_Control"
        self._mdps = mdps
        self._mdp_sampler = mdp_sampler
        self._policies = []

        for _ in range(len(persistences)):
            self._policies.append(copy.deepcopy(policy))

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._state_dim = mdps[0].state_dim
        self._action_dim = mdps[0].action_dim
        self._name = name

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q = DiscreteFittedQ(regressor_type, self._state_dim, actions, **regressor_params)
            else:
                self._policies[i].Q = FittedQ(regressor_type, self._state_dim, self._action_dim, **regressor_params)

        self._perfs = []
        self._stds = []
        self._perfs_1p = []
        self._stds_1p = []
        self._q_k = []
        self._Bell_err = []

        self.reset()

    def _iter(self, sa, s_prime_a, r, s_prime, absorbing, mdp_index, gamma, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self._iteration % self._mdps[mdp_index].env.persistence == 0:
                actions = get_actions(self._policies[mdp_index].Q, self._actions, self._state_dim, sa)
                maxq, _ = self._policies[mdp_index].Q.max(s_prime, actions, absorbing)
                y = r.ravel() + gamma * maxq
            else:
                q_persisted = self._policies[mdp_index].Q.persisted_values(s_prime_a, absorbing)
                y = r.ravel() + gamma * q_persisted

        self._policies[mdp_index].Q.fit(sa, y.ravel(), **fit_params)
        self._iteration += 1

        self._y = y

    def _step_core(self, n_run, callbacks, **kwargs):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        gamma = self._mdp_sampler.gamma
        mdp_index = 0
        t, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        s_prime_a = create_sa(s_prime, a)
        for p in self._persistences:
            self.display("Mdp with persistence {0}".format(p))
            self._iteration = 0
            self._init_perfs_and_q()
            for i in range(self._max_iterations):
                self._iter(sa, s_prime_a, r, s_prime, absorbing, mdp_index, gamma, **kwargs)
                self._update_perfs_and_q(callbacks, mdp_index, r, sa, s_prime, absorbing, gamma, p)
            self._append_perfs_and_q()
            self._result.update_step(n_episodes=self.n_episodes)
            self._result.update_step_with_persistence(persistence=p, n_samples=r.shape[0])
            mdp_index += 1

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)
        self._step_core(n_run, callbacks, **kwargs)
        with open('Perfs_and_qs_' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
            f.write(json.dumps([self._perfs, self._stds, self._perfs_1p, self._stds_1p, self._q_k, self._Bell_err]))
        self._step += 1
        return self._result

    def _init_perfs_and_q(self):
        self._p_perfs = []
        self._p_stds = []
        self._p_perfs_1p = []
        self._p_stds_1p = []
        self._p_q_k = []
        self._p_Bell_err = []

    def _update_perfs_and_q(self, callbacks, mdp_index, r, sa, s_prime, absorbing, gamma, p):
        self._policy = self._policies[mdp_index]
        self._mdp = self._mdps[mdp_index]

        if self._iteration % p == 0:
            for cb in callbacks:
                cb(self)
            self._p_perfs.append(
                self._result.steps[self._step - 1]['perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence)])
            self._p_stds.append(
                self._result.steps[self._step - 1]['perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence)])
            self._mdp = self._mdps[0]
            if mdp_index != 0:
                for cb in callbacks:
                    cb(self)
            self._p_perfs_1p.append(
                self._result.steps[self._step - 1]['perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence)])
            self._p_stds_1p.append(
                self._result.steps[self._step - 1]['perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence)])
            q_k = self._policies[mdp_index].Q.values(sa)
            actions = get_actions(self._policies[mdp_index].Q, self._actions, self._state_dim, sa)
            maxq, _ = self._policies[mdp_index].Q.max(s_prime, actions, absorbing)
            y = r.ravel() + gamma * maxq
            bell_err = np.average(np.abs(q_k - y))
            self._p_q_k.append(np.average(q_k))
            self._p_Bell_err.append(bell_err)

    def _append_perfs_and_q(self):
        self._perfs.append(self._p_perfs)
        self._stds.append(self._p_stds)
        self._q_k.append(self._p_q_k)
        self._Bell_err.append(self._p_Bell_err)
        self._perfs_1p.append(self._p_perfs_1p)
        self._stds_1p.append(self._p_stds_1p)

    def reset(self):

        super().reset()

        from itertools import repeat
        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))


class Double_FQI_SS_EP(Algorithm):
    def __init__(self, mdps, mdp_sampler, policy, actions, batch_size, max_iterations, regressor_type, persistences,
                 name, init_policy=None, verbose=False, save_perfs_and_q=False, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, EpsilonGreedyDoubleQ)
        persistences.sort()

        self._name = "Double_FQI_SS_EP"
        self._mdps = mdps
        self._mdp_sampler = mdp_sampler
        self._policies = []

        for _ in range(len(persistences)):
            self._policies.append(copy.deepcopy(policy))

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._state_dim = mdps[0].state_dim
        self._action_dim = mdps[0].action_dim
        self._name = name
        self._save_perfs_and_q = save_perfs_and_q

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q1 = DiscreteFittedQ(regressor_type, self._state_dim, actions, **regressor_params)
                self._policies[i].Q2 = DiscreteFittedQ(regressor_type, self._state_dim, actions, **regressor_params)
            else:
                self._policies[i].Q1 = FittedQ(regressor_type, self._state_dim, self._action_dim, **regressor_params)
                self._policies[i].Q2 = FittedQ(regressor_type, self._state_dim, self._action_dim, **regressor_params)

        if save_perfs_and_q:
            self._perfs = []
            self._stds = []
            self._best_maxq = []
            self._best_minq = []
            self._second_best_maxq = []
            self._second_best_minq = []

        self.reset()

    def _iter(self, sa1, s_prime_a1, r1, s_prime1, absorbing1, sa2, s_prime_a2, r2, s_prime2, absorbing2,
              mdp_index, gamma, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y1 = r1
            y2 = r2
        else:
            if self._iteration % self._mdps[mdp_index].env.persistence == 0:
                actions1 = get_actions(self._policies[mdp_index].Q1, self._actions, self._state_dim, sa1)
                actions2 = get_actions(self._policies[mdp_index].Q2, self._actions, self._state_dim, sa2)
                _, max_a1 = self._policies[mdp_index].Q2.max(s_prime1, actions1)
                _, max_a2 = self._policies[mdp_index].Q1.max(s_prime2, actions2)
                final_sa1 = create_sa(s_prime1, np.array(max_a1))
                final_sa2 = create_sa(s_prime2, np.array(max_a2))
                q_val1 = self._policies[mdp_index].Q1.values(final_sa1)
                q_val2 = self._policies[mdp_index].Q2.values(final_sa2)
                q_val1[absorbing1 == 1] = 0
                q_val2[absorbing2 == 1] = 0
                y1 = r1.ravel() + gamma * q_val1
                y2 = r2.ravel() + gamma * q_val2
            else:
                q_persisted1 = self._policies[mdp_index].Q1.persisted_values(s_prime_a1, absorbing1)
                q_persisted2 = self._policies[mdp_index].Q2.persisted_values(s_prime_a2, absorbing2)
                y1 = r1.ravel() + gamma * q_persisted1
                y2 = r2.ravel() + gamma * q_persisted2

        self._policies[mdp_index].Q1.fit(sa1, y1.ravel(), **fit_params)
        self._policies[mdp_index].Q2.fit(sa2, y2.ravel(), **fit_params)
        self._iteration += 1

    def _step_core(self, n_run, callbacks, **kwargs):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        self._data1.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self._data2.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data1 = np.concatenate(self._data1)
        data2 = np.concatenate(self._data2)

        gamma = self._mdp_sampler.gamma
        mdp_index = 0
        t1, _, a1, r1, s_prime1, absorbing1, sa1 = split_data(data1, self._mdps[0].state_dim, self._mdps[0].action_dim)
        t2, _, a2, r2, s_prime2, absorbing2, sa2 = split_data(data2, self._mdps[0].state_dim, self._mdps[0].action_dim)
        s_prime_a1 = create_sa(s_prime1, a1)
        s_prime_a2 = create_sa(s_prime2, a2)

        for p in self._persistences:

            if self._save_perfs_and_q:
                self._p_perfs = []
                self._p_stds = []
                self._p_best_maxq = []
                self._p_best_minq = []
                self._p_second_best_maxq = []
                self._p_second_best_minq = []

            self.display("Mdp with persistence {0}".format(p))
            self._iteration = 0
            for i in range(self._max_iterations):
                self._iter(sa1, s_prime_a1, r1, s_prime1, absorbing1, sa2, s_prime_a2, r2, s_prime2, absorbing2,
                           mdp_index, gamma, **kwargs)
                if self._save_perfs_and_q and i % p == p - 1:
                    self.save_perfs_and_q(callbacks, mdp_index, t1, t2)

            if self._save_perfs_and_q:
                self._perfs.append(self._p_perfs)
                self._stds.append(self._p_stds)
                self._best_maxq.append(self._p_best_maxq)
                self._best_minq.append(self._p_best_minq)
                self._second_best_maxq.append(self._p_second_best_maxq)
                self._second_best_minq.append(self._p_second_best_minq)

            self._result.update_step(n_episodes=self.n_episodes)
            self._result.update_step_with_persistence(persistence=p, n_samples=r1.shape[0])
            mdp_index += 1

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)
        self._step_core(n_run, callbacks, **kwargs)
        temp_result = []
        perf_means = []
        perf_std = []

        if self._save_perfs_and_q:
            with open('Perfs_and_qs_' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
                f.write(json.dumps([self._perfs, self._stds, self._best_maxq, self._best_minq,
                                    self._second_best_maxq, self._second_best_minq]))
        else:
            for i in range(len(self._persistences)):
                self._policy = self._policies[i]
                self._mdp = self._mdps[i]
                for cb in callbacks:
                    cb(self)
                if 'perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence) in self._result.steps[0]:
                    perf_means.append(self._result.steps[self._step-1]['perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence)])
                if 'perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence) in self._result.steps[0]:
                    perf_std.append(self._result.steps[self._step-1]['perf_disc_greedy_std'+'_P'+str(self._mdp.env.persistence)])

            with open('TempResults' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
                temp_result.append(perf_means)
                temp_result.append(perf_std)
                f.write(json.dumps(temp_result))

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        self._data1 = []
        self._data2 = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))

    def get_q_vals(self, policy, t1, t2):
        a_idx = 1 + self._mdps[0].state_dim
        r_idx = a_idx + self._mdps[0].action_dim
        s1 = np.concatenate(self._data1)[t1 == 0, 1:a_idx]
        s2 = np.concatenate(self._data2)[t2 == 0, 1:a_idx]
        s = np.vstack((s1, s2))
        sa = np.concatenate(self._data1)[t1 == 0, 1:r_idx]
        actions = get_actions(policy.Q1, self._actions, self._state_dim, sa)
        current_sa = create_sa(s, np.full(len(s), actions[0]))
        vals1 = policy.Q1.values(current_sa)
        vals2 = policy.Q2.values(current_sa)
        for i in range(len(actions)-1):
            current_sa = create_sa(s, np.full(len(s), actions[i+1]))
            vals1 = np.column_stack((vals1, policy.Q1.values(current_sa)))
            vals2 = np.column_stack((vals2, policy.Q2.values(current_sa)))
        return vals1, vals2

    def save_perfs_and_q(self, callbacks, mdp_index, t1, t2):
        self._policy = self._policies[mdp_index]
        self._mdp = self._mdps[mdp_index]
        for cb in callbacks:
            cb(self)
        self._p_perfs.append(
            self._result.steps[self._step - 1]['perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence)])
        self._p_stds.append(
            self._result.steps[self._step - 1]['perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence)])
        vals1, vals2 = self.get_q_vals(self._policy, t1, t2)
        sortedvals1 = np.sort(vals1, axis=1)
        sortedvals2 = np.sort(vals2, axis=1)
        best_maxq = np.amax(np.column_stack((sortedvals1[:, -1], sortedvals2[:, -1])), axis=1)
        best_minq = np.amin(np.column_stack((sortedvals1[:, -1], sortedvals2[:, -1])), axis=1)
        second_best_maxq = np.amax(np.column_stack((sortedvals1[:, -2], sortedvals2[:, -2])), axis=1)
        second_best_minq = np.amin(np.column_stack((sortedvals1[:, -2], sortedvals2[:, -2])), axis=1)
        self._p_best_maxq.append(np.average(best_maxq).item())
        self._p_best_minq.append(np.average(best_minq).item())
        self._p_second_best_maxq.append(np.average(second_best_maxq).item())
        self._p_second_best_minq.append(np.average(second_best_minq).item())


class FQI_P1_VS_1P(Algorithm):

    def __init__(self, mdp_sampler, mdp_persisted, policy, actions, batch_size, max_iterations, regressor_type, name,
                 init_policy=None, verbose=False, **regressor_params):

        assert isinstance(policy, Policy)
        assert isinstance(mdp_sampler.env, PersistentActionWrapperSharedSamples)
        assert isinstance(mdp_persisted.env, PersistentActionWrapper)

        assert mdp_persisted.persistence == mdp_sampler.sampling_persistence

        self._name = "FQI_P1_VS_1P"
        self._policy_P1 = policy
        self._policy_1P = copy.deepcopy(policy)
        self._mdp_sampler = mdp_sampler
        self._mdp_persisted = mdp_persisted
        self._mdp = mdp_persisted
        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistence = mdp_persisted.persistence
        self._regressor_type = regressor_type
        self._n_states = mdp_sampler.state_dim
        self._name = name

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        if isinstance(mdp_persisted.action_space, spaces.Discrete):
            self._policy_P1.Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
            self._policy_1P.Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
        else:
            self._policy_P1.Q = FittedQ(regressor_type, self._n_states, mdp_persisted.action_dim, **regressor_params)
            self._policy_1P.Q = FittedQ(regressor_type, self._n_states, mdp_persisted.action_dim, **regressor_params)

        self.reset()

    def _iter_P1(self, sa, r, s_prime, absorbing, **fit_params):
        """
        In this algorithm the arguments must be preprocessed:
        r is the cumulative estimated reward starting from state s and persisting action i+1 steps
        s_prime is the next state after having persisted the action
        absorbing indicate if the s_prime previously defined is absorbing
        """

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            actions = get_actions(self._policy_P1.Q, self._actions, self._n_states, sa)
            maxq, _ = self._policy_P1.Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdp_persisted.gamma * maxq

        self._policy_P1.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _iter_1P(self, sa, s_prime_a, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self._iteration % self._mdp_sampler.sampling_persistence == 0:
                actions = get_actions(self._policy_1P.Q, self._actions, self._n_states, sa)
                maxq, _ = self._policy_1P.Q.max(s_prime, actions, absorbing)
                y = r.ravel() + self._mdp_sampler.gamma_single_step * maxq
            else:
                q_persisted = self._policy_1P.Q.persisted_values(s_prime_a, absorbing)
                y = r.ravel() + self._mdp_sampler.gamma_single_step * q_persisted

        self._policy_1P.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _step_core(self, **kwargs):

        policy = self._policy_1P if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        _, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)
        if len(a.shape) == 1:
            a_column = np.reshape(a, (len(a), 1))
        else:
            a_column = a
        s_prime_a = np.concatenate((s_prime, a_column), axis=1)
        # training of 1P
        self.display("FQI_(1,{})".format(self._persistence))
        for _ in range(self._max_iterations):
            self._iter_1P(sa, s_prime_a, r, s_prime, absorbing, **kwargs)
        self._result.update_step(n_episodes=self.n_episodes)
        self._result.update_step_with_mdp(mdp_name='1P', n_samples=r.shape[0])
        # training of P1
        self.display("FQI_({},1)".format(self._persistence))
        _, _, _, r, s_prime, absorbing, sa = split_data(data, self._mdp_persisted.state_dim,
                                                        self._mdp_persisted.action_dim, self._mdp_persisted.persistence,
                                                        self._mdp_persisted.gamma_single_step)
        iterations = self._max_iterations // self._persistence
        self._iteration = 0
        for _ in range(iterations):
            self._iter_P1(sa, r, s_prime, absorbing, **kwargs)
        self._result.update_step(n_episodes=self.n_episodes)
        self._result.update_step_with_mdp(mdp_name='P1', n_samples=r.shape[0])

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))

        self._result.add_step(step=self._step)

        self._step_core(**kwargs)

        self._policy = self._policy_1P
        self._policy._name = '_1P'
        for cb in callbacks:
            cb(self)

        self._policy = self._policy_P1
        self._policy._name = '_P1'
        for cb in callbacks:
            cb(self)

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistence=self._persistence,
                                regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policy_1P.__class__.__name__))


class FQI_TOTALLY_SS(Algorithm):

    def __init__(self, mdps, mdp_sampler, policy, actions, batch_size, max_iterations, regressor_type, name,
                 init_policy=None, verbose=False, **regressor_params):
        """
        This FQI version is made to have a series of trajectories created with mdp_sampler, an mdp that persisits an
        action for k steps and for which we record every single step. This data is shared among all other mdps and
        policies that we want to fit.
        The mdps must have a persistence which is a submultiple of the mdp_sampler persistence and for every one of
        them, two policies will be fitted.
        The first kind of policy working on an mdp at persistence k will look just at samples every k steps and apply
        standard FQI on them.
        The second kind of policy working on an mdp at persistence k will look at every single step in the trajectories
        and will apply the persistent Bellman Operator.
        Each step every policy is fitted and evaluated. The last policy to be evaluated is the one with maximum
        persistence fitted with the Persisted Bellman Operator, and will be the policy used to generate samples at each
        successive steps.
        """
        assert isinstance(policy, Policy)
        assert isinstance(mdp_sampler.env, PersistentActionWrapperSharedSamples)
        for i in range(len(mdps)):
            assert isinstance(mdps[i].env, PersistentActionWrapper)
            assert mdp_sampler.sampling_persistence % mdps[i].persistence == 0

        self._name = "FQI_TOTALLY_SS"
        self._mdps = mdps
        self._policy = policy
        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._regressor_type = regressor_type
        self._n_states = mdp_sampler.state_dim
        self._mdp_sampler = mdp_sampler
        self._sampling_persistence = mdp_sampler.sampling_persistence
        self._name = name

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        if isinstance(self._mdp_sampler.action_space, spaces.Discrete):
            self._policy.Q = DiscreteFittedQ(regressor_type, self._n_states, actions, **regressor_params)
        else:
            self._policy.Q = FittedQ(regressor_type, self._n_states, mdp_sampler.action_dim, **regressor_params)

        self.reset()

    def _iter_standard(self, sa, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            actions = get_actions(self._policy.Q, self._actions, self._n_states, sa)
            maxq, _ = self._policy.Q.max(s_prime, actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self._policy.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _iter_persisted_operator(self, sa, s_prime_a, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self._iteration % self._mdp.persistence == 0:
                actions = get_actions(self._policy.Q, self._actions, self._n_states, sa)
                maxq, _ = self._policy.Q.max(s_prime, actions, absorbing)
                y = r.ravel() + self._mdp_sampler.gamma_single_step * maxq
            else:
                q_persisted = self._policy.Q.persisted_values(s_prime_a, absorbing)
                y = r.ravel() + self._mdp_sampler.gamma_single_step * q_persisted

        self._policy.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1

    def _step_core_standard(self, data, label, **kwargs):
        p = self._mdp.persistence
        _, _, _, r, s_prime, absorbing, sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim, p,
                                                        self._mdp_sampler.gamma_single_step)
        self.display("Mdp standard with persistence {0}".format(p))
        max_iterations = self._max_iterations // p
        self._iteration = 0
        for _ in range(max_iterations):
            self._iter_standard(sa, r, s_prime, absorbing, **kwargs)

        self._result.update_step(n_episodes=self.n_episodes)
        self._result.update_step_with_persistence(persistence=p, label=label, n_samples=r.shape[0])

    def _step_core_persisted_operator(self, data, label, **kwargs):
        p = self._mdp.persistence
        _, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)
        if len(a.shape) == 1:
            a_column = np.reshape(a, (len(a), 1))
        else:
            a_column = a
        s_prime_a = np.concatenate((s_prime, a_column), axis=1)
        self.display("Mdp Persisted Operator with persistence {0}".format(p))
        self._iteration = 0
        for _ in range(self._max_iterations):
            self._iter_persisted_operator(sa, s_prime_a, r, s_prime, absorbing, **kwargs)

        self._result.update_step(n_episodes=self.n_episodes)
        self._result.update_step_with_persistence(persistence=p, label=label, n_samples=r.shape[0])

    def step(self, n_run, callbacks=[], **kwargs):
        """
        In this version of FQI trajectory are generated here and shared in each step_core. For every mdp persistence
        step_core is invoked for standard FQI and FQI with Bellman Persisted Operator both.
        The last fitted policy is the one of the mdp_sampler, so that it will be used to generate new episode at every
        step after the first.
        """
        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)

        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        a = data[:, 0]
        b = np.amax(a)
        # FQI standard for every mdp persistence
        temp_result = []
        self._label = 'standard_'
        for mdp in self._mdps:
            self._mdp = mdp
            self._step_core_standard(data, self._label, **kwargs)
            for cb in callbacks:
                cb(self)
            temp_result.append(self._result.steps[self._step-1]['perf_disc_greedy_mean' + '_' + self._label + 'P' +
                               str(mdp.persistence)])

        import json
        with open('TempResultsStandard' + self._name + '_Step' + str(self._step - 1) + 'Run' +
                  str(n_run) + '.json', 'w') as f:
            f.write(json.dumps(temp_result))

        # FQI with Bellman Persisted Operator for every mdp persistence
        temp_result = []
        self._label = 'persisted_operator_'
        for mdp in self._mdps:
            self._mdp = mdp
            self._step_core_persisted_operator(data, self._label, **kwargs)
            for cb in callbacks:
                cb(self)
            temp_result.append(self._result.steps[self._step-1]['perf_disc_greedy_mean' + '_' + self._label + 'P' +
                               str(mdp.persistence)])

        import json
        with open('TempResultsPersistedOperator' + self._name + '_Step' + str(self._step-1) + 'Run' + str(n_run) +
                  '.json', 'w') as f:
            f.write(json.dumps(temp_result))

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                sampling_persistence=self._sampling_persistence,
                                regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policy.__class__.__name__))


def create_sa(s, a):
    if len(a.shape) == 1:
        a_column = np.reshape(a, (len(a), 1))
    else:
        a_column = a
    return np.concatenate((s, a_column), axis=1)


def get_actions(Q, actions, n_states, sa):
    if isinstance(Q, FittedQ) and actions is None:
        return sa[:, n_states:]
    else:
        return actions


class FQI_BOCP(Algorithm):

    def __init__(self, mdps, mdp_sampler, policy, actions, batch_size, max_iterations, regressor_type, persistences,
                 name, init_policy=None, verbose=False, **regressor_params):

        for i in range(len(mdps)):
            assert isinstance(mdps[i], gym.Env)
        assert isinstance(policy, Policy)
        persistences.sort()

        self._name = "FQI_BOCP"
        self._mdps = mdps
        self._mdp_sampler = mdp_sampler
        self._policies = []

        for _ in range(len(persistences)):
            self._policies.append(copy.deepcopy(policy))

        self._verbose = verbose
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._persistences = persistences
        self._regressor_type = regressor_type
        self._state_dim = mdps[0].state_dim
        self._action_dim = mdps[0].action_dim
        self._name = name

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        for i in range(len(self._policies)):
            if isinstance(mdps[i].action_space, spaces.Discrete):
                self._policies[i].Q = DiscreteFittedQ(regressor_type, self._state_dim, actions, **regressor_params)
            else:
                self._policies[i].Q = FittedQ(regressor_type, self._state_dim, self._action_dim, **regressor_params)

        self.reset()

    def _iter(self, sa, s_prime_a, r, s_prime, absorbing, mdp_index, gamma, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self._iteration % self._mdps[mdp_index].env.persistence == 0:
                actions = get_actions(self._policies[mdp_index].Q, self._actions, self._state_dim, sa)
                maxq, _ = self._policies[mdp_index].Q.max(s_prime, actions, absorbing)
                y = r.ravel() + gamma * maxq
            else:
                q_persisted = self._policies[mdp_index].Q.persisted_values(s_prime_a, absorbing)
                y = r.ravel() + gamma * q_persisted

        self._policies[mdp_index].Q.fit(sa, y.ravel(), **fit_params)
        self._iteration += 1

    def _step_core(self, n_run, callbacks, **kwargs):
        policy = self._policies[0] if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp_sampler, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        gamma = self._mdp_sampler.gamma
        mdp_index = 0
        t, _, a, r, s_prime, absorbing, sa = split_data(data, self._mdps[0].state_dim, self._mdps[0].action_dim)
        s_prime_a = create_sa(s_prime, a)
        for p in self._persistences:
            self.display("Mdp with persistence {0}".format(p))
            self._iteration = 0
            for i in range(self._max_iterations):
                self._iter(sa, s_prime_a, r, s_prime, absorbing, mdp_index, gamma, **kwargs)
            self._result.update_step(n_episodes=self.n_episodes)
            self._result.update_step_with_persistence(persistence=p, n_samples=r.shape[0])
            mdp_index += 1

    def step(self, n_run, callbacks=[], **kwargs):

        self.display("Step {0}".format(self._step))
        self._result.add_step(step=self._step)
        self._step_core(n_run, callbacks, **kwargs)
        temp_result_stage1 = []
        perf_means = []
        perf_std = []
        best_perf_idx = 0

        for i in range(len(self._persistences)):
            self._policy = self._policies[i]
            self._mdp = self._mdps[i]
            for cb in callbacks:
                cb(self)
            if 'perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence) in self._result.steps[0]:
                perf_curr = self._result.steps[self._step-1]['perf_disc_greedy_mean'+'_P'+str(self._mdp.env.persistence)]
                perf_means.append(perf_curr)
            if 'perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence) in self._result.steps[0]:
                perf_std.append(self._result.steps[self._step-1]['perf_disc_greedy_std'+'_P'+str(self._mdp.env.persistence)])
            if i == 0:
                best_perf = perf_curr
            else:
                if perf_curr > best_perf:
                    best_perf = perf_curr
                    best_perf_idx = i
        with open('TempResultsStage1' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json', 'w') as f:
            temp_result_stage1.append(perf_means)
            temp_result_stage1.append(perf_std)
            f.write(json.dumps(temp_result_stage1))

        self._policy = self._policies[best_perf_idx]
        temp_result_stage2 = []
        perf_means = []
        perf_std = []

        for i in range(len(self._persistences)):
            self._mdp = self._mdps[i]
            for cb in callbacks:
                cb(self)
            if 'perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence) in self._result.steps[0]:
                perf_means.append(
                    self._result.steps[self._step - 1]['perf_disc_greedy_mean' + '_P' + str(self._mdp.env.persistence)])
            if 'perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence) in self._result.steps[0]:
                perf_std.append(
                    self._result.steps[self._step - 1]['perf_disc_greedy_std' + '_P' + str(self._mdp.env.persistence)])
        with open('TempResultsStage2' + self._name + '_Step' + str(self._step) + 'Run' + str(n_run) + '.json',
                  'w') as f:
            temp_result_stage2.append(perf_means)
            temp_result_stage2.append(perf_std)
            f.write(json.dumps(temp_result_stage2))

        self._step += 1

        return self._result

    def reset(self):

        super().reset()

        from itertools import repeat
        self._data = []
        self._iteration = 0

        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                persistences=self._persistences, regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policies[0].__class__.__name__))