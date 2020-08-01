import json
import numpy as np
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.doubleq_policies import EpsilonGreedyDoubleQ
from trlib.policies.qfunction import ZeroQ
from trlib.environments.wrappers.trlib_wrapper import TrlibWrapper
from trlib.environments.wrappers.persistent_action_wrapper import PersistentActionWrapper
from trlib.environments.wrappers.persistent_action_wrapper_ss import PersistentActionWrapperSharedSamples
from trlib.algorithms.callbacks import get_callback_list_entry
from trlib.algorithms.reinforcement.fqi import FQI, FQI_SS, FQI_SS_EP, Double_FQI_SS_EP, FQI_P1_VS_1P, FQI_TOTALLY_SS,\
    FQI_BOCP, FQI_TrainTest, FQI_SS_EP_1P_Control
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.experiment import RepeatExperiment


class AlgorithmHandler(object):

    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, batch_size=200, n_steps=1, n_runs=1,
                 n_jobs=1, file_name='', pi=None, regressor_type=None, cb_eval=None, generative_setting=False,
                 n_jobs_regressors=1, initial_states=None, **regressor_params):
        self._mdp = mdp
        self._gamma = gamma
        self._max_episode_steps = max_episode_steps
        self._actions = actions
        self._max_iterations = max_iterations
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._n_runs = n_runs
        self._n_jobs = n_jobs
        self._fit_params = {}
        self._file_name = file_name
        self._file_name += '_batch{}'.format(batch_size)
        self._generative_setting = generative_setting
        self._mdp.generative_setting = generative_setting

        if pi is None:
            self._pi = EpsilonGreedy(actions, ZeroQ(), 0.1)
        else:
            self._pi = pi
        if regressor_type is None:
            self._regressor_type = ExtraTreesRegressor
        else:
            self._regressor_type = regressor_type
        if not regressor_params:
            self._regressor_params = {'n_estimators': 100,
                                      'criterion': 'mse',
                                      'min_samples_split': 5,
                                      'min_samples_leaf': 2}
        else:
            self._regressor_params = regressor_params
        self._regressor_params.update({"n_jobs": n_jobs_regressors})
        if cb_eval is None:
            cb_eval = get_callback_list_entry("eval_greedy_policy_callback",
                                              field_name="perf_disc_greedy",
                                              criterion='discounted',
                                              n_episodes=10)
        self._initial_states = initial_states
        self._callback_list = [cb_eval]
        self._pre_callback_list = []

    def handle_algorithm(self):
        raise NotImplementedError

    def get_algorithm(self):
        return self._algorithm

    def get_q_values(self):
        raise NotImplementedError

    def get_q_average(self):
        raise NotImplementedError

    def _valid_states(self, states):
        return all(self._mdp.observation_space.contains(elem) for elem in states)

    def _generate_sa(self, states):
        sa = []
        for action in self._actions:
            a_column = np.ones((states.shape[0], 1)) * action
            s_action = np.hstack((states, a_column))
            sa.append(s_action.tolist())
        return np.concatenate(sa)


class FQI_Handler(AlgorithmHandler):

    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistence=1, batch_size=200, n_steps=1,
                 n_runs=1, n_jobs=1, file_name='', pi=None, regressor_type=None, cb_eval=None, generative_setting=False,
                 **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, generative_setting, **regressor_params)
        self._persistence = persistence
        self._max_iterations = max_iterations // persistence

    def handle_algorithm(self):
        mdp = PersistentActionWrapper(self._mdp,
                                      persistence=self._persistence,
                                      gamma=self._gamma,
                                      max_episode_steps=self._max_episode_steps)
        mdp = TrlibWrapper(mdp)

        self._algorithm = FQI(mdp,
                              self._pi,
                              verbose=True,
                              actions=self._actions,
                              batch_size=self._batch_size,
                              max_iterations=self._max_iterations,
                              regressor_type=self._regressor_type,
                              generative_setting=self._generative_setting,
                              **self._regressor_params)

        experiment = RepeatExperiment("FQI",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)

        result = experiment.run(self._n_jobs)
        name = self._file_name + '_P{}.json'.format(self._persistence)
        result.save_json(name)


class FQI_MultiP_Handler(AlgorithmHandler):

    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1], batch_size=200, n_steps=1,
                 n_runs=1, n_jobs=1, file_name='', pi=None, regressor_type=None, get_q_differences=False, cb_eval=None,
                 **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._max_iterations = max_iterations
        self._get_q_differences = get_q_differences

    def handle_algorithm(self):
        final_perfs = []
        for p in self._persistences:
            mdp = PersistentActionWrapper(self._mdp,
                                          persistence=p,
                                          gamma=self._gamma,
                                          max_episode_steps=self._max_episode_steps)
            mdp = TrlibWrapper(mdp)
            max_iterations = self._max_iterations // p
            algorithm = FQI(mdp,
                            self._pi,
                            verbose=True,
                            actions=self._actions,
                            batch_size=self._batch_size,
                            max_iterations=max_iterations,
                            regressor_type=self._regressor_type,
                            get_q_differences=self._get_q_differences,
                            **self._regressor_params)

            experiment = RepeatExperiment("FQI",
                                          algorithm,
                                          n_steps=self._n_steps,
                                          n_runs=self._n_runs,
                                          callback_list=self._callback_list,
                                          pre_callback_list=self._pre_callback_list,
                                          **self._fit_params)

            result = experiment.run(self._n_jobs)
            for i in range(len(result.runs)):
                final_perfs.append(result.runs[i]['steps'][self._n_steps - 1]['perf_disc_greedy_mean'])

            name = self._file_name + '_P{}.json'.format(p)
            result.save_json(name)

        with open('MultiP_{}_P{}.txt'.format(self._file_name, self._persistences), 'w') as f:
            f.write(json.dumps(final_perfs))


class FQI_SS_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1], batch_size=200, n_steps=1,
                 n_runs=1, n_jobs=1, file_name='', pi=None, regressor_type=None, cb_eval=None, **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._mdps = []
        self._max_iterations = []
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])
            self._max_iterations.append(max_iterations // int(persistences[i]))

        self._mdp_ss = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=self._persistences[-1],
                                                            gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_ss = TrlibWrapper(self._mdp_ss)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        algorithm = FQI_SS(self._mdps,
                           self._pi,
                           mdp_sampler=self._mdp_ss,
                           verbose=True,
                           persistences=self._persistences,
                           actions=self._actions,
                           batch_size=self._batch_size,
                           max_iterations=self._max_iterations,
                           regressor_type=self._regressor_type,
                           name=file_name_final,
                           **self._regressor_params)

        experiment = RepeatExperiment("FQI_SS",
                                      algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_FQISS.json'.format(file_name_final)
        result.save_json(name)
        return result


class FQI_SSEP_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1],
                 sampling_persistence=1, batch_size=200, n_steps=1, n_runs=1, n_jobs=1, file_name='', pi=None,
                 save_perfs_and_q=False, save_step=1, regressor_type=None, cb_eval=None, generative_setting=False,
                 init_policy=None, alternative_policy=None, initial_states=None, n_jobs_regressors=1,
                 **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, generative_setting,  n_jobs_regressors, initial_states,
                         **regressor_params)

        self._persistences = persistences
        self._mdps = []
        self._max_iterations = max_iterations
        self._save_perfs_and_q = save_perfs_and_q
        self._save_step = save_step
        self._init_policy = init_policy
        self._alternative_policy = alternative_policy
        self.n_jobs_regressors = n_jobs_regressors
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])
        self._mdp_ss = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=sampling_persistence,
                                                            gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_ss = TrlibWrapper(self._mdp_ss)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        if self._generative_setting:
            file_name_final += '_gener'
        self._algorithm = FQI_SS_EP(self._mdps,
                                    self._mdp_ss,
                                    self._pi,
                                    verbose=True,
                                    persistences=self._persistences,
                                    actions=self._actions,
                                    batch_size=self._batch_size,
                                    max_iterations=self._max_iterations,
                                    regressor_type=self._regressor_type,
                                    name=file_name_final,
                                    save_perfs_and_q=self._save_perfs_and_q,
                                    save_step=self._save_step,
                                    generative_setting=self._generative_setting,
                                    init_policy=self._init_policy,
                                    alternative_policy=self._alternative_policy,
                                    initial_states=self._initial_states,
                                    n_jobs_regressors=self.n_jobs_regressors,
                                    **self._regressor_params)

        experiment = RepeatExperiment("FQI_FQISSEP",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_FQISSEP.json'.format(file_name_final)
        result.save_json(name)
        return result

    def get_q_values(self, states):
        assert self._valid_states(states)
        sa = self._generate_sa(states)
        Q_functions = [self._algorithm._policies[i].Q for i in range(len(self._algorithm._policies))]
        Q_values = [Q_functions[i].values(sa) for i in range(len(self._algorithm._policies))]
        Q_values = [Q_values[i].tolist() for i in range(len(Q_values))]
        length = len(Q_values[0]) // len(self._actions)
        for i in range(len(Q_values)):
            Q_values[i] = [Q_values[i][j:j + length] for j in range(0, len(Q_values[i]), length)]
        name = 'FQI_FQISSEP_{}_Q_values.json'.format(self._file_name)
        with open(name, "w") as file:
            json.dump(Q_values, file)
        return Q_values

    def get_q_average(self):
        data = np.concatenate(self._algorithm._data)
        a_idx = 1 + self._mdp_ss.state_dim
        r_idx = a_idx + self._mdp_ss.action_dim
        sa = data[:, 1:r_idx]
        Q_functions = [self._algorithm._policies[i].Q for i in range(len(self._algorithm._policies))]
        q_avgs = [np.average(Q_functions[i].values(sa)) for i in range(len(self._algorithm._policies))]

        import json
        with open('Q_averages' + self._file_name + '.json', 'w') as f:
            f.write(json.dumps(q_avgs))

        return q_avgs

    def get_persistences(self):
        return self._persistences

    def get_gamma(self):
        return self._gamma

    def get_q_for_bound(self, policy_idx):
        return self._algorithm.get_q_for_bound(policy_idx, **self._fit_params)

    def get_q_for_bound_2(self):
        return self._algorithm.get_q_for_bound_2()


class Double_FQI_SSEP_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, batch_size=200, persistences=[1],
                 sampling_persistence=1, n_steps=1, n_runs=1, n_jobs=1, file_name='',
                 regressor_type=None, save_perfs_and_q=False, cb_eval=None, **regressor_params):
        policy = EpsilonGreedyDoubleQ(actions, 0, ZeroQ())
        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, policy, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._mdps = []
        self._max_iterations = max_iterations
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])
        assert sampling_persistence in self._persistences
        self._mdp_ss = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=sampling_persistence,
                                                            gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_ss = TrlibWrapper(self._mdp_ss)
        cb_eval = get_callback_list_entry("eval_greedy_policy_doubleq_callback",
                                          field_name="perf_disc_greedy",
                                          criterion='discounted',
                                          n_episodes=10)
        self._callback_list = [cb_eval]
        self._save_perfs_and_q = save_perfs_and_q

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        self._algorithm = Double_FQI_SS_EP(self._mdps,
                                           self._mdp_ss,
                                           policy=self._pi,
                                           verbose=True,
                                           persistences=self._persistences,
                                           actions=self._actions,
                                           batch_size=self._batch_size,
                                           max_iterations=self._max_iterations,
                                           regressor_type=self._regressor_type,
                                           name=file_name_final,
                                           save_perfs_and_q=self._save_perfs_and_q,
                                           **self._regressor_params)

        experiment = RepeatExperiment("Double_FQI_FQISSEP",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_Double_FQISSEP.json'.format(file_name_final)
        result.save_json(name)
        return result

    def get_persistences(self):
        return self._persistences

    def get_data_and_qs(self):
        return self._algorithm.get_data_and_qs()


class FQI_1P_VS_P1_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistence=1, batch_size=200, n_steps=1,
                 n_runs=1, n_jobs=1, file_name='', pi=None, regressor_type=None, cb_eval=None, **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)
        self._persistence = persistence
        self._mdp_sampler = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=persistence, gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_sampler = TrlibWrapper(self._mdp_sampler)
        self._mdp_persisted = PersistentActionWrapper(mdp, persistence=persistence, gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_persisted = TrlibWrapper(self._mdp_persisted)

    def handle_algorithm(self):
        file_name_final = '{}_P{}'.format(self._file_name, int(self._persistence))
        algorithm = FQI_P1_VS_1P(mdp_sampler=self._mdp_sampler,
                                 mdp_persisted=self._mdp_persisted,
                                 policy=self._pi,
                                 verbose=True,
                                 actions=self._actions,
                                 batch_size=self._batch_size,
                                 max_iterations=self._max_iterations,
                                 regressor_type=self._regressor_type,
                                 name=self._file_name,
                                 **self._regressor_params)

        experiment = RepeatExperiment("FQI_P1_VS_1P",
                                      algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_FQI_P1_VS_1P.json'.format(file_name_final)
        result.save_json(name)
        return result


class FQI_TSS_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=1, batch_size=200,
                 n_steps=1, n_runs=1, n_jobs=1, file_name='', pi=None, regressor_type=None, cb_eval=None,
                 **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._mdps = []
        self._max_iterations = max_iterations
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])

        self._mdp_sampler = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=self._persistences[-1],
                                                                 gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_sampler = TrlibWrapper(self._mdp_sampler)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        algorithm = FQI_TOTALLY_SS(mdps=self._mdps,
                                   policy=self._pi,
                                   mdp_sampler=self._mdp_sampler,
                                   verbose=True,
                                   actions=self._actions,
                                   batch_size=self._batch_size,
                                   max_iterations=self._max_iterations,
                                   regressor_type=self._regressor_type,
                                   name=file_name_final,
                                   **self._regressor_params)

        experiment = RepeatExperiment("FQI_TSS",
                                      algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_FQI_TSS.json'.format(file_name_final)
        result.save_json(name)
        return result


class FQI_TrainTest_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1],
                 training_persistence=1, batch_size=200, n_steps=1, n_runs=1, n_jobs=1, file_name='', pi=None,
                 regressor_type=None, cb_eval=None, **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._mdp_test_list = []
        self._max_iterations = max_iterations
        for i in range(len(persistences)):
            self._mdp_test_list.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                               max_episode_steps=max_episode_steps))
            self._mdp_test_list[i] = TrlibWrapper(self._mdp_test_list[i])
        assert training_persistence in self._persistences
        self._mdp_train = PersistentActionWrapper(mdp, persistence=training_persistence,
                                                  gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_train = TrlibWrapper(self._mdp_train)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        self._algorithm = FQI_TrainTest(mdp_test_list=self._mdp_test_list,
                                        mdp_train=self._mdp_train,
                                        policy=self._pi,
                                        verbose=True,
                                        actions=self._actions,
                                        batch_size=self._batch_size,
                                        max_iterations=self._max_iterations,
                                        regressor_type=self._regressor_type,
                                        **self._regressor_params)

        experiment = RepeatExperiment("FQI_TrainTest",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}_FQI_TrainTest.json'.format(file_name_final)
        result.save_json(name)
        return result


class FQI_BOCP_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1],
                 sampling_persistence=1, batch_size=200, n_steps=1, n_runs=1, n_jobs=1, file_name='', pi=None,
                 save_perfs_and_q=False, regressor_type=None, cb_eval=None, **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)
        self._persistences = persistences
        self._mdps = []
        self._max_iterations = max_iterations
        self._save_perfs_and_q = save_perfs_and_q
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])
        self._mdp_ss = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=sampling_persistence,
                                                            gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_ss = TrlibWrapper(self._mdp_ss)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        self._algorithm = FQI_BOCP(self._mdps,
                                   self._mdp_ss,
                                   self._pi,
                                   verbose=True,
                                   persistences=self._persistences,
                                   actions=self._actions,
                                   batch_size=self._batch_size,
                                   max_iterations=self._max_iterations,
                                   regressor_type=self._regressor_type,
                                   name=file_name_final,
                                   **self._regressor_params)

        experiment = RepeatExperiment("FQI_BOCP",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}FQI_BOCP.json'.format(file_name_final)
        result.save_json(name)
        return result


class FQI_SS_EP_1P_Control_Handler(AlgorithmHandler):
    def __init__(self, mdp, gamma, max_episode_steps, actions, max_iterations, persistences=[1],
                 sampling_persistence=1, batch_size=200, n_steps=1, n_runs=1, n_jobs=1, file_name='', pi=None,
                 regressor_type=None, cb_eval=None,  **regressor_params):

        super().__init__(mdp, gamma, max_episode_steps, actions, max_iterations, batch_size, n_steps, n_runs, n_jobs,
                         file_name, pi, regressor_type, cb_eval, **regressor_params)

        self._persistences = persistences
        self._mdps = []
        self._max_iterations = max_iterations
        for i in range(len(persistences)):
            self._mdps.append(PersistentActionWrapper(mdp, persistence=self._persistences[i], gamma=gamma,
                                                      max_episode_steps=max_episode_steps))
            self._mdps[i] = TrlibWrapper(self._mdps[i])
        self._mdp_ss = PersistentActionWrapperSharedSamples(mdp, sampling_persistence=sampling_persistence,
                                                            gamma=gamma, max_episode_steps=max_episode_steps)
        self._mdp_ss = TrlibWrapper(self._mdp_ss)

    def handle_algorithm(self):
        file_name_final = '{}_Pn{}'.format(self._file_name, len(self._persistences))
        self._algorithm = FQI_SS_EP_1P_Control(self._mdps,
                                    self._mdp_ss,
                                    self._pi,
                                    verbose=True,
                                    persistences=self._persistences,
                                    actions=self._actions,
                                    batch_size=self._batch_size,
                                    max_iterations=self._max_iterations,
                                    regressor_type=self._regressor_type,
                                    name=file_name_final,
                                    **self._regressor_params)

        experiment = RepeatExperiment("FQI_FQISSEP1PC",
                                      self._algorithm,
                                      n_steps=self._n_steps,
                                      n_runs=self._n_runs,
                                      callback_list=self._callback_list,
                                      pre_callback_list=self._pre_callback_list,
                                      **self._fit_params)
        result = experiment.run(self._n_jobs)
        name = '{}FQI_FQISSEP1PC.json'.format(file_name_final)
        result.save_json(name)
        return result