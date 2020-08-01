import json
import numpy as np
from trlib.utilities.create_grid import create_grid
from trlib.algorithms.reinforcement.fqi import FQI, FQI_SS_EP

class QValuesProducer(object):

    def __init__(self, mdp, actions, algorithm, file_name):
        self._mdp = mdp
        self._actions = actions
        self._algorithm = algorithm
        self._file_name = file_name

    def _valid_states(self, states):
        return all(self._mdp.observation_space.contains(elem) for elem in states)

    def _generate_sa(self, states):
        sa = []
        for action in self._actions:
            a_column = np.ones((states.shape[0], 1)) * action
            s_action = np.hstack((states, a_column))
            sa.append(s_action.tolist())
        return np.concatenate(sa)

    def _create_states_grid(self, high_vals, low_vals, levels):
        return np.array(create_grid(high_vals, low_vals, levels))

    def _create_states_trajectory(self):
        a_idx = 1 + self._algorithm._mdp.state_dim
        return self._algorithm._data[0][:,1:a_idx]

    def _get_q_values(self, states):
        assert self._valid_states(states)
        sa = self._generate_sa(states)

        if isinstance(self._algorithm, FQI_SS_EP):
            q_values = self._q_from_fqi_ss_ep(sa)
        if isinstance(self._algorithm, FQI):
            q_values = self._q_from_fqi(sa)

        # generate list for every action
        length = len(q_values[0]) // len(self._actions)
        for i in range(len(q_values)):
            q_values[i] = [q_values[i][j:j + length] for j in range(0, len(q_values[i]), length)]

        # save file
        name = 'Q_values_{}.json'.format(self._file_name)
        with open(name, "w") as file:
            json.dump(q_values, file)

        return q_values

    def _q_from_fqi_ss_ep(self, sa):
        # generate list for every policy
        q_functions = [self._algorithm._policies[i].Q for i in range(len(self._algorithm._policies))]
        return [q_functions[i].values(sa).tolist() for i in range(len(self._algorithm._policies))]

    def _q_from_fqi(self, sa):
        return [self._algorithm._policy.Q.values(sa).tolist()]

    def get_q_values_from_grid(self, high_vals, low_vals, levels):
        states = self._create_states_grid(high_vals, low_vals, levels)
        return self._get_q_values(states)

    def get_q_values_from_trajectory(self):
        states = self._create_states_trajectory()
        return self._get_q_values(states)

    def get_q_average(self):
        data = np.concatenate(self._algorithm._data)
        a_idx = 1 + self._mdp.state_dim
        r_idx = a_idx + self._mdp.action_dim
        sa = data[:, 1:r_idx]
        Q_functions = [self._algorithm._policies[i].Q for i in range(len(self._algorithm._policies))]
        q_avgs = [np.average(Q_functions[i].values(sa)) for i in range(len(self._algorithm._policies))]

        import json
        with open('Q_averages' + self._file_name + '.json', 'w') as f:
            f.write(json.dumps(q_avgs))

        return q_avgs
