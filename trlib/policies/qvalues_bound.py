import json
import numpy as np
from trlib.algorithms.algorithm_handler import FQI_SSEP_Handler


def calculate_bound(fqi_ss_ep_handler, file_name, first_sample=False):
    assert isinstance(fqi_ss_ep_handler, FQI_SSEP_Handler)
    q_k_list = []
    errors_list = []
    differences_list = []
    bounds_list = []
    gamma = fqi_ss_ep_handler.get_gamma()
    persistences = fqi_ss_ep_handler.get_persistences()
    number_of_persistences = len(persistences)

    for i in range(number_of_persistences):
        t, sa, q_functions, targets, _ = fqi_ss_ep_handler.get_q_for_bound(i)
        if first_sample:
            sa = sa[t == 0]
            targets = [targets[i][t == 0] for i in range(len(targets))]
        q_functions = [q_functions[j].values(sa) for j in range(len(q_functions))]
        q_k, errors, difference, bound = _get_bound(q_functions, targets, persistences[i], gamma)
        q_k_list.append(q_k)
        errors_list.append(errors)
        differences_list.append(difference)
        bounds_list.append(bound)

    bounds = [q_k_list, errors_list, differences_list, bounds_list]
    with open('Q_bounds' + file_name + '.json', 'w') as f:
        f.write(json.dumps(bounds))


def _get_bound(q_functions, targets, persistence, gamma):
    q_k = np.average(q_functions[0])
    difference = np.average(np.abs(q_functions[-1] - q_functions[0]))
    errors = np.sum(np.array([np.average(np.abs(targets[i] - q_functions[i+1])) * (gamma ** (persistence - 1 - i))
                              for i in range(persistence)]))

    bound = q_k - (1 / (1 - gamma ** persistence)) * (errors + difference)

    return q_k.item(), errors.item(), difference.item(), bound.item()


def _get_bound_2(q_functions, targets, persistence, gamma):
    q_k = np.average(q_functions[0])
    difference = np.average(np.abs(targets[-1] - q_functions[0]))
    errors = np.sum(np.array([np.average(np.abs(targets[i] - q_functions[i+1])) * (gamma ** (persistence - 1 - i))
                              for i in range(persistence - 1)]))

    bound = q_k - (1 / (1 - gamma ** persistence)) * (errors + difference)
    return q_k.item(), errors.item(), difference.item(), bound.item()


def _get_bound_max_norm(q_functions, targets, persistence, gamma):
    q_k = np.average(q_functions[0])
    difference = np.max(np.abs(q_functions[-1] - q_functions[0]))
    errors = np.sum(np.array([np.max(np.abs(targets[i] - q_functions[i+1])) * (gamma ** (persistence - 1 - i))
                              for i in range(persistence)]))

    bound = q_k - (1 / (1 - gamma ** persistence)) * (errors + difference)

    return q_k.item(), errors.item(), difference.item(), bound.item()


def calculate_bound_splitted_dataset(fqi_ss_ep_handler, file_name):
    assert isinstance(fqi_ss_ep_handler, FQI_SSEP_Handler)
    q_k = []
    differences = []
    bounds = []
    gamma = fqi_ss_ep_handler.get_gamma()
    persistences = fqi_ss_ep_handler.get_persistences()
    number_of_persistences = len(persistences)
    sa, q_functions, targets = fqi_ss_ep_handler.get_q_for_bound_2()
    q_functions = [q_functions[i].values(sa) for i in range(len(q_functions))]
    for i in range(number_of_persistences):
        q_k.append(np.average(q_functions[i]))
        differences.append(np.average(np.abs(targets[i] - q_functions[i])))
        bounds.append(q_k[i] - (differences[i] / (1 - gamma)))

    bounds = [q_k, differences, bounds]
    with open('Q_bounds_method_2_' + file_name + '.json', 'w') as f:
        f.write(json.dumps(bounds))


def calculate_bound_greedy_trajectories(fqi_ss_ep_handler, file_name, first_sample=False):
    assert isinstance(fqi_ss_ep_handler, FQI_SSEP_Handler)
    q_k_list = []
    br_term_list = []
    bounds_list = []
    gamma = fqi_ss_ep_handler.get_gamma()
    persistences = fqi_ss_ep_handler.get_persistences()
    number_of_persistences = len(persistences)

    for i in range(number_of_persistences):
        t, sa, q_functions, targets, actions = fqi_ss_ep_handler.get_q_for_bound(i)

        weights = calculate_weights_trajectory(t, sa, actions, q_functions[i], gamma ** persistences[i])
        q_functions = [q_functions[j].values(sa) for j in range(len(q_functions))]
        q_k = np.average(q_functions[0])
        difference = np.abs(q_functions[-1] - q_functions[0])
        errors = np.sum(np.array([np.abs(targets[j] - q_functions[j + 1]) * (gamma ** (persistences[i] - 1 - j))
                                  for j in range(persistences[i])]), axis=0)

        unique, counts = np.unique(t, return_counts=True)
        dictionary = dict(zip(unique, counts))
        n_trajectories = dictionary[0]  # count number of trajectories

        br_term = (1 / (1 - gamma ** persistences[i])) * np.average(weights * (errors + difference)) / n_trajectories
        bound = q_k - br_term

        q_k_list.append(q_k.item())
        br_term_list.append(br_term.item())
        bounds_list.append(bound.item())

    bounds = [q_k_list, br_term_list, bounds_list]
    with open('Q_bounds' + file_name + '.json', 'w') as f:
        f.write(json.dumps(bounds))


def calculate_weights_trajectory(t, sa, actions, q_function, gamma):
    weigths = np.zeros(len(sa))
    still_greedy = True
    greedy_idx = calculate_greedy_idx(sa, actions, q_function)
    current_idx = 0
    for timestep in t:
        if greedy_idx[current_idx] and timestep == 0:
            still_greedy = True
        if not greedy_idx[current_idx]:
            still_greedy = False
        if still_greedy:
            weigths[current_idx] = (gamma * len(actions)) ** timestep
        current_idx += 1
    return weigths


def calculate_greedy_idx(sa, actions, q_function):
    if isinstance(actions[0], int):
        action_dim = 1
    if isinstance(actions[0], list):
        action_dim = len(actions[0])
    s = sa[:, :-action_dim]
    values = q_function.values(sa)
    maxq, _ = q_function.max(s, actions)
    return [values[i] == maxq[i] for i in range(len(values))]
