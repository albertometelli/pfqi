import numpy as np
from joblib import Parallel, delayed

def generate_episodes(mdp, policy, n_episodes = 1, n_threads = 1, use_generative_setting=False, alternative_policy=None):
    """
    Generates episodes in a given mdp using a given policy
    
    Parameters
    ----------
    mdp: the environment to use
    policy: the policy to use
    n_episodes: the number of episodes to generate
    n_threads: the number of threads to use
    
    Returns
    -------
    A matrix where each row corresponds to a single sample (t,s,a,r,s',absorbing)
    """
    if alternative_policy is not None:
        if n_threads == 1:
            episodes = [_single_episode(mdp, alternative_policy) if i % 2 == 0 else _single_episode(mdp, policy)
                        for i in range(n_episodes)]
        elif n_threads > 1:
            episodes = Parallel(n_jobs = n_threads)(delayed(_single_episode)(mdp, policy) if i % 2 == 0
                                                    else delayed(_single_episode)(mdp, alternative_policy)
                                                    for i in range(n_episodes))
    elif use_generative_setting:
        if n_threads == 1:
            episodes = [mdp.unwrapped.get_generative_episode(policy._actions) for _ in range(n_episodes)]
        elif n_threads > 1:
            episodes = Parallel(n_jobs = n_threads)(delayed(mdp.unwrapped.get_generative_episode)
                                                   (policy._actions) for _ in range(n_episodes))
    else:
        if n_threads == 1:
            episodes = [_single_episode(mdp, policy) for _ in range(n_episodes)]
        elif n_threads > 1:
            episodes = Parallel(n_jobs = n_threads)(delayed(_single_episode)(mdp, policy) for _ in range(n_episodes))
        
    return np.concatenate(episodes)


def generate_episodes_imposed_actions(mdp, actions, starting_states, persistence):
    action_index = 0
    episodes = []
    i = 0
    while action_index < len(actions):
        new_episode, action_index = _single_episode_imposed_actions_with_persistence(mdp, starting_states[i,:], actions,
                                                                                     action_index, persistence)
        episodes.append(new_episode)
        i += 1
    return np.concatenate(episodes)


def _single_episode(mdp, policy):
    
    episode = np.zeros((mdp.horizon, 1 + mdp.state_dim + mdp.action_dim + 1 + mdp.state_dim + 1))
    a_idx = 1 + mdp.state_dim
    r_idx = a_idx + mdp.action_dim
    s_idx = r_idx + 1
    
    s = mdp.reset()
    t = 0

    while t < mdp.horizon:
    
        episode[t, 0] = t
        episode[t, 1:a_idx] = s

        a = policy.sample_action(s)
        s,r,done,info = mdp.step(a)
        if 'new_action' in info.keys():
            if info['new_action'] is True:
                prev_a = a
            else:
                a = prev_a
        episode[t,a_idx:r_idx] = a
        episode[t,r_idx] = r
        episode[t,s_idx:-1] = s
        episode[t,-1] = 1 if done else 0
        
        t += 1
        if done:
            break
    
    return episode[0:t,:]


def _single_episode_imposed_actions_with_persistence(mdp, starting_state, actions, action_index, persistence):
    episode = np.zeros((mdp.horizon, 1 + mdp.state_dim + mdp.action_dim + 1 + mdp.state_dim + 1))
    a_idx = 1 + mdp.state_dim
    r_idx = a_idx + mdp.action_dim
    s_idx = r_idx + 1

    mdp.reset()
    mdp.unwrapped.state = starting_state
    s = starting_state
    t = 0
    p = 0

    while t < mdp.horizon:

        episode[t, 0] = t
        episode[t, 1:a_idx] = s

        if p >= persistence:
            action_index += 1
            p = 0

        a = np.array(int(actions[action_index]))
        s, r, done, info = mdp.step(a)

        episode[t, a_idx:r_idx] = a
        episode[t, r_idx] = r
        episode[t, s_idx:-1] = s
        episode[t, -1] = 1 if done else 0

        t += 1
        p += 1
        if done:
            action_index += 1
            break

    return episode[0:t, :], action_index


def split_data(data, state_dim, action_dim, persistence=None, gamma=None):
    """
    Splits the data into (t,s,a,r,s_prime,absorbing,sa)
    """

    assert data.shape[1] == 3 + 2 * state_dim + action_dim

    a_idx = 1 + state_dim
    r_idx = a_idx + action_dim
    s_idx = r_idx + 1

    if persistence is None:
        idx = data[:, 0] * 0 == 0
        idx2 = idx
    else:
        idx = data[:, 0] % persistence == 0
        idx2 = []
        for i in range(len(idx)-1):
            idx2.append(idx[i+1])
        idx2.append(True)

    t = data[idx, 0]
    s = data[idx, 1:a_idx].squeeze()
    a = data[idx, a_idx:r_idx].squeeze()
    r = calculate_rewards(data, idx, r_idx, gamma)
    s_prime = data[idx2, s_idx:-1].squeeze()
    absorbing = data[idx2, -1]
    sa = data[idx, 1:r_idx]

    return t, s, a, r, s_prime, absorbing, sa


def calculate_rewards(data, idx, r_idx, gamma = 1):
    if gamma is None:
        gamma = 1
    r = data[:, r_idx]
    to_return = []
    new_cum_rew = 0
    j = 0
    for i in range(len(r)):
        new_cum_rew += r[i] * gamma ** j
        j += 1
        if i+1 == len(r):
            to_return.append(new_cum_rew)
        else:
            if idx[i+1]:
                to_return.append(new_cum_rew)
                j = 0
                new_cum_rew = 0
    return np.array(to_return)
