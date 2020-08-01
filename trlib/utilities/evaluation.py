import numpy as np
from joblib import Parallel, delayed

def render_policy(mdp, policy, n_episodes = 1, initial_states = None):
    if (initial_states is None or type(initial_states) is np.ndarray):
        [_single_render(mdp, policy, initial_states) for _ in range(n_episodes)]
    elif type(initial_states) is list:
        [_single_render(mdp, policy, initial_states) for _ in range(n_episodes)]

def _single_render(mdp, policy, initial_state):

    score = 0

    s = mdp.reset(initial_state)
    t = 0
    act=[]
    mdp.render()

    while t < mdp.horizon:
        a = policy.sample_action(s)
        act.append(a)
        s,r,done,_ = mdp.step(a)
        mdp.render()
        t += 1
        if done:
            break

def evaluate_policy(mdp, policy, criterion = 'discounted', n_episodes = 1,
                    initial_states = None, n_threads = 1, mul_env=False):
    """
    Evaluates a policy on a given MDP.
    
    Parameters
    ----------
    mdp: the environment to use in the evaluation
    policy: the policy to evaluate
    criterion: either 'discounted' or 'average'
    n_episodes: the number of episodes to generate in the evaluation
    initial_states: either None (i), a numpy array (ii), or a list of numpy arrays (iii)
      - (i) initial states are drawn from the MDP distribution
      - (ii) the given array is used as initial state for all episodes
      - (iii) n_episodes is ignored and the episodes are defined by their initial states
    n_threads: the number of threads to use in the evaluation
    
    Returns
    -------
    The mean of the scores and its confidence interval.
    """
    #n_days = 13
    assert criterion == 'average' or criterion == 'discounted'

    from contextlib import contextmanager
    import time
    @contextmanager
    def timed(msg):
            print(msg)
            tstart = time.time()
            yield
            print("done in %.3f seconds"%(time.time() - tstart))

    if mul_env:
        with timed("evaluation"):
            scores = _multiple_eval(mdp, policy, criterion, initial_states, num_env=n_episodes)
    elif n_threads == 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = [_single_eval(mdp, policy, criterion, initial_states) for _ in range(n_episodes)]
    elif n_threads > 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, initial_states) for _ in range(n_episodes))
    elif n_threads == 1 and type(initial_states) is list:
        scores = [_single_eval(mdp, policy, criterion, init_state) for init_state in initial_states]
    elif n_threads > 1 and type(initial_states) is list:
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, init_state) for init_state in initial_states)
    
    n_episodes = len(initial_states) if type(initial_states) is list else n_episodes
    
    scores, ts, actions = zip(*scores)
    
    scores = np.array(scores)
    actions = np.array(actions)

    scores_mean, scores_std = np.mean(scores), np.std(scores)/ np.sqrt(n_episodes)
    print(scores_mean, scores_std)

    return np.mean(scores), np.std(scores) / np.sqrt(n_episodes), ts

def _single_eval(mdp, policy, criterion, initial_state):

    score = 0
    gamma = mdp.gamma if criterion == "discounted" else 1

    s = mdp.reset(initial_state)
    t = 0
    act=[]
    while t < mdp.horizon:
        a = policy.sample_action(s)
        act.append(a)
        s,r,done,_ = mdp.step(a)
        score += r * gamma**t
        t += 1
        if done:
            break

    return score if criterion == "discounted" else score / t, t, act

def _multiple_eval(mdp, policy, criterion, initial_state, num_env=1):

    import copy
    scores = np.zeros(num_env)
    gamma = mdp.gamma if criterion == "discounted" else 1

    mdps = [copy.deepcopy(mdp) for _ in range(num_env)]
    if initial_state is None:
        states = np.array([m.reset(initial_state) for m in mdps])
    else:  # hardcoded for run_trading 
        states = np.array([m.reset(day_indexes=np.array([initial_state[i]])) for i, m in enumerate(mdps)])
    t = 0
    act=[]

    from contextlib import contextmanager
    import time
    @contextmanager
    def timed(msg):
        print(msg)
        tstart = time.time()
        yield
        print("done in %.3f seconds"%(time.time() - tstart))

    # disable multiprocessing in the regressors, if set
    n_jobs = policy.Q._regressors[0].n_jobs
    for i in policy.Q._regressors:
        policy.Q._regressors[i].set_params(n_jobs=1)

    while t < mdp.horizon:
        a = policy.sample_multiple_actions(states)
        act.append(a)
        states = []
        for i, m in enumerate(mdps):
            s, r, done, _ = m.step(a[i])
            states.append(s)
            scores[i] += r * gamma**t
        t += 1
        if done:  # assuming all the environments finish at the same time
            break
        states = np.array(states)
    # re-enable multiprocessing in the regressors
    for i in policy.Q._regressors:
        policy.Q._regressors[i].set_params(n_jobs=n_jobs)
    act = np.array(act).reshape(num_env, -1)
    return [(s if criterion == "discounted" else s / t, t, act[:, i]) for i, s in enumerate(scores)]