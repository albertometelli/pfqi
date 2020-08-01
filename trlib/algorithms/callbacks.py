from trlib.policies.policy import EpsilonGreedyQN
from trlib.utilities.evaluation import evaluate_policy, render_policy
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.doubleq_policies import EpsilonGreedyDoubleQ
from trlib.algorithms.reinforcement.fqi import FQI_SS, FQI_SS_EP_old, FQI_SS_EP, FQI_P1_VS_1P, FQI_TOTALLY_SS, \
    FQI_BOCP, FQI_SS_EP_1P_Control
def save_json_callback(file_name):
    """
    Generates a callback for saving results in JSON format
    
    Parameters
    ----------
    file_name: the file where to save results
    
    Returns
    -------
    A callback for an algorithm to save results
    """
    
    def fun(algorithm):
        algorithm._result.save_json(file_name)
            
    return fun

def eval_policy_callback(field_name, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1):
    """
    Generates a callback for evaluating a policy.
    
    Parameters
    ----------
    field_name: name of the field in the algorithm's Result object where to store the evaluation
    others: see evaluation.py
    
    Returns
    -------
    A callback for an algorithm to evaluate performance
    """
    
    def fun(algorithm):
        
        perf = evaluate_policy(algorithm._mdp, algorithm._policy, criterion = criterion, n_episodes = n_episodes, initial_states = initial_states, n_threads = n_threads)
        fields = {}
        fields[field_name + "_mean"] = perf[0]
        fields[field_name + "_std"] = perf[1]
        algorithm._result.update_step(**fields)
    
    return fun

def eval_greedy_policy_callback(field_name, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1, mul_env=False):
    """
    Generates a callback for evaluating a policy that is greedy w.r.t. the algorithm's current Q-function
    
    Parameters
    ----------
    field_name: name of the field in the algorithm's Result object where to store the evaluation
    others: see evaluation.py
    
    Returns
    -------
    A callback for an algorithm to evaluate performance
    """
    
    def fun(algorithm):
        
        policy = EpsilonGreedy(algorithm._actions, algorithm._policy.Q, 0)
        perf = evaluate_policy(algorithm._mdp, policy, criterion = criterion, n_episodes = n_episodes, initial_states = initial_states, n_threads = n_threads, mul_env = mul_env)
        fields = {}
        fields[field_name + "_mean"] = perf[0]
        fields[field_name + "_std"] = perf[1]
        fields[field_name + "_steps"] = perf[2]
        if isinstance(algorithm, FQI_SS) or isinstance(algorithm, FQI_SS_EP_old) or isinstance(algorithm, FQI_SS_EP) \
                or isinstance(algorithm, FQI_BOCP) or isinstance(algorithm, FQI_SS_EP_1P_Control):
            algorithm._result.update_step_with_persistence(algorithm._mdp.env.persistence, **fields)
        else:
            if isinstance(algorithm, FQI_P1_VS_1P):
                algorithm._result.update_step_with_mdp(algorithm._policy._name, **fields)
            else:
                if isinstance(algorithm, FQI_TOTALLY_SS):
                    algorithm._result.update_step_with_persistence(algorithm._mdp.env.persistence, algorithm._label, **fields)
                else:
                    algorithm._result.update_step(**fields)
    return fun


def eval_greedy_policy_doubleq_callback(field_name, criterion='discounted', n_episodes=1, initial_states=None, n_threads=1):
    """
    Generates a callback for evaluating a policy that is greedy w.r.t. the algorithm's current double Q-function

    Parameters
    ----------
    field_name: name of the field in the algorithm's Result object where to store the evaluation
    others: see evaluation.py

    Returns
    -------
    A callback for an algorithm to evaluate performance
    """

    def fun(algorithm):

        assert isinstance(algorithm._policy, EpsilonGreedyDoubleQ)
        policy = EpsilonGreedyDoubleQ(algorithm._actions, 0, algorithm._policy.Q1, algorithm._policy.Q2)
        perf = evaluate_policy(algorithm._mdp, policy, criterion=criterion, n_episodes=n_episodes,
                               initial_states=initial_states, n_threads=n_threads)
        fields = {}
        fields[field_name + "_mean"] = perf[0]
        fields[field_name + "_std"] = perf[1]
        fields[field_name + "_steps"] = perf[2]
        algorithm._result.update_step_with_persistence(algorithm._mdp.env.persistence, **fields)

    return fun


def eval_greedy_policy_qn_callback(field_name, low_bounds, high_bounds, criterion='discounted', n_episodes=1,
                                   initial_states=None, n_threads=1):
    """
    Generates a callback for evaluating a policy that is greedy w.r.t. the algorithm's current Q-function using
    quasi newton method to find the maximum

    Parameters
    ----------
    field_name: name of the field in the algorithm's Result object where to store the evaluation
    others: see evaluation.py

    Returns
    -------
    A callback for an algorithm to evaluate performance
    """

    def fun(algorithm):
        policy = EpsilonGreedyQN(algorithm._policy.Q, 0.1, high_bounds, low_bounds)
        perf = evaluate_policy(algorithm._mdp, policy, criterion=criterion, n_episodes=n_episodes,
                               initial_states=initial_states, n_threads=n_threads)
        fields = {}
        fields[field_name + "_mean"] = perf[0]
        fields[field_name + "_std"] = perf[1]
        fields[field_name + "_steps"] = perf[2]
        algorithm._result.update_step(**fields)

    return fun

def render_greedy_policy_callback(n_episodes=1, initial_states=None):
    """
    Generates a callback for rendering a policy that is greedy w.r.t. the algorithm's current Q-function

    Parameters
    ----------
    others: see evaluation.py
    """

    def fun(algorithm):
        policy = EpsilonGreedy(algorithm._actions, algorithm._policy.Q, 0)
        render_policy(algorithm._mdp, policy, n_episodes=n_episodes,
                               initial_states=initial_states)

    return fun

def eval_policy_pre_callback(field_name, policy, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1):
    """
    Generates a pre-callback for evaluating the uniform policy before starting the algorithm
    
    Parameters
    ----------
    field_name: name of the field in the algorithm's Result object where to store the evaluation
    others: see evaluation.py
    
    Returns
    -------
    A callback for an algorithm to evaluate performance
    """
    
    def fun(algorithm):
        
        perf = evaluate_policy(algorithm._mdp, policy, criterion = criterion, n_episodes = n_episodes, initial_states = initial_states, n_threads = n_threads)
        fields = {}
        fields[field_name + "_mean"] = perf[0]
        fields[field_name + "_std"] = perf[1]
        algorithm._result.add_step(step = 0, n_episodes = 0, **fields)
    
    return fun

def render_policy_pre_callback(policy, n_episodes=1, initial_states=None):
    """
    Generates a callback for rendering a policy that is greedy w.r.t. the algorithm's current Q-function

    Parameters
    ----------
    others: see evaluation.py
    """

    def fun(algorithm):
        render_policy(algorithm._mdp, policy, n_episodes=n_episodes,
                               initial_states=initial_states)

    return fun

def get_callbacks(callback_list):
    """
    Returns a list of callbacks given a list of callback specifications.
    A list of callback specifications is a list of tuples (callback_name, **callback_params).
    
    Parameters
    ----------
    callback_list: a list of tuples (callback_name, **callback_params)
    
    Returns
    -------
    A list of callbacks
    """
    
    callbacks = []
    
    for name,params in callback_list:
        callbacks.append(globals()[name](**params))
        
    return callbacks

def get_callback_list_entry(name, **params):
    """
    Builds an entry of a callback specification, i.e., a tuple (callback_name, **callback_params)
    
    Parameters
    ----------
    name: the name of a callback generator function
    params: parameters to the callback generator function
    
    Returns
    -------
    A couple (name,params)
    """
    return (name, params)
    