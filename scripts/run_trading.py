import sys
sys.path.append('.')
import time
import argparse
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from trlib.environments.environment_data import TradingData
from trlib.algorithms.algorithm_handler import FQI_SSEP_Handler
from trlib.utilities.persistnces_creator import powers_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default=time.time())
    parser.add_argument('--persistences', type=int, default=3)
    parser.add_argument('--single_persistence',type=int, default=None)
    parser.add_argument('--sampling_persistence', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--max_iterations', type=int, default=1)
    parser.add_argument('--save_perfs_and_q', dest='save_perfs_and_q', default=False, action='store_true')
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--for_curs', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_jobs_regressors', type=int, default=1)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--env_ver', type=str, default='v1')
    parser.add_argument('--mul_env', dest='mul_env', default=False, action='store_true')
    parser.add_argument('--train_limit', dest='train_limit', default=False, action='store_true')

    args = parser.parse_args(sys.argv[1:])

    data = TradingData(max_iterations=args.max_iterations,
                       name=args.name,
                       gamma=0.9999,
                       for_curs=args.for_curs,
                       eval_episodes=args.eval_episodes,
                       env_ver=args.env_ver,
                       mul_env=args.mul_env,
                       use_train_limit=args.train_limit)

    mdp, actions, max_episode_steps, gamma, max_iterations, file_name = data.get_env_data()
    # file_name += '_' + str(gamma) 
    #len(mdp.data)
    file_name += '_s' + str(args.min_samples_split) 
    file_name += '_es' + str(args.n_estimators) 
    file_name += '_ev' + str(args.eval_episodes) 
    file_name += '_jr' + str(args.n_jobs_regressors) 
    file_name += '_' + str(args.env_ver) 

    if args.single_persistence is not None:
        persistences = [args.single_persistence]
    else:
        persistences = powers_list(args.persistences, 2)
    regressor_params = {'n_estimators': args.n_estimators,
                        'min_samples_split': args.min_samples_split,
                        'min_samples_leaf': 1}
    fqi_handler = FQI_SSEP_Handler(mdp=mdp,
                                   gamma=gamma,
                                   max_episode_steps=max_episode_steps,
                                   actions=actions,
                                   file_name=file_name,
                                   max_iterations=max_iterations,
                                   persistences=persistences,
                                   sampling_persistence=args.sampling_persistence,
                                   batch_size=args.batch_size,
                                   pi=EpsilonGreedy(actions, ZeroQ(), 0),
                                   save_perfs_and_q=args.save_perfs_and_q,
                                   cb_eval=data.get_cb_eval(),
                                   save_step=args.save_step,
                                   n_jobs=args.n_jobs,
                                   n_jobs_regressors=args.n_jobs_regressors,
                                   n_runs=1,
                                   initial_states=None,#data.get_initial_states(),
                                   **regressor_params)
    from contextlib import contextmanager
    import time
    @contextmanager
    def timed(msg):
            print(msg)
            tstart = time.time()
            yield
            print("done in %.3f seconds"%(time.time() - tstart))
    with timed("FQI"):
        fqi_handler.handle_algorithm()