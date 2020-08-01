import sys
sys.path.append('.')
import time
import argparse
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from trlib.environments.environment_data import CartPoleData
from trlib.algorithms.algorithm_handler import FQI_SSEP_Handler
from trlib.utilities.persistnces_creator import powers_list

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default=time.time())
parser.add_argument('--persistences', type=int, default=5)
parser.add_argument('--discretization', type=int, default=1)
parser.add_argument('--sampling_persistence', type=int, default=1)
parser.add_argument('--max_episode_steps', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--iterations_divisor', type=float, default=1)
parser.add_argument('--horizon_factor', type=int, default=4)
parser.add_argument('--tau', type=float, default=0.02)
parser.add_argument('--save_perfs_and_q', type=bool, default=True)
parser.add_argument('--save_step', type=int, default=1)
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--min_samples_split', type=int, default=5)
parser.add_argument('--min_samples_leaf', type=int, default=2)
args = parser.parse_args()

data = CartPoleData(max_episode_steps=args.max_episode_steps,
                    discretization=args.discretization,
                    iterations_divisor=args.iterations_divisor,
                    horizon_factor=args.horizon_factor,
                    name=args.name,
                    tau=args.tau,
                    gamma=0.99)

mdp, actions, max_episode_steps, gamma, max_iterations, file_name = data.get_env_data()
persistences = powers_list(args.persistences, 2)
regressor_params = {'n_estimators': args.n_estimators,
                    'criterion': 'mse',
                    'min_samples_split': args.min_samples_split,
                    'min_samples_leaf': args.min_samples_leaf}

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
                               save_step=args.save_step,
                               **regressor_params)

fqi_handler.handle_algorithm()