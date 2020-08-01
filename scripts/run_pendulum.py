import sys
sys.path.append('.')
import time
import argparse
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from trlib.environments.environment_data import PendulumData
from trlib.algorithms.algorithm_handler import FQI_SSEP_Handler
from trlib.utilities.persistnces_creator import powers_list

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default=time.time())
parser.add_argument('--persistences', type=int, default=3)
parser.add_argument('--dt', type=float, default=.05)
parser.add_argument('--sampling_persistence', type=int, default=1)
parser.add_argument('--max_episode_steps', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--iterations_divisor', type=float, default=4)
parser.add_argument('--horizon_factor', type=int, default=1)
parser.add_argument('--save_perfs_and_q', type=bool, default=False)
parser.add_argument('--save_step', type=int, default=1)
parser.add_argument('--n_actions', type=int, default=3)
args = parser.parse_args()

data = PendulumData(max_episode_steps=args.max_episode_steps,
                    iterations_divisor=args.iterations_divisor,
                    horizon_factor=args.horizon_factor,
                    name=args.name,
                    dt=args.dt,
                    n_actions=args.n_actions)

mdp, actions, max_episode_steps, gamma, max_iterations, file_name = data.get_env_data()
persistences = powers_list(args.persistences, 2)

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
                               save_step=args.save_step)

fqi_handler.handle_algorithm()
