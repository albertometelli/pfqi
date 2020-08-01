import sys
sys.path.append('.')
import time
import argparse
from trlib.environments.environment_data import SwimmerData
from trlib.algorithms.algorithm_handler import FQI_SSEP_Handler
from trlib.utilities.persistnces_creator import powers_list
from trlib.policies.qvalues_bound import calculate_bound
from trlib.policies.qvalues_producer import QValuesProducer
from trlib.utilities.data import save_object

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default=time.time())
parser.add_argument('--persistences', type=int, default=8)
parser.add_argument('--frame_skip', type=int, default=1)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--horizon_factor', type=int, default=1)
parser.add_argument('--sampling_persistence', type=int, default=1)
parser.add_argument('--max_episode_steps', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--iterations_divisor', type=int, default=1)
parser.add_argument('--action_per_dim', type=int, default=2)
parser.add_argument('--save_perfs_and_q', type=bool, default=False)
parser.add_argument('--save_step', type=int, default=1)
args = parser.parse_args()

data = SwimmerData(max_episode_steps=args.max_episode_steps,
                   horizon_factor=args.horizon_factor,
                   iterations_divisor=args.iterations_divisor,
                   frame_skip=args.frame_skip,
                   name=args.name,
                   action_per_dim=args.action_per_dim)

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
                               save_perfs_and_q=args.save_perfs_and_q,
                               save_step=args.save_step)

fqi_handler.handle_algorithm()
