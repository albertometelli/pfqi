import json
import numpy as np
import matplotlib.pyplot as plt

'''
Expect to have a file like that:
list of [perfs, stds, q, q_diff, q_diff_abs, err_proj, err_proj_abs]
perfs, stds, err_proj, err_proj_abs have:
  - a list of [P1, P2, P4...]
  - each with a list of [iter1, iter2, ...]
q, q_diff, q_diff_abs have:
  - a list of [P1, P2, P4...]
  - each with a list of items evaluated in [sa, sa_greedy, s0a_greedy]
  - each with a list of [iter1, iter2, ...]
'''
from pathlib import Path
import os.path as osp
import sys
import os
import re
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    results_path = str(Path(__file__).parent.parent)

    pattern = re.compile(r'^Perfs_and_qs_Trading(.)+')
    if not args.name:
        files = sorted([f for f in os.listdir(results_path)
                        if pattern.match(f)], key=lambda x: os.path.getmtime(osp.join(results_path, x)))
        file_name = files[-1]
    else:
        file_name = args.name
    colors = ['dodgerblue', 'cyan', 'limegreen', 'chartreuse', 'gold', 'orange', 'r', 'm', 'darkorchid']
    dalton_friendly_colors = ['deepskyblue', 'mediumslateblue', 'deeppink', 'darkorange', 'gold']

    final_colors = colors
    sa_index = 0  # 0 for sa, 1 for sa_greedy, 2 for s0a_greedy
    gamma_default = 0.99 ** (1/3)

    with open(file_name, 'r') as f:
        file = json.load(f)

    '''
    save_step added later in position [7], if len(file)==7 there is no save_step
    '''
    if len(file) >= 8:
        save_step = file[7]
    else:
        save_step = 1

    '''
    gamma added later in position [8], if len(file)<9 there is no gamma
    '''
    if len(file) >= 9:
        gamma = file[8]
    else:
        gamma = gamma_default

    persistences_number = len(file[0])

    valid_iterations_perfs = []
    for i in range(persistences_number):
        l = list(range(len(file[0][i])))
        valid_iterations_perfs.append([(j * max(2 ** i, save_step)) + max(2 ** i, save_step) for j in l])

    valid_iterations_bound = []
    for i in range(persistences_number):
        l = list(range(len(file[2][i][0])))
        valid_iterations_bound.append([(j * max(2 ** i, save_step)) + max(2 ** i, save_step) for j in l])

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # plot perf
    # ax[0][0].set(ylabel='performances')
    # for i in range(persistences_number):
    #     ax[0][0].plot(valid_iterations[i], file[0][i], '.', color=final_colors[i], label='P{}'.format(2**i))
    #     ax[0][0].errorbar(valid_iterations[i], file[0][i], yerr=file[1][i], linestyle='', color=final_colors[i])

    ax[0][0].set(ylabel='performances')
    for i in range(persistences_number):
        ax[0][0].plot(valid_iterations_perfs[i], file[0][i], '-', color=final_colors[i], label='P{}'.format(2**i))

    # plot q
    ax[1][0].set(ylabel='Q_functions')
    for i in range(persistences_number):
        ax[1][0].plot(valid_iterations_bound[i], file[2][i][2], '-', color=final_colors[i])

    # plot q_diff_abs
    ax[0][1].set(ylabel='Q_diff_abs')
    for i in range(persistences_number):
        ax[0][1].plot(valid_iterations_bound[i], file[4][i][0], '-', color=final_colors[i])

    # plot bound abs
    ax[1][1].set(ylabel='Q(sa0greedy) - Q_diff_abs(sa)')
    for i in range(persistences_number):
        bound = file[2][i][2] - ((1-gamma**(2**i)) ** -1) * (np.array(file[4][i][0]))
        ax[1][1].plot(valid_iterations_bound[i], bound, '-', color=final_colors[i])

    ax[0][0].legend(ncol=len(file[0]), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')

    plt.show()
