import numpy as np

"""
create_mujoco_actions create a list of discrete actions to apply to mujoco environments. This list contains all the
permutations with repetitions of each value of the action vector.

dim_actions is the dimension of the action space
high_vals is the array of the maximum value each element of the action vector can take
low_vals is the array of the minimum value each element of the action vector can take
elem_per_dim is the number of elements for each dimension. Must be >= 2 (highest and lowest values should be included)
from low_vals, to high_vals. Should take int values >= 0 and populate.

examples:
create_mujoco_actions(2, [1, 1], [-1, -1], 2)
creates [[-1, -1], [1, -1], [-1, 1], [1, 1]]

create_mujoco_actions(2, [1, 1], [-1, -1], 3)
creates [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, -1]]
"""

def create_grid(high_vals, low_vals, elem_per_dim):
    assert len(high_vals) == len(low_vals)
    assert elem_per_dim >= 2

    discr = (high_vals - low_vals) / (elem_per_dim-1)

    return recursive_helper(0, high_vals, low_vals, discr, [], [])

def recursive_helper(dim, high, low, discr, elem, res):

    for i in np.arange(low[dim], high[dim]+discr[dim], discr[dim]):
        new_elem = elem + [i]
        if dim < len(high) - 1:
            res = recursive_helper(dim + 1, high, low, discr, new_elem, res)
        else:
            res.append(new_elem)

    return res
