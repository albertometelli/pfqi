import numpy as np


def powers_list(vector_length, power_base=2):
    a = range(vector_length)
    a = np.power(np.ones(len(a)) * power_base, a)
    p_list = []
    for i in range(len(a)):
        p_list.append(int(a[i]))
    return p_list
