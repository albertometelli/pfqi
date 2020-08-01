import numpy as np

def hes(q_function, state, action):
    h = 0.1
    n = len(action)
    H = np.zeros(n, n)

    for i in range(n):
        delta_i = np.zeros(n)
        delta_i[i] = h
        for j in range(n):
            delta_j = np.zeros(n)
            delta_j[j] = h
            x1 = np.append(state, action + delta_i + delta_j)
            x2 = np.append(state, action + delta_i)
            x3 = np.append(state, action + delta_j)
            x4 = np.append(state, action)
            H[i, j] = (q_function.values(x1) - q_function.values(x2) - q_function.values(x3) + q_function.values(x4)) / h**2

    return np.array(H)