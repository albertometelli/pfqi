import numpy as np

def gradient(q_function, state, action):
    h = 0.1
    n = len(action)
    gradf = np.zeros(n)

    for i in range(n):
        delta = np.zeros(n)
        delta[i] += h
        x1 = [np.append(state, action+delta)]
        x2 = [np.append(state, action)]
        gradf[i] = (q_function.values(x1) - q_function.values(x2)) / h

    return np.array(gradf)

