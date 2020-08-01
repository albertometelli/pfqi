import numpy as np
from numpy import linalg as LA

from trlib.algorithms.optimization.gradient import gradient


def quasi_newton(Qfunction, state, a0, epsilon, maxiterations):

    ak = np.array(a0)
    counter = 0
    error = 1e300

    grad = gradient(Qfunction, state, a0)
    H = np.array(np.eye(len(a0)))

    while np.any(np.linalg.eigvals(H) > 0):  # we look for a local maximum

        while error > epsilon and counter < maxiterations:
            counter += 1
            d = H.dot(grad)
            alpha = 1
            ak_prec = ak
            grad_prec = grad

            ak = ak + alpha * d
            grad = gradient(Qfunction, state, ak)

            error = LA.norm(grad)

            delta = ak - ak_prec
            gamma = grad - grad_prec
            delta_t = np.array(delta)[np.newaxis]
            gamma_t = np.array(gamma)[np.newaxis]

            H = H + ((1 + gamma * H * gamma_t) / (delta * gamma_t)) * (delta_t * delta) / (delta * gamma_t) - (
                        (H * gamma_t * delta) + (delta_t * gamma * H)) / (delta * gamma_t)

            test = np.linalg.eigvals(H)

    return ak
