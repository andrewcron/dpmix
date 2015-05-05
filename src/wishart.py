import numpy as np
import numpy.random as npr
from numpy.linalg import cholesky, solve
from scipy.stats import chi2


def inv(x):
    return solve(x, np.eye(x.shape[0]))


def invwishartrand(nu,phi):
    return inv(wishartrand(nu,phi))


def invwishartrand_prec(nu, phi):
    return inv(wishartrand(nu, inv(phi)))


def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    foo = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i, j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i, j] = npr.normal(0, 1)

    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


if __name__ == '__main__':
    import pymc as pm
    npr.seed(1)
    nu = 5
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    px = np.array([pm.rinverse_wishart(nu, a) for i in range(10000)])
    x = np.array([invwishartrand(nu, a) for i in range(10000)])
    print np.mean(x, 0), '\n', np.mean(px, 0)
