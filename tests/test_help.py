import pymc as pm
import numpy as np
import numpy.random as npr

gen_mean = {
    0 : [0, 5],
    1 : [-5, 0],
    2 : [5,0]
}

gen_sd = {
    0 : [0.5, 0.5],
    1 : [.5, 1],
    2 : [1, .25]
}

gen_corr = {
    0 : 0.5,
    1 : -0.5,
    2 : 0
}

group_weights = [0.4, 0.3, 0.3]

def generate_data(n=1e5, k=2, ncomps=3, seed=1):
    npr.seed(seed)
    data_concat = []
    labels_concat = []

    for j in xrange(ncomps):
        mean = gen_mean[j]
        sd = gen_sd[j]
        corr = gen_corr[j]

        cov = np.empty((k, k))
        cov.fill(corr)
        cov[np.diag_indices(k)] = 1
        cov *= np.outer(sd, sd)

        num = int(n * group_weights[j])
        rvs = pm.rmv_normal_cov(mean, cov, size=num)

        data_concat.append(rvs)
        labels_concat.append(np.repeat(j, num))

    return (np.concatenate(labels_concat),
            np.concatenate(data_concat, axis=0))
