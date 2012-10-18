import sampler
import numpy as np
import wishart
import time

if __name__ == '__main__':
    N = 10000
    nu = 6.0
    phi = 2*np.identity(4)
    phi[1,2]=1; phi[2,1]=1;

    mu = np.zeros(4, dtype=np.float64)
    Sigma = sampler.pyiwishartrand_prec(nu, phi)
    
    res = np.zeros((4,4))
    t1 = time.time()
    for i in range(N):
        res += sampler.pyiwishartrand_prec(nu, phi)
    print res / N
    print time.time() - t1

    t2 = time.time()
    res = np.zeros((4,4))
    for i in range(N):
        res += wishart.invwishartrand_prec(nu,phi)
    print res / N
    print time.time() - t2

    t3 = time.time()
    res = np.zeros(4)
    for i in range(N):
        s = sampler.pymvnorm(mu, Sigma)
        res += s
    print res / N
    print time.time() - t3
