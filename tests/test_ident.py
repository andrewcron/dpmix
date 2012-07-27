#import sys
#sys.path.insert(0, "../src")

import numpy.random as npr
import dpmix

if __name__ == "__main__":
    xs = npr.random((2,100,2))
    for x in xs:
        m1 = dpmix.DPNormalMixture(x, ncomp=128, gpu=2, verbose=10)
        m1.sample(100, nburn=0)

        m2 = dpmix.DPNormalMixture(m1, verbose=10)
        m2.sample(10, nburn=0, ident=True)
