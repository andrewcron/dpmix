"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

#import threading
#import Queue
import multiprocessing
import numpy as np
#import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import to_gpu
#import gpustats
#import gpustats.sampler
#from cuda_functions import *


########### Multi GPU ##########################
class MCMC_Task(object):
    def __init__(self, w, mu, Sigma, relabel=False):
        self.w = w
        self.mu = mu
        self.Sigma = Sigma
        self.relabel = relabel

    def __call__(self, gdata, res_queue, g_ones_long=None,
                 gpustats=None, gutil=None, gsamp=None, cufuncs=None, 
                 cuLA=None, cumath=None, iexp=None):

        densities = gpustats.mvnpdf_multi(gdata, self.mu, self.Sigma,
                                          weights = self.w.flatten(), get=False, logged=True,
                                          order='C')
        labs = gsamp.sample_discrete(densities, logged=True)
        if self.relabel:
            Z = gpu_apply_row_max(densities)[1].get()
        else:
            Z = None
        res_queue.put([labs.copy(), Z])
        densities.gpudata.free()
        del densities
        
class BEM_Task(object):

    def __init__(self, w, mu, Sigma):
        self.w = w
        self.mu = mu
        self.Sigma = Sigma
        self._logmnflt = np.log(1e-37)

    def __call__(self, gdata, res_queue, g_ones_long=None,
                 gpustats=None, gutil=None, gsamp=None, cufuncs=None, 
                 cuLA=None, cumath=None, iexp=None):

        ncomp = len(self.w)
        nobs, ndims = gdata.shape
        g_ones = to_gpu(np.ones((ncomp, 1), dtype=np.float32))

        densities = gpustats.mvnpdf_multi(gdata, self.mu, self.Sigma,
                                          weights = self.w.flatten(), get=False, logged=True)
        tdens = gutil.GPUarray_reshape(densities, (ncomp, nobs), "C")
        ll = cuLA.dot(g_ones, cumath.exp(tdens), "T").get()
        nmzero = np.sum(ll==0)
        ll = np.sum(np.log(ll[ll>0])) + nmzero*self._logmnflt

        nrm, _ = cufuncs.gpu_apply_row_max(densities)
        cufuncs.gpu_sweep_col_diff(densities, nrm)
        iexp(densities); gutil.GPUarray_order(densities, "F")
        nrm = cuLA.dot(g_ones, tdens, "T")
        cufuncs.gpu_sweep_col_div(densities, nrm)

        ct = cuLA.dot(tdens, g_ones_long).get().flatten()
        xbar = cuLA.dot(tdens, gdata).get()
        h_densities = densities.get()

        res = (ll, ct, xbar, h_densities)
        res_queue.put(res)
        ## Free Everything
        g_ones.gpudata.free()
        densities.gpudata.free()
        nrm.gpudata.free()

        
#class GPUWorker(threading.Thread):
class GPUWorker(multiprocessing.Process):

    def __init__(self, data, device):
        #threading.Thread.__init__(self)
        multiprocessing.Process.__init__(self)

        self.data = data
        self.device = device
        self.nobs, self.ndim = data.shape
        
        #self.params = Queue.Queue()
        #self.results = Queue.Queue()
        self.params = multiprocessing.Queue()
        self.results = multiprocessing.Queue()

    def run(self):
        drv.init()
        try:
            self.dev = drv.Device(self.device)
        except:
            raise ValueError("Unable to allocate device " + str(self.device) + "!")
        self.ctx = self.dev.make_context()        

        # imports must be done here ....
        import gpustats
        import gpustats.sampler as gsamp
        import gpustats.util as gutil
        import cuda_functions as cufuncs
        from scikits.cuda import linalg as cuLA; cuLA.init()
        from pycuda import cumath
        from pycuda.elementwise import ElementwiseKernel
        inplace_exp = ElementwiseKernel("float *z", "z[i]=expf(z[i])", "inplexp")
        
        #from cuda_functions import *

        #print 'mem situation device ' + str(self.device) + ' ' + str(drv.mem_get_info())
        
        ## load my portion of data to gpu
        self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))
        self.g_ones_long = to_gpu(np.ones((self.nobs,1), dtype=np.float32))

        ## gets tasks from params queue and puts labels in results queue .. killed
        ## with None poison pill
        while True:
            task = self.params.get()
            if task is None:
                break

            ## Execute Task
            #print 'device ' + str(self.device) + ' started computing'
            task(self.gdata, self.results, self.g_ones_long, gpustats, gutil, gsamp,
                 cufuncs, cuLA, cumath, inplace_exp)

            self.dev_mem = drv.mem_get_info()
            #print 'mem situation device ' + str(self.device) + ' ' + str(drv.mem_get_info())
            #self.params.task_done()

        self.gdata.gpudata.free()
        self.g_ones_long.gpudata.free()
        del self.gdata
        del self.g_ones_long
        #print 'killed thread ' + str(self.device)
        #print 'available mem ' + str(drv.mem_get_info())
        self.ctx.detach()
        del self.ctx


def init_GPUWorkers(data, w, mu, Sigma, devslist=None):

    nobs, ndim = data.shape
    lenpart = nobs / len(devslist)
    partitions = range(0, nobs, lenpart); 
    if len(partitions)==len(devslist):
        partitions.append(nobs)
    else:
        partitions[-1] = nobs
    
    #launch threads
    i=0; workers = []
    for dev in devslist:
        workers.append(GPUWorker(data[partitions[i]:partitions[i+1]], device=int(dev)))
        i+=1
    return workers

def get_labelsGPU(workers, w, mu, Sigma, relabel=False):
    # run all the threads
    nobs, i = 0, 0
    partitions = [0]
    theta = MCMC_Task(w, mu, Sigma, relabel)
    for thd in workers:
        # give new params
        nobs += thd.nobs
        partitions.append(nobs)
        thd.params.put(theta)
    #gather the results
    res = np.zeros(nobs, dtype=np.float32)
    if relabel:
        Z = res.copy()
    else:
        Z = None
    for thd in workers:
        labs = thd.results.get()
        res[partitions[i]:partitions[i+1]] = labs[0]
        if relabel:
            Z[partitions[i]:partitions[i+1]] = labs[1]
        i+=1

    return res, Z

def get_expected_labels_GPU(workers, w, mu, Sigma):
    # run all the threads
    nobs, i = 0, 0
    partitions = [0]
    task = BEM_Task(w, mu, Sigma)
    for thd in workers:
        # give new params
        nobs += thd.nobs
        partitions.append(nobs)
        thd.params.put(task)
    #gather results
    ncomp = len(w); ndim = workers[0].ndim
    dens = np.zeros((nobs, ncomp), dtype=np.float32)
    xbar = np.zeros((ncomp, ndim), dtype=np.float32)
    ct = np.zeros(ncomp, dtype=np.float32)
    ll = 0
    for thd in workers:
        res = thd.results.get()
        ll += res[0]
        ct += res[1]
        xbar += res[2]
        dens[partitions[i]:partitions[i+1], :] = res[3].copy()
        i+=1
        
    return ll, ct, xbar, dens

def kill_GPUWorkers(workers):
    for thd in workers:
        thd.params.put(None)
    
        


