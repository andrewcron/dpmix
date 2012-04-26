"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

#import threading
#import Queue
import multiprocessing
import Queue as pQueue
import numpy as np
#import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import to_gpu
#import gpustats
#import gpustats.sampler
#from cuda_functions import *

## WIERDNESS: libraries must be loaded inside of "run" and passed to
## functions to work properly ... 

########### Multi GPU ##########################
class MCMC_Task(object):
    def __init__(self, w, mu, Sigma, relabel=False):
        self.w = w
        self.mu = mu
        self.Sigma = Sigma
        self.relabel = relabel

    def __call__(self, gdata, res_queue, g_ones_long=None,
                 gpustats=None, gsamp=None, cufuncs=None, gutil=None,
                 cuLA=None, cumath=None, iexp=None):

        densities = gpustats.mvnpdf_multi(gdata, self.mu, self.Sigma,
                                          weights = self.w.flatten(), get=False, logged=True,
                                          order='C')

        labs = gsamp.sample_discrete(densities, logged=True)
        if self.relabel:
            Z = cufuncs.gpu_apply_row_max(densities)[1].get()
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
                 gpustats=None, gsamp=None, cufuncs=None, gutil=None,
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
        # imports must be done here ....
        import pycuda.tools as pytools
        pytools.clear_context_caches()
        import gpustats
        import gpustats.sampler as gsamp
        import gpustats.util as gutil
        import cuda_functions as cufuncs
        from scikits.cuda import linalg as cuLA
        from pycuda import cumath
        from pycuda.elementwise import ElementwiseKernel
        inplace_exp = ElementwiseKernel("float *z", "z[i]=expf(z[i])", "inplexp")

        ## some of these libraries initialize a context by default ... dumb
        ctx = drv.Context.get_current()
        if ctx is not None:
            ctx = ctx.detach()
            del ctx

        try:
            self.dev = drv.Device(self.device)
        except:
            raise ValueError("Unable to allocate device " + str(self.device) + "!")
        self.ctx = self.dev.make_context()     
        cuLA.init()   

        #print 'mem situation device ' + str(self.device) + ' ' + str(drv.mem_get_info())
        
        ## load my portion of data to gpu
        self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))
        self.g_ones_long = to_gpu(np.ones((self.nobs,1), dtype=np.float32))

        ## put empty object in result queue to indicate readiness
        self.results.put(None)

        ## gets tasks from params queue and puts labels in results queue .. killed
        ## with None poison pill
        while True:
            task = self.params.get()
            if task is None:
                break

            ## Execute Task
            #print 'device ' + str(self.device) + ' started computing'
            task(self.gdata, self.results, self.g_ones_long, gpustats, gsamp,
                 cufuncs, gutil, cuLA, cumath, inplace_exp)

            #self.dev_mem = drv.mem_get_info()
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
        ## put None in results to indicate destruction ... forces block
        self.results.put(None)


def init_GPUWorkers(data, devslist=None):
    ## dpmix and BEM
    if type(data) == np.ndarray:
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
    else: ## HDP .. one or more datasets per GPU
        ndev = len(devslist)
        i=0; workers = []
        for dt in data:
            workers.append(GPUWorker(dt, device=int(devslist[i%ndev])))
            i += 1
    return workers

def start_GPUWorkers(workers):
    for thd in workers:
        thd.start()
        try:
            thd.results.get(timeout=60)
        except pQueue.Empty:
            ## thread got hung up ... kill them all and raise exception
            for deadthd in workers:
                deadthd.terminate()
            raise MemoryError("Bad things happened with GPU ... ")
    
            
def get_hdp_labels_GPU(workers, w, mu, Sigma, relabel=False):
    labels = []; Z = [];
    #import pdb; pdb.set_trace()
    i = 0
    for thd in workers:
        thd.params.put(MCMC_Task(w[i], mu, Sigma, relabel))
        i += 1
    for thd in workers:
        res = thd.results.get()
        labels.append(res[0])
        Z.append(res[1])
    return labels, Z 

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
        try:
            thd.results.get(timeout=60)
        except pQueue.Empty:
            thd.terminate()
    
        


