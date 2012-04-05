"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

import threading
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import to_gpu
import gpustats
import gpustats.sampler
from cuda_functions import *

class GPUWorker(threading.Thread):

    def __init__(self, data, w, mu, Sigma, relabel, device):
        threading.Thread.__init__(self)

        self.data = data
        self.w = w.flatten()
        self.mu = mu
        self.Sigma = Sigma
        self.device = device
        self.relabel = relabel
        self.nobs, self.ndim = data.shape
        self.ncomp = len(w)
        self.condition = threading.Condition()
        self.end_sampler = False
        self.new_params = False

    def run(self):

        self.condition.acquire()
        self.dev = drv.Device(self.device)
        self.ctx = self.dev.make_context()
        ## load my portion of data to gpu
        self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))
        self.condition.release()

        ## this can only be killed by externally setting end_sampler to True
        ## and releasing the lock
        while not self.end_sampler:
            ## wait for new params ... the model params should have changed
            self.condition.acquire()
            while not self.new_params and not self.end_sampler:
                self.condition.wait()
            if self.end_sampler:
                break
            ## get new samples and maybe row max
            densities = gpustats.mvnpdf_multi(self.gdata, self.mu, self.Sigma,
                                              weights = self.w, get=False, logged=True,
                                              order='C')
            labs = gpustats.sampler.sample_discrete(densities, logged=True)
            if self.relabel:
                Z = gpu_apply_row_max(densities)[1].get()
            else:
                Z = None
                
            self.new_params = False
            self.condition.release()
            ## host thread should gather new labels and proceed. 

        del self.gdata
        del densities
        self.ctx.pop()
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
        workers.append(GPUWorker(data[partitions[i]:partitions[i+1]], 
                                 w=w, mu=mu, Sigma=Sigma, relabel=False,
                                 device=dev))
        workers[-1].start()
        i+=1
    return workers

def get_labels(workers, w, mu, Sigma):
    # run all the threads
    nobs, i = 0, 0
    partitions = [0]
    for thd in workers:
        # get control of thread
        thd.condition.acquire()
        # give new params
        nobs += thd.nobs
        partitions.append(nobs)
        thd.w = w
        thd.mu = mu
        thd.Sigma = Sigma
        thd.new_params = True
        thd.condition.notify()
        thd.condition.release()
    #gather the results
    res = np.zeros(nobs, dtype=np.float32)
    for thd in workers:
        thd.condition.acquire() # wait until finished
        res[partitions[i]:partitions[i+1]] = thd.labs.copy()
        thd.condition.release() 
        i+=1
    return res

def kill_workers(workers):
    for thd in workers:
        thd.condition.acquire()
        thd.end_sampler = True
        thd.condition.notify()
        thd.condition.release()
    
        


