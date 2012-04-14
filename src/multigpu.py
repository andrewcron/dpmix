"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

import threading
import Queue
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import to_gpu
import gpustats
import gpustats.sampler
from cuda_functions import *

class Theta(object):
    def __init__(self, w, mu, Sigma, relabel=False):
        self.w = w
        self.mu = mu
        self.Sigma = Sigma
        self.relabel = relabel
        

class GPUWorker(threading.Thread):

    def __init__(self, data, device):
        threading.Thread.__init__(self)

        self.data = data
        self.device = device
        self.nobs, self.ndim = data.shape
        
        self.params = Queue.Queue()
        self.results = Queue.Queue()

    def run(self):

        self.dev = drv.Device(self.device)
        self.ctx = self.dev.make_context()
        ## load my portion of data to gpu
        self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))

        ## gets tasks from params queue and puts labels in results queue .. killed
        ## with None poison pill
        while True:
            theta = self.params.get()
            if theta is None:
                break

            ## get new samples and maybe row max
            #print 'device ' + str(self.device) + ' started computing'
            densities = gpustats.mvnpdf_multi(self.gdata, theta.mu, theta.Sigma,
                                              weights = theta.w.flatten(), get=False, logged=True,
                                              order='C')
            self.labs = gpustats.sampler.sample_discrete(densities, logged=True)
            #print 'mem situation device ' + str(self.device) + ' ' + str(drv.mem_get_info())
            if theta.relabel:
                Z = gpu_apply_row_max(densities)[1].get()
            else:
                Z = None

            densities.gpudata.free() # avoid leaks
            self.results.put(self.labs.copy())
            self.params.task_done()


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
        workers.append(GPUWorker(data[partitions[i]:partitions[i+1]], device=dev))
        workers[-1].start()
        i+=1
    return workers

def get_labels(workers, w, mu, Sigma):
    # run all the threads
    nobs, i = 0, 0
    partitions = [0]
    theta = Theta(w, mu, Sigma)
    for thd in workers:
        # give new params
        nobs += thd.nobs
        partitions.append(nobs)
        thd.params.put(theta)
    #gather the results
    res = np.zeros(nobs, dtype=np.float32)
    for thd in workers:
        labs = thd.results.get()
        res[partitions[i]:partitions[i+1]] = labs
        i+=1
    return res

def kill_workers(workers):
    for thd in workers:
        thd.params.put(None)
    
        


