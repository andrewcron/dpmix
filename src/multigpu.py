"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

#import threading
import multiprocessing
import numpy as np

#import pycuda.autoinit
#import gpustats
#import gpustats.sampler
#from cuda_functions import *

class GPUWorker(multiprocessing.Process):

    def __init__(self, data, device):
        multiprocessing.Process.__init__(self)

        self.data = data
        self.nobs, self.ndim = self.data.shape
        self.device = device
        self.tasks = multiprocessing.Queue()
        self.results = multiprocessing.Queue()

    def run(self):

        import pycuda.autoinit
        from pycuda.gpuarray import to_gpu
        #import pycuda.driver as drv
        #drv.init()
        #self.dev = drv.Device(self.device)
        #self.ctx = self.dev.make_context()



        ## load my portion of data to gpu
        self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))
        densities = 0

        ## this can only be killed by externally by ending None in the queue
        while True:
            ## wait for new params ...
            new_task = self.tasks.get()
            if new_task is None:
                break # poison pill

            ## get new samples and maybe row max
            densities = gpustats.mvnpdf_multi(self.gdata, new_task.mu, new_task.Sigma,
                                              weights = new_task.w.flatten(), get=False, 
                                              logged=True, order='C')
            self.labs = gpustats.sampler.sample_discrete(densities, logged=True)
            #print self.labs
            #print densities
            if new_task.relabel:
                Z = gpu_apply_row_max(densities)[1].get()
            else:
                Z = None                
            ## host thread should gather new labels and proceed. 
            self.results.put(self.labs.copy())

        del self.gdata
        del densities
        self.ctx.pop()
        del self.ctx

def init_GPUWorkers(data, w, mu, Sigma, devslist=None):
    # partitions data
    nobs, ndim = data.shape
    lenpart = nobs / len(devslist)
    partitions = range(0, nobs, lenpart); 
    if len(partitions)==len(devslist):
        partitions.append(nobs)
    else:
        partitions[-1] = nobs
    
    #launch threads uploading GPU data
    i=0; workers = []
    for dev in devslist:
        workers.append(GPUWorker(data[partitions[i]:partitions[i+1]], 
                                 device=dev))
        workers[-1].start()
        i+=1
    return workers

# simple object to go into task queue
class Task(object):
    def __init__(self, w, mu, Sigma):
        self.w = w
        self.mu = mu
        self.Sigma = Sigma

# callable function of new params and init threads
def get_labels(workers, w, mu, Sigma):
    # run all the threads
    nobs, i = 0, 0
    partitions = [0]
    for thd in workers:
        # give new params
        nobs += thd.nobs
        partitions.append(nobs)
        thd.tasks.put(Task(w, mu, Sigma))
    #gather the results
    res = np.zeros(nobs, dtype=np.float32)
    for thd in workers:
        labs = thd.results.get()
        res[partitions[i]:partitions[i+1]] = labs
        i+=1
    return res

# feeds poison pills to each thread
def kill_workers(workers):
    for thd in workers:
        thd.tasks.put(None)
    
        


