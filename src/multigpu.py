"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

from mpi4py import MPI
import numpy as np
import sys; import os
from utils import BEM_Task, MCMC_Task, Init_Task

########### Multi GPU ##########################

def init_GPUWorkers(data, devslist=None):
    worker_file = os.path.dirname(__file__) + os.sep + 'gpuworker.py'
    ndev = len(devslist)
    if type(data)==list:
        ndev = len(data)

    workers = MPI.COMM_SELF.Spawn(sys.executable, args=[worker_file], maxprocs = ndev)
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
        for i in xrange(ndev):
            todat = np.asarray(data[partitions[i]:partitions[i+1]], dtype='d')
            task = Init_Task(todat.shape[0], todat.shape[1], int(devslist[i]))
            workers.isend(task, dest=i, tag=11)
            workers.Send([todat, MPI.DOUBLE], dest=i, tag=12)
            workers.recv(source=i, tag=13)

            i+=1
    else: ## HDP .. one or more datasets per GPU
        for i in xrange(ndev):
            todat = np.asarray(data[i], dtype='d')
            task = Init_Task(todat.shape[0], todat.shape[1], int(devslist[i%len(devslist)]))
            workers.isend(task, dest=i, tag=11)
            workers.Send([todat, MPI.DOUBLE], dest=i, tag=12)
            workers.recv(source=i, tag=13)

    return workers

def get_hdp_labels_GPU(workers, w, mu, Sigma, relabel=False):
    labels = []; Z = [];
    #import pdb; pdb.set_trace()
    ndev = workers.remote_group.size
    for i in xrange(ndev):
        theta = MCMC_Task(w[i], mu, Sigma, relabel)
        workers.isend(theta, dest=i, tag=11)

    for i in xrange(ndev):
        theta = workers.recv(source=i, tag=13)
        labels.append(theta.labs)
        Z.append(theta.Z)

    return labels, Z 

def get_labelsGPU(workers, w, mu, Sigma, relabel=False):
    # run all the threads
    ndev = workers.remote_group.size
    nobs, i = 0, 0
    partitions = [0]
    theta = MCMC_Task(w, mu, Sigma, relabel)
    for i in xrange(ndev):
        # give new params
        workers.isend(theta, dest=i, tag=11)
    #gather the results
    theta = []
    for i in xrange(ndev):
        theta.append(workers.recv(source=i, tag=13))
        nobs += theta[-1].nobs
        partitions.append(nobs)

    res = np.zeros(nobs, dtype=np.float32)
    if relabel:
        Z = res.copy()
    else:
        Z = None

    for i in xrange(ndev):
        res[partitions[i]:partitions[i+1]] = theta[i].labs
        if relabel:
            Z[partitions[i]:partitions[i+1]] = theta[i].Z

    return res, Z

def get_expected_labels_GPU(workers, w, mu, Sigma):
    # run all the threads
    ndev = workers.remote_group.size
    nobs, i = 0, 0
    partitions = [0]
    theta = BEM_Task(w, mu, Sigma)
    for i in xrange(ndev):
        # give new params
        workers.isend(theta, dest=i, tag=11)
    #gather results
    theta = []; 
    for i in xrange(ndev):
        theta.append(workers.recv(source=i, tag=13))
        nobs += theta[-1].nobs
        partitions.append(nobs)

    ncomp = len(w); ndim = theta[0].ndim
    dens = np.zeros((nobs, ncomp), dtype=np.float32)
    xbar = np.zeros((ncomp, ndim), dtype=np.float32)
    ct = np.zeros(ncomp, dtype=np.float32)
    ll = 0
    #import pdb; pdb.set_trace()
    for i in xrange(ndev):
        ll += theta[i].ll
        ct += theta[i].ct
        xbar += theta[i].xbar
        dens[partitions[i]:partitions[i+1], :] = theta[i].dens
        
    return ll, ct, xbar, dens

def kill_GPUWorkers(workers):
    #poison pill to each child 
    ndev = workers.remote_group.size
    msg = None
    for i in xrange(ndev):
        workers.isend(msg, dest=i, tag=11)
    workers.Disconnect()

    
        


