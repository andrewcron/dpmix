"""
Support for multi-GPU via threading for dpmix MCMC
Written by: Andrew Cron
"""

from mpi4py import MPI
import numpy as np
import sys; import os
from utils import BEM_Task, MCMC_Task, Init_Task

########### Multi GPU ##########################
_datadevmap = {}
_dataind = {}

def init_GPUWorkers(data, devslist=None):

    worker_file = os.path.dirname(__file__) + os.sep + 'gpuworker.py'
    ndev = len(devslist)

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
        ndata = len(data)
        for i in xrange(ndata):
            dev = int(devslist[i%len(devslist)])
            thd = i%len(devslist)
            todat = np.asarray(data[i], dtype='d')
            task = Init_Task(todat.shape[0], todat.shape[1], dev)
            workers.isend(task, dest=thd, tag=11)
            workers.Send([todat, MPI.DOUBLE], dest=thd, tag=12)
            _dataind[i] = workers.recv(source=thd, tag=13)
            _datadevmap[i] = thd

    return workers

def get_hdp_labels_GPU(workers, w, mu, Sigma, relabel=False):

    #import pdb; pdb.set_trace()
    ndev = workers.remote_group.size
    ndata = len(_datadevmap)

    tasks = []
    for _i in xrange(ndev): tasks.append([])
    labels = []; Z = [];
    for _i in xrange(ndata): labels.append(None); Z.append(None)

    ## setup task
    for i in xrange(ndata):
        tasks[_datadevmap[i]].append(MCMC_Task(Sigma.shape[0], relabel, _dataind[i], i))

    for i in xrange(ndev):
        workers.isend(tasks[i], dest=i, tag=11)
        for tsk in tasks[i]:
            gid = tsk.gid
            workers.Send([np.asarray(w[gid].copy(),dtype='d'), MPI.DOUBLE], dest=i, tag=21)
            workers.Send([np.asarray(mu, dtype='d'), MPI.DOUBLE], dest=i, tag=22)
            workers.Send([np.asarray(Sigma, dtype='d'), MPI.DOUBLE], dest=i, tag=23)

    for i in xrange(ndev):
        results = workers.recv(source=i, tag=13)
        for res in results:
            labs = np.empty(res.nobs, dtype='i')
            workers.Recv([labs, MPI.INT], source=i, tag=21)
            labels[res.gid] = labs
            if relabel:
                cZ = np.empty(res.nobs, dtype='i')
                workers.Recv([cZ, MPI.INT], source=i, tag=22)
                Z[res.gid] = cZ


    return labels, Z 

def get_labelsGPU(workers, w, mu, Sigma, relabel=False):
    # run all the threads
    ndev = workers.remote_group.size
    nobs, i = 0, 0
    partitions = [0]
    theta = [MCMC_Task(len(w), relabel)]
    for i in xrange(ndev):
        # give new params
        workers.isend(theta, dest=i, tag=11)
        workers.Send([np.asarray(w,dtype='d'), MPI.DOUBLE], dest=i, tag=21)
        workers.Send([np.asarray(mu,dtype='d'), MPI.DOUBLE], dest=i, tag=22)
        workers.Send([np.asarray(Sigma,dtype='d'), MPI.DOUBLE], dest=i, tag=23)
    #gather the results
    theta = []; labs=[]; Zs=[];
    for i in xrange(ndev):
        theta.append(workers.recv(source=i, tag=13))
        nobs += theta[-1][0].nobs
        partitions.append(nobs)
        lab = np.empty(theta[-1][0].nobs, dtype='i')
        workers.Recv([lab, MPI.INT], source=i, tag=21)
        labs.append(lab)
        if relabel:
            Z = np.empty(theta[-1][0].nobs, dtype='i')
            workers.Recv([Z, MPI.INT], source=i, tag=22)
            Zs.append(Z)

    res = np.zeros(nobs, dtype='i')
    if relabel:
        Z = res.copy()
    else:
        Z = None

    for i in xrange(ndev):
        res[partitions[i]:partitions[i+1]] = labs[i]
        if relabel:
            Z[partitions[i]:partitions[i+1]] = Zs[i]

    return res, Z

def get_expected_labels_GPU(workers, w, mu, Sigma):
    # run all the threads
    ndev = workers.remote_group.size
    ncomp = len(w)
    ndim = Sigma.shape[1]
    nobs, i = 0, 0
    partitions = [0]
    theta = [BEM_Task(len(w))]
    for i in xrange(ndev):
        # give new params
        workers.isend(theta, dest=i, tag=11)
        workers.Send([np.asarray(w,dtype='d'), MPI.DOUBLE], dest=i, tag=21)
        workers.Send([np.asarray(mu,dtype='d'), MPI.DOUBLE], dest=i, tag=22)
        workers.Send([np.asarray(Sigma,dtype='d'), MPI.DOUBLE], dest=i, tag=23)

    #gather results
    theta = []; xbars = []; densities=[]; cts = []
    ll = 0

    for i in xrange(ndev):
        theta.append(workers.recv(source=i, tag=13))
        nobs += theta[-1][0].nobs
        ll += theta[-1][0].ll
        partitions.append(nobs)
        ct = np.empty(ncomp, dtype='d')
        workers.Recv([ct, MPI.DOUBLE], source=i, tag=21)
        cts.append(ct)
        xbar = np.empty(ncomp*ndim, dtype='d')
        workers.Recv([xbar, MPI.DOUBLE], source=i, tag=22)
        xbars.append(xbar.reshape(ncomp, ndim))
        dens = np.empty(theta[-1][0].nobs*ncomp, dtype='d')
        workers.Recv([dens, MPI.DOUBLE], source=i, tag=23)
        densities.append(dens.reshape(theta[-1][0].nobs, ncomp))

    dens = np.zeros((nobs, ncomp), dtype='d')
    xbar = np.zeros((ncomp, ndim), dtype='d')
    ct = np.zeros(ncomp, dtype='d')
    #import pdb; pdb.set_trace()
    for i in xrange(ndev):
        ct += cts[i]
        xbar += xbars[i]
        dens[partitions[i]:partitions[i+1], :] = densities[i]
        
    return ll, ct, xbar, dens

def kill_GPUWorkers(workers):
    #poison pill to each child 
    ndev = workers.remote_group.size
    msg = None
    for i in xrange(ndev):
        workers.isend(msg, dest=i, tag=11)
    workers.Disconnect()

    
        


