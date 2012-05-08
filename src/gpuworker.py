#!/usr/bin/env python
from mpi4py import MPI

import numpy as np
import time

import pycuda.driver as drv
import gpustats
import gpustats.sampler as gsamp
import gpustats.util as gutil
import cuda_functions as cufuncs
from pycuda import cumath
from pycuda.elementwise import ElementwiseKernel
import pycuda.tools as pytools
from pycuda.gpuarray import to_gpu

import sys; import os;
homepath = sys.path[0]
if os.path.basename(homepath)=='dpmix':
    sys.path[0] = os.path.dirname(sys.path[0])
    from dpmix.utils import MCMC_Task, BEM_Task, Init_Task
else:
    from utils import MCMC_Task, BEM_Task, Init_Task


comm = MPI.Comm.Get_parent()

# MPI Tag Key:
# 11 -- new task
# 12 -- ctypes streams
# 13 -- completed task

_init = False
_logmnflt = np.log(1e-37)
iexp = ElementwiseKernel("float *z", "z[i] = (z[i] < -40.0) ? 0.0 : expf(z[i]);", "inplexp")
### Code needs to be moved out of tasks ... pretty sure ...
while True:
    # get task ... manual wait to decrease CPU impact 2% load
    while True:
        if comm.Iprobe(source=0, tag=11):
            break
        time.sleep(0.001)
    task = comm.recv(source=0, tag=11)

    # process task or pill
    if task is None:
        break #poison pill 
    elif isinstance(task, Init_Task):
        # no reinit for 2nd dataset ... 
        if _init is False:
            dev_num = task.dev_num
            gutil.threadSafeInit(task.dev_num)
            alldata = []
            gdata = []
            dataind = 0
            _init = True
        else:
            dataind = len(alldata)
            
        data = np.empty(task.nobs*task.ndim, dtype='d')
        comm.Recv([data, MPI.DOUBLE], source=0, tag=12)
        data = data.reshape(task.nobs, task.ndim)
        alldata.append(data)
        gdata.append(to_gpu(np.asarray(data, dtype=np.float32)))

        task = dataind
        comm.send(task, dest=0, tag=13)
        #print 'memory on dev ' + str(dev_num) + ': ' + str(drv.mem_get_info())
    elif isinstance(task, list):
        for subtask in task:
            if isinstance(subtask, MCMC_Task):
                ## do GPU work ... 
                dataind = subtask.dataind
                nobs, ndom = alldata[dataind].shape

                densities = gpustats.mvnpdf_multi(gdata[dataind], subtask.mu, subtask.Sigma,
                                                  weights = subtask.w.flatten(), get=False, logged=True,
                                                  order='C')
                labs = gsamp.sample_discrete(densities, logged=True)
                if subtask.relabel:
                    Z = cufuncs.gpu_apply_row_max(densities)[1].get()
                else:
                    Z = None
        
                subtask.labs = labs
                subtask.Z = Z
                subtask.nobs = nobs
                del subtask.mu
                del subtask.w
                del subtask.Sigma

                densities.gpudata.free()
                del densities
                #comm.send(task, dest=0, tag=13) # return it
                #print 'memory on dev ' + str(dev_num) + ': ' + str(drv.mem_get_info())

            elif isinstance(subtask, BEM_Task):

                ncomp = len(subtask.w)
                nobs, ndim = alldata[dataind].shape

                densities = gpustats.mvnpdf_multi(gdata[dataind], subtask.mu, subtask.Sigma,
                                                  weights = subtask.w.flatten(), get=False, logged=True)

                dens = np.asarray(densities.get(), dtype=np.float, order='C')
                dens = np.exp(dens)
                norm = dens.sum(1)
                subtask.ll = np.sum(np.log(norm))
                dens = (dens.T / norm).T
                subtask.ct = dens.sum(0)
                subtask.xbar = np.dot(dens.T, alldata[dataind])
                subtask.dens = dens


                subtask.nobs = nobs
                subtask.ndim = ndim
                # subtask.dens = h_densities
                del subtask.mu, subtask.Sigma, subtask.w

                ## Free Everything
                densities.gpudata.free()

        comm.send(task, dest=0, tag=13)


## the end 
comm.Disconnect()

