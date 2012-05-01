#!/usr/bin/env python
from mpi4py import MPI

import numpy as np
import time

import pycuda.driver as drv
import gpustats
import gpustats.sampler as gsamp
import gpustats.util as gutil
import cuda_functions as cufuncs
from scikits.cuda import linalg as cuLA
from pycuda import cumath
from pycuda.elementwise import ElementwiseKernel
iexp = ElementwiseKernel("float *z", "z[i]=expf(z[i])", "inplexp")
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
# 13 -- completed task

gdata = None
g_ones_long = None

_logmnflt = np.log(1e-37)

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
        gutil.threadSafeInit(task.dev_num)
        cuLA.init()
        gdata = to_gpu(np.asarray(task.data, dtype=np.float32))
        nobs, ndim = task.data.shape
        g_ones_long = to_gpu(np.ones((nobs,1), dtype=np.float32))
        task = None
        comm.send(task, dest=0, tag=13)
    elif isinstance(task, MCMC_Task):
        ## do GPU work ... 
        densities = gpustats.mvnpdf_multi(gdata, task.mu, task.Sigma,
                                          weights = task.w.flatten(), get=False, logged=True,
                                          order='C')
        labs = gsamp.sample_discrete(densities, logged=True)
        if task.relabel:
            Z = cufuncs.gpu_apply_row_max(densities)[1].get()
        else:
            Z = None
        
        task.labs = labs
        task.Z = Z
        task.nobs = nobs
        del task.mu
        del task.w
        del task.Sigma

        densities.gpudata.free()
        del densities
        comm.send(task, dest=0, tag=13) # return it

    elif isinstance(task, BEM_Task):

        ncomp = len(task.w)
        g_ones = to_gpu(np.ones((ncomp, 1), dtype=np.float32))
        densities = gpustats.mvnpdf_multi(gdata, task.mu, task.Sigma,
                                          weights = task.w.flatten(), get=False, logged=True)
        tdens = gutil.GPUarray_reshape(densities, (ncomp, nobs), "C")

        ll = cuLA.dot(g_ones, cumath.exp(tdens), "T").get()

        nmzero = np.sum(ll==0)
        ll = np.sum(np.log(ll[ll>0])) + nmzero*_logmnflt

        nrm, _ = cufuncs.gpu_apply_row_max(densities)
        cufuncs.gpu_sweep_col_diff(densities, nrm)
        iexp(densities); gutil.GPUarray_order(densities, "F")
        nrm = cuLA.dot(g_ones, tdens, "T")
        cufuncs.gpu_sweep_col_div(densities, nrm)

        ct = cuLA.dot(tdens, g_ones_long).get().flatten()
        xbar = cuLA.dot(tdens, gdata).get()
        h_densities = densities.get()

        # store result in task
        task.ll = ll
        task.ct = ct
        task.xbar = xbar
        task.nobs = nobs
        task.ndim = ndim
        task.dens = h_densities
        del task.mu, task.Sigma, task.w

        ## Free Everything
        g_ones.gpudata.free()
        densities.gpudata.free()
        nrm.gpudata.free()

        comm.send(task, dest=0, tag=13)


## the end 
comm.Disconnect()

