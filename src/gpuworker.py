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
    
    task = np.array(0, dtype='i')
    comm.Recv([task, MPI.INT], source=0, tag=11) # -1 -- kill, 0 -- init
    print 'got task'
    print task

    # process task or pill
    if task == -1:
        break #poison pill 
    elif task == 0:
        params = np.empty(3, dtype='i')
        comm.Recv([params, MPI.INT], source=0, tag=12)
        print 'got params'
        dim0, dim1, n_dev_num = params
        print dim0
        print dim1
        print n_dev_num
        # no reinit for 2nd dataset ... 
        if _init is False:
            dev_num = int(n_dev_num)
            gutil.threadSafeInit(dev_num)
            alldata = []
            gdata = []
            dataind = 0
            _init = True
        else:
            dataind = len(alldata)
            
        data = np.empty(dim0*dim1, dtype='d')
        comm.Recv([data, MPI.DOUBLE], source=0, tag=13)
        print data
        print 'got data'
        data = data.reshape(dim0, dim1)
        alldata.append(data)
        gdata.append(to_gpu(np.asarray(data, dtype=np.float32)))

        task = np.array(dataind, dtype='i')
        comm.Send([task, MPI.INT], dest=0, tag=14)
        print 'sent result'
        #print 'memory on dev ' + str(dev_num) + ': ' + str(drv.mem_get_info())
    elif task == 1:
        results = []

        # get number of subtasks
        subtasknum = np.array(0, dtype='i')
        comm.Recv([subtasknum, MPI.INT], source=0, tag=12)

        #for subtask in task:
        for it in range(subtasknum):
            # get task parameters
            params = np.empty(4, dtype='i')
            comm.Recv([params, MPI.INT], source=0, tag=13)
            dataind, ncomp, ttype, gid = params
            nobs, ndim = alldata[dataind].shape
            
            ## get other inputs via ctype streams! 
            # w
            w = np.empty(ncomp, dtype='d')
            comm.Recv([w, MPI.DOUBLE], source=0, tag=21)
            # mu
            mu = np.empty(ncomp*ndim, dtype='d')
            comm.Recv([mu, MPI.DOUBLE], source=0, tag=22); 
            mu = mu.reshape(ncomp, ndim)
            # Sigma
            Sigma = np.empty(ncomp*ndim*ndim, dtype='d')
            comm.Recv([Sigma, MPI.DOUBLE], source=0, tag=23); 
            Sigma = Sigma.reshape(ncomp, ndim, ndim)
            
            if ttype>0: # 1 -- just densities ... 2 -- relabel too
                ## do GPU work ... 

                densities = gpustats.mvnpdf_multi(gdata[dataind], mu, Sigma,
                                                  weights = w.flatten(), get=False, logged=True,
                                                  order='C')

                labs = np.asarray(gsamp.sample_discrete(densities, logged=True), dtype='i')
                subresult = [np.array(nobs, dtype='i'), labs, np.array(gid, dtype='i')]
                if ttype==2:
                    Z = np.asarray(cufuncs.gpu_apply_row_max(densities)[1].get(), dtype='i')
                    subresult.append(Z)
                else:
                    Z = None

                results.append(subresult)

                #subtask.labs = labs
                #subtask.Z = Z
                #subtask.nobs = nobs
                #del subtask.mu
                #del subtask.w
                #del subtask.Sigma

                densities.gpudata.free()
                del densities

                #comm.send(task, dest=0, tag=13) # return it
                #print 'memory on dev ' + str(dev_num) + ': ' + str(drv.mem_get_info())

            elif ttype==0:
                print 'starting bem'
                densities = gpustats.mvnpdf_multi(gdata[dataind], mu, Sigma,
                                                  weights = w.flatten(), get=False, logged=True,
                                                  order='C')

                dens = np.asarray(densities.get(), dtype='d')
                dens = np.exp(dens)
                norm = dens.sum(1)
                ll = np.sum(np.log(norm))
                dens = (dens.T / norm).T

                ct = np.asarray(dens.sum(0), dtype='d')
                xbar = np.asarray(np.dot(dens.T, alldata[dataind]), dtype='d')
                dens = dens.copy('C')

                subresult = [np.array(nobs,dtype='i'), 
                             ct, xbar, dens, 
                             np.array(ll,dtype='d'),
                             np.array(gid, dtype='i')]
                results.append(subresult)


                #subtask.nobs = nobs
                #subtask.ndim = ndim
                # subtask.dens = h_densities
                #del subtask.mu, subtask.Sigma, subtask.w

                ## Free Everything
                densities.gpudata.free()
                
        # send results summary
        numres = np.array(len(results), dtype='i')
        comm.Send(numres, dest=0, tag=13)
        # send details
        for subresult in results:
            tag = 21
            for res in subresult:
                print 'sending results tag ' + str(tag)
                if np.issubdtype(res.dtype, 'float'):
                    comm.Send([res, MPI.DOUBLE], dest=0, tag=tag)
                else:
                    comm.Send([res, MPI.INT], dest=0, tag=tag)
                tag += 1

## the end 
comm.Disconnect()

