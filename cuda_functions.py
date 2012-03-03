from kernels import CUDA_Kernels
import numpy as np
from pycuda.gpuarray import to_gpu
from pycuda.gpuarray import GPUArray
from gpustats.util import DeviceInfo

devinfo = DeviceInfo()

def gpu_sweep_col_diff(X, y):
    """ X - y = X across the columns """
    if type(X)==GPUArray:
        gX = X
    else:
        gX = to_gpu(np.asarray(X, dtype=np.float32))

    if type(y)==GPUArray:
        gy = y
    else:
        gy = to_gpu(np.asarray(y, dtype=np.float32))

    dims = np.asarray(X.shape, dtype=np.int32)
    if devinfo.max_block_threads >= 1024:
        blocksize = 32
    else:
        blocksize = 16

    gridsize = int(dims[0] / blocksize) + 1
    shared = 4*16

    if gX.flags.c_contiguous:
        func = CUDA_Kernels.get_function("sweep_columns_diff")
    else:
        func = CUDA_Kernels.get_function("sweep_columns_diff_cm")

    func(gX, gy, dims[0], dims[1], block=(blocksize, blocksize,1),
         grid = (gridsize,1), shared = shared)

    if type(y)!=GPUArray:
        X = gX.get()


def gpu_sweep_col_div(X, y):
    """ X / y = X across the columns """
    if type(X)==GPUArray:
        gX = X
    else:
        gX = to_gpu(np.asarray(X, dtype=np.float32))

    if type(y)==GPUArray:
        gy = y
    else:
        gy = to_gpu(np.asarray(y, dtype=np.float32))

    dims = np.asarray(X.shape, dtype=np.int32)
    if devinfo.max_block_threads >= 1024:
        blocksize = 32
    else:
        blocksize = 16

    gridsize = int(dims[0] / blocksize) + 1
    shared = 4*16

    if gX.flags.c_contiguous:
        func = CUDA_Kernels.get_function("sweep_columns_div")
    else:
        func = CUDA_Kernels.get_function("sweep_columns_div_cm")

    func(gX, gy, dims[0], dims[1], block=(blocksize, blocksize,1),
         grid = (gridsize,1), shared = shared)

    if type(y)!=GPUArray:
        X = gX.get()


def gpu_sweep_row_diff(X, y):
    """ X - y = X down the rows """
    if type(X)==GPUArray:
        gX = X
    else:
        gX = to_gpu(np.asarray(X, dtype=np.float32))

    if type(y)==GPUArray:
        gy = y
    else:
        gy = to_gpu(np.asarray(y, dtype=np.float32))

    dims = np.asarray(X.shape, dtype=np.int32)
    if devinfo.max_block_threads >= 1024:
        blocksize = 32
    else:
        blocksize = 16

    gridsize = int(dims[0] / blocksize) + 1
    shared = 4*dim[1]

    if gX.flags.c_contiguous:
        func = CUDA_Kernels.get_function("sweep_columns_diff")
    else:
        func = CUDA_Kernels.get_function("sweep_columns_diff_cm")
    func(gX, gy, dims[0], dims[1], block=(blocksize, blocksize,1),
         grid = (gridsize,1), shared = shared)

    if type(y)!=GPUArray:
        X = gX.get()

def gpu_sweep_row_div(X, y):
    """ X / y = X down the rows """
    if type(X)==GPUArray:
        gX = X
    else:
        gX = to_gpu(np.asarray(X, dtype=np.float32))

    if type(y)==GPUArray:
        gy = y
    else:
        gy = to_gpu(np.asarray(y, dtype=np.float32))

    dims = np.asarray(X.shape, dtype=np.int32)
    if devinfo.max_block_threads >= 1024:
        blocksize = 32
    else:
        blocksize = 16

    gridsize = int(dims[0] / blocksize) + 1
    shared = 4*dim[1]

    if gX.flags.c_contiguous:
        func = CUDA_Kernels.get_function("sweep_columns_div")
    else:
        func = CUDA_Kernels.get_functions("sweep_columns_div_cm")

    func(gX, gy, dims[0], dims[1], block=(blocksize, blocksize,1),
         grid = (gridsize,1), shared = shared)

    if type(y)!=GPUArray:
        X = gX.get()

def gpu_apply_row_max(X):
    """ 
    max(X) = y across the rows 

    returns the gpuarray, y
    """
    if type(X)==GPUArray:
        gX = X
    else:
        gX = to_gpu(np.asarray(X, dtype=np.float32))

    dims = np.asarray(X.shape, dtype=np.int32)

    gy = to_gpu(np.zeros(dims[0], dtype=np.float32))

    if devinfo.max_block_threads >= 1024:
        blocksize = 32
    else:
        blocksize = 16

    gridsize = int(dims[0] / blocksize) + 1
    shared = 4*blocksize*(blocksize+1)

    if gX.flags.c_contiguous:
        func = CUDA_Kernels.get_function("apply_rows_max")
    else:
        func = CUDA_Kernels.get_function("apply_rows_max_cm")

    func(gX, gy, dims[0], dims[1], block=(blocksize, blocksize,1),
         grid = (gridsize,1), shared = shared)

    return gy




    

        
