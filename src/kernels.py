import pycuda.autoinit
from pycuda.compiler import SourceModule

import os.path as pth

cu_file_path = pth.join(pth.abspath(pth.split(__file__)[0]), 'cufiles')

files = ("helpers.cu", "sweep_rows.cu", "sweep_columns.cu", "apply_rows_max.cu")

full_code = open(pth.join(cu_file_path,"helpers.cu")).read()
full_code += open(pth.join(cu_file_path,"sweep_rows.cu")).read() % { 'name' : 'diff' }
full_code += open(pth.join(cu_file_path,"sweep_rows.cu")).read() % { 'name' : 'div' }
full_code += open(pth.join(cu_file_path,"sweep_rows.cu")).read() % { 'name' : 'mult' }
full_code += open(pth.join(cu_file_path,"sweep_columns.cu")).read() % { 'name' : 'diff' }
full_code += open(pth.join(cu_file_path,"sweep_columns.cu")).read() % { 'name' : 'div' }
full_code += open(pth.join(cu_file_path,"sweep_columns.cu")).read() % { 'name' : 'mult' }
full_code += open(pth.join(cu_file_path,"apply_rows_max.cu")).read()
full_code += open(pth.join(cu_file_path,"sweep_rows_cm.cu")).read() % { 'name' : 'diff' }
full_code += open(pth.join(cu_file_path,"sweep_rows_cm.cu")).read() % { 'name' : 'div' }
full_code += open(pth.join(cu_file_path,"sweep_rows_cm.cu")).read() % { 'name' : 'mult' }
full_code += open(pth.join(cu_file_path,"sweep_columns_cm.cu")).read() % { 'name' : 'diff' }
full_code += open(pth.join(cu_file_path,"sweep_columns_cm.cu")).read() % { 'name' : 'div' }
full_code += open(pth.join(cu_file_path,"sweep_columns_cm.cu")).read() % { 'name' : 'mult' }
full_code += open(pth.join(cu_file_path,"apply_rows_max_cm.cu")).read()

CUDA_Kernels = SourceModule(full_code)
