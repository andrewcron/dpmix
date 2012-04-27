import pycuda.driver as drv
if drv.Context.get_current() is None:
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

class Compiled_Kernels(object):
    """
    Small wrapper to SourceModule that will check for
    and handle context changes!
    """
    def __init__(self, src):
        self.src = src
        self.modules = { drv.Context.get_current() : SourceModule(self.src) }
        #self.curDevice = drv.Context.get_device()

    def get_function(self, fn_str):
        context = drv.Context.get_current()
        try:
            mod = self.modules[context]
        except KeyError:
            self.modules[context] = SourceModule(self.src)
            mod = self.modules[context]
        return mod.get_function(fn_str)
        #curDevice = drv.Context.get_device()
        #if self.curDevice != curDevice:
        #    self.module = SourceModule(self.src)
        #    self.curDevice = drv.Context.get_device()
        #return self.module.get_function(fn_str)


CUDA_Kernels = Compiled_Kernels(full_code)
