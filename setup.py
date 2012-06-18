'''
Created on Mar 15, 2012

@author: Jacob Frelinger
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

setup(name='dpmix',
      version='0.1',
      packages=['dpmix'],
      package_dir={'dpmix': 'src'},
      description='Optimized (and optionally gpu enhaced) fitting of Gaussian Mixture Models',
      maintainer='Jacob Frelinger',
      maintainer_email='jacob.frelinger@duke.edu',
      author='Andrew Cron',
      author_email='andrew.cron@duke.edu',
      url='https://github.com/andrewcron/pycdp',
      requires=['numpy (>=1.3.0)',
                'scipy (>=0.6)',
                'matplotlib (>=1.0)',
                'cython (>=0.15.1)'],
      package_data={'dpmix': ['cufiles/*.cu']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("dpmix.munkres", 
                ["src/munkres.pyx", "src/cpp/Munkres.cpp"],
                include_dirs = [get_include(), 'src/cpp'],
                language='c++')],
      )
