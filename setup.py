'''
Created on Mar 15, 2012

@author: Jacob Frelinger
'''

from distutils.core import setup

setup(name='dpmix',
      version='0.1',
      packages=['dpmix'],
      package_dir={'dpmix': 'src'},
      description='Optimized (and optionally gpu enhaced) fitting of Gaussian Mixture Models',
      maintainer='Jacob Frelinger',
      maintainer_email='jacob.frelinger@duke.edu',
      url='https://github.com/andrewcron/pycdp',
      requires=['numpy (>=1.3.0)',
                'scipy (>=0.6)',
                'matplotlib (>=1.0)',
                'pymc (>=2.1)',
                'cython (>=0.15.1)'],
      package_data={'dpmix': ['cufiles/*.cu']}
      )
