=====
dpmix
=====

dpmix is a library for understanding posterior distributions for
Dirichlet and heirarchical Dirichlet mixtures of normal distributions
represented by truncated stick breaking.

Requirements
------------

* NumPy
* SciPy
* Cython
* PyCUDA
* cyarma
* cyrand
* scikits.cuda
* gpustats
* mpi4py

Installation and testing
------------------------

Install via

::

   python setup.py install

To test, run the scripts in the "test" subfolder.

Usage
-----

Check out the class docstrings for more info.

MPI
---

The multigpu facilities are developed using MPI. Therefore, 
using multiple machines is possible. However, note that the
machines must be configured the same way. (Python)

Running the code on multiple machines requires *mpiexec*:

::

   mpiexec -hostfile my_hosts -np 1 python tests/test_dpmix.py --gpu MPI

Where the *my_hosts* file looks like 

::

   host1 slots=3
   host2 slots=2

I'm assuming here that the master instance of python is running on host1
and that host1 and host2 have 2 GPUs each. Note, an extra slot needs to be
reserved for the master on host1. Furthermore, we need to specify which
devices to use on each host. The *gpu* argument in the class constructors
must be a dictionary like

::

  gpu={'host1': [0,1], 'host2': [0,1]}

The keys must match the result of a call to *os.uname()* to get the
host string.



