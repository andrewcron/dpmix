
__global__ void
sweep_rows_%(name)s(float* X, /** matrix to sweep in place **/
	               	      float* y, /** row vector to remove **/
			      int rows,
			      int cols
  ) {
  // NOTE: assumes all of y can be held in shared mem ....

  unsigned int thidx = threadIdx.x;
  unsigned int thidy = threadIdx.y;
  unsigned int bid = blockIdx.x;
  unsigned int bdx = blockDim.x; // assumed equal to blockDim.y .. 16 or 32 ..
  unsigned int tid = thidy*bdx+thidx;

  int currow = bdx*bid;

  // flexible block size 
  extern __shared__ float shared_data[];

  // get the row to sweep ....
  for(int chunk = 0; chunk < cols; chunk+=bdx*bdx){
  	  if(tid+chunk<cols){
	    	  shared_data[tid+chunk] = y[tid+chunk];
	  }
  }
  __syncthreads();

  if(currow + thidy < rows) {
    for(int chunk = 0; chunk < cols; chunk+=bdx){
    	  // get some values chunking accross rows ...
	  if(chunk + thidx < cols){
	     X[(currow + thidy)*cols + chunk + thidx] = \
	     	l_%(name)s(X[(currow + thidy)*cols + chunk + thidx], shared_data[chunk+thidx]);
	  }	
    }
  }
}

