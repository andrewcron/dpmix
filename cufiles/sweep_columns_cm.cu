
__global__ void
sweep_columns_%(name)s_cm(float* X, /** matrix to sweep in place **/
	               	      float* y, /** column vector to remove **/
			      int rows,
			      int cols) {

  unsigned int thidx = threadIdx.x;
  unsigned int thidy = threadIdx.y;
  unsigned int bid = blockIdx.x;
  unsigned int bdx = blockDim.x; // assumed equal to blockDim.y .. 16 or 32 ..

  int currow = bdx*bid;

  // flexible block size 
  extern __shared__ float shared_data[];
  
  if( currow + thidx < rows){  
    shared_data[thidx] = y[currow+thidx];}
  __syncthreads();

  for(int chunk = 0; chunk < cols; chunk+=bdx){
  	  // get some values chunking accross rows ...
	  if(currow + thidx < rows && chunk + thidy < cols){
	  	    X[currow + thidx + rows*(chunk + thidy)] = \
	  	    	l_%(name)s(X[currow + thidx + rows*(chunk + thidy)], shared_data[thidx]);
	  }
  }
}
