
__global__ void
apply_rows_max_cm(float* X, /** matrix to apply ... column major **/
	               	      float* y, /** result vector  **/
			      int rows,
			      int cols
  ) {

  unsigned int thidx = threadIdx.x;
  unsigned int thidy = threadIdx.y;
  unsigned int bid = blockIdx.x;
  unsigned int bdx = blockDim.x; // assumed equal to blockDim.y .. 16 or 32 ..

  int currow = bdx*bid;

  // flexible block size 
  extern __shared__ float shared_data[];
  float *sh_max = shared_data + bdx*bdx;

  // initialize ...
  if( thidy == 0 && thidx + currow < rows ){
      y[currow+thidx] = -1e37;
      sh_max[thidx] = -1e37;
  }
  __syncthreads();  

  float cur_val;
  for(int chunk = 0; chunk < cols; chunk+=bdx){
  	  // get some values chunking accross rows ...
	  if( thidx + currow < rows && chunk + thidy < cols ){
	      shared_data[thidy*bdx + thidx] = X[thidx + currow + (chunk + thidy)*rows];	
	  }
	  __syncthreads();
	  // get maximum in chunk ...
  	  if( thidy == 0 && thidx + currow < rows ){
	      for( int i = 0; i < bdx; i++){
	      	   if(chunk + i < cols){
	      	      cur_val = sh_max[thidx];
	      	      sh_max[thidx] = fmax(cur_val, shared_data[i*bdx + thidx]);
                   }
	      }
	  }
	  __syncthreads();
  }

  // save results
  if(thidx + currow < rows && thidy==0){
    y[currow + thidx] = sh_max[thidx];}

}

