
__global__ void
apply_rows_max(float* X, /** matrix to apply .. row major **/
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

  if( thidy == 0 && thidx + currow < rows  ){
      y[currow+thidx] = -1e37;
      sh_max[thidx] = -1e37;
  }
  __syncthreads();  
  
  float cur_val;
  for(int chunk = 0; chunk < cols; chunk+=bdx){
  	  // get some values chunking accross rows ...
	  if(currow+thidy < rows && chunk + thidx < cols){
	  	shared_data[thidx*bdx + thidy] = X[(currow + thidy)*cols + chunk + thidx];}
	  __syncthreads();
	  // get maximum in chunk ...
  	  if( thidy == 0 && thidx + currow < rows ){
	      for( int i = 0; i < bdx; i++){
	      	   if(chunk + i < cols){
	      	      cur_val = sh_max[thidx];
	      	      sh_max[thidx]  = fmax(cur_val, shared_data[i*bdx + thidx]);
                   }
	      }
	  }
	  __syncthreads();
  }
  // save values
  if(thidx + currow < rows && thidy==0){
    y[currow+thidx] = sh_max[thidx];}

}

   