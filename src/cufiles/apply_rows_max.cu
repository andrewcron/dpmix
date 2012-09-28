
__global__ void
apply_rows_max(float* X, /** matrix to apply .. row major **/
	               	      float* y, /** result vector  **/
 			      int* iy, 
			      int rows,
			      int cols
  ) {

  unsigned int thidx = threadIdx.x;
  unsigned int thidy = threadIdx.y;
  unsigned int bid = blockIdx.x;
  unsigned int bdx = blockDim.x; // assumed equal to blockDim.y .. 16 or 32 ..

  unsigned int stride = bdx + 1; // shared mem padded for bank conflicts
  unsigned int currow = bdx*bid;

  // flexible block size 
  extern __shared__ float shared_data[];
  float *sh_max = shared_data + bdx*stride;

//  if( thidy == 0 && thidx + currow < rows  ){
//      sh_max[thidx] = -1e37;
//  }
//  __syncthreads();  
  
  float cur_val; float new_val; int argmax=0;
  for(int chunk = 0; chunk < cols; chunk+=bdx){
  	  // get some values chunking accross rows ...
	  if(currow+thidy < rows && chunk + thidx < cols){
	  	shared_data[thidx*stride + thidy] = X[(currow + thidy)*cols + chunk + thidx];}
	  __syncthreads();
	  // get maximum in chunk ...
  	  if( thidy == 0 && thidx + currow < rows ){
	      // if first val, it's the max
	      if( chunk==0 ){
	         sh_max[thidx] = shared_data[thidx];
	      }
              // get maximmum in chunk ... 
	      for( int i = 0; i < bdx; i++){
	      	   if(chunk + i < cols){
	      	      cur_val = sh_max[thidx];
		      new_val = shared_data[i*stride + thidx];
		      if( cur_val < new_val ){
		         sh_max[thidx] = new_val;
			 argmax = chunk + i;
		      }
                   }
	      }
	  }
	  __syncthreads();
  }
  // save values
  if(thidx + currow < rows && thidy==0){
    y[currow+thidx] = sh_max[thidx];
    iy[currow+thidx] = argmax;
  }

}

   