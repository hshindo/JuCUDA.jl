// reduce_float

template <typename T>
__device__ void reduce(T *x, T *y, int n, int stride) {

  extern __shared__ T sdata[];
  int blockSize = 128;
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockSize + threadIdx.x;
  int gridSize = blockSize * gridDim.x;

  T sum = x[idx];
  //idx += gridSize;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  //while (idx < n) {
  //  sum += x[idx];
  //  idx += gridSize;
  //}
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
  //  if (nIsPow2 || i + blockSize < n) mySum += g_idata[i+blockSize];
  //  i += gridSize;
  //}

  sdata[tid] = sum;
  __syncthreads();
  if ((blockSize >= 512) && (tid < 256)) sdata[tid] = sum = sum + sdata[tid+256];
  __syncthreads();
  if ((blockSize >= 256) &&(tid < 128)) sdata[tid] = sum = sum + sdata[tid+128];
   __syncthreads();
  if ((blockSize >= 128) && (tid <  64)) sdata[tid] = sum = sum + sdata[tid+64];
  __syncthreads();
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) sdata[tid] = sum = sum + sdata[tid+32];
  __syncthreads();
  if ((blockSize >= 32) && (tid < 16)) sdata[tid] = sum = sum + sdata[tid+16];
  __syncthreads();
  if ((blockSize >= 16) && (tid < 8)) sdata[tid] = sum = sum + sdata[tid+8];
  __syncthreads();
  if ((blockSize >= 8) && (tid < 4)) sdata[tid] = sum = sum + sdata[tid+4];
  __syncthreads();
  if ((blockSize >= 4) && (tid < 2)) sdata[tid] = sum = sum + sdata[tid+2];
  __syncthreads();
  if ((blockSize >= 2) && (tid < 1)) sdata[tid] = sum = sum + sdata[tid+1];
  __syncthreads();
  // write result for this block to global mem
  if (tid == 0) y[blockIdx.x] = sum;
}

extern "C" {
  __global__ void reduce_float(float *x, float *y, int n, int stride) {
    reduce(x, y, n, stride);
  }
}
