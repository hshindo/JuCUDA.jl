template<typename T>
__device__ void blockReduce(T *in, T *out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for (int i = idx; i < N; i += blockDim.x*gridDim.x) sum += in[i];
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);

  //write reduced value to shared memory
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);
  return val;
}
