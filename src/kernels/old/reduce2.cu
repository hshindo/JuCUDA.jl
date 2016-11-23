#define NDCOPY_INDEX \
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x; \
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y; \
  int idx_z  = threadIdx.z + blockIdx.z * blockDim.z; \
  if (idx_x >= size1 || idx_y >= src_size2 || idx_z >= size3) return; \
  int src_idx = idx_x + size1*(idx_y + src_offset + src_size2*idx_z); \
  int dst_idx = idx_x + size1*(idx_y + dst_offset + dst_size2*idx_z)

typedef struct CUarray_st {
  void *value;
  int *stride;
} CUarray;

template <typename T>
__device__ void ndcopy3(CUndarray &src, CUndarray &dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  x[idx] = a.stride1;
}

template <typename T>
__device__ void ndcopy2(T *x, CUndarray &a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  x[idx] = a.stride1;
}

#define NDCOPY_INDEX \
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x; \
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y; \
  int idx_z  = threadIdx.z + blockIdx.z * blockDim.z; \
  if (idx_x >= size1 || idx_y >= src_size2 || idx_z >= size3) return; \
  int src_idx = idx_x + size1*(idx_y + src_offset + src_size2*idx_z); \
  int dst_idx = idx_x + size1*(idx_y + dst_offset + dst_size2*idx_z)

template <typename T>
__device__ void ndcopy(T *src, T *dst, int size1, int src_size2, int size3,
  int dst_size2, int src_offset, int dst_offset) {
  NDCOPY_INDEX;
  dst[dst_idx] = src[src_idx];
}

#define NDCOPY_FUN(dtype) \
  __global__ void ndcopy ## _ ## dtype(dtype *src, dtype *dst, int size1, \
    int src_size2, int size3, int dst_size2, int src_offset, int dst_offset) { \
    ndcopy(src, dst, size1, src_size2, size3, dst_size2, src_offset, dst_offset); \
  }

extern "C" {
  NDCOPY_FUN(float)
  NDCOPY_FUN(double)
  __global__ void ndcopy2_float(float *x, CUndarray a) { ndcopy2(x,a); }
}
