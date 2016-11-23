template<typename T, int N>
struct Array {
  T *data;
  const int dims[N];
  const int strides[N];
public:
  __device__ int length() {
    int n = dims[0];
    for (int i = 1; i < N; i++) n *= dims[i];
    return n;
  }
  __device__ T &operator[](int idx) { return data[idx]; }
  __device__ T &operator()(int i0) {
    if (N == 1) return data[i0*strides[0]];

    int cumdims[N];
    cumdims[0] = 1;
    for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

    int temp = i0;
    for (int i = N-1; i >= 0; i--) {
      int a = temp / cumdims[i];
      temp -= a * cumdims[i];
      cumdims[i] = a;
    }
    return (*this)(cumdims);
  }
  __device__ T &operator()(int i0, int i1) {
    return data[i0*strides[0] + i1*strides[1]];
  }
  __device__ T &operator()(int i0, int i1, int i2) {
    return data[i0*strides[0] + i1*strides[1] + i2*strides[2]];
  }
  __device__ T &operator()(int idxs[]) {
    int idx = 0;
    for (int i = 0; i < N; i++) idx += idxs[i] * strides[i];
    return data[idx];
  }
};

template<int N>
struct Dims {
  const int data[N];
public:
  __device__ int operator[](int idx) { return data[idx]; }
};
