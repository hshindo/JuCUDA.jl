function Base.sum{T}(x::CuArray{T}, dim::Int)
    CT = cstring(T)
    f = @nvrtc "reduce_$CT" """
        __global__ void reduce_$CT(Array<$CT,1> x, Array<$CT,1> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        $CT val = 0;
        for (int i = idx; i < x.length(); i += blockDim.x*gridDim.x) val += x[i];
        for (int i = warpSize/2; i >= 1; i /= 2) val += __shfl_down(val, i);
        if (threadIdx.x % warpSize == 0) atomicAdd(y.data, val);
        }
        """
    y = zeros(x)
    culaunch(f, (length(x),), (x,y))
    py
end

function Base.max{T}(x::CuArray{T}, dim::Int)
    CT = cstring(T)
    f = @nvrtc "max_$CT" """
        $array_h
        extern "C" __global__ void max_$CT(Array<$CT,1> x, Array<$CT,1> y) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            $CT val = 0;
            for (int i = idx; i < x.length(); i += blockDim.x*gridDim.x) val += x[i];
            for (int i = warpSize/2; i >= 1; i /= 2) {
                val = max(val, __shfl_down(val, i));
            }
            if (threadIdx.x % warpSize == 0) atomicMax(y.data, val);
        }
        """
    y = CuArray(zeros(Array(x)))
    culaunch(f, (length(x),), (x,y))
    y
end
