@generated function Base.copy!{T}(dest::AbstractCuArray{T,2}, src::AbstractCuArray{T,2})
    CT = cstring(T)
    f = compile("""
    copy(Array<$CT,2> dest, Array<$CT,2> src) {
        int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
        int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
        if (idx0 >= src.dims[0] || idx1 >= src.dims[1]) return;
        dest(idx0,idx1) = src(idx0,idx1);
    }""")
    quote
        @assert size(dest) == size(src)
        launch($f, size(src), (dest,src))
        dest
    end
end

@generated function Base.copy!{T}(dest::AbstractCuArray{T,3}, src::AbstractCuArray{T,3})
    CT = cstring(T)
    f = compile("""
    copy(Array<$CT,$N> dest, Array<$CT,$N> src) {
        int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
        int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
        int idx2 = threadIdx.z + blockIdx.z * blockDim.z;
        if (idx0 >= src.dims[0] || idx1 >= src.dims[1] || idx2 >= src.dims[2]) return;
        dest(idx0,idx1,idx2) = src(idx0,idx1,idx2);
    }""")
    quote
        @assert size(dest) == size(src)
        launch($f, size(src), (dest,src))
        dest
    end
end

#Base.copy{T}(src::CuSubArray{T}) = copy!(similar(src), src)
