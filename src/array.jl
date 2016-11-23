export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn, device
export reshape3, shiftcopy!

type CuArray{T,N} <: AbstractCuArray{T,N}
    ptr::CuPtr{T}
    dims::NTuple{N,Int}
end

typealias CuVector{T} CuArray{T,1}
typealias CuMatrix{T} CuArray{T,2}
typealias CuVecOrMat{T} Union{CuVector{T},CuMatrix{T}}

CuArray{T,N}(::Type{T}, dims::NTuple{N,Int}) = CuArray(CUDA.malloc(T,prod(dims)), dims)
CuArray{T}(::Type{T}, dims::Int...) = CuArray(T, dims)
CuArray{T,N}(src::Array{T,N}) = copy!(CuArray(T,size(src)), src)

Base.length(a::CuArray) = prod(a.dims)

Base.size(a::CuArray) = a.dims
Base.size(a::CuArray, d::Int) = a.dims[d]

Base.ndims{T,N}(a::CuArray{T,N}) = N

Base.eltype{T}(a::CuArray{T}) = T

function Base.stride(a::CuArray, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(a,i)
    end
    d
end

Base.strides{T}(a::CuArray{T,1}) = (1,)
Base.strides{T}(a::CuArray{T,2}) = (1,size(a,1))
Base.strides{T}(a::CuArray{T,3}) = (1,size(a,1),size(a,1)*size(a,2))
Base.strides{T}(a::CuArray{T,4}) = (1,size(a,1),size(a,1)*size(a,2),size(a,1)*size(a,2)*size(a,3))
function Base.strides{T,N}(a::CuArray{T,N})
    throw("Not implemented yet.")
end

Base.similar{T}(a::CuArray{T}) = CuArray(T, size(a))
Base.similar{T}(a::CuArray{T}, dims::NTuple) = CuArray(T, dims)
Base.similar{T}(a::CuArray{T}, dims::Int...) = similar(a, dims)

Base.convert{T}(::Type{Ptr{T}}, a::CuArray) = Ptr{T}(a.ptr)
Base.convert(::Type{CUDA.CUdeviceptr}, a::CuArray) = CUDA.CUdeviceptr(a.ptr)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::CuArray) = Ptr{T}(a.ptr)
Base.unsafe_convert(::Type{CUDA.CUdeviceptr}, a::CuArray) = CUDA.CUdeviceptr(a.ptr)

Base.zeros{T,N}(a::CuArray{T,N}) = zeros(CuArray{T}, a.dims)
Base.zeros{T}(::Type{CuArray{T}}, dims::Int...) = zeros(CuArray{T}, dims)
Base.zeros{T}(::Type{CuArray{T}}, dims) = fill(CuArray, T(0), dims)

Base.ones{T}(a::CuArray{T}) = ones(CuArray{T}, a.dims)
Base.ones{T}(::Type{CuArray{T}}, dims::Int...) = ones(CuArray{T}, dims)
Base.ones{T}(::Type{CuArray{T}}, dims) = fill(CuArray, T(1), dims)

function Base.copy!{T}(dest::Array{T}, src::CuArray{T}; stream=C_NULL)
    cucopy!(dest, src, CUDA.cuMemcpyDtoHAsync, length(src), stream)
end
function Base.copy!{T}(dest::CuArray{T}, src::Array{T}; stream=C_NULL)
    cucopy!(dest, src, CUDA.cuMemcpyHtoDAsync, length(src), stream)
end
function Base.copy!{T}(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL)
    cucopy!(dest, src, CUDA.cuMemcpyDtoDAsync, length(src), stream)
end
function Base.copy!{T}(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL)
    CUDA.cuMemcpyDtoDAsync(cupointer(dest,doffs), cupointer(src,soffs), n*sizeof(T), stream)
    dest
end

Base.copy(src::CuArray) = copy!(similar(src),src)

Base.pointer{T}(a::CuArray{T}, index::Int=1) = Ptr{T}(a) + sizeof(T) * (index-1)

Base.Array{T,N}(src::CuArray{T,N}) = copy!(Array(T,size(src)), src)

Base.isempty(a::CuArray) = length(a) == 0

Base.vec(a::CuArray) = ndims(a) == 1 ? a : CuArray(a.ptr, (length(a),))

Base.fill{T}(::Type{CuArray}, value::T, dims::NTuple) = fill!(CuArray(T,dims), value)

@generated function Base.fill!{T}(x::CuArray{T}, value::T)
    CT = cstring(T)
    f = compile("""
    fill(Array<$CT,1> x, $CT value) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < x.length()) x[idx] = value;
    }""")
    quote
        launch($f, (length(x),1,1), (vec(x),value))
        x
    end
end

Base.reshape{T,N}(a::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(a.ptr, dims)
Base.reshape{T}(a::CuArray{T}, dims::Int...) = reshape(a, dims)

Base.getindex(a::CuArray, key...) = copy!(view(a, key...))

function Base.setindex!{T,N}(y::CuArray{T,N}, x::CuArray{T,N}, indexes...)
    if N <= 3
        shift = [0,0,0]
        for i = 1:length(indexes)
            idx = indexes[i]
            if typeof(idx) == Colon
            elseif typeof(idx) <: Range
                # TODO: more range check
                @assert length(idx) == size(x,i)
                shift[i] = start(idx) - 1
            else
                throw("Invalid range: $(idx)")
            end
        end
        shiftcopy!(reshape3(y), reshape3(x), (shift[1],shift[2],shift[3]))
    else
        throw("Not implemented yet.")
    end
    #xx = view(x, key...)
    #copy!(xx, value)
end
function Base.setindex!{T}(x::CuArray{T}, value::T, key::Int)
    throw("Not implemented yet.")
end

function curand{T,N}(::Type{T}, dims::NTuple{N,Int})
    # TODO: use curand library
    CuArray(rand(T, dims))
end
curand(T::Type, dims::Int...) = curand(T, dims)

function curandn{T,N}(::Type{T}, dims::NTuple{N,Int})
    # TODO: use curand library
    CuArray(Array{T}(randn(dims)))
end
curandn(T::Type, dims::Int...) = curandn(T, dims)

device(a::CuArray) = a.ptr.dev

cupointer{T}(a::CuArray{T}, index::Int=1) = CuPtr{T}(pointer(a,index), device(a))

function cucopy!(dest, src, f::Function, n::Int, stream)
    nbytes = n * sizeof(eltype(src))
    f(dest, src, nbytes, stream)
    dest
end

function reshape3(x::CuArray, dim::Int)
    d1, d2, d3 = 1, size(x,dim), 1
    for i = 1:dim-1
        d1 *= size(x,i)
    end
    for i = dim+1:ndims(x)
        d3 *= size(x,i)
    end
    reshape(x, (d1,d2,d3))
end
reshape3{T}(x::CuArray{T,1}) = reshape(x, size(x,1), 1, 1)
reshape3{T}(x::CuArray{T,2}) = reshape(x, size(x,1), size(x,2), 1)
reshape3{T}(x::CuArray{T,3}) = x

@generated function shiftcopy!{T}(dest::CuArray{T,3}, src::CuArray{T,3}, shift::NTuple{3,Int})
    CT = cstring(T)
    f = compile("""
    shiftcopy(Array<$CT,3> dest, Array<$CT,3> src, Dims<3> shift) {
        int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
        int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
        int idx2 = threadIdx.z + blockIdx.z * blockDim.z;
        if (idx0 >= src.dims[0] || idx1 >= src.dims[1] || idx2 >= src.dims[2]) return;
        dest(idx0+shift[0], idx1+shift[1], idx2+shift[2]) = src(idx0, idx1, idx2);
    }""")
    quote
        launch($f, size(src), (dest,src,shift))
        dest
    end
end
