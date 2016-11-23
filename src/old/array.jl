const cuda_memstep = 1024
const cuda_ptrs = Dict{Int,Vector{Ptr{Void}}}()

type CudaArray{T,N}
  ptr::Ptr{T}
  dims::NTuple{N,Int}
  dev::Int
end

function allocate!{T}(a::CudaArray{T})
  id = length(a) * sizeof(T) ÷ cuda_memstep + 1
  if haskey(cuda_ptrs, id)
    ptrs = cuda_ptrs[id]
  else
    ptrs = Ptr{Void}[]
    cuda_ptrs[id] = ptrs
  end
  if length(ptrs) == 0
    p = Ptr{Void}[0]
    cudaMalloc(p, id * cuda_memstep)
    a.ptr = p[1]
  else
    a.ptr = convert(Ptr{T}, pop!(ptrs))
  end
end

function release{T}(a::CudaArray{T})
  id = length(a) * sizeof(T) ÷ cuda_memstep + 1
  p = convert(Ptr{Void}, a.ptr)
  push!(cuda_ptrs[id], p)
end

function device_gc()
  for ptrs in cuda_ptrs
    for p in ptrs
      cudaFree(p)
    end
  end
end

function CudaArray{T,N}(::Type{T}, dims::NTuple{N,Int})
  a = CudaArray(Ptr{T}(0), dims, 1)
  allocate!(a)
  finalizer(a, release)
  a
end
CudaArray{T}(::Type{T}, dims...) = CudaArray(T, dims)

typealias CudaVector{T} CudaArray{T,1}
typealias CudaMatrix{T} CudaArray{T,2}

Base.length(a::CudaArray) = prod(a.dims)
Base.size(a::CudaArray) = a.dims
Base.ndims{T,N}(a::CudaArray{T,N}) = N
Base.eltype{T}(a::CudaArray{T}) = T
Base.stride(a::CudaArray, dim::Int) = prod(size(a)[1:dim-1])
Base.pointer(a::CudaArray) = a.ptr

Base.similar{T}(a::CudaArray{T}) = CudaArray(T, size(a))

Base.unsafe_convert(::Type{Ptr{Void}}, a::CudaArray) = convert(Ptr{Void}, a.ptr)

Base.Array{T}(a::CudaArray{T}) = copy!(Array(T, size(a)), a)

# To limit the likelihood of ambiguity warnings with other packages,
# these should be the only two-argument definitions of copy!
#copy!(dst::Array, src::HostOrDevArray; stream=null_stream) = _copy!(dst, src, stream)
#copy!(dst::HostOrDevArray, src::HostOrDevArray; stream=null_stream) = _copy!(dst, src, stream)

function copy!{T}(dest::Array{T}, src::CudaArray{T}, stream=nullstream)
  if length(dest) != length(src)
    throw(ArgumentError("Inconsistent array length."))
  end
  nbytes = length(src) * sizeof(T)
  cudaMemcpyAsync(dest, src, nbytes, cudaMemcpyDeviceToHost, stream)
  dest
end
#_copy!{T}(dst::ContiguousArray{T}, src::ContiguousArray, stream) = _copy!(dst, to_eltype(T, src), stream)
#_copy!{T}(dst::AbstractCudaArray{T}, src, stream) = _copy!(dst, copy!(Array(T, size(src)), src), stream)
