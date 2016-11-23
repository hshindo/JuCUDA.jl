cubox(p::CuPtr) = cubox(CUdeviceptr(p))
cubox{T}(x::T) = T[x]

const blocksize = (128, 1, 1)

function gridsize(dim1::Int, dim2::Int, dim3::Int)
  x = ceil(dim1 / blocksize[1])
  y = ceil(dim2 / blocksize[2])
  z = ceil(dim3 / blocksize[3])
  x, y, z
end
gridsize(dims::NTuple{1,Int}) = gridsize(dims[1],1,1)
gridsize(dims::NTuple{2,Int}) = gridsize(dims[1],dims[2],1)
gridsize(dims::NTuple{3,Int}) = gridsize(dims[1],dims[2],dims[3])

function Base.launch(f::CuFunction, dims::Tuple, args::Tuple;
  sharedMemBytes::Int=4, stream=C_NULL)

  kargs = Any[cubox(a) for a in args]
  cuLaunchKernel(f, gridsize(dims)..., blocksize..., sharedMemBytes, stream, kargs, C_NULL)
end
