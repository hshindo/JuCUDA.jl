export CuSubArray
export view

type CuSubArray{T,N} <: AbstractCuArray{T,N}
    parent::CuArray{T}
    indexes::Tuple
    offset::Int
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}
end

typealias CuSubVector{T} CuArray{T,1}
typealias CuSubMatrix{T} CuArray{T,2}
typealias CuSubVecOrMat{T} Union{CuSubVector{T}, CuSubMatrix{T}}

Base.size(a::CuSubArray) = a.dims
Base.size(a::CuSubArray, dim::Int) = a.dims[dim]

Base.strides(a::CuSubArray) = a.strides
Base.strides(a::CuSubArray, dim::Int) = a.strides[dim]

Base.length(a::CuSubArray) = prod(a.dims)

Base.similar(a::CuSubArray) = similar(a, size(a))
Base.similar(a::CuSubArray, dims::Int...) = similar(a, dims)
Base.similar{T,N}(a::CuSubArray{T}, dims::NTuple{N,Int}) = CuArray(T, dims)

function Base.pointer(a::CuSubArray, index::Int=1)
    index == 1 && return pointer(a.parent, a.offset+index)
    throw("Not implemented yet.")
end

function cupointer(a::CuSubArray, index::Int=1)
    index == 1 && return cupointer(a.parent, a.offset+index)
    throw("Not implemented yet.")
end

function view{T,N}(a::CuArray{T,N}, indexes::Union{UnitRange{Int},Colon,Int}...)
    dims = Int[]
    strides = Int[]
    stride = 1
    offset = 0
    for i = 1:length(indexes)
        r = indexes[i]
        t = typeof(r)
        if t == Colon
            push!(dims, size(a,i))
            push!(strides, stride)
        elseif t == Int
            offset += stride * (r-1)
        else
            push!(dims, length(r))
            push!(strides, stride)
            offset += stride * (first(r)-1)
        end
        stride *= size(a,i)
    end
    CuSubArray(a, indexes, offset, tuple(dims...), tuple(strides...))
end

Base.Array(a::CuSubArray) = Array(a.parent)[a.indexes...]
Base.SubArray(a::CuSubArray) = view(Array(a.parent), a.indexes)
