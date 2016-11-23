"""
Interop types for CUDA native.
"""
immutable CUarray{T,N}
    ptr::Ptr{T}
    dims::N
    strides::N
end

CUDA.cubox(x::Union{CuArray,CuSubArray}) = CUDA.cubox(CUarray(x))
CUDA.cubox(t::NTuple{1,Int}) = CUDA.cubox(Cint1(t[1]))
CUDA.cubox(t::NTuple{2,Int}) = CUDA.cubox(Cint2(t[1],t[2]))
CUDA.cubox(t::NTuple{3,Int}) = CUDA.cubox(Cint3(t[1],t[2],t[3]))
CUDA.cubox(t::NTuple{4,Int}) = CUDA.cubox(Cint4(t[1],t[2],t[3],t[4]))
CUDA.cubox(t::NTuple{5,Int}) = CUDA.cubox(Cint5(t[1],t[2],t[3],t[4],t[5]))

CUarray(x::Union{CuArray,CuSubArray}) = CUarray(pointer(x), cint(size(x)), cint(strides(x)))

function CUarray3(x::CuArray, dim::Int)
    dims = Cint[1, size(x,dim), 1]
    for i = 1:dim-1
        dims[1] *= size(x, i)
    end
    for i = dim+1:ndims(x)
        dims[3] *= size(x, i)
    end
    cdims = Cint3(dims[1],dims[2],dims[3])
    cstrides = Cint3(Cint(1), cdims.i1, cdims.i1*cdims.i2)
    CUarray(pointer(x), cdims, cstrides)
end

immutable Cint1
    i1::Cint
end

immutable Cint2
    i1::Cint
    i2::Cint
end

immutable Cint3
    i1::Cint
    i2::Cint
    i3::Cint
end

immutable Cint4
    i1::Cint
    i2::Cint
    i3::Cint
    i4::Cint
end

immutable Cint5
    i1::Cint
    i2::Cint
    i3::Cint
    i4::Cint
    i5::Cint
end

cint(t::NTuple{1,Int}) = Cint1(t[1])
cint(t::NTuple{2,Int}) = Cint2(t[1],t[2])
cint(t::NTuple{3,Int}) = Cint3(t[1],t[2],t[3])
cint(t::NTuple{4,Int}) = Cint4(t[1],t[2],t[3],t[4])
cint(t::NTuple{5,Int}) = Cint5(t[1],t[2],t[3],t[4],t[5])
cint(i::Int...) = cint(i)
