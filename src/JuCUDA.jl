module JuCUDA

import Compat: String, view

include("cuda/CUDA.jl")
include("CURAND.jl")
include("NVRTC.jl")

using .CUDA, .CURAND, .NVRTC

const array_h = open(readstring, joinpath(Pkg.dir("JuCUDA"),"src/kernels/array.h"))

function compile(src::String)
    name = src[1:searchindex(src,"(")-1]
    src = """
    #include "array.h"
    extern \"C\" __global__ void $(src)
    """
    ptx = NVRTC.compile(src, [array_h], ["array.h"])
    CuFunction(CuModule(ptx), name)
end

#=
macro nvrtc(expr, T...)
    sym = [gensym()]
    quote
        local s = Symbol($sym[1], $(T...))
        get!(functions, s) do
            local src = $(esc(expr))
            local name = src[1:searchindex(src,"(")-1]
            src = """
            #include "array.h"
            extern \"C\" __global__ void $(src)
            """
            ptx = NVRTC.compile(src, [array_h], ["array.h"])
            CuFunction(CuModule(ptx), name)
        end
    end
end
=#

abstract AbstractCuArray{T,N}
include("abstractcuarray.jl")
include("array.jl")
include("arraymath.jl")
include("subarray.jl")
include("cuarray.jl")
#include("reduce.jl")

include("CUBLAS.jl")
using .CUBLAS

#import CUDA: CuFunction, CuModule
const functions = Dict{Symbol,CuFunction}()

cstring(::Type{Float16}) = "half"
cstring(::Type{Float32}) = "float"
cstring(::Type{Float64}) = "double"
cstring(::Type{Int32}) = "int"
cstring(::Type{Int64}) = "long long int"

function read_function(src_path::String)
    src = open(readlines, src_path)
    # TODO: avoid fixed headers
    ptx = NVRTC.compile(join(src), [array_h], ["array.h"])
    m = CuModule(ptx)
    for l in src
        startswith(l, "// ") || break
        items = split(l)
        for i = 2:length(items)
            s = Symbol(items[i])
            functions[s] = CuFunction(m, string(s))
        end
    end
end

for name in [
    #"array.cu",
    "arraymath.cu",
    #"vector.cu",
    #"reduce.cu"
    ]
    read_function(joinpath(Pkg.dir("JuCUDA"),"src/kernels/$name"))
end

end
