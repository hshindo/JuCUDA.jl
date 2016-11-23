import Base: +, .+, -, .-, .*, *

function +{T,N}(x1::CuArray{T,N}, x2::CuArray{T,N})
    @assert length(x1) == length(x2)
    x1 .+ x2
end
.+(x1::CuArray, x2::CuArray) = mathop(:add, x1, x2)

function -{T,N}(x1::CuArray{T,N}, x2::CuArray{T,N})
    @assert length(x1) == length(x2)
    x1 .- x2
end
.-(x1::CuArray, x2::CuArray) = mathop(:sub, x1, x2)

function .*{T,N}(x1::CuArray{T,N}, x2::CuArray{T,N})
    @assert length(x1) == length(x2)
    mathop(:mul, x1, x2)
end

*(x1::CuMatrix, x2::CuMatrix) = BLAS.gemm('N', 'N', x1, x2)

Base.exp(x::CuArray) = mathop(:exp, x)
Base.log(x::CuArray) = mathop(:log, x)

function mathop{T}(op::Symbol, x1::CuArray{T}, x2::CuArray{T})
    s = Symbol(op, :_, cstring(T))
    f = functions[s]
    x = length(x1) >= length(x2) ? x1 : x2
    y = similar(x)
    launch(f, (length(y),), (CUvector(x1),CUvector(x2),CUvector(y)))
    y
end

function mathop{T}(op::Symbol, x::CuArray{T})
    s = Symbol(op, :_, cstring(T))
    f = functions[s]
    y = similar(x)
    launch(f, (length(y),), (CUvector(x),CUvector(y)))
    y
end
