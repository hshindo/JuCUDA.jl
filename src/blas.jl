module CUBLAS

importall Base.LinAlg.BLAS

typealias CuPtrOrArray{T} Union{CuPtr{T},CuArray{T}}

# convert BlasChar {N,T,C} to cublasOperation_t
function cublasop(t::Char)
    t == 'N' && return CUBLAS_OP_N
    t == 'T' && return CUBLAS_OP_T
    t == 'C' && return CUBLAS_OP_C
    throw("Unknown cublas operation: $(t).")
end

# Level 1
for (fname,elty) in [(:cublasDcopy,:Float64), (:cublasScopy,:Float32)]
    @eval begin
        function blascopy!(n::Int, x::CuPtrOrArray{$elty}, incx::Int,
            y::CuPtrOrArray{$elty}, incy::Int)

            h = cublas_handle(device(x))
            $fname(h, n, x, incx, y, incy)
            y
        end
    end
end

for (fname,elty) in [(:cublasDaxpy,:Float64), (:cublasSaxpy,:Float32)]
    @eval begin
        function axpy!(n::Int, alpha::$elty, dx::CuPtrOrArray{$elty}, incx::Int,
            dy::CuPtrOrArray{$elty}, incy::Int)

            h = cublas_handle(device(dx))
            $fname(h, n, [alpha], dx, incx, dy, incy)
            dy
        end
    end
end

function axpy!{T}(alpha, x::CuArray{T}, y::CuArray{T})
    length(x) == length(y) || throw(DimensionMismatch(""))
    axpy!(length(x), T(alpha), x, 1, y, 1)
end

function axpy!{T}(alpha, x::CuArray{T}, rx::Range{Int}, y::CuArray{T}, ry::Range{Int})
    length(rx) == length(ry) || throw(DimensionMismatch(""))
    (minimum(rx) < 1 || maximum(rx) > length(x)) && throw(BoundsError())
    (minimum(ry) < 1 || maximum(ry) > length(y)) && throw(BoundsError())
    axpy!(length(rx), T(alpha), pointer(x,first(rx)-1), step(rx), pointer(y,first(ry)-1), step(ry))
end

# Level 3
## (GE) general matrix-matrix multiplication
for (fname, elty) in [(:cublasDgemm,:Float64), (:cublasSgemm,:Float32)]
    @eval begin
        function gemm!(tA::Char, tB::Char,
            alpha::$elty, A::CuVecOrMat{$elty}, B::CuVecOrMat{$elty},
            beta::$elty, C::CuVecOrMat{$elty})

            @assert device(A) == device(B) == device(C)
            h = cublas_handle(device(A))
            m = size(A, tA == 'N' ? 1 : 2)
            k = size(A, tA == 'N' ? 2 : 1)
            n = size(B, tB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1, stride(A,2))
            ldb = max(1, stride(B,2))
            ldc = max(1, stride(C,2))
            $fname(h, cublasop(tA), cublasop(tB), m, n, k,
            [alpha], A, lda, B, ldb, [beta], C, ldc)
            C
        end
        function gemm(tA::Char, tB::Char, alpha::$elty, A::CuVecOrMat{$elty}, B::CuVecOrMat{$elty})
            C = similar(B, size(A, tA=='N' ? 1 : 2), size(B, tB=='N' ? 2 : 1))
            gemm!(tA, tB, alpha, A, B, $elty(0), C)
        end
        function gemm(tA::Char, tB::Char, A::CuVecOrMat{$elty}, B::CuVecOrMat{$elty})
            gemm(tA, tB, $elty(1), A, B)
        end
    end
end

for (fname,elty) in [(:cublasDgemmBatched,:Float64), (:cublasSgemmBatched,:Float32)]
    @eval begin
        function gemm_batched!(tA::Char, tB::Char,
            alpha::$elty, As::Vector{CuMatrix{$elty}}, Bs::Vector{CuMatrix{$elty}},
            beta::$elty, Cs::Vector{CuMatrix{$elty}})

            if (length(As) != length(Bs) || length(As) != length(Cs))
                throw(DimensionMismatch(""))
            end
            for i = 1:length(As)
                A, B, C = As[i], Bs[i], Cs[i]
                m = size(A, tA == 'N' ? 1 : 2)
                k = size(A, tA == 'N' ? 2 : 1)
                n = size(B, tB == 'N' ? 2 : 1)
                if m != size(C,1) || n != size(C,2) || k != size(B, tB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end
            h = cublas_handle
            m = size(As[1], tA == 'N' ? 1 : 2)
            k = size(As[1], tA == 'N' ? 2 : 1)
            n = size(Bs[1], tB == 'N' ? 2 : 1)
            lda = max(1, stride(As[1],2))
            ldb = max(1, stride(Bs[1],2))
            ldc = max(1, stride(Cs[1],2))
            Aptrs = map(a -> Ptr{$elty}(a.ptr), As)
            Bptrs = map(a -> Ptr{$elty}(a.ptr), Bs)
            Cptrs = map(a -> Ptr{$elty}(a.ptr), Cs)
            $fname(h, cublasop(tA), cublasop(tB), m, n, k, [alpha], pointer(Aptrs)
            lda, pointer(Bptrs), ldb, [beta], pointer(Cptrs), ldc, length(As))
            Cs
        end
        function gemm_batched(tA::Char, tB::Char,
            alpha::$elty, A::Vector{CuVecOrMat{$elty}}, B::Vector{CuVecOrMat{$elty}})
            C = CuMatrix{$elty}[similar(B[1], (size(A[1], tA=='N' ? 1 : 2), size(B[1], tB=='N' ? 2 : 1))) for i in 1:length(A)]
            gemm_batched!(tA, tB, alpha, A, B, $elty(0), C)
        end
        function gemm_batched(tA::Char, tB::Char, A::Vector{CuVecOrMat{$elty}}, B::Vector{CuVecOrMat{$elty}})
            gemm_batched(tA, tB, $elty(1), A, B)
        end
    end
end

end
