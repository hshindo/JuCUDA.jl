export CuPtr

type CuPtr{T}
    ptr::CUdeviceptr
    dev::Int
    id::Int

    function CuPtr(ptr, dev, id)
        p = new(ptr, dev, id)
        finalizer(p, free)
        p
    end
end

Base.convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr
Base.convert{T}(::Type{Ptr{T}}, p::CuPtr) = Ptr{T}(p.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::CuPtr) = Ptr{T}(p.ptr)

const memstep = 1024
#const alloc_ptrs = CUdeviceptr[]
const free_ptrs = Dict{Int,Vector{CUdeviceptr}}[]

function malloc(T::Type, n::Int)
    dev = getdevice()
    while length(free_ptrs) < dev+1
        push!(free_ptrs, Dict{Int,Vector{CUdeviceptr}}())
    end

    id = sizeof(T) * n ÷ memstep + 1
    ptrs = get!(free_ptrs[dev+1], id, CUdeviceptr[])
    if length(ptrs) == 0
        p = CUdeviceptr[0]
        cuMemAlloc(p, id*memstep)
        #push!(alloc_ptrs, p[1])
        CuPtr{T}(p[1], dev, id)
    else
        CuPtr{T}(pop!(ptrs), dev, id)
    end
end

free(p::CuPtr) = push!(free_ptrs[p.dev+1][p.id], p.ptr)

function devicegc()
    gc()
    _dev = getdevice()
    for dev = 0:ndevices()-1
        setdevice(dev)
        for (id,ptrs) in free_ptrs[dev+1]
            for p in ptrs
                cuMemFree(p)
            end
            empty!(ptrs)
        end
    end
    setdevice(_dev)
    run(`nvidia-smi`)
end
