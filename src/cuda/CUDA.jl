module CUDA

using Compat
import Compat.String
importall ..JuCUDA

include(joinpath(Pkg.dir("JuCUDA"),"lib/7.5/libcuda.jl"))
include(joinpath(Pkg.dir("JuCUDA"),"lib/7.5/libcuda_types.jl"))

@compat if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
isempty(libcuda) && error("CUDA library cannot be found.")

function check_curesult(status)
    status == CUDA_SUCCESS && return nothing
    warn("CUDA error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    p = Ptr{UInt8}[0]
    cuGetErrorString(status, p)
    throw(bytestring(p[1]))
end

function version()
    p = Cint[0]
    cuDriverGetVersion(p)
    Int(p[1])
end

include("context.jl")
include("device.jl")
include("stream.jl")
include("pointer.jl")
include("module.jl")
include("function.jl")
include("execution.jl")

const contexts = CUcontext[]
const streams = CuStream[]

info("Initializing CUDA...")
cuInit(0)
for dev = 0:ndevices()-1
    p = CUcontext[0]
    cuCtxCreate(p, 0, dev)
    push!(contexts, p[1])
end
setdevice(0)
info("CUDA driver version: $(version())")

end
