module NVRTC

using Compat
import Compat: String
importall ..JuCUDA

include(joinpath(Pkg.dir("JuCUDA"),"lib/7.5/libnvrtc.jl"))
include(joinpath(Pkg.dir("JuCUDA"),"lib/7.5/libnvrtc_types.jl"))

@compat if is_windows()
    const libnvrtc = Libdl.find_library(
    ["nvrtc64_75", "nvrtc64_70"])
else
    const libnvrtc = Libdl.find_library(["libnvrtc"])
end
isempty(libnvrtc) && error("NVRTC library cannot be found.")

function check_nvrtcresult(status)
    status == NVRTC_SUCCESS && return nothing
    warn("NVRTC error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    throw(bytestring(nvrtcGetErrorString(status)))
end

function getlog(prog::Ptr{Void})
    logsize = Csize_t[0]
    nvrtcGetProgramLogSize(prog, logsize)
    log = Array(UInt8, logsize[1])
    nvrtcGetProgramLog(prog, log)
    bytestring(log)
end

function compile(src::String, headers::Vector=[], include_names::Vector=[])
    p = Ptr{Void}[0]
    headers = map(pointer, headers)
    nvrtcCreateProgram(p, src, "", length(headers), headers, include_names)

    prog = p[1]
    options = ["--gpu-architecture=compute_50"]
    nvrtcCompileProgram(prog, length(options), options)

    ptxsize = Csize_t[0]
    nvrtcGetPTXSize(prog, ptxsize)
    ptx = Array(UInt8, ptxsize[1])
    nvrtcGetPTX(prog, ptx)
    #log = getlog(prog)
    #println(log)
    nvrtcDestroyProgram([prog])
    ptx
end

end
