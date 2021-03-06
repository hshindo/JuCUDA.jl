function device()
  ret = Cint[0]
  cudaGetDevice(ret)
  Int(ret[1])
end

function device_count()
  ret = Cint[0]
  cudaGetDeviceCount(ret)
  Int(ret[1])
end

setdevice(dev::Integer) = cudaSetDevice(dev)

#=
function device_reset(dev::Integer)
  todelete = Any[]
  for (p,pdev) in cuda_ptrs
    if pdev == dev
      finalize(p)
      push!(todelete, p)
    end
  end
  for p in todelete
    delete!(cuda_ptrs, p)
  end
  # Reset the device
  device(dev)
  cudaDeviceReset()
end
device_reset() = device_reset(device())
=#

device_sync() = cudaDeviceSynchronize()

function device_properties(dev::Integer)
  aprop = Array(cudaDeviceProp, 1)
  cudaGetDeviceProperties(aprop, dev)
  aprop[1]
end

function attribute(dev::Integer, code::Integer)
  ret = Cint[0]
  cudaDeviceGetAttribute(ret, code, dev)
  Int(ret[1])
end

capability(dev::Integer) = (attribute(dev,cudaDevAttrComputeCapabilityMajor),
                            attribute(dev,cudaDevAttrComputeCapabilityMinor))

name(p::cudaDeviceProp) = bytestring(convert(Ptr{UInt8}, pointer([p.name])))

# criteria = dev -> Bool
function devices(criteria::Function; nmax::Integer = typemax(Int), status=:any)
  devlist = find(map(criteria, 0:devcount().-1)).-1
  if status == :any
  elseif status == :free
    devlist = filter_free(devlist)
  end
  isempty(devlist) && error("No suitable devices found")
  devlist[1:min(nmax, length(devlist))]
end

# do syntax, f(devlist)
function devices(f::Function, criteria::Function;
                 nmax::Integer = typemax(Int), status = :any)
  devlist = devices(criteria, nmax=nmax, status=status)
  devices(f, devlist)
end

function devices(f::Function, devlist::Union{Integer,AbstractVector})
  local ret
  try
    init(devlist)
    ret = f(devlist)
  finally
    close(devlist)
  end
  ret
end

function filter_free(devlist)
    smi = readall(`nvidia-smi`)
    if contains(smi, "No running compute processes")
        return devlist
    end
    idx = search(smi, "Compute processes:")
    isempty(idx) && error("Neither search string found")
    lines = split(smi[last(idx):end], '\n')
    free = Set(devlist)
    for line in lines
        m = match(r"^\| +([0-9]+)", line)
        if m != nothing
            length(m.captures) == 1 || error("Expected only a single capture")
            dev = parse(Int, m.captures[1])
            delete!(free, dev)
        end
    end
    collect(free)
end

function wait_free(devlist; check_interval=10)
    f = filter_free(devlist)
    length(f) < length(devlist) && warn("Waiting for GPU devices ", devlist)
    while length(f) < length(devlist)
        sleep(check_interval)
        f = filter_free(devlist)
    end
end

# A cache of useful CUDA kernels that gets loaded and closed by devices(f, devlist)
immutable PtxUtils
  mod::CuModule
  fns::Dict{Any,CuFunction}
end
const global ptxdict = Dict{Integer,PtxUtils}()

function init(devlist::Union{Integer,AbstractVector})
  funcnames = ["fill_contiguous", "fill_pitched"]
  funcexts  = ["double","float","int64","uint64","int32","uint32","int16","uint16","int8","uint8"]
  datatypes = [Float64,Float32,Int64,UInt64,Int32,UInt32,Int16,UInt16,Int8,UInt8]
  utilfile  = joinpath(Pkg.dir(), "CUDArt/deps/utils.ptx")

  # initialize all devices
  for dev in devlist
    # It has already been initialized.
    haskey(ptxdict, dev) && continue

    device(dev)
    # allocate and destroy memory to force initialization
    free(malloc(UInt8, 1))
    # Load the utility functions.
    ptx = PtxUtils(CuModule(utilfile, false), Dict{Any,CuFunction}())
    for func in funcnames
      for (dtype,ext) in zip(datatypes, funcexts)
        ptx.fns[(func, dtype)] = CuFunction(ptx.mod, func*"_"*ext)
      end
    end
    ptx.fns["clock_block"] = CuFunction(ptx.mod, "clock_block")
    ptxdict[dev] = ptx
  end
end

function close(devlist::Union{Integer,AbstractVector})
  for dev in devlist
    if haskey(ptxdict, dev)
      unload(ptxdict[dev].mod)
      delete!(ptxdict, dev)
    end
    device_reset(dev)
  end
end
