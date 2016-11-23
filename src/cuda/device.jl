function capability(dev::Int)
  major, minor = Cint[0], Cint[0]
  cuDeviceComputeCapability(major, minor, dev)
  Int(major[1]), Int(minor[1])
end

function ndevices()
  p = Cint[0]
  cuDeviceGetCount(p)
  Int(p[1])
end

function totalmem(dev::Int)
  bytes = Csize_t[0]
  cuDeviceTotalMem(bytes, dev)
  Int(bytes[1])
end

function attribute(attrib::Int, dev::Int)
  p = Cint[0]
  cuDeviceGetAttribute(p, attrib, dev)
  Int(p[1])
end

function attributes(dev::Int)
  "MAX_THREADS_PER_BLOCK" => attribute(1, dev),
  "MAX_BLOCK_DIM_X" => attribute(2, dev),
  "MAX_BLOCK_DIM_Y" => attribute(3, dev),
  "MAX_BLOCK_DIM_Z" => attribute(4, dev),
  "MAX_GRID_DIM_X" => attribute(5, dev),
  "MAX_GRID_DIM_Y" => attribute(6, dev),
  "MAX_GRID_DIM_Z" => attribute(7, dev),
  "MAX_SHARED_MEMORY_PER_BLOCK" => attribute(8, dev),
  "TOTAL_CONSTANT_MEMORY" => attribute(9, dev),
  "WARP_SIZE" => attribute(10, dev),
  "MAX_PITCH" => attribute(11, dev),
  "MAX_REGISTERS_PER_BLOCK" => attribute(12, dev)
end

function properties(dev::Int)
  p = Array(CUdevprop, 1)
  cuDeviceGetProperties(p, dev)
  p[1]
end

function Base.show(io::IO, p::CUdevprop)
  names = fieldnames(CUdevprop)
  for i = 1:nfields(CUdevprop)
    name = string(names[i])
    f = getfield(p, i)
    println(io, "$(name): $(f)")
  end
end
