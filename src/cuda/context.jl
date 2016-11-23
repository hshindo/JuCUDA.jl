export getdevice, setdevice, sync

function getdevice()
  p = CUdevice[0]
  cuCtxGetDevice(p)
  Int(p[1])
end

setdevice(dev::Int) = cuCtxSetCurrent(contexts[dev+1])

sync() = cuCtxSynchronize()
