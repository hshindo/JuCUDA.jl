export CuModule

type CuModule
  ptr::CUmodule

  function CuModule(ptr::CUmodule)
    m = new(ptr)
    finalizer(m, cuModuleUnload)
    m
  end
end

function CuModule(filename::String)
  p = CUmodule[0]
  cuModuleLoad(p, filename)
  CuModule(p[1])
end

function CuModule(image::Vector{UInt8})
  p = CUmodule[0]
  cuModuleLoadData(p, image)
  #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
  CuModule(p[1])
end

Base.unsafe_convert(::Type{CUmodule}, m::CuModule) = m.ptr
