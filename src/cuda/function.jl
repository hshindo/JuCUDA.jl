export CuFunction

type CuFunction
    m::CuModule # avoid CuModule gc-ed
    ptr::CUfunction
end

function CuFunction(m::CuModule, name::String)
    p = CUfunction[0]
    cuModuleGetFunction(p, m, name)
    CuFunction(m, p[1])
end

Base.unsafe_convert(::Type{CUfunction}, f::CuFunction) = f.ptr
