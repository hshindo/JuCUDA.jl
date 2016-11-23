export CuStream

type CuStream
  ptr::CUstream

  function CuStream(ptr)
    s = new(ptr)
    finalizer(s, cuStreamDestroy)
    s
  end
end

function CuStream()
  p = CUstream[0]
  cuStreamCreate(p, 0)
  CuStream(p[1])
end

Base.unsafe_convert(::Type{CUstream}, s::CuStream) = s.ptr

function test_stream(f::Function, num::Int)
  while length(custreams) < num
    push!(custreams, CuStream())
  end
  results = []
  @sync begin
    for i = 1:num
      @async begin
        while true
          s = custreams[i]
          results[i] = f()
        end
      end
    end
  end
end
