type Stream
  ptr::cudaStream_t
end

function Stream()
  p = cudaStream_t[0]
  cudaStreamCreate(p)
  p = p[1]
  Stream(p)
end

const nullstream = Stream()

Base.unsafe_convert(::Type{cudaStream_t}, s::Stream) = s.ptr

destroy(s::Stream) = cudaStreamDestroy(s)

synchronize(s::Stream) = cudaStreamSynchronize(s)
