# JuCUDA.jl
CUDA bindings for Julia.

## Support
CUDA 7.0 or higher.

Check compute capability from [here](https://developer.nvidia.com/cuda-gpus)

## Install
```julia
julia> Pkg.clone("https://github.com/hshindo/JuCUDA.jl.git")
```

## Usage
`CuArray{T,N}` is analogous to `Base.Array{T,N}` in Julia.

###
```julia
x = curand(Float32,2,2,2)
y = zeros(CuArray{Float32},4,4,4)
y[2:3,2:3,3:4]
y[2:3,2:3,3:4] = x
```
