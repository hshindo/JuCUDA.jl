# CURAND

| | Support |
|:---:|:---:|
| curandCreateGenerator | ✓ |
| curandCreateGeneratorHost | ✓ |
| curandDestroyGenerator | ✓ |
| curandGetVersion | ✓ |
| curandSetStream | |
| curandSetPseudoRandomGeneratorSeed | ✓ |
| curandSetGeneratorOffset | ✓ |
| curandSetGeneratorOrdering | ✓ |
| curandSetQuasiRandomGeneratorDimensions | ✓ |
| curandGenerate | ✓ |
| curandGenerateLongLong | ✓ |
| curandGenerateUniform | ✓ |
| curandGenerateUniformDouble | ✓ |
| curandGenerateNormal | ✓ |
| curandGenerateNormalDouble | ✓ |
| curandGenerateLogNormal | ✓ |
| curandGenerateLogNormalDouble | ✓ |
| curandCreatePoissonDistribution | ✓ |
| curandDestroyDistribution | ✓ |
| curandGeneratePoisson | ✓ |
| curandGeneratePoissonMethod | |
| curandGenerateSeeds | ✓ |
| curandGetDirectionVectors32 | |
| curandGetScrambleConstants32 | |
| curandGetDirectionVectors64 | |
| curandGetScrambleConstants64 | | |

## Test
```julia
julia> using JuCUDA                                                                          
# 省略
julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_QUASI_DEFAULT)                     
Ptr{Void} @0x000000000d48f740

julia> hrng = JuCUDA.CURAND.curandCreateGeneratorHost(rng,JuCUDA.CURAND.CURAND_RNG_PSEUDO_DEFAULT)                                                                                        

julia> JuCUDA.CURAND.curandDestroyGenerator(rng)                                             

julia> JuCUDA.CURAND.curandgetversion()                                                      
7050

julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_PSEUDO_DEFAULT)                    
Ptr{Void} @0x000000000e29c550

julia> JuCUDA.CURAND.cusrand(rng,1)                                                          

julia> JuCUDA.CURAND.curandordering(rng,JuCUDA.CURAND.CURAND_ORDERING_PSEUDO_BEST)           

julia> JuCUDA.CURAND.curandDestroyGenerator(rng)                                             

julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_QUASI_DEFAULT)                     
Ptr{Void} @0x000000000d5c5b70

julia> JuCUDA.CURAND.curandSetQuasiRandomGeneratorDimensions(rng,3)                          

julia> JuCUDA.CURAND.curandDestroyGenerator(rng)                                             

julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_PSEUDO_DEFAULT)                    
Ptr{Void} @0x000000000e290d60

julia> a = JuCUDA.CURAND.curand(rng,UInt32,10)                                               
JuCUDA.CuArray{UInt32,1}(JuCUDA.CUDA.CuPtr{UInt32}(0x0000000b0c900600,0,1),(10,))

julia> JuCUDA.CURAND.curandDestroyGenerator(rng)                                             

julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_QUASI_SOBOL64)                     
Ptr{Void} @0x000000000e290d60

julia> a = JuCUDA.CURAND.curand(rng,UInt64,10)                                               
JuCUDA.CuArray{UInt64,1}(JuCUDA.CUDA.CuPtr{UInt64}(0x0000000b0c900a00,0,1),(10,))

julia> JuCUDA.CURAND.curandDestroyGenerator(rng)                                             

julia> rng = JuCUDA.CURAND.curng(JuCUDA.CURAND.CURAND_RNG_PSEUDO_DEFAULT)                    
Ptr{Void} @0x000000000d5a6450

julia> a = JuCUDA.CURAND.curand(rng,Float32,10)                                              
JuCUDA.CuArray{Float32,1}(JuCUDA.CUDA.CuPtr{Float32}(0x0000000b0c900e00,0,1),(10,))

julia> a = JuCUDA.CURAND.curand(rng,Float64,10)                                              
JuCUDA.CuArray{Float64,1}(JuCUDA.CUDA.CuPtr{Float64}(0x0000000b0c900600,0,1),(10,))

julia> a = JuCUDA.CURAND.curandn(rng,Float32,10, mean=1, stddev=10)                          
JuCUDA.CuArray{Float32,1}(JuCUDA.CUDA.CuPtr{Float32}(0x0000000b0c900a00,0,1),(10,))

julia> a = JuCUDA.CURAND.curandn(rng,Float64,10, mean=1, stddev=10)                          
JuCUDA.CuArray{Float64,1}(JuCUDA.CUDA.CuPtr{Float64}(0x0000000b0c901200,0,1),(10,))

julia> a = JuCUDA.CURAND.curandlogn(rng,Float32,10, mean=1, stddev=10)                       
JuCUDA.CuArray{Float32,1}(JuCUDA.CUDA.CuPtr{Float32}(0x0000000b0c901600,0,1),(10,))

julia> a = JuCUDA.CURAND.curandlogn(rng,Float64,10, mean=1, stddev=10)                       
JuCUDA.CuArray{Float64,1}(JuCUDA.CUDA.CuPtr{Float64}(0x0000000b0c901a00,0,1),(10,))

julia> dist = JuCUDA.CURAND.curandcreatedist(1.)                                             
Ptr{Void} @0x0000000b0c902600

julia> JuCUDA.CURAND.curandDestroyDistribution(dist)                                         

julia> a = JuCUDA.CURAND.curandpoisson(rng,10,1.)                                            
JuCUDA.CuArray{UInt32,1}(JuCUDA.CUDA.CuPtr{UInt32}(0x0000000b0c901e00,0,1),(10,))

julia> JuCUDA.CURAND.curandGenerateSeeds(rng)                                                

julia> JuCUDA.CURAND.curandDestroyGenerator(rng) 
```
