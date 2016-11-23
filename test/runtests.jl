using JuCUDA
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

files = ["array", "arraymath"]
const T = Float32

for f in files
    include("$(f).jl")
end
