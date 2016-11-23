@testset "array" for i = 1:5

x = CuArray(T,10,5)
x = zeros(x)
x = similar(x)
x = ones(x)

end
