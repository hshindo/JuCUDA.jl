#=
const sources = [""]

@compat if is_windows()
    flags = ["-fopenmp", "-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.dll"
    cmd = `nvcc `
    run(cmd)
elseif is_apple()
    flags = ["-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.so"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
elseif is_linux()
    flags = ["-fopenmp", "-fPIC", "-Wall", "-O3", "-shared", "-march=native"]
    libname = "libmerlin.so"
    cmd = `$compiler $flags -o $libname $sources`
    println("Running $cmd")
    run(cmd)
else
    throw("Unknown OS.")
end
=#
