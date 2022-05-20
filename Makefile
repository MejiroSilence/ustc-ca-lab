all: mkdir avxMatrixMul basicMatrixMul blockedAvxMatrixMul blockedGpuMatrixMul gpuMatrixMul

OPT = -O3

mkdir:
	mkdir build

avxMatrixMul: cpu/avxMatrixMul/avxMatrixMul.cpp include/include.hpp
	g++ -mfma $(OPT) -o build/avxMatrixMul cpu/avxMatrixMul/avxMatrixMul.cpp

basicMatrixMul: cpu/basicMatrixMul/basicMatrixMul.cpp include/include.hpp
	g++ $(OPT) -o build/basicMatrixMul cpu/basicMatrixMul/basicMatrixMul.cpp 

blockedAvxMatrixMul: cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp include/include.hpp
	g++ -mfma $(OPT) -o build/blockedAvxMatrixMul cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp

blockedGpuMatrixMul: gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu include/include.hpp
	nvcc $(OPT) -o build/blockedGpuMatrixMul gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu

gpuMatrixMul: gpu/gpuMatrixMul/gpuMatrixMul.cu include/include.hpp
	nvcc $(OPT) -o build/gpuMatrixMul gpu/gpuMatrixMul/gpuMatrixMul.cu 

clean:
	rm build