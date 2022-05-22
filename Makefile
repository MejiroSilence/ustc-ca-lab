all: avxMatrixMul basicMatrixMul blockedAvxMatrixMul blockedGpuMatrixMul gpuMatrixMul

FLAGS = -O1 -I include

avxMatrixMul: cpu/avxMatrixMul/avxMatrixMul.cpp include/include.hpp
	g++ -mfma $(FLAGS) -o build/avxMatrixMul cpu/avxMatrixMul/avxMatrixMul.cpp

basicMatrixMul: cpu/basicMatrixMul/basicMatrixMul.cpp include/include.hpp
	g++ $(FLAGS) -o build/basicMatrixMul cpu/basicMatrixMul/basicMatrixMul.cpp 

blockedAvxMatrixMul: cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp include/include.hpp
	g++ -mfma $(FLAGS) -o build/blockedAvxMatrixMul cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp

blockedGpuMatrixMul: gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu include/include.hpp
	nvcc $(FLAGS) -o build/blockedGpuMatrixMul gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu

gpuMatrixMul: gpu/gpuMatrixMul/gpuMatrixMul.cu include/include.hpp
	nvcc $(FLAGS) -o build/gpuMatrixMul gpu/gpuMatrixMul/gpuMatrixMul.cu 

clean:
	rm build