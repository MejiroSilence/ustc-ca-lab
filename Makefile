all: avxMatrixMul basicMatrixMul blockedAvxMatrixMul blockedGpuMatrixMul gpuMatrixMul

OPT = -O3

avxMatrixMul: cpu/avxMatrixMul/avxMatrixMul.cpp _include.o
	g++ -mfma -o avxMatrixMul $(OPT) cpu/avxMatrixMul/avxMatrixMul.cpp _include.o

basicMatrixMul: cpu/basicMatrixMul/basicMatrixMul.cpp _include.o
	g++ -o basicMatrixMul $(OPT) cpu/basicMatrixMul/basicMatrixMul.cpp _include.o

blockedAvxMatrixMul: cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp _include.o
	g++ -mfma -o blockedAvxMatrixMul $(OPT) cpu/blockedAvxMatrixMul/blockedAvxMatrixMul.cpp _include.o

blockedGpuMatrixMul: gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu _include.o
	nvcc -o blockedGpuMatrixMul $(OPT) gpu/blockedGpuMatrixMul/blockedGpuMatrixMul.cu _include.o

gpuMatrixMul: gpu/gpuMatrixMul/gpuMatrixMul.cu _include.o
	nvcc -o gpuMatrixMul $(OPT) gpu/gpuMatrixMul/gpuMatrixMul.cu _include.o

_include.o: include/include.hpp
	g++ -c $(OPT) -o _include.o include/include.hpp
