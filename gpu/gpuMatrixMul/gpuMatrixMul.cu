#include <iostream>
#include <ctime>
#include <fstream>
#include "include.hpp"
#include <string>

#define VERIFY

using namespace std;

int N_;
int N;
int matrixSize;

__global__ void gemm_baseline(float *A, float *B, float *C, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= N || y >= N)
        return;

    float temp = 0;
    auto aptr = A + x * N;
    auto bptr = B + y;
    for (int i = 0; i < N; ++i)
    {
        // float += A[x * N + i] * B[i * N + y];
        temp += (*aptr) * (*bptr);
        aptr += 1;
        bptr += N;
    }
    C[x * N + y] = temp;
}

void cpu_gemm_baseline(float *A, float *B, float *C)
{
    float *now = C;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            auto a = A + i * N;
            auto b = B + j;
            for (int k = 0; k < N; ++k)
            {
                sum += (*a) * (*b);
                a += 1;
                b += N;
            }
            *now = sum;
            now += 1;
        }
    }
}

bool gemm_verify(float *A, float *B, float *C)
{
    auto base_c = new float[matrixSize];
    cpu_gemm_baseline(A, B, base_c);
    auto end = C + matrixSize;

    for (float *p1 = C, *p2 = base_c; p1 != end; ++p1, ++p2)
    {
        if (*p1 != *p2)

            return 0;
    }

    return 1;
};

int main(int argc, char **argv)
{
    N_ = stoi(argv[1]);
    N = 1 << N_;
    matrixSize = N * N;
    auto a = new float[matrixSize];
    auto b = new float[matrixSize];
    auto c = new float[matrixSize];
    randInit(a, matrixSize, 10);
    randInit(b, matrixSize, 10);
    float *d_a, *d_b, *d_c;
    auto nBytes = matrixSize * sizeof(float);
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, nBytes);
    cudaMemcpy((void *)d_a, (void *)a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (void *)b, nBytes, cudaMemcpyHostToDevice);
    dim3 blockSize(8, 8);
    dim3 gridSize((N + 7) / 8, (N + 7) / 8);
    gemm_baseline<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaMemcpy((void *)c, (void *)d_c, nBytes, cudaMemcpyDeviceToHost);

#ifdef VERIFY
    cout << (gemm_verify(a, b, c) ? "true" : "false");
#endif
}
