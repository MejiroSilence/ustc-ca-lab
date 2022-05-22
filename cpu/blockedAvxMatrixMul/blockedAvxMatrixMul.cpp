#include <immintrin.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include "include.hpp"
#include <string>
#include <ctime>

#define blockN 16
#define VERIFY

using namespace std;

int N_;
int N;
int matrixSize;

void gemm_baseline(float *A, float *B, float *C)
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
    gemm_baseline(A, B, base_c);
    auto end = C + matrixSize;

    for (float *p1 = C, *p2 = base_c; p1 != end; ++p1, ++p2)
    {
        if (*p1 != *p2)

            return 0;
    }
    return 1;
}

void addDot(float *A, float *B, float *C, int si, int sj, int sk)
{
    for (int i = si; i < si + blockN; i += 1)
    {
        for (int j = sj; j < sj + blockN; j += 8)
        {
            auto c = _mm256_loadu_ps(C + i * N + j);

            for (int k = sk; k < sk + blockN; ++k)
            {
                auto a = _mm256_set1_ps(A[i * N + k]);
                auto b = _mm256_loadu_ps(B + k * N + j);
                c = _mm256_fmadd_ps(a, b, c);
            }
            _mm256_storeu_ps(C + i * N + j, c);
        }
    }
}

void gemm_avx_block(float *A, float *B, float *C)
{
    for (int i = 0; i < N; i += blockN)
        for (int j = 0; j < N; j += blockN)
            for (int k = 0; k < N; k += blockN)
                addDot(A, B, C, i, j, k);
}

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
    auto begin_t = clock();
    gemm_avx_block(a, b, c);
    cout << "time: " << 1000 * (clock() - begin_t) / (double)CLOCKS_PER_SEC << "ms" << endl;
#ifdef VERIFY
    cout << (gemm_verify(a, b, c) ? "true" : "false");
#endif
}