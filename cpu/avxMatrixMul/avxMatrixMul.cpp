#include <immintrin.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <ctime>
#include "include.hpp"

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

    for (auto p1 = C, p2 = base_c; p1 != end; ++p1, ++p2)
    {
        if (*p1 != *p2)

            return 0;
    }
    return 1;
}

void gemm_avx(float *A, float *origin_B, float *C)
{
    auto B = rotate(origin_B, N);
    int r = N % 8;
    int m = N - r;
    auto cptr = C;

    auto aptr = A;
    for (int i = 0; i < N; ++i)
    {
        auto bptr = B;
        for (int j = 0; j < N; ++j)
        {
            auto sum = _mm256_setzero_ps();
            auto aa = aptr;
            auto bb = bptr;
            for (int k = 0; k < m; k += 8)
            {
                auto va = _mm256_loadu_ps(aa);
                auto vb = _mm256_loadu_ps(bb);
                sum = _mm256_fmadd_ps(va, vb, sum);
                aa += 8;
                bb += 8;
            }
            *cptr = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

            bptr += N;
            cptr += 1;
        }
        aptr += N;
    }
    if (r)
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float temp = 0;
                for (int k = 0; k < r; ++k)
                {
                    temp += A[i * N + m + k] * B[j * N + m + k];
                }
                C[i * N + j] += temp;
            }
        }
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
    gemm_avx(a, b, c);
    cout << "time: " << 1000 * (clock() - begin_t) / (double)CLOCKS_PER_SEC << "ms" << endl;
#ifdef VERIFY
    cout << "verify " << (gemm_verify(a, b, c) ? "true" : "false");
#endif
}