#include <immintrin.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include "include.hpp"
#include <string>

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
    // ofstream baseout("base");
    // ofstream avxout("avx");
    // printMatrix(C, N, avxout);
    // printMatrix(base_c, N, baseout);

    for (float *p1 = C, *p2 = base_c; p1 != end; ++p1, ++p2)
    {
        if (*p1 != *p2)

            return 0;
    }
    /*
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
        if (C[i * N + j] != base_c[i * N + j])
        {
            cout << '(' << i << ',' << j << ')' << endl;
        }
        */
    return 1;
}

// int dbgcnt = 0;

void addDot8x8(float *A, float *B, float *C)
{
    auto b_p_ptr = B;
    auto c0 = _mm256_setzero_ps();
    auto c1 = _mm256_setzero_ps();
    auto c2 = _mm256_setzero_ps();
    auto c3 = _mm256_setzero_ps();
    auto c4 = _mm256_setzero_ps();
    auto c5 = _mm256_setzero_ps();
    auto c6 = _mm256_setzero_ps();
    auto c7 = _mm256_setzero_ps();
    for (int p = 0; p < N; ++p)
    {
        auto a_0p = _mm256_set1_ps(A[p]);
        auto a_1p = _mm256_set1_ps(A[p + N]);
        auto a_2p = _mm256_set1_ps(A[p + N * 2]);
        auto a_3p = _mm256_set1_ps(A[p + N * 3]);
        auto a_4p = _mm256_set1_ps(A[p + N * 4]);
        auto a_5p = _mm256_set1_ps(A[p + N * 5]);
        auto a_6p = _mm256_set1_ps(A[p + N * 6]);
        auto a_7p = _mm256_set1_ps(A[p + N * 7]);
        auto b_p = _mm256_loadu_ps(b_p_ptr);
        b_p_ptr += N;
        c0 = _mm256_fmadd_ps(a_0p, b_p, c0);
        c1 = _mm256_fmadd_ps(a_1p, b_p, c1);
        c2 = _mm256_fmadd_ps(a_2p, b_p, c2);
        c3 = _mm256_fmadd_ps(a_3p, b_p, c3);
        c4 = _mm256_fmadd_ps(a_4p, b_p, c4);
        c5 = _mm256_fmadd_ps(a_5p, b_p, c5);
        c6 = _mm256_fmadd_ps(a_6p, b_p, c6);
        c7 = _mm256_fmadd_ps(a_7p, b_p, c7);
    }
    _mm256_store_ps(C, c0);
    _mm256_store_ps(C + N, c1);
    _mm256_store_ps(C + N * 2, c2);
    _mm256_store_ps(C + N * 3, c3);
    _mm256_store_ps(C + N * 4, c4);
    _mm256_store_ps(C + N * 5, c5);
    _mm256_store_ps(C + N * 6, c6);
    _mm256_store_ps(C + N * 7, c7);
}

void gemm_avx_block(float *A, float *B, float *C)
{
    int r = N % 8;
    int m = N - r;
    if (N > 7)
        for (int i = 0; i < m; i += 8)
            for (int j = 0; j < m; j += 8)
                addDot8x8(A + i * N, B + j, C + i * N + j);
    for (int i = m; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int p = 0; p < N; ++p)
            {
                sum += A[i * N + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    for (int i = 0; i < m; ++i)
        for (int j = m; j < N; ++j)
        {
            float sum = 0;
            for (int p = 0; p < N; ++p)
            {
                sum += A[i * N + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
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
    gemm_avx_block(a, b, c);
    cout << (gemm_verify(a, b, c) ? "true" : "false");
}