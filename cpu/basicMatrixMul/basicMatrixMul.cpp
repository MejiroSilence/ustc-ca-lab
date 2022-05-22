#include <iostream>
#include <ctime>
#include <string>
#include <ctime>
#include "include.hpp"

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
    gemm_baseline(a, b, c);
    cout << "time: " << 1000 * (clock() - begin_t) / (double)CLOCKS_PER_SEC << "ms" << endl;
}
