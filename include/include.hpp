#ifndef _INCLUDE_HPP_
#define _INCLUDE_HPP_ 1

#include <cstdlib>
#include <ctime>
#include <fstream>

void randInit(float *a, int size, int max)
{
    srand(time(0));
    auto end = a + size;
    for (auto p = a; p != end; ++p)
        *p = rand() % max;
}

inline float get(float *a, int n, int i, int j)
{
    return a[i * n + j];
}

inline void set(float *a, int n, int i, int j, float key)
{
    a[i * n + j] = key;
}

float *rotate(float *a, int n)
{
    auto ans = new float[n * n];
    auto now = ans;
    // auto p = a;
    for (int i = 0; i < n; ++i)
    {
        auto p = a + i;
        for (int j = 0; j < n; ++j)
        {
            *now = *p;
            ++now;
            p += n;
        }
    }
    return ans;
}

void printMatrix(float *a, int n, std::ofstream &fout)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            fout << a[i * n + j] << '\t';
        fout << std::endl;
    }
}

#endif