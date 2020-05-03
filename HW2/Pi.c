#include <stdio.h>
#include <omp.h>

int main()
{
    const size_t N = 100000;
    double step;
    int i;

    double x, pi, sum = 0.0;

    step = 1 / (double)N;

#pragma omp parallel private(i,x) 
    {
    #pragma omp for reduction(+:sum) schedule(static)
      for (i = 0; i < N; ++i)
        {
            x = (i + 0.5) * step;
            sum += 4.0 / (1. + x * x);
        }
    }
    pi = step * sum;

    printf("pi = %.16f\n", pi);

    return 0;
}
