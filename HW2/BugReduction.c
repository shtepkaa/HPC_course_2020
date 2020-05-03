#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

float dotprod(float * a, float * b, size_t N, size_t threads)
{
    int i, tid;
    float sum, psum;

    sum = 0.0;

    #pragma omp parallel private(i,tid,psum) num_threads(threads) 
    {
        psum = 0.0;
        tid = omp_get_thread_num();

        #pragma omp for reduction(+:sum)
        for (i = 0; i < threads*N; ++i)
        {
            sum += a[i] * b[i];
            //printf("tid = %d i = %d\n", tid, i);
            psum = sum;
        }

        printf("thread id = %d partial sum = %f\n", tid, psum);
    }

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    const size_t threads = 8;
    
    int i;
    float sum;
    float *a, *b;

    a = (float*) malloc (threads*N*sizeof(float));
    b = (float*) malloc (threads*N*sizeof(float));

    for (i = 0; i < threads*N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    sum = 0.0;

    sum = dotprod(a, b, N, threads);

    printf("Sum = %f\n",sum);

    return 0;
}
