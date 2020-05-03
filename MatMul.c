#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>


void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}

int main()
{
    const size_t N = 1000; // size of an array

    clock_t start, end;   
 
    double ** A, ** B, ** C; // matrices
    int i,j,n;

    double omp_get_wtime(void);

    struct timeval tv1, tv2, tv3, tv4, tv5, tv6;
    struct timezone tz;
    double elapsed;

    omp_set_num_threads(omp_get_num_procs());

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime(); //clock();
    gettimeofday(&tv1, &tz);


//  matrix multiplication algorithm
    #pragma omp parallel for private(i,j,n) shared(A,B,C)
    for (i = 0; i < N; i++)
    {    
        for (j = 0; j < N; j++)
        {
            for (n = 0; n < N; n++)
                C[i][j]+=A[i][n]*B[n][j];
        }

    }

    end = omp_get_wtime(); //clock();
    gettimeofday(&tv2, &tz);
    
    printf("Time elapsed (ijn): %f seconds.\n", (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6);
    //printf("Time elapsed (ijn)_old: %f seconds.\n", (end - start) / CLOCKS_PER_SEC);
    printf("Time elapsed (ijn)_omp_get_wtime: %f seconds.\n~", (double)(end - start));
    
    zero_init_matrix(C, N);
 
    gettimeofday(&tv3, &tz);
    #pragma omp parallel for private(j,i,n) shared(A,B,C)
    for (j = 0; j < N; j++)
    {    
        for (i = 0; i < N; i++)
        {
            for (n = 0; n < N; n++)
                C[i][j]+=A[i][n]*B[n][j];
        }

    }
    gettimeofday(&tv4, &tz);
    printf("Time elapsed (jin): %f seconds.\n", (double) (tv4.tv_sec-tv3.tv_sec) + (double) (tv4.tv_usec-tv3.tv_usec) * 1.e-6);

    zero_init_matrix(C, N);

    gettimeofday(&tv5, &tz);
    #pragma omp parallel for private(n,i,j) shared(A,B,C)
    for (n = 0; n < N; n++)
    {    
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
                C[i][j]+=A[i][n]*B[n][j];
        }

    }
    gettimeofday(&tv6, &tz);
    printf("Time elapsed (nij): %f seconds.\n", (double) (tv6.tv_sec-tv5.tv_sec) + (double) (tv6.tv_usec-tv5.tv_usec) * 1.e-6);

    
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
