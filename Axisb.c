#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

void matrix_filled(float **matrix, const int n) 
{
	*matrix = (float *) malloc(n * n * sizeof(float));

	for (int i = 0; i < n*n; i++) 
    {
		(*matrix)[i] = ((float)rand()/(float)(RAND_MAX))*100;
	}
}

void gs_solver(float **matrix, const int n, const int threads, const int max_cells_per_thread) 
{
	float diff;
	int condition = 0;
	int iterations = 0;

	while (!condition&&(iterations < 100)) 
    {	
        diff = 0;

		#pragma omp parallel for num_threads(threads) schedule(static, max_cells_per_thread) reduction(+:diff)
		
        for (int i = 1; i < n-1; i++) 
        {
            for (int j = 1; j < n-1; j++) 
            {
		        const int pos = (i * n) + j;

		        const float temp = (*matrix)[pos];

		        (*matrix)[pos] = 0.2*((*matrix)[pos] + (*matrix)[pos-1]+ (*matrix)[pos-n]+ (*matrix)[pos+1]+ (*matrix)[pos+n]);

		        diff += abs((*matrix)[pos] - temp);
		    }
	    }

		if (diff/n*n < 0.000001) condition = 1;
		
        iterations ++;
	}

	printf("Solver converged after %d iterations\n", iterations);
}


int main(int argc, char *argv[]) {

	if (argc < 2) 
    {
		printf("Write matrix size (e.g. 64, 128, 256, 1024...)\n");
		exit(1);
	}

	const int n = atoi(argv[1]);
    printf("Matrix size = %d \n", n);
	
    float *matrix;

	matrix_filled(&matrix, n);

	const int max_threads = omp_get_max_threads();

	const int max_rows = (int)(ceil((n-2) / max_threads) + 2);

	const int max_cells = max_rows * (n-2);

	gs_solver(&matrix, n, max_threads, max_cells);

	free(matrix);
}
