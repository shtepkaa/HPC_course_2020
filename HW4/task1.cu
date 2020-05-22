// /usr/local/cuda/bin/nvcc task1.cu -o task1
// nvcc task1.cu -o task1
//./task1

#include <stdio.h>  
#include <stdlib.h>
#include <math.h>

__global__ void solver(double *T_new, const double *T_old, int cols, int rows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
  
    int idx = (row*cols + col);
          
    if (row < rows-1 && col < cols-1 && row > 0 && col > 0) 
    {
        T_new[idx] =  0.25*(T_old[idx+cols] + T_old[idx-cols] + T_old[idx-1] + T_old[idx+1]);
    }
}

int main()
{
    int M = 200; //rows
    int N = 200; //columns
    int arrSize = M * N * sizeof(double);
    int maxiter = 100;
    double err;
    int iter;

    double *T_new = (double *) malloc(arrSize);
    double *T_old = (double *) malloc(arrSize);

    int i, j;
    double temp = 1.0;

    for(i = 0; i < M; i++)                 
    {   
        for (j = 0; j < N; j++) 
        {
            T_old[i*N + j] = 0.0;
            T_new[i*N + j] = 0.0;
        }
    }

    for (j = 0; j < N; j++) 
    {
        T_old[0*N + j] = temp;
        T_new[0*N + j] = temp;
    }
    
    for (j = 0; j < N; j++) 
    {
        T_old[(N-1)*N + j] = 0.0;
        T_new[(N-1)*N + j] = 0.0;
    }
    
    for (i = 0; i < M; i++) 
    {
        T_old[i*N + 0] = 0.0;
        T_new[i*N + 0] = 0.0;
    }
    
    for (i = 0; i < M; i++) 
    {
        T_old[i*N + N-1] = 0.0;
        T_new[i*N + N-1] = 0.0;
    }
    
    double *T_new_d, *T_old_d;
    cudaMalloc(&T_new_d, arrSize);
    cudaMalloc(&T_old_d, arrSize);
    printf("Allocate Device memory for matrices\n");

    cudaMemcpy(T_new_d, T_new, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(T_old_d, T_old, arrSize, cudaMemcpyHostToDevice);
    printf("Copy matrices from the host memory to the CUDA device\n");

    const dim3 BLOCK_DIM(32, 32); // 1024 threads
    const dim3 GRID_DIM( (N-1)/BLOCK_DIM.x+1, (M-1)/BLOCK_DIM.y+1);
    printf("CUDA kernel launch with BLOCK_DIM[%d %d] GRID_DIM[%d %d]\n", BLOCK_DIM.x, BLOCK_DIM.y, GRID_DIM.x, GRID_DIM.y);
    		
    for (iter = 0; iter < maxiter; iter++)
    {
        solver<<<GRID_DIM, BLOCK_DIM>>>(T_new_d, T_old_d, N, M);  
        solver<<<GRID_DIM, BLOCK_DIM>>>(T_old_d, T_new_d, N, M); 
        if (iter%10 == 0)
        { 
        	cudaMemcpy(T_new, T_new_d, arrSize, cudaMemcpyDeviceToHost);
    		cudaMemcpy(T_old, T_old_d, arrSize, cudaMemcpyDeviceToHost);
        	err = 0.0; 
        	for(i = 1; i < (M-1); i++)                  
        	{
        	    for(j = 1; j < (N-1); j++)               
        	    {
        	       if (fabs(T_old[i*N + j]-T_new[i*N + j]) > err) err = fabs(T_old[i*N + j]-T_new[i*N + j]);
        	    }
        	}
        	printf("|%d| %f\n", iter, err);
        }
    }    
    cudaDeviceSynchronize();

    printf("Done solving\n");

    cudaMemcpy(T_new, T_new_d, arrSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(T_old, T_old_d, arrSize, cudaMemcpyDeviceToHost);
    printf("Copy matrices from the CUDA device to the host memory\n");

    cudaFree(T_new_d);
    cudaFree(T_old_d);
    free(T_new);
    free(T_old);
    printf("Free device and device memory\n");
}
