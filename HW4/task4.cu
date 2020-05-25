#include <iostream>
#include <stdio.h>
#include<stdlib.h>
#include <math.h>
#include<valarray>

using namespace std;

int nsamples = 6000;

float float_rand( float min, float max )
{    
    return min + (rand() / (RAND_MAX/(max-min)) );
}

__global__ void solver(float* res, const int n, const float* rand_x, const float* rand_y) 
{

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    float x,y;
    float val = 0.0;

    for (int i = 0; i < n; i++)
    {
        x = rand_x[i];
        y = rand_y[i];
        val += expf(-x*x-y*y);
    }

    __syncthreads();

    res[tid] = val/n/(1.0/16.0);
}

int main() 
{
    int blocks = 300;
    int threads = 300; 
  
    float h_res[threads*blocks]; 
    float *d_res;    
    float *h_x = new float[nsamples];
    float *h_y = new float[nsamples];

	srand(time(NULL));
    for (int i = 0; i < nsamples; i++)
    {
        h_x[i] = float_rand(-2, 2);
        h_y[i] = float_rand(-2, 2);
    }
    
    float *d_x, *d_y; 
    cudaMalloc(&d_res, threads*blocks*sizeof(float));
    cudaMalloc(&d_x, nsamples*sizeof(float));
    cudaMalloc(&d_y, nsamples*sizeof(float));

    cudaMemcpy(d_x, h_x, nsamples*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nsamples*sizeof(float), cudaMemcpyHostToDevice);

    solver<<<threads,blocks>>>(d_res, nsamples, d_x, d_y);

    cudaMemcpy(h_res, d_res, threads*blocks*sizeof(float), cudaMemcpyDeviceToHost);

    valarray<float> myvalarray(h_res,threads*blocks);
    
    float result = myvalarray.sum()/float(threads*blocks);

    cout<<"Result: "<<result<<endl;

    cudaFree(d_res);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
