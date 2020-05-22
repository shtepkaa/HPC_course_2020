#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_W 16 
#define BLOCK_H 16

//--------------------------------------------------------------------------------------------------------------------
__global__ void median_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h) {

    __shared__ unsigned short surround[BLOCK_W*BLOCK_H][9];

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   

    if( (x >= (w - 1)) || (y >= h - 1) || (x == 0) || (y == 0)) return;

    // Fill shared memory
    int iter = 0;
    for (int r = x - 1; r <= x + 1; r++) 
    {
        for (int c = y - 1; c <= y + 1; c++) 
        {
            surround[tid][iter] = in[c * w + r];
            iter++;
        }
    }

    // Sort shared memory to find the median using Bubble Short
    for (int i = 0; i < 5; i++) 
    {

        // Find the position of the minimum element
        int minval = i;
        for (int l = i+1; l<9; l++) if (surround[tid][l] < surround[tid][minval]) minval = l;

        // Put found minimum element in its place
        unsigned short temp = surround[tid][i];
        surround[tid][i] = surround[tid][minval];
        surround[tid][minval]=temp;
    }

    // Pick the middle one
    out[(y*w)+x] = surround[tid][4]; 

    __syncthreads();

}
//--------------------------------------------------------------------------------------------------------------------
const unsigned int imgw = 100;
const unsigned int imgh = 100;
void loadImg(unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *ch){
  *w = imgw;
  *h = imgh;
  *ch = 1;
  *data = (unsigned char *)malloc(imgw*imgh*sizeof(unsigned char));
  for (int i = 0; i < imgw*imgh; i++) (*data)[i] = i%9;
  }
//--------------------------------------------------------------------------------------------------------------------
int main()
{
    unsigned char *data = NULL, *d_idata = NULL, *d_odata = NULL;
    unsigned int w, h, channels;

    unsigned int numElements;
    size_t datasize;

    loadImg(&data, &w, &h, &channels);
    printf("Loaded input file with   w:%d   h:%d   channels:%d \n",w, h, channels);
    
    printf("input:\n");
    for (int i = 0; i < BLOCK_W; i++)
    {
      for (int j = 0; j < BLOCK_H; j++) printf("%d ", data[i*w+j]);
      printf("\n");
    }

    numElements = w*h*channels;
    datasize = numElements * sizeof(unsigned char);
    
    cudaMalloc(&d_idata, datasize);
    cudaMalloc(&d_odata, datasize);
    printf("Allocate Devicememory for data\n");

    cudaMemcpy(d_idata, data, datasize, cudaMemcpyHostToDevice);
    printf("Copy input data from the host memory to the CUDA device\n");

    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);
    dim3 blocksPerGrid((w+threadsPerBlock.x-1)/threadsPerBlock.x, (h+threadsPerBlock.y-1)/threadsPerBlock.y);
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n", blocksPerGrid.x, blocksPerGrid.y,  
                                                                          threadsPerBlock.x, threadsPerBlock.y);
   
    median_filter<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, w, h);

    cudaMemcpy(data, d_odata, datasize, cudaMemcpyDeviceToHost);
    printf("Copy output data from the CUDA device to the host memory\n");

    printf("output:\n");
    for (int i = 0; i < BLOCK_W; i++)
    {
      for (int j = 0; j < BLOCK_H; j++) printf("%d ", data[i*w+j]);
      printf("\n");
    }

    free(data);
    cudaFree(d_idata);
    cudaFree(d_odata);
    printf("Free device and host memory\n");

}
