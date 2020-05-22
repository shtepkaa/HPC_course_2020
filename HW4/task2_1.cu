/*reference https://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf*/

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TILE_W      10
#define TILE_H      10
#define R           2                   // filter radius
#define D           (R*2+1)             // filter diameter
#define S           (D*D)               // filter size
#define BLOCK_W     (TILE_W+(2*R))
#define BLOCK_H     (TILE_H+(2*R))
//--------------------------------------------------------------------------------------------------------------------
__global__ void box_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h)
{
    __shared__ unsigned char smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x*TILE_W + threadIdx.x - R;
    int y = blockIdx.y*TILE_H + threadIdx.y - R;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, w-1);
    y = max(y, 0);
    y = min(y, h-1);

    unsigned int index = y * w + x;
    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    // each thread copies its pixel of the block to shared memory
    smem[bindex] = in[index];
    __syncthreads();

    // only threads inside the apron will write results
    if ((threadIdx.x >= R) && (threadIdx.x < (BLOCK_W-R)) && (threadIdx.y >= R) && (threadIdx.y < (BLOCK_H-R))) 
    {
        float sum = 0;
        for(int dy = -R; dy <= R; dy++) 
        {
            for(int dx = -R; dx <= R; dx++) 
            {
                float i = smem[bindex + (dy*blockDim.x) + dx];
                sum += i;
            }
        }
        out[index] = sum / S;
    }
}
//--------------------------------------------------------------------------------------------------------------------
const unsigned int imgw = 100;
const unsigned int imgh = 100;
void loadImg(unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *ch){
  *w = imgw;
  *h = imgh;
  *ch = 1;
  *data = (unsigned char *)malloc(imgw*imgh*sizeof(unsigned char));
  for (int i = 0; i < imgw*imgh; i++) (*data)[i] = i%8;
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
    for (int i = 0; i < TILE_W; i++)
    {
      for (int j = 0; j < TILE_H; j++) printf("%d ", data[i*w+j]);
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
   
    box_filter<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, w, h);

    cudaMemcpy(data, d_odata, datasize, cudaMemcpyDeviceToHost);
    printf("Copy output data from the CUDA device to the host memory\n");

    printf("output:\n");
    for (int i = 0; i < TILE_W; i++)
    {
      for (int j = 0; j < TILE_H; j++) printf("%d ", data[i*w+j]);
      printf("\n");
    }

    free(data);
    cudaFree(d_idata);
    cudaFree(d_odata);
    printf("Free device and host memory\n");

}
