//scp -P 2200 ./input.ppm o.shtepa@sandbox.zhores.net:./
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define HISTOGRAM_LENGTH 512
#define BLOCK_SIZE 512
#define PGMHeaderSize           0x40
//--------------------------------------------------------------------------------------------------------------------
inline bool loadPPM(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels)
{
    FILE *fp = NULL;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    char header[PGMHeaderSize];

    fp = fopen(file, "rb");
         if (!fp) {
              fprintf(stderr, "__LoadPPM() : unable to open file\n" );
                return false;
         }
    if (fgets(header, PGMHeaderSize, fp) == NULL)
    {
        fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
        return false;
    }
    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else
    {
        fprintf(stderr,"__LoadPPM() : File is not a PPM or PGM image\n" );
        *channels = 0;
        return false;
    }

    while (i < 3)
    {
        if (fgets(header, PGMHeaderSize, fp) == NULL)
        {
            fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
            return false;
        }
        if (header[0] == '#')
        {
            continue;
        }
        if (i == 0)
        {
            i += sscanf(header, "%u %u %u", &width, &height, &maxval);
        }
        else if (i == 1)
        {
            i += sscanf(header, "%u %u", &height, &maxval);
        }
        else if (i == 2)
        {
            i += sscanf(header, "%u", &maxval);
        }
    }

    if (NULL != *data)
    {
        if (*w != width || *h != height)
        {
            fprintf(stderr, "__LoadPPM() : Invalid image dimensions.\n" );
        }
    }
    else
    {
        *data = (unsigned char *) malloc(sizeof(unsigned char) * width * height * *channels);
        if (!data) {
         fprintf(stderr, "Unable to allocate hostmemory\n");
         return false;
        }
        *w = width;
        *h = height;
    }

    if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) == 0)
    {
        fprintf(stderr, "__LoadPPM() : read data returned error.\n" );
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}
//--------------------------------------------------------------------------------------------------------------------
__global__ void grayscale(unsigned char * input, unsigned char * output, int size) 
{
	unsigned char r, g, b;

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < size) 
    {
		r = input[3 * i];
		g = input[3 * i + 1];
		b = input[3 * i + 2];

		output[i] = (unsigned char)(0.21*(float)r + 0.71*(float)g + 0.07*(float)b);

		/*if (i == 0) 
        { 
            printf("grayscale i: %d\t", i); 
            printf("(r,g,b): (%u,%u,%u)\t", r, g, b); 
            printf("output[i]: %u\n", output[i]); 
        }*/
	}
}
//--------------------------------------------------------------------------------------------------------------------
__global__ void histogram(unsigned char *input, unsigned int *histo, int size) 
{    
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

	if (threadIdx.x < HISTOGRAM_LENGTH) histo_private[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// stride is the total number of threads in grid (threads/block*blocks/grid)
	int stride = blockDim.x * gridDim.x;

	// All threads handle blockDim.x * gridDim.x consecutive elements
	while (i<size) 
    {
		atomicAdd(&(histo[input[i]]), 1);
		i += stride;
	}

	__syncthreads();

	if (threadIdx.x < HISTOGRAM_LENGTH) 
    {
		atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
	}

	/*if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
		printf("histogram (bx,tx)=(%d,%d): %u\n", blockIdx.x, threadIdx.x, histo[threadIdx.x]);
	}*/
}
//--------------------------------------------------------------------------------------------------------------------
int main()
{
    unsigned char *data = NULL, *d_idata = NULL, *grayImage = NULL;
    unsigned int *d_odata = NULL;
    unsigned int w, h, channels;
    unsigned int numElements;
    size_t datasize;

    if(!loadPPM("input.ppm", &data, &w, &h, &channels))
    {
        fprintf(stderr, "Failed to open File\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded file with   w:%d   h:%d   channels:%d \n",w, h, channels);

    numElements = w*h*channels;
    datasize = numElements * sizeof(unsigned char);

	cudaMalloc(&d_idata, datasize);
    cudaMalloc(&grayImage, datasize);
    cudaMalloc(&d_odata, datasize);
    printf("Allocate Devicememory for data\n");

    cudaMemcpy(d_idata, data, datasize, cudaMemcpyHostToDevice);
    printf("Copy input data from the host memory to the CUDA device\n");

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((numElements-1)/ BLOCK_SIZE + 1 );
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n", blocksPerGrid.x, blocksPerGrid.y,  
                                                                          threadsPerBlock.x, threadsPerBlock.y);

	grayscale<<<blocksPerGrid, threadsPerBlock>>>(d_idata, grayImage, w*h);
    printf("Done grayscale\n");

    /*cudaMemcpy(data, grayImage, datasize, cudaMemcpyDeviceToHost);
    printf("Copy grayImage from the CUDA device to the host memory\n");*/

	histogram<<<blocksPerGrid, threadsPerBlock>>>(grayImage, d_odata, w*h);
    printf("Done histogram\n");

    cudaMemcpy(data, d_odata, datasize, cudaMemcpyDeviceToHost);
    printf("Copy output data from the CUDA device to the host memory\n");

    free(data);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(grayImage);
    printf("Free device and host memory\n");
   
}




