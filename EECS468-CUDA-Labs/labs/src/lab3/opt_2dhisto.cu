#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

static inline __device__ void atomicAdd(uint8_t *address, uint8_t val) {
  unsigned int * address_as_ui = (unsigned int *) (address - ((size_t)address & 0x3));
  unsigned int old = *address_as_ui;
  unsigned int shift = (((size_t)address & 0x3) << 3);
  unsigned int sum;
  unsigned int assumed;

  do {
    assumed = old;
    sum = val + static_cast<uint8_t>((old >> shift) & 0xff);
    // do not rollover
    if (sum > UINT8_MAX) return;
    old = (old & ~(0x000000ff << shift)) | (sum << shift);
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

__global__ void opt_2dhisto_kernel(uint32_t *input, size_t *inputHeight, size_t *inputWidth, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
    // get indexes
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int size = *inputHeight * *inputWidth;
    int sectionSize = 32;
    int start = tid*sectionSize;

    uint8_t sub_bins[32] = {};

    for(int i = 0; i < sectionSize; i++){
    	if((start + i) < size){
    		int idx = input[start + i];

    		// ensure < 256
    		if(bins[idx] < 255){
    			atomicAdd(&bins[idx], 1);
    		}
    	}
    }



}

uint32_t * allocCopyInput(uint32_t **input, size_t width, size_t height)
{
    // solution from http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
    uint32_t** flattenedRepresentation = new uint32_t*[height];
    flattenedRepresentation[0] = new uint32_t[height * width];
    for (int i = 1; i < height; ++i) flattenedRepresentation[i] = flattenedRepresentation[i-1] + width;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            flattenedRepresentation[i][j] = input[i][j];
        }
    }

    uint32_t *input_d;
//    uint32_t *input_device;
    int sizeInput = width*height*sizeof(uint32_t);
    cudaError_t allocError = cudaMalloc((void **)&input_d, sizeInput);
    printf("input alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(input_d, flattenedRepresentation[0], sizeInput, cudaMemcpyHostToDevice);
    //delete [] flattenedRepresentation;
    printf("input cpy error: %s\n", cudaGetErrorString(cpyError));
    return input_d;
}

uint8_t * allocCopyBin(uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
    uint8_t *bins_d;
    int sizeBins = HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t);
    cudaError_t allocError = cudaMalloc((void **)&bins_d, sizeBins);
    printf("bin alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(bins_d, bins, sizeBins, cudaMemcpyHostToDevice);
    printf("bin cpy error: %s\n", cudaGetErrorString(cpyError));
    cudaError_t memSetError = cudaMemset(bins_d, 0, sizeBins);
    printf("bin mem set error: %s\n", cudaGetErrorString(cpyError));
    return bins_d;
}

size_t * allocCopyDim(size_t inputDim)
{
    size_t *inputDim_d;
    cudaError_t allocError = cudaMalloc((void **) &inputDim_d, sizeof(size_t));
    printf("dim alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(inputDim_d, &inputDim, sizeof(size_t), cudaMemcpyHostToDevice);
    printf("dim cpy error: %s\n", cudaGetErrorString(cpyError));
    return inputDim_d;
}

void copyBinsFromDevice(uint8_t h_bins[HISTO_HEIGHT*HISTO_WIDTH], uint8_t d_bins[HISTO_HEIGHT*HISTO_WIDTH]){
	int sizeBins = HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t);
	cudaError_t cpyError = cudaMemcpy(h_bins, d_bins, sizeBins, cudaMemcpyDeviceToHost);
    printf("Copy D to H error: %s\n", cudaGetErrorString(cpyError));
}

void freeMemory(uint32_t *input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] ){
	printf("Freeing memory\n");
	cudaFree(input);
	input = NULL;
	cudaFree(height);
	height = NULL;
	cudaFree(width);
	width = NULL;
	cudaFree(bins);
	bins = NULL;
}


void opt_2dhisto( uint32_t *input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] )
{
    //dim3 DimGrid(31872, 1);
    float numThreads = 128.0;
    float numBlocks = 3984.0;

    cudaMemset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH);

    // TODO: make numBlocks dynamic
    // (width * height) / 512 = 318
    //int numBlocks = ceil((float)(*width * *height) / numThreads);
    //printf("num blocks: %i", numBlocks);//
    //dim3 DimBlock();

//    unsigned int BIN_COUNT= HISTO_HEIGHT*HISTO_WIDTH;
    opt_2dhisto_kernel<<<numBlocks,numThreads>>>(input, height, width, bins);

    cudaThreadSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
//    printf("error: %s\n", cudaGetErrorString(error));

}

/* Include below the implementation of any other functions you need */ //


