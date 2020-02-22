#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void opt_2dhisto_kernel(uint32_t *input, size_t *inputHeight, size_t *inputWidth, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
    printf("Hello I am Kernel %u\n", (unsigned)input[1001 * *inputWidth + 200]);
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

void freeMemory(uint32_t *input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] ){
	printf("Freeing memory\n");
	cudaFree(&input);
	cudaFree(height);
	cudaFree(width);
	cudaFree(bins);
}


void opt_2dhisto( uint32_t *input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] )
{
    dim3 DimGrid(1,1);
//    dim3 DimBlock(HISTO_HEIGHT, HISTO_WIDTH);
    dim3 DimBlock(1, 1);

    unsigned int BIN_COUNT= HISTO_HEIGHT*HISTO_WIDTH;
    printf("\n\n--- starting kernel ---\n\n");
    opt_2dhisto_kernel<<<DimGrid,DimBlock>>>(input, height, width, bins);

    cudaThreadSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    printf("error: %s\n", cudaGetErrorString(error));
}

/* Include below the implementation of any other functions you need */ //



