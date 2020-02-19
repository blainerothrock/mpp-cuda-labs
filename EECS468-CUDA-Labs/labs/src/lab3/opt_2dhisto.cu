#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void opt_2dhisto_kernel(uint32_t **input, size_t *inputHeight, size_t *inputWidth, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
    printf("Hello I am Kernel %i, %i\n", *inputWidth, *inputHeight);
}

uint32_t ** allocCopyInput(uint32_t **input, size_t width, size_t height)
{
    // TRIED TO FLATTEN TO FIX SEG FAULT, DIDN'T WORK
//    uint32_t flattenedInput[width*height];
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            flattenedInput[i * width + j] = 0;
//        }
//    }

    //printf("starting cudamalloc");
    ////

    uint32_t **input_d;
    uint32_t *input_device;
    int sizeInput = width*height*sizeof(uint32_t);
    cudaError_t allocError = cudaMalloc((void **)&input_d, sizeInput);
    printf("input alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(&input_device, input, sizeInput, cudaMemcpyHostToDevice);
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

void freeMemory(uint32_t **input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] ){
	printf("Freeing memory\n");
	cudaFree(&input);
	cudaFree(height);
	cudaFree(width);
	cudaFree(bins);
}


void opt_2dhisto( uint32_t **input, size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] )
{
    dim3 DimGrid(1,1);
    dim3 DimBlock(HISTO_HEIGHT, HISTO_WIDTH);

    unsigned int BIN_COUNT= HISTO_HEIGHT*HISTO_WIDTH;
    printf("\n\n--- starting kernel ---\n\n");
    opt_2dhisto_kernel<<<DimGrid,DimBlock>>>(input, height, width, bins);

    cudaThreadSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    printf("error: %s\n", cudaGetErrorString(error));
}

/* Include below the implementation of any other functions you need */ //



