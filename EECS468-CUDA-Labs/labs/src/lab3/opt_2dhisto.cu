#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"


__global__ void opt_2dhisto_kernel(uint32_t **input, uint8_t *bins, int BIN_COUNT )
{
    // TODO:
}


void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t *bins,  uint8_t HIST_HEIGHT, uint8_t HIST_WIDTH)
{
/* This function should only contain a call to the GPU
   histogramming kernel. Any memory allocations and
   transfers must be done outside this function */

    uint32_t **input_d;
    int sizeInput = width*height*sizeof(uint32_t);
    cudaMalloc((void **)&input_d, sizeInput);

    uint8_t *kernel_bins_d;
    int sizeKernelBins = HIST_HEIGHT*HIST_WIDTH*sizeof(uint8_t);
    cudaMalloc((void **)&kernel_bins_d, sizeKernelBins);

    cudaMemcpy(input_d, input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_bins_d, bins, sizeKernelBins, cudaMemcpyHostToDevice);

    dim3 DimGrid(1,1);
    dim3 DimBlock(HIST_HEIGHT, HIST_WIDTH);

    unsigned int BIN_COUNT= HIST_HEIGHT*HIST_WIDTH;
    opt_2dhisto_kernel<<<DimGrid,DimBlock>>>(input_d, kernel_bins_d,BIN_COUNT);

    cudaMemcpy(kernel_bins_d, bins, sizeKernelBins, cudaMemcpyDeviceToHost);
}

/* Include below the implementation of any other functions you need */

