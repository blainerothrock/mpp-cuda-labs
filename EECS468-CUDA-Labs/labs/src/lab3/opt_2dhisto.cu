#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void opt_2dhisto_kernel(uint32_t *input, size_t *inputHeight, size_t *inputWidth, uint32_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockIdx.x * blockDim.x;
	const int binSize = HISTO_HEIGHT*HISTO_WIDTH;

	__shared__ uint32_t sBins[binSize];

	for ( int pos = threadIdx.x; pos < binSize; pos += blockDim.x )
		sBins[pos] = 0;

	__syncthreads();

	int binIdx = input[tid];
	atomicAdd(&sBins[binIdx], 1);

	__syncthreads();

	for ( int pos = threadIdx.x; pos < binSize; pos += blockDim.x ) {
		atomicAdd(&bins[pos], sBins[pos]);
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
    //printf("input alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(input_d, flattenedRepresentation[0], sizeInput, cudaMemcpyHostToDevice);
    //delete [] flattenedRepresentation;
    //printf("input cpy error: %s\n", cudaGetErrorString(cpyError));
    return input_d;
}

uint32_t * allocCopyBin()
{
    uint32_t *bins_d;
    int sizeBins = HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t);
    cudaError_t allocError = cudaMalloc((void **)&bins_d, sizeBins);
    //printf("bin alloc error: %s\n", cudaGetErrorString(allocError));
//    cudaError_t cpyError = cudaMemcpy(bins_d, bins, sizeBins, cudaMemcpyHostToDevice);
//    printf("bin cpy error: %s\n", cudaGetErrorString(cpyError));
    cudaError_t memSetError = cudaMemset(bins_d, 0, sizeBins);
    //printf("bin mem set error: %s\n", cudaGetErrorString(memSetError));
    return bins_d;
}

size_t * allocCopyDim(size_t inputDim)
{
    size_t *inputDim_d;
    cudaError_t allocError = cudaMalloc((void **) &inputDim_d, sizeof(size_t));
    //printf("dim alloc error: %s\n", cudaGetErrorString(allocError));
    cudaError_t cpyError = cudaMemcpy(inputDim_d, &inputDim, sizeof(size_t), cudaMemcpyHostToDevice);
    //printf("dim cpy error: %s\n", cudaGetErrorString(cpyError));
    return inputDim_d;
}

void copyBinsFromDevice(uint8_t h_bins[HISTO_HEIGHT*HISTO_WIDTH], uint32_t d_bins[HISTO_HEIGHT*HISTO_WIDTH]){
	uint32_t tmpBins[HISTO_HEIGHT*HISTO_WIDTH];

	int sizeTmpBins = HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t);
	cudaError_t cpyError = cudaMemcpy(tmpBins, d_bins, sizeTmpBins, cudaMemcpyDeviceToHost);
    //printf("Copy D to H error: %s\n", cudaGetErrorString(cpyError));

    for ( int i = 0; i < HISTO_HEIGHT*HISTO_WIDTH; i++ ) {
    	if (tmpBins[i] > 255) tmpBins[i] = 255;
    	h_bins[i] = static_cast<uint8_t>(tmpBins[i]);
    }
}

void freeMemory(uint32_t *input, size_t *height, size_t *width, uint32_t bins[HISTO_HEIGHT*HISTO_WIDTH] ){
	//printf("Freeing memory\n");
	cudaFree(input);
	input = NULL;
	cudaFree(height);
	height = NULL;
	cudaFree(width);
	width = NULL;
	cudaFree(bins);
	bins = NULL;
}


void opt_2dhisto( uint32_t *input, size_t *height, size_t *width, uint32_t bins[HISTO_HEIGHT*HISTO_WIDTH] )
{
    //dim3 DimGrid(31872, 1);
    float numThreads = 1024.0;
    float inputSize = INPUT_HEIGHT * INPUT_WIDTH;
    float numBlocks = ceilf(inputSize / numThreads);

    // set the bins count to 0
    cudaMemset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH);

//    printf("\nprinting first 5 values of bins:");
//    printf(bins[0]);
//    for(int i = 0; i < 5; i++){
//        printf("\t %i", bins[i]);
//        printf("\n");
//    }

//    unsigned int BIN_COUNT= HISTO_HEIGHT*HISTO_WIDTH;
    opt_2dhisto_kernel<<<numBlocks,numThreads>>>(input, height, width, bins);

    cudaThreadSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    //printf("error: %s\n", cudaGetErrorString(error));

}

/* Include below the implementation of any other functions you need */ //



