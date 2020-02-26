#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__device__ void atomicAdd(uint8_t *address, uint8_t val) {
	// convert to uint
	uint32_t * uAddress = (uint32_t *) (address - ((size_t)address & 0x3));
	uint32_t old = *uAddress;
	uint32_t shift = (((size_t)address & 0x3) << 3);
	uint32_t sum;
	uint32_t assumed;

	do {
		assumed = old;
		sum = val + static_cast<uint8_t>((old >> shift) & 0xff);
		if (sum > UINT8_MAX)
			sum = UINT8_MAX;
		old = (old & ~(0x000000ff << shift)) | (sum << shift);
		old = atomicCAS(uAddress, assumed, old);
	} while (assumed != old);
}

//__device__ uint32_t gBins[HISTO_HEIGHT*HISTO_WIDTH];

__global__ void opt_2dhisto_kernel(uint32_t *input, size_t *inputHeight, size_t *inputWidth, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockIdx.x * blockDim.x;
	const int binSize = HISTO_HEIGHT*HISTO_WIDTH;

	__shared__ uint8_t sBins[binSize];

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
    float numThreads = 1024.0;
    float inputSize = INPUT_HEIGHT * INPUT_WIDTH;
    float numBlocks = ceilf(inputSize / numThreads);

    // set the bins count to 0
    cudaMemset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH);

//    unsigned int BIN_COUNT= HISTO_HEIGHT*HISTO_WIDTH;
    opt_2dhisto_kernel<<<numBlocks,numThreads>>>(input, height, width, bins);

    cudaThreadSynchronize();
//    cudaError_t error;
//    error = cudaGetLastError();
//    printf("error: %s\n", cudaGetErrorString(error));

}

/* Include below the implementation of any other functions you need */ //



