#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define MAX_THREADS 512
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512

#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// Lab4: Host Helper Functions (allocate your own data structure...)

// Lab4: Device Functions

// Lab4: Kernel Functions
__global__ void prescanKernel(float *outArray, float *inArray, float *blockSums, int numElements){

    // scan arr in shared mem
    extern __shared__ float scanArray[];

    if (blockIdx.x == 0 && threadIdx.x == 0)
        scanArray[threadIdx.x] = 0;
    else
        scanArray[threadIdx.x] = inArray[blockIdx.x * blockDim.x + threadIdx.x - 1];

    __syncthreads();
    // exclusive

    // reduction step
    int stride = 1;

    while (stride < blockDim.x){
        int index;
        // exclusive for the first block
        if (blockIdx.x == 0)
            index = (threadIdx.x + 1) * stride * 2;
            // inclusive for remaining blocks
        else
            index = (threadIdx.x + 1) * stride * 2 - 1;

        if (index < blockDim.x)
            scanArray[index] += scanArray[index-stride];

        stride *= 2;

        __syncthreads();
    }

    // post-scan step
    stride = blockDim.x >> 1;

    while (stride > 0){
        // don't need to check block, inclusive
        int index;
        if (blockIdx.x == 0)
            index = (threadIdx.x + 1) * stride * 2;
        else
            index = (threadIdx.x + 1) * stride * 2 - 1;

        if (index < blockDim.x)
            scanArray[index+stride] += scanArray[index];

        stride = stride >> 1;

        __syncthreads();

    }

    __syncthreads();

    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = scanArray[blockDim.x - 1];
    }

    outArray[blockIdx.x * blockDim.x + threadIdx.x] = scanArray[threadIdx.x];

}

__global__ void prescanKernel_bank(float *outArray, float *inArray, float *blockSums, int numElements){

	// scan arr in shared mem
	extern __shared__ float scanArray[];

    int aIdx = threadIdx.x;
    int bIdx = aIdx + (numElements / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(aIdx);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bIdx);

    scanArray[aIdx + bankOffsetA] = inArray[aIdx];
    scanArray[bIdx + bankOffsetB] = inArray[bIdx];

//    printf("%i ", threadIdx.x);

    __syncthreads();
    // exclusive

    // reduction step
	int stride = 1;

	while (stride < blockDim.x){
        int ai = stride*(2*threadIdx.x+1)-1;
        int bi = stride*(2*threadIdx.x+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        scanArray[bi] += scanArray[ai];

		stride *= 2;

		__syncthreads();
	}

    if (threadIdx.x==0) { scanArray[numElements - 1 + CONFLICT_FREE_OFFSET(numElements - 1)] = 0; }

	// post-scan step
	stride = blockDim.x >> 1;

	while (stride > 0){
        int ai = stride*(2*threadIdx.x+1)-1;
        int bi = stride*(2*threadIdx.x+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        float t = scanArray[ai];
        scanArray[ai] = scanArray[bi];
        scanArray[bi] += t;

		stride = stride >> 1;

		__syncthreads();
	}

	__syncthreads();

    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = scanArray[blockDim.x - 1];
    }

//	outArray[blockIdx.x * blockDim.x + threadIdx.x] = scanArray[threadIdx.x];

    outArray[2*threadIdx.x] = scanArray[2*threadIdx.x]; // write results to device memory
    outArray[2*threadIdx.x+1] = scanArray[2*threadIdx.x+1];
}

__global__ void scanKernel(float *inArray) {
	extern __shared__ float scanArray[];
	int numT = blockDim.x;

	if (blockIdx.x == 0 && threadIdx.x == 0)
		scanArray[threadIdx.x] = 0;
	else
		scanArray[threadIdx.x] = inArray[blockIdx.x * blockDim.x + threadIdx.x - 1];

	__syncthreads();
	// exclusive

	// reduction step
	int stride = 1;

	while (stride < numT){
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numT)
			scanArray[index] += scanArray[index-stride];

		stride *= 2;
	}

	// post-scan step
	stride = numT >> 1;

	while (stride > 0){
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numT)
			scanArray[index+stride] += scanArray[index];

		stride = stride >> 1;

	}

	inArray[threadIdx.x] = scanArray[threadIdx.x];
	__syncthreads();
}

__global__ void bigBoyAdder(float * outArray, float * blockSums) {
    outArray[blockIdx.x * blockDim.x + threadIdx.x] += blockSums[blockIdx.x];
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
    int threads = numElements;
    if (threads > MAX_THREADS)
        threads = MAX_THREADS;

    int blockSumSize = ceil((float)numElements/(float)threads); // 15,625
    int sharedMemSize = threads * 1.45 * sizeof(float);

    printf("numThreads: %i\n", threads);
    printf("numElements: %i\n", numElements);
    printf("blockSumSize: %i\n", blockSumSize);
    printf("sharedMemSize: %i\n", sharedMemSize);

    float * d_blockSums = NULL;
    cudaMalloc( (void**) &d_blockSums, blockSumSize * sizeof(float));

    dim3 DimBlock(blockSumSize);

    prescanKernel<<<DimBlock, threads, sharedMemSize>>>(outArray, inArray, d_blockSums, numElements);
    cudaThreadSynchronize();

    if (blockSumSize > threads) { // only do one step
        int blockSumSize1 = ceil(float(blockSumSize)/(float)threads); // 16
        printf("blockSumSize1: %i\n", blockSumSize1);
        float * d_blockSums1 = NULL;
        cudaMalloc( (void**) &d_blockSums1, blockSumSize1 * sizeof(float)); // 16 * 4

        float * d_tmpOut = NULL;
        cudaMalloc( (void**) &d_tmpOut, blockSumSize * sizeof(float)); // 15,625 * 4

        dim3 DimBlock1(blockSumSize1);
        int numElements1 = (blockSumSize1 - 1) * blockSumSize % threads;
        prescanKernel<<<DimBlock1, threads, sharedMemSize>>>(d_tmpOut, d_blockSums, d_blockSums1, numElements1);
        cudaThreadSynchronize();

//        float * h_tmpOut = (float *)malloc(blockSumSize1 * sizeof(float));
//        cudaError_t allocError = cudaMemcpy( h_tmpOut, d_tmpOut, blockSumSize1 * sizeof(float), cudaMemcpyDeviceToHost);
//        printf("copy error: %s\n", cudaGetErrorString(allocError));
//        printf("d_tmpOut: %.1f\n", h_tmpOut[1024]);

//        float * d_blockSums2 = NULL;
//        cudaMalloc( (void**) &d_blockSums2, sizeof(float));
//
//        float * d_tmpOut2 = NULL;
//        cudaMalloc( (void**) &d_tmpOut2, blockSumSize1 * sizeof(float)); // 16 * 4

//        prescanKernel<<<1, blockSumSize1, blockSumSize1 * sizeof(float)>>>(d_tmpOut2, d_blockSums1, d_blockSums2, 16);

        scanKernel<<<1, blockSumSize1, blockSumSize1 * sizeof(float)>>>(d_blockSums1);
        cudaThreadSynchronize();

//        float * h_blockSums1 = (float *)malloc(blockSumSize1 * sizeof(float));
//        cudaError_t bsError = cudaMemcpy( h_blockSums1, d_blockSums1, blockSumSize1 * sizeof(float), cudaMemcpyDeviceToHost);
//        printf("copy bs error: %s\n", cudaGetErrorString(bsError));
//        for (int i = 0; i < blockSumSize1; i++) {
//            printf("%.1f ", h_blockSums1[i]);
//        }
//        printf("\n");

        bigBoyAdder<<<DimBlock1, threads>>>(d_tmpOut, d_blockSums1); // d_tmpOut2
        cudaThreadSynchronize();

        bigBoyAdder<<<DimBlock, threads>>>(outArray, d_tmpOut);
    } else {
        scanKernel<<<1, blockSumSize, sharedMemSize>>>(d_blockSums);
        cudaThreadSynchronize();
        bigBoyAdder<<<DimBlock, blockSumSize>>>(outArray, d_blockSums);
    }
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
