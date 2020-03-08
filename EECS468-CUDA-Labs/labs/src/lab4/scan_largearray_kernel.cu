#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define numThreads 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions

__global__ void prescanKernel(float *outArray, float *inArray, int numElements){

	// scan arr in shared mem
	extern __shared__ float scanArray[];

	//printf("threadidx.x: %i\n", threadIdx.x);

	scanArray[threadIdx.x] = (threadIdx.x > 0) ? inArray[blockIdx.x * numThreads + threadIdx.x - 1] : 0;


	__syncthreads();
	// exclusive

    // reduction step
	int stride = 1;

	while (stride < numThreads){
        __syncthreads();
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numThreads)
			scanArray[index] += scanArray[index-stride];

		stride *= 2;

	}


	// post-scan step
	stride = numThreads >> 1;

	while (stride > 0){
        __syncthreads();
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numThreads)
			scanArray[index+stride] += scanArray[index];

		stride = stride >> 1;

	}


    __syncthreads();
	outArray[blockIdx.x * numThreads + threadIdx.x] = scanArray[threadIdx.x];

}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

	printf("smem size: %i\n", numThreads * sizeof(float));

//    dim3 DimGrid(1,1);
    printf("num elements: %i\n", numElements);

    //TODO:Make dynamic for numElements not divisible by 1024 (Multiple and Remainder)
    const int numBlocks = numElements/numThreads;
    dim3 DimBlock(numBlocks);
    int sharedMemSize = numThreads * sizeof(float);

    prescanKernel<<<DimBlock, numThreads,sharedMemSize>>>(outArray, inArray, numElements);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
