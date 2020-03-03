#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


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
	scanArray[threadIdx.x] = (threadIdx.x > 0) ? inArray[threadIdx.x - 1] : 0;


	__syncthreads();
	// exclusive

    // reduction step
	int stride = 1;

	while (stride < numElements){
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numElements)
			scanArray[index] += scanArray[index-stride];

		stride *= 2;

		__syncthreads();
	}


	// post-scan step
	stride = numElements >> 1;

	while (stride > 0){
		int index = (threadIdx.x + 1) * stride * 2;

		if (index < numElements)
			scanArray[index+stride] += scanArray[index];

		stride = stride >> 1;

		__syncthreads();
	}


	outArray[threadIdx.x] = scanArray[threadIdx.x];
	__syncthreads();
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

	printf("smem size: %i\n", numElements * sizeof(float));

    dim3 DimGrid(1,1);
    printf("num elements: %i\n", numElements);
    dim3 DimBlock(numElements);
    int sharedMemSize = numElements * sizeof(float);

    prescanKernel<<<DimGrid, DimBlock, sharedMemSize>>>(outArray, inArray, numElements);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
