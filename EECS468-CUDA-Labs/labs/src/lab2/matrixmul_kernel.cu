/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__constant__ int stride;
__constant__ int height;
__constant__ int width;

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P, int *stride)
{

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

	extern __shared__ float smem[];

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float pValue = 0.0f;
//	printf("stride: %i\n", *stride);

	for (int k = 0; k < M.width / *stride; ++k) {
	    smem[(ty * *stride) + tx] = M.elements[((by) * blockDim.y + ty) + (k * *stride) + tx];
	    smem[(*stride * blockDim.y) + (ty * blockDim.x) + tx] = N.elements[((k * *stride + ty) + (bx * blockDim.x + tx))];

	    __syncthreads();

//        printf("k: %i\n", k);

//        for (int i, j = 0; i<blockDim.x, j<blockDim.y; ++i, ++j) {
//            pValue += smem[ty * i] * smem[(*stride * blockDim.y) + i * tx];
//        }
//        __syncthreads();
	}
    P.elements[row * P.width + col] = pValue;

//	for (int i = 0; i < M.width; ++i) {
//		for (int j = blockIdx.x * blockDim.x; j < (blockIdx.x * blockDim.x) + blockDim.x; ++j) {
//			subM[i][j] = M.elements[i * M.width * j];
//			__syncthreads();
//		}
//	}
//
//	for (int i = 0; i < blockDim.x; ++i) {
//		for (int j = blockIdx.y * blockDim.y; j < (blockIdx.y * blockDim.y) + blockDim.y; ++j) {
//			subN[i][j] = N.elements(j * N.width + i);
//			__syncthreads();
//		}
//	}


//
//	int count = 0;
//	for (int i = 0, j = 0; i<M.width , j<N.height; ++i, ++j) {
////		pValue += subM[i][row*M.width] * subN[j]
////		pValue += M.elements[row*M.width+i] * N.elements[j*N.width+col];
//		count++;
//	}

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
