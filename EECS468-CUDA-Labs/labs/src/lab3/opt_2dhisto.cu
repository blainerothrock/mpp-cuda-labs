//#include <stdint.h>
//#include <stdlib.h>
//#include <string.h>
//
//#include <cutil.h>
//#include "util.h"
//#include "ref_2dhisto.h"
//#include "opt_2dhisto.h"
//
//
//__global__ void opt_2dhisto_kernel(uint32_t **input, uint8_t *bins, int BIN_COUNT ){
//
//
//
//
//}
//
//
//void opt_2dhisto(uint32_t *input, size_t INPUT_HEIGHT, size_t INPUT_WIDTH, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH],  uint8_t HIST_HEIGHT, uint8_t HIST_WIDTH)
//{
//    /* This function should only contain a call to the GPU
//       histogramming kernel. Any memory allocations and
//       transfers must be done outside this function */
//
//    uint32_t **input_d;
//    (uint32_t**)cudaMalloc(&input_d, INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
//
//    uint8_t *kernel_bins_d;
//    (uint8_t*)cudaMalloc(HIST_HEIGHT*HIST_WIDTH*sizeof(uint8_t));
//
//    cudaMemcpy(input_d, input,INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t), cudaMemcpyHostToDevice);
//
//    cudaMemcpy(kernel_bins_d, kernel_bins,HIST_HEIGHT*HIST_WIDTH*sizeof(uint8_t), cudaMemcpyHostToDevice);
//
//    dim3 DimGrid(1,1);
//    dim3 DimBlock(HISTO_HEIGHT, HISTO_WIDTH);
//
//    unsigned int BIN_COUNT= HIST_HEIGHT*HIST_WIDTH;
//    opt_2dhisto_kernel<<<DimGrid,DimBlock>>>(input_d, kernel_bins_d,BIN_COUNT);
//
//    cudaMemcpy(kernel_bins_d, kernel_bins, HIST_HEIGHT*HIST_WIDTH*sizeof(uint8_t), cudaMemcpyHostToDevice);
//
//}
//
///* Include below the implementation of any other functions you need */
//
