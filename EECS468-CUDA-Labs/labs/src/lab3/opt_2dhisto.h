#ifndef OPT_KERNEL
#define OPT_KERNEL

#define HISTO_WIDTH  1024
#define HISTO_HEIGHT 1
#define HISTO_LOG 10

#define UINT8_MAX 255

uint32_t * allocCopyInput(uint32_t *input[], size_t width, size_t height);
uint8_t * allocCopyBin(uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);
size_t * allocCopyDim(size_t inputDim);

void opt_2dhisto(uint32_t input[], size_t *height, size_t *width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

/* Include below the function headers of any other functions that you implement */


#endif
