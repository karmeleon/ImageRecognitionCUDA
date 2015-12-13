#define IMAGE_SIZE 28
// the average pixel difference to trigger a feature
#define THRESHOLD 150

// features
#define HEDGE 0
#define VEDGE 1
#define HLINE 2
#define VLINE 3
#define RECT4 4

#define wbCheck(stmt) do {                                                \
	cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
		printf("ERROR: Failed to run stmt %s\n", #stmt);                       \
        printf("ERROR: Got CUDA error ...  %s\n", cudaGetErrorString(err));    \
		}                                                                     \
} while(0)

#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math_functions.h>
#include <thread>
//#include <omp.h>
#include <sys/timeb.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#ifndef _FEATURE
typedef struct _feature {
	uint8_t x1, y1, x2, y2, type;
	int32_t mag;
} feature;
#define _FEATURE
#endif