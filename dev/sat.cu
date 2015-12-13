#include "sat.cuh"

#define POWER2_BLOCK_SIZE 32

// Performs a 1d scan on the given memory. share should be shared memory for best results.
__device__ void scan(uint32_t* share) {
	// reduce
	for (uint32_t stride = 1; stride <= POWER2_BLOCK_SIZE; stride *= 2) {
		//__syncthreads();
		uint32_t index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * POWER2_BLOCK_SIZE)
			share[index] += share[index - stride];
	}

	// post reduce
	for (uint32_t stride = POWER2_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		//__syncthreads();
		uint32_t index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * POWER2_BLOCK_SIZE)
			share[index + stride] += share[index];
	}
	//__syncthreads();
}

// assumes image is in global memory
__device__ void scan2d(uint32_t* image) {
	// shared memory
	__shared__ uint32_t share[IMAGE_SIZE];

	// scan across rows first...
	for (int i = 0; i < IMAGE_SIZE; i++) {
		if (threadIdx.x < IMAGE_SIZE)
			share[threadIdx.x] = image[threadIdx.x + IMAGE_SIZE * i];
		else
			share[threadIdx.x] = 0;

		if (threadIdx.x + POWER2_BLOCK_SIZE < IMAGE_SIZE)
			share[threadIdx.x + POWER2_BLOCK_SIZE] = image[threadIdx.x + POWER2_BLOCK_SIZE + IMAGE_SIZE * i];
		else
			share[threadIdx.x + POWER2_BLOCK_SIZE] = 0;

		scan(share);

		// write back to image
		if (threadIdx.x < IMAGE_SIZE)
			image[threadIdx.x + IMAGE_SIZE * i] = share[threadIdx.x];
		if (threadIdx.x + POWER2_BLOCK_SIZE < IMAGE_SIZE)
			image[threadIdx.x + POWER2_BLOCK_SIZE + IMAGE_SIZE * i] = share[threadIdx.x + POWER2_BLOCK_SIZE];
		//__syncthreads();
	}

	/// ...then scan down columns.
	for (int i = 0; i < IMAGE_SIZE; i++) {
		if (threadIdx.x < IMAGE_SIZE)
			share[threadIdx.x] = image[threadIdx.x * IMAGE_SIZE + i];
		else
			share[threadIdx.x] = 0;

		if (threadIdx.x + POWER2_BLOCK_SIZE < IMAGE_SIZE)
			share[threadIdx.x + POWER2_BLOCK_SIZE] = image[(threadIdx.x + POWER2_BLOCK_SIZE) * IMAGE_SIZE + i];
		else
			share[threadIdx.x + POWER2_BLOCK_SIZE] = 0;

		scan(share);

		if (threadIdx.x < IMAGE_SIZE)
			image[threadIdx.x * IMAGE_SIZE + i] = share[threadIdx.x];
		if (threadIdx.x + POWER2_BLOCK_SIZE < IMAGE_SIZE)
			image[(threadIdx.x + POWER2_BLOCK_SIZE) * IMAGE_SIZE + i] = share[threadIdx.x + POWER2_BLOCK_SIZE];
		//__syncthreads();
	}
	// all done!
}

void scan2dSerial(uint32_t* image, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 1; j < dim; j++) {
			image[j + i * dim] += image[j - 1 + i * dim];
		}
	}

	for (int i = 0; i < dim; i++) {
		for (int j = 1; j < dim; j++) {
			image[i + j * dim] += image[i + (j - 1) * dim];
		}
	}
}