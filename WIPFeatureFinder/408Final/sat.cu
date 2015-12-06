#include "sat.cuh"

// Performs a 1d scan on the given memory. share should be shared memory for best results.
__device__ void scan(uint32_t* share) {
	// reduce
	for (uint32_t stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		uint32_t index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * blockDim.x)
			share[index] += share[index - stride];
	}

	// post reduce
	for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
		__syncthreads();
		uint32_t index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * blockDim.x)
			share[index + stride] += share[index];
	}
	__syncthreads();
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

		if (threadIdx.x + blockDim.x < IMAGE_SIZE)
			share[threadIdx.x + blockDim.x] = image[threadIdx.x + blockDim.x + IMAGE_SIZE * i];
		else
			share[threadIdx.x + blockDim.x] = 0;

		scan(share);

		// write back to image
		if (threadIdx.x < IMAGE_SIZE)
			image[threadIdx.x + IMAGE_SIZE * i] = share[threadIdx.x];
		if (threadIdx.x + blockDim.x < IMAGE_SIZE)
			image[threadIdx.x + blockDim.x + IMAGE_SIZE * i] = share[threadIdx.x + blockDim.x];
		__syncthreads();
	}

	/// ...then scan down columns.
	for (int i = 0; i < IMAGE_SIZE; i++) {
		if (threadIdx.x < IMAGE_SIZE)
			share[threadIdx.x] = image[threadIdx.x * IMAGE_SIZE + i];
		else
			share[threadIdx.x] = 0;

		if (threadIdx.x + blockDim.x < IMAGE_SIZE)
			share[threadIdx.x + blockDim.x] = image[(threadIdx.x + blockDim.x) * IMAGE_SIZE + i];
		else
			share[threadIdx.x + blockDim.x] = 0;

		scan(share);

		if (threadIdx.x < IMAGE_SIZE)
			image[threadIdx.x * IMAGE_SIZE + i] = share[threadIdx.x];
		if (threadIdx.x + blockDim.x < IMAGE_SIZE)
			image[(threadIdx.x + blockDim.x) * IMAGE_SIZE + i] = share[threadIdx.x + blockDim.x];
		__syncthreads();
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

/*
void main() {
float* hostImage;
float* deviceImage;

// generate test image
hostImage = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++)
hostImage[i] = .01f;

// alloc buffers
wbCheck(cudaMalloc((void**)&deviceImage, IMG_SIZE * IMG_SIZE * sizeof(float)));

// fill buffers
wbCheck(cudaMemcpy(deviceImage, hostImage, IMG_SIZE * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice));

// this is a single-block algorithm that assumes square images of up to size 2048*2048.
// as a result the occupancy is terrible and so is performance when only working on one block
// but with 40 000 images being scanned at once, this should be a non-issue
dim3 dimGrid(1, 1, 1);
dim3 dimBlock((unsigned int)ceil((float)IMG_SIZE / 2.0f), 1, 1);

struct timeb start, end;
int diff;
ftime(&start);
scan2d <<<dimGrid, dimBlock, IMG_SIZE * sizeof(float) >>>(deviceImage, IMG_SIZE);

cudaDeviceSynchronize();
ftime(&end);
diff = (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
printf("Computation took %d ms\n", diff);

// copy buffer
wbCheck(cudaMemcpy(hostImage, deviceImage, IMG_SIZE * IMG_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

// free buffer
wbCheck(cudaFree(deviceImage));


// check to make sure output is correct
/*
for (int i = 0; i < IMG_SIZE; i++) {
for (int j = 0; j < IMG_SIZE; j++) {
float diff = abs(hostImage[j + IMG_SIZE * i] - (.01f * (i + 1) * (j + 1)));
if (diff > .01)
printf("%d, %d is off by %f (is %f, should be %f)\n", j, i, diff, hostImage[j + IMG_SIZE * i], .01f * (i + 1) * (j + 1));
}
}


free(hostImage);

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaError_t cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceReset failed!");
return 1;
}

return 0;
}
*/