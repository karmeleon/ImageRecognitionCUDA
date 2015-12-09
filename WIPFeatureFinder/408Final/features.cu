#include "features.cuh"

// a 28*28 image has 90 000 rectangular regions with dimensions >= 4x4, times 5 possible features per region
// doing much less than that because the chances every region has all 5 features is virtually 0
#define FEATURES_PER_IMAGE (70000 * 5)
#define FEATURE_BUFFER_SIZE_PER_IMAGE (FEATURES_PER_IMAGE * sizeof(feature))
#define TOTAL_FEATURE_BUFFER_SIZE (FEATURE_BUFFER_SIZE_PER_IMAGE * concImages)
#define TOTAL_FEATURE_BUFFER_COUNT (FEATURES_PER_IMAGE * concImages)
// fraction of free VRAM to use
#define FREE_VRAM_USAGE .6
// the average pixel difference to trigger a feature
#define THRESHOLD 10
#define THREADS_PER_BLOCK 128

__global__ void findFeatures(uint32_t* imageBuffer, feature* featureBuffer, uint32_t* featureIndex) {
	uint32_t imgId = blockIdx.x;
	uint32_t* img = &(imageBuffer[IMAGE_SIZE * IMAGE_SIZE * imgId]);

	// build the SAT
	if (threadIdx.x < 32)
		scan2d(img);
	__syncthreads();

	// copy the SAT to shared memory
	__shared__ uint32_t SAT[IMAGE_SIZE * IMAGE_SIZE];
	memcpy(SAT, img, IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

	__syncthreads();

	// find haar-like features
	haarfinder(SAT, featureBuffer, THRESHOLD, featureIndex);
}

void printFeature(feature feat) {
	switch (feat.type) {
	case HEDGE:
		printf("Horiz Edge\n");
		break;
	case VEDGE:
		printf("Vert Edge\n");
		break;
	case HLINE:
		printf("Horiz Line\n");
		break;
	case VLINE:
		printf("Vert Line\n");
		break;
	case RECT4:
		printf("4-Rect\n");
		break;
	default:
		printf("Something else: %d\n", feat.type);
		break;
	}

	printf("Mag: %d\n", feat.mag);
	printf("(%d, %d) -> (%d, %d)\n", feat.x1, feat.y1, feat.x2, feat.y2);
}

feature* findFeatures(uint32_t* hostImageBuffer, uint32_t count, uint32_t* numFeatures) {
	// get the amount of vram we can allocate for this step
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("CUDA memory: Total: %d MB, free: %d MB\n", totalMem/1024/1024, freeMem/1024/1024);

	// compute number of images we can process at once
	int32_t concImages = freeMem * FREE_VRAM_USAGE / (FEATURE_BUFFER_SIZE_PER_IMAGE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	printf("Computing up to %d images at once using %lu MB of memory and %d kernels\n", concImages, concImages * (FEATURE_BUFFER_SIZE_PER_IMAGE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t)) / 1024 / 1024, (int)ceil((float)count / concImages));

	printf("Readying kernel 0\n");

	//feature* hostFeatureBuffer = (feature*)malloc(FEATURE_BUFFER_SIZE * concImages);
	uint32_t hostFeatureIndex;
	feature* deviceFeatureBuffer;
	uint32_t* deviceImageBuffer, *deviceFeatureIndex;

	wbCheck(cudaMalloc((void**)&deviceFeatureBuffer, TOTAL_FEATURE_BUFFER_SIZE));
	wbCheck(cudaMalloc((void**)&deviceImageBuffer, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));
	wbCheck(cudaMalloc((void**)&deviceFeatureIndex, sizeof(uint32_t)));

	uint32_t handledImages = 0;

	bool kernelRunning = concImages < count;
	bool willTerminate = false;

	// this will be expanded if necessary
	int32_t finishedFeatureBufferSize = concImages * FEATURES_PER_IMAGE;
	int32_t numFinishedFeatures = 0;
	feature* finishedFeatures = (feature*)malloc(finishedFeatureBufferSize * sizeof(feature));

	// the CUDA part

	// we want lots of shared memory, not so much L1
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	// clear feature buffer and copy first batch of images to device
	wbCheck(cudaMemset(deviceFeatureBuffer, 0, TOTAL_FEATURE_BUFFER_SIZE));
	wbCheck(cudaMemset(deviceImageBuffer, 0, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));
	wbCheck(cudaMemset(deviceFeatureIndex, 0, sizeof(uint32_t)));
	wbCheck(cudaMemcpy(deviceImageBuffer, &(hostImageBuffer[handledImages * IMAGE_SIZE * IMAGE_SIZE]), IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	dim3 dimGrid(min(concImages, count - handledImages), 1, 1);
	dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

	printf("Launching kernel 0\n");

	findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer, deviceFeatureIndex);

	handledImages += min(concImages, count - handledImages);

	uint32_t kernels = 0;

	do {
		kernels++;
		// copy feature buffer from device
		// cudaMemcpy blocks until the previous kernel finishes
		// see how much we have to copy
		wbCheck(cudaMemcpy(&hostFeatureIndex, deviceFeatureIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		printf("Kernel %u output sized %u elements (%u MB, %u%% full)\n", kernels, hostFeatureIndex, hostFeatureIndex * sizeof(feature) / 1024 / 1024, (uint32_t)((float)hostFeatureIndex * 100 / TOTAL_FEATURE_BUFFER_COUNT));

		if (hostFeatureIndex > TOTAL_FEATURE_BUFFER_COUNT)
			printf("Buffer overflow by %u features, increase FEATURES_PER_IMAGE or THRESHOLD\n", hostFeatureIndex - TOTAL_FEATURE_BUFFER_COUNT);

		// then copy it, but make sure it'll fit first
		if (numFinishedFeatures + hostFeatureIndex > finishedFeatureBufferSize) {
			printf("Resizing host buffer to %u elements (%u MB)\n", finishedFeatureBufferSize + TOTAL_FEATURE_BUFFER_COUNT, (finishedFeatureBufferSize * sizeof(feature) + TOTAL_FEATURE_BUFFER_SIZE) / 1024 / 1024);
			finishedFeatures = (feature*)realloc(finishedFeatures, finishedFeatureBufferSize * sizeof(feature) + TOTAL_FEATURE_BUFFER_SIZE);
			finishedFeatureBufferSize += TOTAL_FEATURE_BUFFER_COUNT;
		}
		printf("Copying buffer to host\n");
		wbCheck(cudaMemcpy(&(finishedFeatures[numFinishedFeatures]), deviceFeatureBuffer, hostFeatureIndex * sizeof(feature), cudaMemcpyDeviceToHost));
		numFinishedFeatures += hostFeatureIndex;
		
		willTerminate = !kernelRunning;

		// if there are more images to analyze, start them doing so
		if (handledImages < count) {
			printf("Readying kernel %u\n", kernels + 1);
			// clear feature buffer and copy next batch of images to device
			wbCheck(cudaMemset(deviceFeatureBuffer, 0, TOTAL_FEATURE_BUFFER_SIZE));
			wbCheck(cudaMemset(deviceImageBuffer, 0, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));
			wbCheck(cudaMemset(deviceFeatureIndex, 0, sizeof(uint32_t)));
			wbCheck(cudaMemcpy(deviceImageBuffer, &(hostImageBuffer[handledImages * IMAGE_SIZE * IMAGE_SIZE]), IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

			dim3 dimGrid(min(concImages, count - handledImages), 1, 1);
			dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

			printf("Launching kernel %u\n", kernels + 1);

			findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer, deviceFeatureIndex);

			handledImages += min(concImages, count - handledImages);
		} else
			kernelRunning = false;

	} while (!willTerminate);

	// cleanup, cleanup, everybody everywhere

	// C cleanup
	finishedFeatures = (feature*)realloc(finishedFeatures, numFinishedFeatures * sizeof(feature));

	// CUDA cleanup
	wbCheck(cudaFree(deviceFeatureBuffer));
	wbCheck(cudaFree(deviceImageBuffer));
	wbCheck(cudaFree(deviceFeatureIndex));

	*numFeatures = numFinishedFeatures;
	return finishedFeatures;
}