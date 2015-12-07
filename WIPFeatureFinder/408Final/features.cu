#include "features.cuh"

// 6 MiB should be enough
#define FEATURE_BUFFER_SIZE 1024 * 1024 * 6
#define FEATURES_IN_BUFFER FEATURE_BUFFER_SIZE / sizeof(feature)
// fraction of free VRAM to use
#define FREE_VRAM_USAGE .6
// the average pixel difference to trigger a feature
#define THRESHOLD 10
#define THREADS_PER_BLOCK 32

__global__ void findFeatures(uint32_t* imageBuffer, feature* featureBuffer) {
	uint32_t imgId = blockIdx.x;
	uint32_t* img = &(imageBuffer[IMAGE_SIZE * IMAGE_SIZE * imgId]);
	feature* features = &(featureBuffer[FEATURES_IN_BUFFER * imgId]);

	// build the SAT
	if (threadIdx.x < 32)
		scan2d(img);

	// copy the SAT to shared memory
	__shared__ uint32_t SAT[IMAGE_SIZE * IMAGE_SIZE];
	memcpy(SAT, img, IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

	// find haar-like features
	haarfinder(SAT, features, THRESHOLD);
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

//uint64_t packFeature(feature unpacked) {
//	// 7 bits each for x1, y1, x2, y2
//	uint64_t out = 0;
//	// bitwise operations are so satisfying
//	out |= ((uint64_t)unpacked.x1) << 57;
//	out |= ((uint64_t)unpacked.y1) << 50;
//	out |= ((uint64_t)unpacked.x2) << 43;
//	out |= ((uint64_t)unpacked.y2) << 36;
//	out |= ((uint64_t)unpacked.type) << 33;
//	((int32_t*)&out)[0] = ((int32_t)unpacked.mag);
//	return out;
//}

// unpacks a bitpacked CUDA feature into a C feature struct
//inline feature unpackFeature(uint64_t packed) {
//	feature out;
//	out.x1 = (packed & 0xfe00000000000000) >> 57;
//	out.y1 = (packed & 0x1fc000000000000) >> 50;
//	out.x2 = (packed & 0x003f80000000000) >> 43;
//	out.y2 = (packed & 0x00007f000000000) >> 36;
//	out.type = (packed & 0xe00000000) >> 33;
//	out.mag = (packed & 0xffffffff);
//	return out;
//}

feature* findFeatures(uint32_t* hostImageBuffer, uint32_t count, uint32_t* numFeatures) {
	// get the amount of vram we can allocate for this step
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("CUDA memory: Total: %d MB, free: %d MB\n", totalMem/1024/1024, freeMem/1024/1024);

	// compute number of images we can process at once
	int32_t concImages = freeMem * FREE_VRAM_USAGE / (FEATURE_BUFFER_SIZE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	printf("Computing %d images at once using %lu MB of memory\n", concImages, concImages * (FEATURE_BUFFER_SIZE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t))/1024/1024);

	feature* hostFeatureBuffer = (feature*)malloc(FEATURE_BUFFER_SIZE * concImages);
	feature* deviceFeatureBuffer;
	uint32_t* deviceImageBuffer;

	wbCheck(cudaMalloc((void**)&deviceFeatureBuffer, FEATURE_BUFFER_SIZE * concImages));
	wbCheck(cudaMalloc((void**)&deviceImageBuffer, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));

	uint32_t handledImages = 0;

	bool kernelRunning = true;
	bool willTerminate = false;

	// this will be expanded if necessary
	uint32_t finishedFeatureBufferSize = concImages * FEATURES_IN_BUFFER / 2;
	uint32_t numFinishedFeatures = 0;
	feature* finishedFeatures = (feature*)malloc(finishedFeatureBufferSize * sizeof(feature));

	// the CUDA part

	// we want lots of shared memory, not so much L1
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	// clear feature buffer and copy first batch of images to device
	wbCheck(cudaMemset(deviceFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages));
	wbCheck(cudaMemset(deviceImageBuffer, 0, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));
	wbCheck(cudaMemcpy(deviceImageBuffer, &(hostImageBuffer[handledImages * IMAGE_SIZE * IMAGE_SIZE]), IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	dim3 dimGrid(min(concImages, count - handledImages), 1, 1);
	// one warp per block, we can crank this later if it helps
	dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

	findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer);

	handledImages += min(concImages, count - handledImages);

	//uint32_t OMPUnpackedImages = 0;

	do {
		// copy feature buffer from device
		// cudaMemcpy blocks until the previous kernel finishes
		wbCheck(cudaMemcpy(hostFeatureBuffer, deviceFeatureBuffer, FEATURE_BUFFER_SIZE * concImages, cudaMemcpyDeviceToHost));
		
		willTerminate = !kernelRunning;

		// if there are more images to analyze, start them doing so
		if (handledImages < count) {
			// clear feature buffer and copy next batch of images to device
			wbCheck(cudaMemset(deviceFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages));
			wbCheck(cudaMemset(deviceImageBuffer, 0, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));
			wbCheck(cudaMemcpy(deviceImageBuffer, &(hostImageBuffer[handledImages * IMAGE_SIZE * IMAGE_SIZE]), IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

			dim3 dimGrid(min(concImages, count - handledImages), 1, 1);
			// one warp per block, we can crank this later if it helps
			dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

			findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer);

			handledImages += min(concImages, count - handledImages);
		} else
			kernelRunning = false;

		#pragma omp parallel for
		for (int32_t i = 0; i < concImages; i++) {
			uint32_t id = omp_get_thread_num();
			for (int32_t j = 0; j < FEATURES_IN_BUFFER; j++) {
				feature packed = hostFeatureBuffer[j + FEATURES_IN_BUFFER * i];
				if (packed.mag == 0 && j > 0) {
					// there are no more features in this buffer, dump it into the combined buffer
					#pragma omp critical
					{
						//printf("found %d features in image %d\n", j, i);
						int32_t spaceInBuffer = finishedFeatureBufferSize - numFinishedFeatures;
						// make the combined buffer bigger if necessary
						if (spaceInBuffer < j) {
							finishedFeatures = (feature*)realloc(finishedFeatures, (finishedFeatureBufferSize + concImages * FEATURES_IN_BUFFER) * sizeof(feature));
							finishedFeatureBufferSize += concImages * FEATURES_IN_BUFFER;
						}
						//memcpy(&(finishedFeatures[numFinishedFeatures]), threadFeatureBuffers[id], j * sizeof(feature));
						memcpy(&(finishedFeatures[numFinishedFeatures]), &(hostFeatureBuffer[FEATURES_IN_BUFFER * i]), j * sizeof(feature));
						numFinishedFeatures += j;
					}
					break;
				}
				else if (packed.mag == 0)
					break;
			}
		}
		//OMPUnpackedImages += concImages;
		//printf("%d%% complete (%u / %u), found %u features so far (%u MB feat buffer)\n", OMPUnpackedImages * 100 / count, OMPUnpackedImages, count, numFinishedFeatures, finishedFeatureBufferSize / 1024 / 1024 * sizeof(feature));

		// clear the host feature buffer
		memset(hostFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages);

	} while (!willTerminate);

	// cleanup, cleanup, everybody everywhere

	// C cleanup
	finishedFeatures = (feature*)realloc(finishedFeatures, numFinishedFeatures * sizeof(feature));

	free(hostFeatureBuffer);

	// CUDA cleanup
	wbCheck(cudaFree(deviceFeatureBuffer));
	wbCheck(cudaFree(deviceImageBuffer));

	*numFeatures = numFinishedFeatures;
	return finishedFeatures;
}