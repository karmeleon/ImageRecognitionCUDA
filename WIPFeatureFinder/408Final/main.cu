#include "main.cuh"

int main() {
	printf("Loading images from disk. This will take a few seconds.\n");
	uint32_t count, numFeatures;
	uint32_t* images = loadFromFile(&count);
	printf("Loaded %u images from file (%u MB).\n", count, count * sizeof(uint32_t*) * IMAGE_SIZE * IMAGE_SIZE / 1024 / 1024);

	// start timer
	struct timeb start, end;
	int diff;
	ftime(&start);

	// find features
	feature* features = findFeatures(images, count, &numFeatures);

	// stop timer
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	printf("Computation took %d ms\n", diff);

	printf("Found %u features.\n", numFeatures);
	
	free(features);
	free(images);
}