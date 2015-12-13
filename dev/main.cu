#include "main.cuh"

int main() {
	printf("Loading images from disk. This will take a few seconds.\n");
	/*uint32_t count, numFeatures;
	uint32_t* images = loadAllFromFile(&count);
	printf("Loaded %u images from file (%u MB).\n", count, count * sizeof(uint32_t*) * IMAGE_SIZE * IMAGE_SIZE / 1024 / 1024);*/

	uint32_t positiveCount, negativeCount, numPositiveFeatures, numNegativeFeatures;
	uint8_t label = 0;
	unsigned char* labels;
	uint32_t* positiveImages, *negativeImages;

	loadLabelFromFile(labels, label, &positiveImages, &positiveCount, &negativeImages, &negativeCount);
	printf("Loaded %u positive images and %u negative images from file (%u MB).\n", positiveCount, negativeCount, (positiveCount + negativeCount) * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t) / 1024 / 1024);

	// start timer
	struct timeb start, end;
	int diff;
	ftime(&start);

	// find features
	//feature* features = findFeatures(images, count, &numFeatures);
	printf("Finding features in positive set.\n");
	feature* positiveFeatures = findFeatures(positiveImages, positiveCount, &numPositiveFeatures);
	//feature* positiveFeatures = serialFindFeatures(positiveImages, positiveCount, &numPositiveFeatures);
	printf("Found %u positive features\n", numPositiveFeatures);
	printf("Finding features in negative set.\n");
	feature* negativeFeatures = findFeatures(negativeImages, negativeCount, &numNegativeFeatures);
	//feature* negativeFeatures = serialFindFeatures(negativeImages, negativeCount, &numNegativeFeatures);
	printf("Found %u negative features\n", numNegativeFeatures);

	// stop timer
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	printf("Computation took %d ms\n", diff);

	//printf("Found %u features.\n", numFeatures);

	ftime(&start);
	train_cascade(positiveFeatures, negativeFeatures, labels, numPositiveFeatures, numNegativeFeatures, positiveCount, negativeCount);
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	printf("Computation took %d ms\n", diff);
	
	free(positiveFeatures);
	free(negativeFeatures);
	free(positiveImages);
	free(negativeImages);
	/*free(features);
	free(images);*/
}