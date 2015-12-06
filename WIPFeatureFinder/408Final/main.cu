#include "main.cuh"

void addPureWhite(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 0;
}

void addPureBlack(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 1;
}

uint32_t* generateImages(int num) {
	uint32_t* images = (uint32_t*)malloc(num * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	for (int i = 0; i < num; i += 2) {
		addPureBlack(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
		addPureWhite(&images[(i + 1) * IMAGE_SIZE * IMAGE_SIZE]);
	}
	return images;
}

int main() {
	int count = 330;
	uint32_t* images = generateImages(count);
	uint32_t numFeatures;
	feature* features = findFeatures(images, count, &numFeatures);

	for (uint32_t i = 0; i < min(10, numFeatures); i++) {
		printFeature(features[i]);
		printf("\n");
	}

	printf("Found %u features\n", numFeatures);

	free(features);
	free(images);
}