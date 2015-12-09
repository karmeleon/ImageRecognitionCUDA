#include "main.cuh"

void addPureWhite(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 0;
}

void addPureBlack(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 255;
}

void addVertLine(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			if (j > IMAGE_SIZE / 3 && j < 2 * IMAGE_SIZE / 3)
				buf[j + IMAGE_SIZE * i] = 1;
			else
				buf[j + IMAGE_SIZE * i] = 0;
		}
	}
}

uint32_t* generateImages(int num) {
	uint32_t* images = (uint32_t*)malloc(num * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

	/*for (int i = 0; i < num; i += 2) {
		addPureBlack(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
		addPureWhite(&images[(i + 1) * IMAGE_SIZE * IMAGE_SIZE]);
	}*/

	for (int i = 0; i < num; i += 3) {
		addPureBlack(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
		addPureWhite(&images[(i + 1) * IMAGE_SIZE * IMAGE_SIZE]);
		addVertLine(&images[(i + 2) * IMAGE_SIZE * IMAGE_SIZE]);
	}

	/*for (int i = 0; i < num; i++) {
		addPureBlack(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
	}*/

	/*for (int i = 0; i < num; i++) {
		addVertLine(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
	}*/
	return images;
}

int main() {
	int count = 348;
	uint32_t* images = generateImages(count);
	uint32_t numFeatures;
	struct timeb start, end;
	int diff;
	ftime(&start);
	feature* features = findFeatures(images, count, &numFeatures);
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	printf("Computation took %d ms\n", diff);

	for (uint32_t i = 0; i < min(10, numFeatures); i++) {
		printFeature(features[i]);
		printf("\n");
	}

	printf("Found %u features, displaying first %d\n", numFeatures, min(10, numFeatures));

	free(features);
	free(images);
}