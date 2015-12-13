#include <detection.cuh>
void detectionAlgorithm(){
	uint32_t count;
	//do I need to malloc here? covered by dataloader?
	uint32_t* images = loadAllFromFile(&count, "test_set_images");
	//just one image to test on 
	uint32_t* image = (uint32_t*)malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	int offset = 0; //probably something to pass in, to select different images

	for(int i =0; i<IMAGE_SIZE; i++)
		image[i]=images[i+offset];

	for (uint32_t xSize = 4; xSize < IMAGE_SIZE; xSize++) {
		for (uint32_t ySize = 4; ySize < IMAGE_SIZE; ySize++) {

			// the total number of regions to process
			uint32_t xRegions = IMAGE_SIZE - xSize;
			uint32_t yRegions = IMAGE_SIZE - ySize;
			uint32_t numRegions = xRegions * yRegions;

			// number of regions each thread processes
			

			for (uint32_t i = 0; i < numRegions; i++) {
					uint8_t x1 = idx % xRegions;
					uint8_t y1 = idx / xRegions;
					uint8_t x2 = x1 + xSize;
					uint8_t y2 = y1 + ySize;
					classify(image, x1, y2);
			}
		}
	}
}