#include "main.cuh"
using namespace std;
void addPureWhite(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 0;
}

void addPureBlack(uint32_t* buf) {
	for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
		buf[i] = 1;
}

int ReverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/*uint32_t* generateImages() {
	ifstream file("C:\Users\Charles\Downloads\train-images-idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		uint32_t* images = (uint32_t*)malloc(number_of_images * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));


		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i<number_of_images; ++i)
		{
			uint32_t temp = 0;
			file.read((uint32_t*)&temp, sizeof(temp));
			images[i] = temp;
			/*for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{

					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					arr[i][(n_rows*r) + c] = (double)temp;
				}
			}
		}
	}
	
	/*uint32_t* images = (uint32_t*)malloc(num * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	for (int i = 0; i < num; i += 2) {
		addPureBlack(&images[i * IMAGE_SIZE * IMAGE_SIZE]);
		addPureWhite(&images[(i + 1) * IMAGE_SIZE * IMAGE_SIZE]);
	}
	return images;
} */

int main() {
	//int count = 60000;
	//uint32_t* images = generateImages(count);

	printf("point 1\n");
	ifstream file;
	file.open("C:\\Users\\Charles\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte"); //,ios::binary
	if (file.is_open())
	{
		printf("point 2\n");
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		uint32_t* images = (uint32_t*)malloc(number_of_images * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));


		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			uint32_t temp = 0;
			file.read((char*)&temp, sizeof(temp));
			images[i] = temp;

		}


		printf("point 3\n");
		int count = number_of_images;
		//uint32_t* images = generateImages();
		uint32_t numFeatures;
		feature* features = findFeatures(images, count, &numFeatures);
		printf("point 4\n");
		for (uint32_t i = 0; i < min(10, numFeatures); i++) {
			printFeature(features[i]);
			printf("\n");
		}

		printf("Found %u features\n", numFeatures);

		free(features);
		free(images);
	}
}