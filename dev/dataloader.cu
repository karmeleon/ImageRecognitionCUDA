#include "dataloader.cuh"

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

int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

uint32_t* loadAllFromFile(uint32_t* count) {
	std::ifstream file;
	file.open("training_set_images", std::ifstream::in | std::ifstream::binary);
	if (file.is_open()) {
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
		for (int i = 0; i < number_of_images * IMAGE_SIZE * IMAGE_SIZE; ++i) {
			uint32_t temp = 0;
			file.read((char*)&temp, sizeof(char));
			images[i] = temp;
		}
		
		file.close();

		*count = number_of_images;
		return images;
	}
	else {
		printf("Couldn't open file.\n");
		return NULL;
	}
}

void loadLabelFromFile(unsigned char* labels, uint8_t label, uint32_t** positive, uint32_t* positiveCount, uint32_t** negative, uint32_t* negativeCount) {
	// read labels
	labels = read_mnist_labels();

	// read training set header
	std::ifstream file;
	file.open("training_set_images", std::ifstream::in | std::ifstream::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		//uint32_t* images = (uint32_t*)malloc(number_of_images * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
		uint32_t* inSet = (uint32_t*)malloc(number_of_images * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
		uint32_t* outOfSet = (uint32_t*)malloc(number_of_images * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

		uint32_t inSetCount = 0, outOfSetCount = 0;

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; i++) {
			uint32_t* writeSet;
			if (labels[i] == label)
				writeSet = &(inSet[(inSetCount++) * IMAGE_SIZE * IMAGE_SIZE]);
			else
				writeSet = &(outOfSet[(outOfSetCount++) * IMAGE_SIZE * IMAGE_SIZE]);

			for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
				uint32_t temp = 0;
				file.read((char*)&temp, sizeof(char));
				writeSet[j] = temp;
			}
		}

		file.close();

		inSet = (uint32_t*)realloc(inSet, inSetCount * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
		outOfSet = (uint32_t*)realloc(outOfSet, outOfSetCount * IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

		*positive = inSet;
		*negative = outOfSet;

		*positiveCount = inSetCount;
		*negativeCount = outOfSetCount;
	}
	else {
		printf("Couldn't open file.\n");
	}
}

unsigned char* read_mnist_labels() {
	typedef unsigned char uchar;

	std::ifstream file("training_set_labels");

	if (file.is_open()) {
		int magic_number = 0;
		int number_of_labels = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		if (magic_number != 2049) printf("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = ReverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
			/*if (_dataset[i] == 0)
				_dataset[i] = 1;
			else _dataset[i] = 0;*/
		}
		return _dataset;
	}
	else {
		printf("Couldn't open file.\n");
		return NULL;
	}
}