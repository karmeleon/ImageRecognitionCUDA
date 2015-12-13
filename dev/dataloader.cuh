#include "common.cuh"
#include <fstream>

void addPureWhite(uint32_t* buf);

void addPureBlack(uint32_t* buf);

void addVertLine(uint32_t* buf);

uint32_t* loadAllFromFile(uint32_t* count);

void loadLabelFromFile(unsigned char* labels, uint8_t label, uint32_t** positive, uint32_t* positiveCount, uint32_t** negative, uint32_t* negativeCount);

unsigned char* read_mnist_labels();