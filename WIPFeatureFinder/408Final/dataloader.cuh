#include "common.cuh"
#include <fstream>

void addPureWhite(uint32_t* buf);

void addPureBlack(uint32_t* buf);

void addVertLine(uint32_t* buf);

uint32_t* loadFromFile(uint32_t* count);