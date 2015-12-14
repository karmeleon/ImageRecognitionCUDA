#include "common.cuh"

int train_cascade(feature* pos_features, feature* neg_features, unsigned char* label, uint32_t num_pos_features, uint32_t num_neg_features, uint32_t num_pos_examples, uint32_t num_neg_examples);