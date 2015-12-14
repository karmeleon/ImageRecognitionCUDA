#include <cstdlib>
#include <vector>
#include <iostream>
#include "common.cuh"
using namespace std;

#define BLOCK_SIZE 	 1024
#define NUM_EXAMPLES 60000

__constant__ unsigned char l[NUM_EXAMPLES];

//////////////////////////////////////////////////

//Strong classifier class
class _strong_classifier
{
public:
	struct _weak_classifier
	{
		feature _feature;
		uint16_t _threshold;
		float _error;
	};

	_weak_classifier* weak_classifier;
	//std::vector<float> alpha;
	float* alpha;

	void init(uint8_t stages)
	{
		/*weak_classifier = (_weak_classifier*)malloc(stages*sizeof(_weak_classifier));
		weak_classifier->_feature.x1 = 0;
		weak_classifier->_feature.x2 = 0;
		weak_classifier->_feature.y1 = 0;
		weak_classifier->_feature.y2 = 0;
		weak_classifier->_feature.type = 0;
		weak_classifier->_feature.mag = 0;
		weak_classifier->_threshold = 0;
		weak_classifier->_error = 0;*/
		//alpha.reserve(stages);
	}

	void append_classifier(int t)
	{
		_weak_classifier tmp;
		weak_classifier[t] = tmp;
	}

	void append_alpha(int t)
	{
		//alpha.push_back(tmp);
		//alpha[t] = t;
	}

	/*void classify(int i)
	{
	std::cout << "Hello World! " << i << std::endl;
	}*/
};

//////////////////////////////////////////////////

//CUDA weak_classifier
__global__ void weak_classifier(_strong_classifier* strong_classifier, int* global_min, float* w, feature* f, uint16_t range, uint32_t num_examples, int t)
{
//	uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	//Theta variables
//	uint16_t theta = range;
//	uint16_t min_theta = 0;
//
//	//Haar properties
//	int32_t perfect_haar;
//	uint32_t dist;
//
//	//Error values
//	uint32_t min_misclassified = INT_MAX;
//	uint32_t misclassified = 0;
//
//	//Weighted error values
//	float weighted_error = 0.0;
//	float min_weighted_error = FLT_MAX;
//
//	//Global min errors and values
//	float global_min_weighted_error = FLT_MAX;
//	uint32_t global_min_misclassified = 0;
//	uint32_t global_min_theta = 0;
//	uint32_t global_min_idx = 0;
//
//	int32_t perfect_haar = 127 * (f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);
//
//	int32_t mag = f[i].mag;
//	const int32_t diffFromPerfect = abs(mag - perfect_haar);
//
//	//Loop through the theta values
//	while (theta > 0)
//	{
//		//Reset misclassified and weighted_error
//		misclassified = 0;
//		weighted_error = 0.0;
//
//		//Loop through each image
//		for (int j = 0; j < num_examples; j++)
//		{
//			//Check decision for errors
//			if (dist < theta && l[j] != 0)
//			{
//				//Classified as + but label is -
//				misclassified++;
//				weighted_error += w[j];
//			}
//			else
//			{
//				if (l[j] == 0)
//				{
//					//Classified as - but label is +
//					misclassified++;
//					weighted_error += w[j];
//				}
//			}
//		}
//
//
//
//		//Keep track of the minimum weighted error for the best theta value
//		if (misclassified*weighted_error < min_weighted_error)
//		{
//			min_misclassified = misclassified;
//			min_weighted_error = misclassified*weighted_error;
//			min_theta = theta;
//		}
//
//		theta--;
//	}
//
//	//Find the minimum weighted error over all the features for the best feature
//	if (min_misclassified*min_weighted_error < global_min_weighted_error)
//	{
//		global_min_misclassified = min_misclassified;
//		global_min_weighted_error = min_misclassified*min_weighted_error;
//		global_min_theta = min_theta;
//		global_min_idx = i;
//	}
//
////	//Find the minimum weighted error over all the features for the best feature
////	if (min_misclassified*min_weighted_error < global_min_weighted_error)
////	{
////		global_min_misclassified = min_misclassified;
////		global_min_weighted_error = min_misclassified*min_weighted_error;
////		global_min_theta = min_theta;
////		global_min_idx = i;
////	}
////}
////
////strong_classifier->weak_classifier[global_min_idx]._threshold = global_min_theta;
////strong_classifier->weak_classifier[global_min_idx]._feature = f[global_min_idx];
////strong_classifier->weak_classifier[global_min_idx]._error = global_min_weighted_error;
//
//	__syncthreads();
//
//	//Check for the minimum weighted error
//	if (weighted_int == atomicMin(global_min, weighted_int))
//	{
//		printf("thread: %i\n", i);
//		strong_classifier->weak_classifier[t]._threshold = min_theta;
//		strong_classifier->weak_classifier[t]._feature = f[i];
//		strong_classifier->weak_classifier[t]._error = min_weighted;
//	}

	uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;

	uint16_t theta = 1;
	uint16_t min_theta = 0;
	uint32_t minimum = INT_MAX;
	uint32_t error = 0;
	float weighted = 0;
	float min_weighted = 0;
	int32_t perfect_haar = 127 * (f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);

	int32_t mag = f[i].mag;
	const int32_t diffFromPerfect = abs(mag - perfect_haar);
	if (i < num_examples) {
		while (theta > 0)
		{
			error = 0;
			weighted = 0;
			for (int j = 0; j < num_examples; j++)
			{
				uint8_t label = l[j];
				float weight = w[j];
				if (diffFromPerfect < theta && label != 0)
				{
					weighted += weight;
					error++;
				}
				else if (l[j] == 0)
				{
					weighted += weight;
					error++;
				}
			}

			printf("thread %d error = %u, minimum = %u\n", i, error, minimum);

			//Track the minimum error
			if (error < minimum)
			{
				minimum = error;
				min_weighted = weighted;
				min_theta = theta;
			}

			theta--;
		}
	}
	

	//Find the globally minimum weighted error from all trained features and set weak classifier to that
	//Convert global_min and min_weighted to int using __float_as_int() device function
	int weighted_int = __float_as_int(min_weighted);
	__syncthreads();
	printf("syncthreads passed\n");

	//Check for the minimum weighted error
	if (i < num_examples && weighted_int == atomicMin(global_min, weighted_int))
	{
		printf("thread %d new min\n", i);
		strong_classifier->weak_classifier[t]._threshold = min_theta;
		strong_classifier->weak_classifier[t]._feature = f[i];
		strong_classifier->weak_classifier[t]._error = min_weighted;
	}
}

//////////////////////////////////////////////////

void serial_weak_classifier(_strong_classifier* strong_classifier, feature* f, unsigned char* l, float* w, uint32_t num_features, uint32_t num_examples, uint16_t range)
{
	//Theta variables
	uint16_t theta = range;
	uint16_t min_theta = 0;

	//Haar properties
	int32_t perfect_haar;
	uint32_t dist;

	//Error values
	uint32_t min_misclassified = INT_MAX;
	uint32_t misclassified = 0;

	//Weighted error values
	float weighted_error = 0.0;
	float min_weighted_error = FLT_MAX;

	//Global min errors and values
	float global_min_weighted_error = FLT_MAX;
	uint32_t global_min_misclassified = 0;
	uint32_t global_min_theta = 0;
	uint32_t global_min_idx = 0;

	//Loop through each feature
	for (int i = 0; i < num_features; i++)
	{
		//Compute perfect haar and current feature's distance from that
		perfect_haar = 127 * (f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);
		dist = abs(f[i].mag - perfect_haar);

		theta = range;

		//Loop through the theta values
		while (theta > 0)
		{
			//Reset misclassified and weighted_error
			misclassified = 0;
			weighted_error = 0.0;

			//Loop through each image
			for (int j = 0; j < num_examples; j++)
			{
				//Check decision for errors
				if (dist < theta && l[j] != 0)
				{
					//Classified as + but label is -
					misclassified++;
					weighted_error += w[j];
				}
				else
				{
					if (l[j] == 0)
					{
						//Classified as - but label is +
						misclassified++;
						weighted_error += w[j];
					}
				}
			}



			//Keep track of the minimum weighted error for the best theta value
			if (misclassified*weighted_error < min_weighted_error)
			{
				min_misclassified = misclassified;
				min_weighted_error = misclassified*weighted_error;
				min_theta = theta;
			}

			theta--;
		}

		printf("minMisclassifiedWeightedError=%f\n", min_misclassified*min_weighted_error);

		//Find the minimum weighted error over all the features for the best feature
		if (min_misclassified*min_weighted_error < global_min_weighted_error)
		{
			global_min_misclassified = min_misclassified;
			global_min_weighted_error = min_misclassified*min_weighted_error;
			global_min_theta = min_theta;
			global_min_idx = i;
		}
	}

	strong_classifier->weak_classifier[global_min_idx]._threshold = global_min_theta;
	strong_classifier->weak_classifier[global_min_idx]._feature = f[global_min_idx];
	strong_classifier->weak_classifier[global_min_idx]._error = global_min_weighted_error;
}

/*void serial_weak_classifier(_strong_classifier* strong_classifier, feature* f, unsigned char* l, float* w, uint32_t num_features, uint32_t num_examples, uint16_t range)
{
uint16_t theta = range;
uint16_t min_theta = 0;
uint32_t minimum = INT_MAX;
uint32_t error = 0;
float weighted = 0;
float min_weighted = 0;
float global_min = INT_MAX;
uint32_t global_min_error = 0;
float global_min_weighted = 0;
uint16_t global_min_theta = 0;
uint32_t global_min_idx = 0;
int32_t perfect_haar;

//Loop through all features
for (int i = 0; i < num_features; i++)
{
perfect_haar = 127 * (f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);
uint32_t diff = abs(f[i].mag - perfect_haar);
min_weighted = 0;
theta = range;

minimum = INT_MAX;

//Loop through theta values
while (theta > 0)
{
error = 0;
weighted = 0;

//Loop through each image
for (int j = 0; j < num_examples; j++)
{
if (diff < theta && l[j] != 0)
{
weighted += w[j];
//printf("weighted is now %f (i=%d, j=%d)\n", weighted, i, j);
error++;
}
else
{
if (l[j] == 0)
{
weighted += w[j];
//printf("weighted is now %f (i=%d, j=%d)\n", weighted, i, j);
error++;
}
}
}

//printf("error is %u at i=%d, theta=%u\n", error, i, theta);

//Track the minimum error
if (error < minimum)
{
minimum = error;
min_weighted = weighted;
min_theta = theta;
}

theta--;
}

printf("min_weighted for i=%d is %f\n", i, min_weighted);

if (min_weighted < global_min)
{
global_min = min_weighted;
global_min_error = minimum;
global_min_weighted = min_weighted;
global_min_theta = min_theta;
global_min_idx = i;
}
}

strong_classifier->weak_classifier[global_min_idx]._threshold = global_min_theta;
strong_classifier->weak_classifier[global_min_idx]._feature = f[global_min_idx];
strong_classifier->weak_classifier[global_min_idx]._error = global_min_weighted;
}*/

//////////////////////////////////////////////////

_strong_classifier* adaboost(feature* features, unsigned char* labels, uint32_t num_features, uint32_t num_examples, uint8_t stages)
{
	//Initialize error rate
	float error = 0;

	//Initialize example weights
	float* w = (float*)malloc(num_examples*sizeof(float));
	float tot_w = 0;

	//Classifier threshold range
	//uint16_t theta = 5000;
	uint16_t theta = 1;

	//Strong classifier
	_strong_classifier* strong_classifier = (_strong_classifier*)malloc(sizeof(_strong_classifier));
	strong_classifier->weak_classifier = (_strong_classifier::_weak_classifier*)malloc(sizeof(_strong_classifier::_weak_classifier));
	strong_classifier->alpha = (float*)malloc(stages * sizeof(float));
	//strong_classifier->init(stages);


	_strong_classifier* mirrored_strong_classifier = (_strong_classifier*)malloc(sizeof(_strong_classifier));
	//mirrored_strong_classifier->init(stages);

	_strong_classifier::_weak_classifier* cudaPtr;
	wbCheck(cudaMalloc((void**)&cudaPtr, stages * sizeof(_strong_classifier::_weak_classifier)));
	float* cudaPtr2;
	wbCheck(cudaMalloc((void**)&cudaPtr2, stages * sizeof(float)));

	mirrored_strong_classifier->weak_classifier = cudaPtr;
	mirrored_strong_classifier->alpha = cudaPtr2;

	//CUDA allocations
	dim3 dimGrid(ceil((num_features) / BLOCK_SIZE), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	//dim3 dimGrid(1, 1, 1);
	//dim3 dimBlock(32, 1, 1);

	feature* f;
	//unsigned char* l;
	_strong_classifier* strong_classifier_dev;
	float* w_dev;
	int* global_min = (int*)malloc(sizeof(int));
	*global_min = INT_MAX;
	int* global_min_dev;

	//Allocate space on GPU
	cudaMemcpyToSymbol(l, labels, NUM_EXAMPLES*sizeof(unsigned char));
	cudaError_t cuda_error[5];
	cuda_error[0] = cudaMalloc((void**)&strong_classifier_dev, sizeof(_strong_classifier));
	cuda_error[1] = cudaMalloc((void**)&f, num_features*sizeof(feature));
	//cuda_error[2] = cudaMalloc((void**)&l, num_examples*sizeof(unsigned char));
	cuda_error[2] = cudaMalloc((void**)&w_dev, num_examples*sizeof(float));
	cuda_error[3] = cudaMalloc((void**)&global_min_dev, sizeof(int));
	for (int i = 0; i < 4; i++)
	{
		if (cuda_error[i] != 0)
		{
			std::cout << "cudaMalloc error " << i << ": " << cudaGetErrorString(cuda_error[i]) << std::endl;
		}
	}

	//Initialize weak_classifier from device
	/*weak_classifier_dev->_feature.x1 = 0;
	weak_classifier_dev->_feature.x2 = 0;
	weak_classifier_dev->_feature.y1 = 0;
	weak_classifier->_feature.y2 = 0;
	weak_classifier->_feature.type = 0;
	weak_classifier->_feature.mag = 0;
	weak_classifier->_threshold = 0;
	weak_classifier->_error = 0;*/

	//Copy data to GPU
	//cuda_error[0] = cudaMemcpy(strong_classifier_dev.weak_classifier, strong_classifier.weak_classifier, numFeatures*sizeof(_weak_classifier));
	cuda_error[0] = cudaMemcpy(f, features, num_features*sizeof(feature), cudaMemcpyHostToDevice);
	//cuda_error[1] = cudaMemcpy(l, labels, num_examples*sizeof(unsigned char), cudaMemcpyHostToDevice);
	for (int i = 0; i < 1; i++)
	{
		if (cuda_error[i] != 0)
		{
			std::cout << "cudaMemcpy host2dev error " << i << ": " << cudaGetErrorString(cuda_error[i]) << std::endl;
		}
	}

	//Set the global min value on the GPU
	/*cuda_error[0] = cudaMemset(global_min, INT_MAX, sizeof(int));
	if(cuda_error[0] != 0)
	{
	std::cout << "cudaMemset error: " << cudaGetErrorString(cuda_error[0]) << std::endl;
	}*/

	//Initialize weight distribution
	for (int i = 0; i < num_examples; i++)
	{
		w[i] = 1 / num_examples;
	}

	//Training stage. Loop T stages or until error rate less than target error rate
	for (int t = 0; t < stages; t++)
	{
		//Append a new weak classifier to the strong classifier
		//strong_classifier->append_classifier(t);

		//Normalize weights to produce a distribution
		for (int i = 0; i < num_examples; i++)
		{
			tot_w += w[i];
		}
		for (int i = 0; i < num_examples; i++)
		{
			w[i] /= tot_w;
		}

		//Train weak classifier h_j for each feature j
		cuda_error[0] = cudaMemcpy(strong_classifier_dev, mirrored_strong_classifier, sizeof(_strong_classifier), cudaMemcpyHostToDevice);
		cuda_error[1] = cudaMemcpy(w_dev, w, num_examples*sizeof(float), cudaMemcpyHostToDevice);
		cuda_error[2] = cudaMemcpy(global_min_dev, global_min, sizeof(int), cudaMemcpyHostToDevice);
		for (int i = 0; i < 3; i++)
		{
			if (cuda_error[i] != 0)
			{
				std::cout << "cudaMemcpy host2dev error " << i << ": " << cudaGetErrorString(cuda_error[i]) << std::endl;
			}
		}

		weak_classifier << <dimGrid, dimBlock >> >(strong_classifier_dev, global_min_dev, w_dev, f, theta, num_examples, t);
		cudaDeviceSynchronize();
		wbCheck(cudaPeekAtLastError());
		cuda_error[0] = cudaMemcpy(strong_classifier, strong_classifier_dev, sizeof(_strong_classifier), cudaMemcpyDeviceToHost);
		cuda_error[1] = cudaMemcpy(strong_classifier->weak_classifier, mirrored_strong_classifier->weak_classifier, stages * sizeof(_strong_classifier::_weak_classifier), cudaMemcpyDeviceToHost);
		cuda_error[2] = cudaMemcpy(strong_classifier->alpha, mirrored_strong_classifier->alpha, stages * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 3; i++)
		{
			if (cuda_error[i] != 0)
			{
				std::cout << "cudaMemcpy dev2host error " << i << ": " << cudaGetErrorString(cuda_error[i]) << std::endl;
			}
		}

		//Update error and weights
		for (int i = 0; i < num_examples; i++)
		{
			if (abs(features[i].mag - 127 * (features[i].x2 - features[i].x1)*(features[i].y2 - features[i].y1)) < strong_classifier->weak_classifier[t]._threshold)
			{
				if (labels[i] == 1)
				{
					w[i] *= 0.5*(error / (1 - error));
				}
				else
				{
					w[i] *= 0.5*(1 / error);
				}
			}
			else
			{
				if (labels[i] == 1)
				{
					w[i] *= 0.5*(1 / error);
				}
				else
				{
					w[i] *= 0.5*(error / (1 - error));
				}
			}
		}

		strong_classifier->alpha[t] = log((1 - error) / error);

		//Display info about training stage
		cout << "*****************************************************" << endl;
		cout << "* Stage " << t << " done with error rate: " << strong_classifier->weak_classifier[t]._error << endl;
		cout << "*****************************************************" << endl;
	}

	cudaFree(strong_classifier_dev);
	cudaFree(f);
	cudaFree(l);
	cudaFree(w_dev);

	return strong_classifier;
}

//////////////////////////////////////////////////

_strong_classifier* serial_adaboost(feature* features, unsigned char* labels, uint32_t num_features, uint32_t num_examples, uint8_t stages)
{
	//Initialize error rate
	float error = 0;

	//Initialize example weights
	float* w = (float*)malloc(num_examples*sizeof(float));
	float tot_w = 0;

	//Classifier threshold range
	uint16_t theta = 5000;

	//Strong classifier
	_strong_classifier* strong_classifier = (_strong_classifier*)malloc(sizeof(_strong_classifier));
	strong_classifier->weak_classifier = (_strong_classifier::_weak_classifier*)malloc(sizeof(_strong_classifier::_weak_classifier));
	strong_classifier->alpha = (float*)malloc(stages * sizeof(float));

	//Initialize weight distribution
	for (int i = 0; i < num_examples; i++)
	{
		w[i] = 1.0 / num_examples;
	}

	//Training stage. Loop T stages or until error rate less than target error rate
	for (int t = 0; t < stages; t++)
	{
		//Append a new weak classifier to the strong classifier
		//strong_classifier->append_classifier(t);

		//Normalize weights to produce a distribution
		for (int i = 0; i < num_examples; i++)
		{
			tot_w += w[i];
		}
		for (int i = 0; i < num_examples; i++)
		{
			w[i] /= tot_w;
		}

		//Train weak classifier h_j for each feature j
		serial_weak_classifier(strong_classifier, features, labels, w, num_features, num_examples, theta);

		//Update the error and weights
		error = strong_classifier->weak_classifier[t]._error;

		for (int i = 0; i < num_examples; i++)
		{
			if (abs(features[i].mag - 127 * (features[i].x2 - features[i].x1)*(features[i].y2 - features[i].y1)) < strong_classifier->weak_classifier[t]._threshold)
			{
				if (labels[i] == 1)
				{
					w[i] *= 0.5*(error / (1 - error));
				}
				else
				{
					w[i] *= 0.5*(1 / error);
				}
			}
			else
			{
				if (labels[i] == 1)
				{
					w[i] *= 0.5*(1 / error);
				}
				else
				{
					w[i] *= 0.5*(error / (1 - error));
				}
			}
		}

		strong_classifier->alpha[t] = log((1 - error) / error);

		//Display info about training stage
		cout << "*****************************************************" << endl;
		cout << "* Stage " << t << " done with error rate: " << strong_classifier->weak_classifier[t]._error << endl;
		cout << "*****************************************************" << endl;
	}

	return strong_classifier;
}

//////////////////////////////////////////////////

int train_cascade(feature* pos_features, feature* neg_features, unsigned char* label, uint32_t num_pos_features, uint32_t num_neg_features, uint32_t num_pos_examples, uint32_t num_neg_examples)
{
	feature* features = (feature*)malloc((num_pos_features + num_neg_features)*sizeof(feature));
	/*for(uint32_t i = 0; i < num_pos_features; i++)
	{
	features[i] = pos_features[i];
	}
	for(uint32_t i = 0; i < num_neg_features; i++)
	{
	features[num_pos_features + i] = neg_features[i];
	}*/

	memcpy(features, pos_features, num_pos_examples*sizeof(feature));
	memcpy(&features[num_pos_examples], neg_features, num_neg_features*sizeof(feature));

	uint32_t num_features = num_pos_features + num_neg_features;
	uint32_t num_examples = num_pos_examples + num_neg_examples;

	//Train cascade classifier
	uint8_t stages = 50;
	_strong_classifier* classifier = (_strong_classifier*)malloc(sizeof(_strong_classifier));
	classifier->init(stages);

	cout << "-------------------" << endl;
	cout << "Training classifier" << endl;
	cout << "-------------------" << endl;

	//classifier = serial_adaboost(features, label, num_features, num_examples, stages);
	classifier = adaboost(features, label, num_features, num_examples, stages);

	free(features);
	return 0;
}