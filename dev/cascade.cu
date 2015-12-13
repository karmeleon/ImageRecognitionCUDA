#include <iostream>
#include <climits>
#include <cstdlib>
#include <vector>
#include "common.cuh"

#define BLOCK_SIZE 1024

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
	std::vector<float> alpha;

	void init(uint8_t stages)
	{
		weak_classifier = (_weak_classifier*)malloc(stages*sizeof(_weak_classifier));
		alpha.reserve(stages);
	}

	void append_classifier(int t)
	{
		_weak_classifier tmp;
		weak_classifier[t] = tmp;
	}

	void append_alpha(float tmp)
	{
		alpha.push_back(tmp);
	}

	/*void classify(int i)
	{
		std::cout << "Hello World! " << i << std::endl;
	}*/
};

//////////////////////////////////////////////////

//CUDA weak_classifier
__global__ void weak_classifier(_strong_classifier* strong_classifier, int* global_min, float* w, feature* f, unsigned char* l, uint16_t range, uint32_t num_examples, int t)
{
	uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;

	uint16_t theta = range;
	uint16_t min_theta = 0;
	uint32_t minimum = INT_MAX;
	uint32_t error = 0;
	float weighted = 0;
	float min_weighted = 0;
	int32_t perfect_haar = 127*(f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);

	while(theta > 0)
	{
		error = 0;
		weighted = 0;
		for(int j = 0; j < num_examples; j++)
		{
			if(abs(f[i].mag - perfect_haar) < theta)
			{
				if(l[i] == 0)
				{
					weighted += w[i];
					error++;

				}
			}
			else
			{
				if(l[i] == 1)
				{
					weighted += w[i];
					error++;
				}
			}
		}

		//Track the minimum error
		if(error < minimum)
		{
			minimum = error;
			min_weighted = weighted;
			min_theta = theta;
		}

		theta--;
	}

	//Find the globally minimum weighted error from all trained features and set weak classifier to that
	//Convert global_min and min_weighted to int using __float_as_int() device function
	int weighted_int = __float_as_int(min_weighted);
	__syncthreads();

	//Check for the minimum weighted error
	if(weighted_int == atomicMin(global_min, weighted_int))
	{
		strong_classifier->weak_classifier[t]._threshold = min_theta;
		strong_classifier->weak_classifier[t]._feature = f[i];
		strong_classifier->weak_classifier[t]._error = min_weighted;
	}
}

//////////////////////////////////////////////////

_strong_classifier* adaboost(feature* features, unsigned char* labels, uint32_t num_features, uint32_t num_examples, uint8_t stages)
{
	//Initialize error rate
	float error = 0;

	//Initialize example weights
	float* w = (float*)malloc(num_examples*sizeof(float));
	float tot_w = 0;

	//Classifier threshold range
	uint16_t theta = 5000;

	//Strong classifier
	_strong_classifier* strong_classifier = (_strong_classifier*)malloc(sizeof(_strong_classifier));;
	strong_classifier->init(stages);

	//CUDA allocations
	dim3 dimGrid(ceil((num_features)/BLOCK_SIZE), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	feature* f;
	unsigned char* l;
	_strong_classifier* strong_classifier_dev = (_strong_classifier*)malloc(sizeof(_strong_classifier));;
	strong_classifier_dev->init(stages);
	float* w_dev;
	int* global_min = (int*)malloc(sizeof(int));
	*global_min = INT_MAX;
	int* global_min_dev;

	//Allocate space on GPU
	cudaError_t cuda_error[5];
	cuda_error[0] = cudaMalloc((void**) &strong_classifier_dev, sizeof(_strong_classifier));
	cuda_error[1] = cudaMalloc((void**) &f, num_features*sizeof(feature));
	cuda_error[2] = cudaMalloc((void**) &l, num_examples*sizeof(unsigned char));
	cuda_error[3] = cudaMalloc((void**) &w_dev, num_examples*sizeof(float));
	cuda_error[4] = cudaMalloc((void**) &global_min_dev, sizeof(int));
	for(int i = 0; i < 5; i++)
	{
		if(cuda_error[i] != 0)
		{
			std::cout << "cudaMalloc error: " << cudaGetErrorString(cuda_error[i]) << std::endl;
		}
	}

	//Copy data to GPU
	//cuda_error[0] = cudaMemcpy(strong_classifier_dev.weak_classifier, strong_classifier.weak_classifier, numFeatures*sizeof(_weak_classifier));
	cuda_error[0] = cudaMemcpy(f, features, num_features*sizeof(feature), cudaMemcpyHostToDevice);
	cuda_error[1] = cudaMemcpy(l, labels, num_examples*sizeof(unsigned char), cudaMemcpyHostToDevice);
	for(int i = 0; i < 2; i++)
	{
		if(cuda_error[i] != 0)
		{
			std::cout << "cudaMemcpy host2dev error: " << cudaGetErrorString(cuda_error[i]) << std::endl;
		}
	}

	//Set the global min value on the GPU
	/*cuda_error[0] = cudaMemset(global_min, INT_MAX, sizeof(int));
	if(cuda_error[0] != 0)
	{
		std::cout << "cudaMemset error: " << cudaGetErrorString(cuda_error[0]) << std::endl;
	}*/

	//Initialize weight distribution
	for(int i = 0; i < num_examples; i++)
	{
		w[i] = 1/num_examples;
	}

	//Training stage. Loop T stages or until error rate less than target error rate
	for(int t = 0; t < stages; t++)
	{
		//Append a new weak classifier to the strong classifier
		strong_classifier->append_classifier(t);

		//Normalize weights to produce a distribution
		for(int i = 0; i < num_examples; i++)
		{
			tot_w += w[i];
		}
		for(int i = 0; i < num_examples; i++)
		{
			w[i] /= tot_w;
		}

		//Train weak classifier h_j for each feature j
		cuda_error[0] = cudaMemcpy(strong_classifier_dev, strong_classifier, sizeof(_strong_classifier), cudaMemcpyHostToDevice);
		cuda_error[1] = cudaMemcpy(w_dev, w, num_examples*sizeof(float), cudaMemcpyHostToDevice);
		cuda_error[2] = cudaMemcpy(global_min_dev, global_min, sizeof(int));
		for(int i = 0; i < 3; i++)
		{
			if(cuda_error[i] != 0)
			{
				std::cout << "cudaMemcpy host2dev error: " << cudaGetErrorString(cuda_error[i]) << std::endl;
			}
		}

		weak_classifier<<<dimGrid, dimBlock>>>(strong_classifier_dev, global_min_dev, w_dev, f, l, theta, num_examples, t);

		cudaDeviceSynchronize();
		cuda_error[0] = cudaMemcpy(strong_classifier, strong_classifier_dev, sizeof(_strong_classifier), cudaMemcpyDeviceToHost);
		if(cuda_error[0] != 0)
		{
			std::cout << "cudaMemcpy dev2host error: " << cudaGetErrorString(cuda_error[0]) << std::endl;
		}

		//Update error and weights
		for(int i = 0; i < num_examples; i++)
		{
			if(abs(features[i].mag - 127*(features[i].x2 - features[i].x1)*(features[i].y2 - features[i].y1)) < strong_classifier->weak_classifier[t]._threshold)
			{
				if(labels[i] == 1)
				{
					w[i] *= 0.5*(error/(1 - error));
				}
				else
				{
					w[i] *= 0.5*(1/error);
				}
			}
			else
			{
				if(labels[i] == 1)
				{
					w[i] *= 0.5*(1/error);
				}
				else
				{
					w[i] *= 0.5*(error/(1 - error));
				}
			}
		}

		strong_classifier->append_alpha(log((1 - error)/error));

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

	classifier = adaboost(features, label, num_features, num_examples, stages);

	free(features);
	return 0;
}