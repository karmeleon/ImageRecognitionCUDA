#include <iostream>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <vector>

//Include other CUDA files
#include "main.cuh"
#include "features.cuh"
#include "haarfinder.cuh"
#include "sat.cuh"
#include "common.cuh"
using namespace std;

#define NUM_EXAMPLES 	 60000

#define BLOCK_SIZE		 1024

//Define constant memory for reading from features and labels
__constant__ feature f[numFeatures];
__constant__ unsigned char l[NUM_EXAMPLES];

//Strong classifier class
class _strong_classifier
{
private:
	struct _weak_classifier
	{
		uint16_t threshold;
		float e;
	};

public:
	_weak_classifier weak_classifier[numFeatures];
	float* alpha;

	void classify(int i)
	{
		cout << "Hello World! " << i << endl;
	}
};

//CUDA weak_classifier
__global__ weak_classifier(float* w, _weak_classifier* weak_classifier, uint16_t range, uint32_t num_examples)
{
	i = blockDim.x*blockIdx.x + threadIdx.x;

	uint8_t theta = range;
	uint32_t minimum = MAX_INT;
	uint32_t error = 0;
	float e = 0;
	float min_error = 0;
	int32_t perfect_haar = 255*0.5*(f[i].x2 - f[i].x1)*(f[i].y2 - f[i].y1);

	while(theta > 0)
	{
		error = 0;
		e = 0;
		for(int j = 0; j < num_examples; j++)
		{
			if(abs(f[i].mag - perfect_haar) < theta)
			{
				if(l[i] == 0)
				{
					e += w[i];
					error++;

				}
			}
			else
			{
				if(l[i] == 1)
				{
					e += w[i];
					error++;
				}
			}
		}

		__syncthreads();

		//Track the minimum error
		if(error < minimum)
		{
			minimum = error;
			min_error = e;
			min_theta = theta;
		}

		theta--;
	}

	//Set the weak classifier
	weak_classifier.threshold = min_theta;
	weak_classifier.e = min_error;
}

int main(int argc, char *argv[])
{
	//Initialize error rates and targets
	float error_rate = 0;
	float error_target = atof(argv[1]);
	uint16_t min_error = 0;

	//Initialize example weights
	float w[NUM_EXAMPLES];
	float tot_w;

	//Classifier threshold range
	uint16_t theta = 5000;

	//Strong classifier
	_strong_classifier* strong_classifier_host;

	//Allocate space for CUDA
	dim3 dimGrid(ceil((numFeatures)/BLOCK_SIZE), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	_strong_classifier* strong_classifier_dev;

	//Allocate space for cumulative error and weak classifier structure
	cudaError_t cuda_error[2];
	cuda_error[0] = cudaMalloc((void**) &strong_classifier_dev.weak_classifier, numFeatures*sizeof(_weak_classifier));
	cuda_error[1] = cudaMalloc((void**) &w_dev, NUM_EXAMPLES*sizeof(float));
	for(int i = 0; i < 2; i++)
	{
		if(cuda_error[i] != 0)
		{
			cout << "cudaMalloc error (" << (i == 0)?("error"):("weights") << "): " << cudaGetErrorString(cuda_error[i]) << endl;
		}
	}

	//Copy weak classifier structure to GPU
	cuda_error[0] = cudaMemcpy(strong_classifier_dev.weak_classifier, strong_classifier_host.weak_classifier, numFeatures*sizeof(_weak_classifier));
	if(cuda_error[0] != 0)
	{
		cout << "cudaMemcpy error (weak_classifier): " << cudaGetErrorString(cuda_error[0]) << endl;
	}

	//Copy constant memory to GPU
	cuda_error[0] = cudaMemcpyToSymbol(f, features, numFeatures*sizeof(feature));
	cuda_error[1] = cudaMemcpyToSymbol(l, labels, numLabels*sizeof(unsigned char));
	for(int i = 0; i < 2; i++)
	{
		if(cuda_error[i] != 0)
		{
			cout << "cudaMemcpyToSymbol error (" << (i == 0)?("f"):("l") << "): " << cudaGetErrorString(cuda_error[i]) << endl;
		}
	}

	//Training stage. Loop T stages as input from user
	for(int t = 0; t < T; t++)
	{
		//Initialize weights
		while(error_rate > error_target)
		{
			for(int i = 0; i < NUM_EXAMPLES; i++)
			{
				w[i] = 1/NUM_EXAMPLES;
			}

			//Train weak classifier h_j for each feature j
			cuda_error[0] = cudaMemcpy(strong_classifier_dev.weak_classifier, strong_classifier_host.weak_classifier, numFeatures*sizeof(_weak_classifier), cudaMemcpyHostToDevice);
			cuda_error[1] = cuamMemcpy(w_dev, w, NUM_EXAMPLES*sizeof(float), cudaMemcpyHostToDevice);
			for(int i = 0; i < 2; i++)
			{
				if(cuda_error[i] != 0)
				{
					cout << "cudaMemcpyToSymbol error (" << (i == 0)?("weak_classifier"):("weights") << "): " << cudaGetErrorString(cuda_error[i]) << endl;
				}
			}

			weak_classifier<<<dimGrid, dimBlock>>>(w_dev, strong_classifier_dev.weak_classifier, theta, count);

			cudaDeviceSynchronize();
			cuda_error[0] = cudaMemcpy(strong_classifier_host.weak_classifier, strong_classifier_dev.weak_classifier, numFeatures*sizeof(_weak_classifier), cudaMemcpyDeviceToHost);
			if(cuda_error[0] != 0)
			{
				cout << "cudaMemcpy error (error): " << cudaGetErrorString(error[0]) << endl;
			}

			//Pick feature with minimized weighted error
			for(int i = 0; i < numFeatures; i++)
			{
				if(strong_classifier_host.weak_classifier[i].e < min_error)
				{
					min_error = error;
				}
			}

			//Update error and weights
			for(int i = 0; i < NUM_EXAMPLES; i++)
			{
				if(features[i].mag < strong_classifier_host.weak_classifier.threshold)
				{
					if(labels[i] == 1)
					{
						w[i] *= error/(1-error);
					}
				}
			}

			//Normalize weights
			for(int i = 0; i < NUM_EXAMPLES; i++)
			{
				tot_w += w[i];
			}
			for(int i = 0; i < NUM_EXAMPLES; i++)
			{
				w[i] /= tot_w;
			}

			strong_classifier_host.alpha[i] = log((1 - error)/error);
		}
	}

	return 0;
}