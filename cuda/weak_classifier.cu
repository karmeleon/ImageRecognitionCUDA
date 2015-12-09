#include <iostream>
#include <cmath>
#include <cstdlib>
#include <climits>

#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

#define SIZE 14*14
#define X_SIZE 14

#define NUM_EXAMPLES 3
#define BLOCK_SIZE 3

//Function prototype
//int SAT(int, int, int, int, int);

//Global variable
//Create big ass array for test char
	unsigned char image[3*SIZE] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 191, 127, 255, 255, 191, 127, 20 , 255, 255, 255,
					 		       255, 255, 255, 255, 127, 0  , 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 127, 0  , 40 , 40 , 230, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 175, 127, 127, 127, 235, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 245, 220, 245, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 127, 120, 180, 125, 120, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 245, 255, 255, 255, 100, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 110, 252, 252, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 120, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 128, 0  , 45 , 127, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 250, 20 , 240, 30 , 252, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 250, 15 , 15 , 235, 205, 135, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		 	   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 150, 50 , 255, 255, 191, 127, 30 , 255, 255, 255,
					 		       255, 255, 255, 255, 120, 0  , 255, 235, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 113, 0  , 40 , 40 , 230, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 175, 150, 127, 127, 235, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 127, 0  , 235, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 45 , 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
					 		       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

//Define constant memory for device image
__constant__ unsigned char _image[3*SIZE];
__constant__ int _label[NUM_EXAMPLES];

//Function SAT
__device__ int SAT(int x1, int y1, int x2, int y2, int img)
{
	unsigned int area = 0;
	for(int j = y1; j <= y2; j++)
	{
		for(int i = x1; i <= x2; i++)
		{
			area += _image[img*SIZE + j*X_SIZE + i];
		}
	}
	return area;
}

__global__ void compute(int *error, int x, int y, int w, int h, int theta, int f, int comp)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	//Compute features and classify with them
	f = SAT(x, y, x+w-1, y+h-1, i) - SAT(x+w, y, x+2*w-1, y+h-1, i);
	comp = abs(f - 255*w*h);
	if(comp < theta)
	{
		if(_label[i] == 0)
		{
			//printf("label[%i] == 0\n", i);
			atomicAdd(error, 1);
		}
	}
	else
	{
		if(_label[i] == 1)
		{
			//printf("label[%i] == 1\n", i);
			atomicAdd(error, 1);
		}
	}

	__syncthreads();
	//printf("error: %i\n", error[0]);
}

int main(int argc, char*argv[])
{
	int label[3];
	label[0] = atoi(argv[1]);
	label[1] = atoi(argv[2]);
	label[2] = atoi(argv[3]);

	int hi = atoi(argv[4]);

	int i = 0;

	int theta = 0;
	int min_theta = 0;

	int e[1] = {0};
	int min_e = INT_MAX;

	int f = 0;
	int comp = 0;

	//CUDA stuffs
	dim3 dimGrid(ceil(NUM_EXAMPLES/BLOCK_SIZE), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	unsigned char* dev_in;
	unsigned char* dev_out;
	int* dev_e = (int*)malloc(sizeof(int));

	int size = 3*SIZE*sizeof(unsigned char);

	cudaError_t error[3];
	error[0] = cudaMalloc((void**) &dev_in, size);
	error[1] = cudaMalloc((void**) &dev_out, size);
	error[2] = cudaMalloc((void**) &dev_e, sizeof(int));
	cout << "cudaMalloc dev_in error: " << cudaGetErrorString(error[0]) << endl;
	cout << "cudaMalloc dev_out error: " << cudaGetErrorString(error[1]) << endl;
	cout << "cudaMalloc dev_e error: " << cudaGetErrorString(error[2]) << endl;

	//error[0] = cudaMemcpy(dev_e, e, sizeof(int), cudaMemcpyHostToDevice);
	error[1] = cudaMemcpyToSymbol(_image, image, size);
	error[2] = cudaMemcpyToSymbol(_label, label, 3*sizeof(int));
	//cout << "cudaMemcpy error: " << cudaGetErrorString(error[0]) << endl;
	cout << "cudaMemcpyToSymbol image error: " << cudaGetErrorString(error[1]) << endl;
	cout << "cudaMemcpyToSymbol label error: " << cudaGetErrorString(error[2]) << endl;
	//END CUDA stuffs
		
	for(int x = 0; x < 14; x++)
	{
		for(int y = 0; y < 14; y++)
		{
			for(int h = 1; h <= 15 - y; h++)
			{
				for(int w = 1; w <= (15 - x)/2; w++)
				{
					theta = hi;
					i++;

					while(theta > 0)
					{
						e[0] = 0;
						error[0] = cudaMemcpy(dev_e, e, sizeof(int), cudaMemcpyHostToDevice);
						//cout << "cudaMemcpy error: " << cudaGetErrorString(error[0]) << endl;

						//Call the cuda kernel
						compute<<<dimGrid, dimBlock>>>(dev_e, x, y, w, h, theta, f, comp);
						cudaDeviceSynchronize();
						error[0] = cudaMemcpy(e, dev_e, sizeof(int), cudaMemcpyDeviceToHost);
						//cout << "cudaMemcpy error: " << cudaGetErrorString(error[0]) << endl;

						//Keep track of current best theta value
						if(e[0] <= min_e)
						{
							min_e = e[0];
							min_theta = theta;
						}
							
						//Compute new threshold bounds based on number of misclassifications
						theta--;

						/*//Display classification info
						cout << "\tNumber misclassified:" << endl;
						cout << "\t---------------------" << endl;
						cout << "\te: " << e[0] << endl;
				
						cout << "\tNew theta bounds:" << endl;
						cout << "\t-----------------" << endl;
						cout << "\ttheta: " << theta << endl << endl;*/
					}

					cout << "==========================" << endl;
					cout << "feature(" << x << ", " << y << ", " << w << ", " << h << ")" << endl;
					cout << "Best theta classifier: " << min_theta << endl;
					cout << "Number misclassified:  " << min_e << endl;
					cout << "==========================" << endl << endl;
				}
			}
		}
	}

	cout << "Total number of features: " << i << endl;
	cout << "Memory size for array to hold feature values: " << i*sizeof(int) << " B" << endl << endl;

	return 0;
}