/**
 * This is a very basic sample that finds the coefficients of a regression line y = mx + b.
 * x and y are input vectors with numElements elements.
 */

#include <stdio.h>
#include <chrono>  // for high_resolution_clock struct and now()
using namespace std::chrono;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel device code
 * vectorMult computes the element by element product of vectors A and B,
 *   storing the individual products in vector C.
 * Parameters
 *   C           - address of output vector of doubles.
 *   A           - address of first input vector of doubles.
 *   B           - address of second vector array of doubles.
 *   numElements - number of elements (doubles) in each vector.
 * No return value.
 * All 3 vectors are assumed to have the same number of elements.
 */
__global__ void
vectorMult(double *C, const double * A, const double * B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] * B[i];
	}
}


/**
* vectorSum computes the sum of the elements of a vector C.
* Parameters
*   C           - address of the vector of doubles.
*   numElements - number of elements in vector C.
* Returns the calculated sum of the elements of the passed vector.
*/
double vectorSum(const double *C, int numElements)
{
	double tempSum = 0.0;

	for (int i = 0; i < numElements; i += 8)
	{
		tempSum += C[i] + C[i + 1] + C[i + 2] + C[i + 3];
		tempSum += C[i + 4] + C[i + 5] + C[i + 6] + C[i + 7];
	}
	return tempSum;
}

/**
 * Host main routine
 */
int
main(void)
{
	// Local variables
	double h_sumX;  // Host-side sum of the x coordinates.
	double h_sumY;  // Host-side sum of the y coordinates.
	double h_sumXY; // Host-side sum of the xy pairs.
	double h_sumXX; // Host-side sum of the x^2 values.
	double m;       // Slope of the regression line.
	double b;       // y-intercept of the regression line.

	// Error code to check return values for CUDA calls.
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size.
	int numElements = 512 * 512;	// = 262144, or 2^18
	size_t size = numElements * sizeof(double);
	printf("[Linear regression of %d points]\n", numElements);

	// Allocate the host input vectors X, Y, XY, XX.
	// h_XY will eventually contain the term-by-term product of vectors X and Y.
	// h_XX will eventually contain the term-by-term product of vector X with itself.
	double *h_X = (double *)malloc(size);
	double *h_Y = (double *)malloc(size);
	double *h_XY = (double *)malloc(size);
	double *h_XX = (double *)malloc(size);


	// Verify that allocations succeeded.
	if (h_X == NULL || h_Y == NULL || h_XY == NULL || h_XX == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Declare and allocate the device input vector X.
	double *d_X = NULL;
	err = cudaMalloc((void **)&d_X, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Declare and allocate the device input vector Y.
	double *d_Y = NULL;
	err = cudaMalloc((void **)&d_Y, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Declare and allocate the device output vector XY.
	double *d_XY = NULL;
	err = cudaMalloc((void **)&d_XY, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector XY (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Declare and allocate the device output vector XX
	double *d_XX = NULL;
	err = cudaMalloc((void **)&d_XX, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector XX (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors.
	// All points lie on the line y = 1.0*x + 0.5
	const double slope = 1.0;
	const double y_int = 0.5;

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	
	// Fill the host-side vectors h_X and h_Y with data points.
	for (int i = 0; i < numElements; i += 4)
	{
		h_X[i] = slope * i;
		h_Y[i] = h_X[i] + y_int;
		h_X[i + 1] = slope * i;
		h_Y[i + 1] = h_X[i + 1] + y_int;
		h_X[i + 2] = slope * (i + 2);
		h_Y[i + 2] = h_X[i + 2] + y_int;
		h_X[i + 3] = slope * (i + 3);
		h_Y[i + 3] = h_X[i + 3] + y_int;
	}

	// Define a time point for a start time.
	high_resolution_clock::time_point t0 = high_resolution_clock::now();

	// Copy the host input vector X in host memory to the device input vector in
	// device memory.
	
	err = cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector X from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vector Y in host memory to the device input vector in
	// device memory.
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector Y from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaDeviceSynchronize();

	// Launch the vectorMult CUDA Kernel to calculate the vector XY.
	vectorMult <<<blocksPerGrid, threadsPerBlock >>> (d_XY, d_X, d_Y, numElements);
	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorMult kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}		

	cudaDeviceSynchronize();
	
	// Launch the vectorMult CUDA Kernel to calculate the vector XX.
	vectorMult <<<blocksPerGrid, threadsPerBlock >>> (d_XX, d_X, d_X, numElements);
	cudaDeviceSynchronize();	
	
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorMult kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device vectors XY and XX to the corresponding host vectors.
	err = cudaMemcpy(h_XY, d_XY, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector XY from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(h_XX, d_XX, size, cudaMemcpyDeviceToHost);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	auto et_vec_products = t1 - t0;
	auto et_vec_products_usec = et_vec_products / 1000;
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector XX from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Calculate the sums of the four vectors.
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	h_sumX = vectorSum(h_X, numElements);
	h_sumY = vectorSum(h_Y, numElements);
	h_sumXY = vectorSum(h_XY, numElements);
	h_sumXX = vectorSum(h_XX, numElements);	
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	auto et_vec_sums = t3 - t2;
	auto et_vec_sums_usec = et_vec_sums / 1000;

	printf("Sum of x: %f\n", h_sumX);
	printf("Sum of y: %f\n", h_sumY);
	printf("Sum of xy: %f\n", h_sumXY);
	printf("Sum of x^2: %f\n", h_sumXX);
	printf("Processed %d points\n", numElements);
	m = (numElements * h_sumXY - h_sumX * h_sumY) / (numElements * h_sumXX - h_sumX * h_sumX);
	b = (h_sumY - h_sumX) / numElements;

	// Display times.
	long long et_prods = et_vec_products.count();
	long long et_sums = et_vec_sums.count();

	printf("Time to calculate XY and XX: %lld nsec. = %lld usec.\n", et_prods, et_prods / 1000);
	printf("Time to calculate vector sums: %lld nsec. = %lld usec.\n", et_sums, et_sums / 1000);
	
	// Verify that the results are correct.
	printf("Predicted value of m: %lf\n", slope);
	printf("Computed value of m: %0.10lf\n", m);
	printf("Predicted value of b: %lf\n", y_int);
	printf("Computed value of b: %0.10lf\n", b);

	if (fabs(m - slope) > 1e-7 || fabs(b - y_int) > 1e-4)
	{
		fprintf(stderr, "Result verification failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("Test PASSED\n");

	// Free device global memory for the vectors d_X, d_Y, d_XY, and d_XX.
	err = cudaFree(d_X);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector X (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_Y);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector Y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_XY);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector XY (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_XX);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector XX (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory - vectors h_X, h_Y, h_XY, and h_XX.
	free(h_X);
	free(h_Y);
	free(h_XY);
	free(h_XX);

	// Reset the device and exit.
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits.
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
}
