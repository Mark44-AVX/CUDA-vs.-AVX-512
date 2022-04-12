/**
 * Finds the coefficients of a regression line y = mx + b for a set of points (X, Y).
 * X and Y are input vectors with numElements elements.
 * Formulas:
 * m = (N * sum(XY) - sum(X)sum(Y))/(N * sum(X^2) - (sum(X))^2)
 * b = (sum(Y) - sum(X))/N
 */

#include <cstdio>

#include <chrono>   // for high_resolution_clock::time_point
#include <iostream>
using namespace std::chrono;

extern "C" double sumVector(double[], int);
extern "C" void vecProduct(double output[], double arr1[], double arr2[], int count);

const int NUMELEMENTS = 512 * 512;
// Allocate the input vectors X, Y, XY, XX
__declspec(align(64)) double X[NUMELEMENTS];
__declspec(align(64)) double Y[NUMELEMENTS];
__declspec(align(64)) double XX[NUMELEMENTS];
__declspec(align(64)) double XY[NUMELEMENTS];

int main(void)
{
	double m;  // Slope of regression line
	double b;  // y-intercept of regr. line
	
	// Print the vector length to be used, and compute its size
	//int numElements = 16;// 262144;   // 512 * 512
	size_t size = NUMELEMENTS * sizeof(double);
	printf("[Linear regression of %d points]\n", NUMELEMENTS);

	double sumX = 0;
	double sumY = 0;
	double sumXX = 0;
	double sumXY = 0;
	
	// Verify that allocations succeeded
	if (X == NULL || Y == NULL || XY == NULL || XX == NULL)
	{
		fprintf(stderr, "Failed to allocate input vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the input vectors
	// All points lie on the line y = 1*x + 0.5
	const double slope = 1.0;
	const double y_int = 0.5;

	for (int i = 0; i < NUMELEMENTS; i += 4)
	{
		X[i] = slope * i;
		Y[i] = X[i] + y_int;
		X[i+1] = slope * (i+1);
		Y[i+1] = X[i+1] + y_int;
		X[i+2] = slope * (i+2);
		Y[i+2] = X[i+2] + y_int;
		X[i+3] = slope * (i+3);
		Y[i+3] = X[i+3] + y_int;		
	}

	high_resolution_clock::time_point t0 = high_resolution_clock::now();	

	// Calculate vector product X*Y.	
	vecProduct(XY, X, Y, NUMELEMENTS);

	// Calculate vector product X*X.
	vecProduct(XX, X, X, NUMELEMENTS);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	auto et_vec_products = t1 - t0;
	auto et_vec_products_usec = et_vec_products / 1000;

	// Calculate the four sums: sum(X), sum(Y), sum(XY), and sum(X^2)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	sumX = sumVector(X, NUMELEMENTS);
	sumY = sumVector(Y, NUMELEMENTS);
	sumXY = sumVector(XY, NUMELEMENTS);
	sumXX = sumVector(XX, NUMELEMENTS);
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	auto et_vec_sums = t3 - t2;
	auto et_vec_sums_usec = et_vec_sums / 1000;

	printf("Sum of x: %f\n", sumX);
	printf("Sum of y: %f\n", sumY);
	printf("Sum of xy: %f\n", sumXY);
	printf("Sum of x^2: %f\n",sumXX);
	printf("Processed %d points\n", NUMELEMENTS);

	m = (double)(NUMELEMENTS * sumXY - sumX * sumY) / (NUMELEMENTS * sumXX - sumX * sumX);
	b = (double)(sumY - sumX) / NUMELEMENTS;

	//Display times.
	long long et_prods = et_vec_products.count();
	long long et_sums = et_vec_sums.count();
	printf("Time to calculate XY and XX: %lld nsec. = %lld usec.\n", et_prods, et_prods/1000);
	printf("Time to calculate vector sums: %lld nsec. = %lld usec.\n", et_sums, et_sums / 1000);

	// Verify that the results are correct.
	printf("Predicted value of m: %lf\n", slope);
	printf("Computed value of m: %0.10lf\n", m);
	printf("Predicted value of b: %lf\n", y_int);
	printf("Computed value of b: %0.10lf\n", b);
}








