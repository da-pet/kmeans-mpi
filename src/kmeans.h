#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


double get_distance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);
int get_cluster(const double* const x, const double* const c, const int m, int k);
int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k);
char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k);
int* kmeans(const double* const X, const int n, const int m, const int k);
int* kmeans_ws(const double* const x, const int n, const int m, const int k);

#endif