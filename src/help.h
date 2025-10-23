#ifndef HELP_H_
#define HELP_H_

#include <stdlib.h>
#include <stdio.h>

void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double p);
void fscanf_splitting(const char *fn, int *y, const int n);
double get_precision(int *x, int *y, const int n);

#endif
