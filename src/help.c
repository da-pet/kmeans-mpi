#include "help.h"

void fscanf_data(const char *fn, double *x, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i = 0;
	while (i < n && !feof(fl)) {
		if (fscanf(fl, "%lf", x + i) == 0) {}
		++i;
	}
	fclose(fl);
}

void fprintf_result(const char *fn, const int* const y, const int n) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s result file...\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of k-means clustering...\n");
	int i;
	for (i = 0; i < n; ++i) {
		fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
	}
	fputc('\n', fl);
	fclose(fl);
}

void fprintf_full_result(const char *fn, const int* const y, const int n, const double p) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s result file...\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of k-means clustering...\nPrecision of k-means clustering = %.5lf;\n", p);
	int i;
	for (i = 0; i < n; ++i) {
		fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
	}
	fputc('\n', fl);
	fclose(fl);
}

void fscanf_splitting(const char *fn, int *y, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Can't access %s file with ideal splitting for reading...\n", fn);
		exit(1);
	}
	int i = 0;
	while (i < n && !feof(fl)) {
		if (fscanf(fl, "%d", y + i) == 0) {
			printf("Error in reading the perfect partition from %s file\n", fn);
			exit(1);
		}
		++i;
	}
	fclose(fl);
}

double get_precision(int *x, int *y, const int n) {
	unsigned long yy = 0ul, ny = 0ul;
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {
			if (x[i] == x[j] && y[i] == y[j]) ++yy;
			if (x[i] != x[j] && y[i] == y[j]) ++ny;
		}
	}
	return yy == 0ul && ny == 0ul ? 0.0 : (double)yy / (double)(yy + ny);
}
