/*
 * K-Means Clustering - Versão Sequencial
 * 
 * Implementação básica do algoritmo K-Means sem paralelização.
 * Serve como baseline para comparação de desempenho.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <unistd.h> /* getpid if needed */

/* ----------------- help.h (inlined) ----------------- */
void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double p);
void fscanf_splitting(const char *fn, int *y, const int n);
double get_precision(int *x, int *y, const int n);

/* ----------------- kmeans.h (inlined) ----------------- */
double get_distance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);
int get_cluster(const double* const x, const double* const c, const int m, int k);
int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k);
char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k);
int* kmeans(const double* const X, const int n, const int m, const int k);
int* kmeans_ws(const double* const x, const int n, const int m, const int k);
                
/* ----------------- help.c (inlined) ----------------- */
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

/* ----------------- kmeans.c (inlined) ----------------- */

double get_distance(const double *x1, const double *x2, int m) {
    double d, r = 0.0;
    for (; m > 0; --m) {
        d = *x1 - *x2;
        r += d * d;
        ++x1;
        ++x2;
    }
    return r;
}

void autoscaling(double* const x, const int n, const int m) {
    const double* const end = x + n * m;
    int j;
    for (j = 0; j < m; ++j) {
        double sd, Ex = 0.0, Exx = 0.0, *ptr;
        for (ptr = x + j; ptr < end; ptr += m) {
            sd = *ptr;
            Ex += sd;
            Exx += sd * sd;
        }
        Exx /= n;
        Ex /= n;
        sd = fabs(Exx - Ex * Ex); 
        if (sd == 0.0) sd = 1.0; 
        sd = 1.0 / sqrt(sd);      
        for (ptr = x + j; ptr < end; ptr += m) {
            *ptr = (*ptr - Ex) * sd; 
        }
    }
}

char constr(const int *y, const int val, int s) {
    for (; s > 0; --s) {
        if (*y == val) return 1;
        ++y;
    }
    return 0;
}

/* Inicializa centros aleatoriamente selecionando k pontos distintos do dataset */
void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
    int *nums = (int*)malloc(k * sizeof(int));
    srand((unsigned int)time(NULL));
    int i;
    for (i = 0; i < k; ++i) {
        int val = rand() % n;
        while (constr(nums, val, i)) {
            val = rand() % n;
        }
        nums[i] = val;
        memcpy(c + i * m, x + val * m, m * sizeof(double));
    }
    free(nums);
}

/* Atribui cada ponto ao cluster mais próximo (partição inicial) */
int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k) {

    int *y = (int*)malloc(n * sizeof(int)); 
    memset(nums, 0, k * sizeof(int)); 
    
    while (n > 0) {
        --n;
        const int l = get_cluster(x + n * m, c, m, k);
        y[n] = l;
        ++nums[l];
    }
    return y;
}

int get_cluster(const double* const x, const double* const c, const int m, int k) {
    int res = --k;
    double min_d = get_distance(x, c + k * m, m);
    while (k > 0) {
        --k;
        const double cur_d = get_distance(x, c + k * m, m);
        if (cur_d < min_d) {
            min_d = cur_d;
            res = k;
        }
    }
    return res;
}

/* 
 * Recalcula centros dos clusters e verifica se houve mudança na partição
 * Retorna 1 se algum ponto mudou de cluster, 0 se convergiu
 */
char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k) {
    memset(c, 0, k * m * sizeof(double));
    int i, j;

    /* Soma coordenadas dos pontos de cada cluster */
    for (i = 0; i < n; ++i) {
        double* const c_yi = c + y[i] * m;
        const double* const x_i = x + i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j];
        }
    }

    /* Calcula novos centros como média dos pontos */
    for (i = 0; i < k; ++i) {
        const double f = nums[i];
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f;
        }
    }

    memset(nums, 0, k * sizeof(int));
    char flag = 0; 

    /* Reatribui pontos aos clusters com novos centros */
    for (i = 0; i < n; ++i) {
        const int f = get_cluster(x + i * m, c, m, k);
        if (y[i] != f) flag = 1;
        y[i] = f;
        ++nums[f];
    }

    return flag;
}


int* kmeans(const double* const X, const int n, const int m, const int k) {
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    double *c = (double*)malloc(k * m * sizeof(double));

    det_cores(x, c, n, m, k);
    int *nums = (int*)malloc(k * sizeof(int));

    int *y = det_start_partition(x, c, nums, n, m, k);
    if (!y) return NULL;

    while (check_partition(x, c, y, nums, n, m, k));
    free(x);
    free(c);
    free(nums);
    return y;
}

int* kmeans_ws(const double* const x, const int n, const int m, const int k) {
    double *c = (double*)malloc(k * m * sizeof(double));
    det_cores(x, c, n, m, k);
    int *nums = (int*)malloc(k * sizeof(int));
    int *y = det_start_partition(x, c, nums, n, m, k);
    if (!y) return NULL;
    while (check_partition(x, c, y, nums, n, m, k));
    free(c);
    free(nums);
    return y;
}

/* ----------------- main.c (inlined) ----------------- */
int main(int argc, char **argv) {
    
    if (argc < 6) {
        puts("Not enough parameters...");
        exit(1);
    }
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    if (n < 1 || m < 1 || k < 1 || k > n) {
        puts("Values of input parameters are incorrect...");
        exit(1);
    }
    double *x = (double*)malloc(n * m * sizeof(double));
    if (!x) {
        puts("Memory allocation error...");
        exit(1);
    }	
    fscanf_data(argv[1], x, n * m);
    
    clock_t cl = clock();
    int *y = kmeans(x, n, m, k);	
    cl = clock() - cl;
    if (!y) {
        puts("Memory allocation error...");
        free(x);
        exit(1);
    }
    if (argc > 6) {
        int *ideal = (int*)malloc(n * sizeof(int));
        if (!ideal) fprintf_result(argv[5], y, n);
        else {
            fscanf_splitting(argv[6], ideal, n);
            const double p = get_precision(ideal, y, n);
            printf("Precision of k-means clustering = %.5lf;\n", p);
            fprintf_full_result(argv[5], y, n, p);
            free(ideal);
        }
    } else fprintf_result(argv[5], y, n);
    printf("Time for k-means clustering = %lf s.;\nThe work of the program is completed...\n", (double)cl / CLOCKS_PER_SEC);
    free(x);
    free(y);
    return 0;
}