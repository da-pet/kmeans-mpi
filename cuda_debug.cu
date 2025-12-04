/*
 * K-Means Clustering -  CUDA com DEBUG

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Macro para checagem de erros do CUDA
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

/* ----------------- Declarações ----------------- */
void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double p);
void fscanf_splitting(const char *fn, int *y, const int n);
double get_precision(int *x, int *y, const int n);
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);

/* ----------------- Kernels CUDA  ----------------- */


__device__ int get_cluster_gpu(const double *x, const double *c, int m, int k)
{
    int best = 0;
    double best_d = 1e300;
    for (int j = 0; j < k; ++j) {
        double d = 0.0;
        for (int t = 0; t < m; ++t) {
            double diff = x[t] - c[j*m + t];
            d += diff * diff;
        }
        if (d < best_d) {
            best_d = d;
            best = j;
        }
    }
    return best;
}

__global__ void kernel_start_partition(const double *d_x, const double *d_c,
                                        int *d_y, int *d_nums, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double *xi = d_x + (size_t)i * m;
    int cluster = get_cluster_gpu(xi, d_c, m, k);
    d_y[i] = cluster;
    atomicAdd(&d_nums[cluster], 1);
}

__global__ void kernel_check_partition(const double *d_x, const double *d_c,
                                        int *d_y, int *d_nums, int *d_changed, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double *xi = d_x + (size_t)i * m;
    int old = d_y[i];
    int nc = get_cluster_gpu(xi, d_c, m, k);
    
    if (nc != old) {
        atomicExch(d_changed, 1);
    }
    d_y[i] = nc;
    atomicAdd(&d_nums[nc], 1);
}

/* ----------------- Launchers ----------------- */

void launch_start_partition(double *d_x, double *d_c, int *d_y, int *d_nums, 
                            int n, int m, int k) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    kernel_start_partition<<<gs, bs>>>(d_x, d_c, d_y, d_nums, n, m, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

char launch_check_partition(double *d_x, double *d_c, int *d_y, int *d_nums, int *d_changed,
                            double *h_c, int n, int m, int k) {
    
    size_t sc = (size_t)k * m * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_c, h_c, sc, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_nums, 0, (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

    int bs = 256;
    int gs = (n + bs - 1) / bs;

    kernel_check_partition<<<gs, bs>>>(d_x, d_c, d_y, d_nums, d_changed, n, m, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_changed = 0;
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    
    return (char)h_changed;
}

/* ----------------- Função Principal K-Mean ----------------- */

int* kmeans(const double* const X, const int n, const int m, const int k) {
    
    printf("[DEBUG] Entrando no kmeans... n=%d m=%d k=%d\n", n, m, k);

    // Preparações CPU
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    
    printf("[DEBUG] Chamando autoscaling...\n");
    autoscaling(x, n, m); 
    printf("[DEBUG] autoscaling concluido.\n");

    double *h_c = (double*)malloc(k * m * sizeof(double));
    
    printf("[DEBUG] Selecionando centros iniciais...\n");
    det_cores(x, h_c, n, m, k); 
    printf("[DEBUG] Centros iniciais definidos.\n");

    int *h_y = (int*)malloc(n * sizeof(int));
    int *h_nums = (int*)malloc(k * sizeof(int));

    // ALOCAÇÃO GPU
    double *d_x, *d_c;
    int *d_y, *d_nums, *d_changed;

    size_t sx = (size_t)n * m * sizeof(double);
    size_t sc = (size_t)k * m * sizeof(double);
    size_t sy = (size_t)n * sizeof(int);
    size_t sn = (size_t)k * sizeof(int);

    printf("[DEBUG] Alocando memoria na GPU e copiando dados...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_x, sx));
    CUDA_CHECK(cudaMalloc((void**)&d_c, sc));
    CUDA_CHECK(cudaMalloc((void**)&d_y, sy));
    CUDA_CHECK(cudaMalloc((void**)&d_nums, sn));
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    // COPIA O DATASET
    CUDA_CHECK(cudaMemcpy(d_x, x, sx, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, sc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_nums, 0, sn));

    // PARTIÇÃO INICIAL
    printf("[DEBUG] Calculando particao inicial (GPU)...\n");
    launch_start_partition(d_x, d_c, d_y, d_nums, n, m, k);
    
    // Traz resultados iniciais para CPU para printar debug
    CUDA_CHECK(cudaMemcpy(h_y, d_y, sy, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nums, d_nums, sn, cudaMemcpyDeviceToHost));

    // -- PRINTS DE DEBUG INICIAIS --
    printf("[DEBUG] Primeiros valores de y apos particao inicial:\n");
    for (int i = 0; i < 10; i++) printf("y[%d] = %d\n", i, h_y[i]);

    printf("[DEBUG] nums apos particao inicial:\n");
    for (int i = 0; i < k; i++) printf("nums[%d] = %d\n", i, h_nums[i]);
    
    printf("[DEBUG] Particao inicial calculada.\n");

    // LOOP PRINCIPAL
    int iter = 0;
    while (1) {
        printf("[DEBUG] Iteracao %d - chamando check_partition...\n", iter);

        // Recalcula Centróides na CPU
        memset(h_c, 0, k * m * sizeof(double));
        for (int i = 0; i < n; ++i) {
            double* const c_yi = h_c + h_y[i] * m;
            const double* const x_i = x + (size_t)i * m;
            for (int j = 0; j < m; ++j) c_yi[j] += x_i[j];
        }
        for (int i = 0; i < k; ++i) {
            const double f = h_nums[i] ? (double)h_nums[i] : 1.0;
            double* const c_i = h_c + i * m;
            for (int j = 0; j < m; ++j) c_i[j] /= f;
        }

        // Roda Kernel
        char changed = launch_check_partition(d_x, d_c, d_y, d_nums, d_changed, h_c, n, m, k);

        printf("[DEBUG] Iteracao %d finalizada. changed=%d\n", iter, changed);

        if (!changed) break;

        // Traz resultados para CPU (Necessário para o cálculo dos centróides na próxima iteração)
        CUDA_CHECK(cudaMemcpy(h_y, d_y, sy, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nums, d_nums, sn, cudaMemcpyDeviceToHost));

        iter++;
    }
    printf("[DEBUG] Convergiu apos %d iteracoes!\n", iter);

    // Limpeza
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_nums));
    CUDA_CHECK(cudaFree(d_changed));
    
    free(x);
    free(h_c);
    free(h_nums);
    
    return h_y;
}

/* ----------------- Main ----------------- */

int main(int argc, char **argv) {
    if (argc < 6) {
        puts("Uso: ./cuda <data> <n> <m> <k> <result> [ideal]");
        exit(1);
    }
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    
    double *x = (double*)malloc(n * m * sizeof(double));
    if (!x) { puts("Memory allocation error..."); exit(1); }
    
    fscanf_data(argv[1], x, n * m);

    clock_t cl = clock();
    unsigned int seed = 123;          
    srand(seed);
    int *y = kmeans(x, n, m, k);
    cl = clock() - cl;
    
    if (!y) { puts("Erro Fatal."); free(x); exit(1); }

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
    
    printf("Time for k-means clustering = %lf s.;\n", (double)cl / CLOCKS_PER_SEC);
    free(x);
    free(y);
    return 0;
}

/* ----------------- Funções Auxiliares ----------------- */

void fscanf_data(const char *fn, double *x, const int n) {
    FILE *fl = fopen(fn, "r");
    if (!fl) { printf("Error opening %s\n", fn); exit(1); }
    int i = 0;
    while (i < n && !feof(fl)) {
        if (fscanf(fl, "%lf", x + i) == 0) {}
        ++i;
    }
    fclose(fl);
}

void fprintf_result(const char *fn, const int* const y, const int n) {
    FILE *fl = fopen(fn, "w");
    if (!fl) { printf("Error opening result %s\n", fn); exit(1); }
    fprintf(fl, "Result of k-means clustering...\n");
    for (int i = 0; i < n; ++i) fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
    fputc('\n', fl);
    fclose(fl);
}

void fprintf_full_result(const char *fn, const int* const y, const int n, const double p) {
    FILE *fl = fopen(fn, "w");
    if (!fl) { printf("Error opening result %s\n", fn); exit(1); }
    fprintf(fl, "Result of k-means clustering...\nPrecision of k-means clustering = %.5lf;\n", p);
    for (int i = 0; i < n; ++i) fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
    fputc('\n', fl);
    fclose(fl);
}

void fscanf_splitting(const char *fn, int *y, const int n) {
    FILE *fl = fopen(fn, "r");
    if (!fl) { printf("Error opening splitting %s\n", fn); exit(1); }
    int i = 0;
    while (i < n && !feof(fl)) {
        if (fscanf(fl, "%d", y + i) == 0) { exit(1); }
        ++i;
    }
    fclose(fl);
}

// Otimização O(N)
double get_precision(int *x, int *y, const int n) {
    int max_x = 0, max_y = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_x) max_x = x[i];
        if (y[i] > max_y) max_y = y[i];
    }
    int rows = max_x + 1;
    int cols = max_y + 1;

    unsigned long *buf = (unsigned long*)calloc((size_t)rows * cols, sizeof(unsigned long));
    for (int i = 0; i < n; ++i) {
        int r = x[i], c = y[i];
        buf[r * cols + c]++;
    }

    unsigned long yy = 0;
    unsigned long yy_plus_ny = 0;

    for (int j = 0; j < cols; ++j) {
        unsigned long total_in_cluster = 0;
        for (int i = 0; i < rows; ++i) {
            unsigned long count = buf[i * cols + j];
            total_in_cluster += count;
            if (count > 1) yy += (count * (count - 1)) / 2;
        }
        if (total_in_cluster > 1) yy_plus_ny += (total_in_cluster * (total_in_cluster - 1)) / 2;
    }

    free(buf);
    return yy_plus_ny == 0 ? 0.0 : (double)yy / (double)yy_plus_ny;
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
    for (int i = 0; i < s; ++i) {
        if (y[i] == val) return 1;
    }
    return 0;
}

void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
    int *nums = (int*)malloc(k * sizeof(int));
    clock_t cl = clock();
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