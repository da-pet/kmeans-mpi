/*
 * K-Means Clustering - Versão Paralela com OpenMP para GPU
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

/* ----------------- help.h ----------------- */
void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double p);
void fscanf_splitting(const char *fn, int *y, const int n);
double get_precision(int *x, int *y, const int n);

/* ----------------- kmeans.h----------------- */
#pragma omp declare target
double get_distance(const double *x1, const double *x2, int m);
int get_cluster(const double* const x, const double* const c, const int m, int k);
#pragma omp end declare target
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);
int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k);
char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k);
int* kmeans(const double* const X, const int n, const int m, const int k);
int* kmeans_ws(const double* const x, const int n, const int m, const int k);

/* ----------------- help.c----------------- */
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
   
    
    
    int max_x = 0, max_y = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_x) max_x = x[i];
        if (y[i] > max_y) max_y = y[i];
    }
    int rows = max_x + 1; // Classes reais
    int cols = max_y + 1; // Clusters encontrados

    
    unsigned long **counts = (unsigned long**)calloc(rows, sizeof(unsigned long*));
    for(int i=0; i<rows; i++) counts[i] = (unsigned long*)calloc(cols, sizeof(unsigned long));

    // Preencher a matriz (Passada única O(N))
    for (int i = 0; i < n; ++i) {
        counts[x[i]][y[i]]++;
    }

    
    unsigned long yy = 0;      // Pares que estão juntos em X e juntos em Y
    unsigned long yy_plus_ny = 0; // Pares que estão juntos em Y (independente de X)

    
    for (int j = 0; j < cols; ++j) {
        unsigned long total_in_cluster = 0;
        
        
        for (int i = 0; i < rows; ++i) {
            unsigned long count = counts[i][j];
            total_in_cluster += count;
            
            
            if (count > 1) {
                yy += (count * (count - 1)) / 2;
            }
        }
        
        
        if (total_in_cluster > 1) {
            yy_plus_ny += (total_in_cluster * (total_in_cluster - 1)) / 2;
        }
    }

    // Limpeza
    for(int i=0; i<rows; i++) free(counts[i]);
    free(counts);

    // Retorna precisão: (Verdadeiros Positivos) / (Todos Preditos como Positivos)
    return yy_plus_ny == 0 ? 0.0 : (double)yy / (double)yy_plus_ny;
}

/*double get_precision(int *x, int *y, const int n) {
    unsigned long yy = 0ul, ny = 0ul;
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            if (x[i] == x[j] && y[i] == y[j]) ++yy;
            if (x[i] != x[j] && y[i] == y[j]) ++ny;
        }
    }
    return yy == 0ul && ny == 0ul ? 0.0 : (double)yy / (double)(yy + ny);
}*/

/* ----------------- kmeans.c----------------- */

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
    //srand((unsigned int)time(NULL));
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


int* det_start_partition(const double* const x, const double* const c,
                         int* const nums, int n, const int m, const int k) {
    /* --- INICIO DO TESTE DE GPU --- */
    int running_on_cpu = 1;
    #pragma omp target map(from:running_on_cpu)
    {
        running_on_cpu = omp_is_initial_device();
    }
    if (running_on_cpu) {
        printf("\n[ALERTA CRITICO] O codigo esta rodando na CPU (Fallback)!\n");
    } else {
        printf("\n[SUCESSO] O codigo esta rodando DENTRO DA GPU!\n\n");
    }
    /* --- FIM DO TESTE DE GPU --- */

    int *y = (int*)malloc(n * sizeof(int));
    if (!y) return NULL;

    memset(nums, 0, k * sizeof(int));


    #pragma omp target data map(from: y[0:n]) \
                            map(tofrom: nums[0:k])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < n; ++i) {
            int l = get_cluster(x + (size_t)i * m, c, m, k);
            y[i] = l;
            #pragma omp atomic
            nums[l]++;
        }
    }

    return y;
}


char check_partition(const double* const x, double* const c,
                     int* const y, int* const nums,
                     const int n, const int m, const int k)
{
    /*Recalcula os centróides na CPU */
    memset(c, 0, k * m * sizeof(double));
    int i, j;

    for (i = 0; i < n; ++i) {
        double* const c_yi = c + y[i] * m;
        const double* const x_i = x + (size_t)i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j];
        }
    }

    for (i = 0; i < k; ++i) {
        const double f = nums[i] ? (double)nums[i] : 1.0;
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f;
        }
    }

    /*Prepara para reatribuição na GPU */
    memset(nums, 0, k * sizeof(int));
    int changed_arr[1] = {0}; 

    
    #pragma omp target update to(c[0:k*m])

    
    #pragma omp target data map(tofrom: y[0:n]) \
                            map(tofrom: nums[0:k]) \
                            map(tofrom: changed_arr[0:1])
    {
        #pragma omp target teams distribute parallel for


        for (int ii = 0; ii < n; ++ii) {
            int new_cluster = get_cluster(x + (size_t)ii * m, c, m, k);
            
            if (y[ii] != new_cluster) {
                changed_arr[0] = 1;
            }
            
            y[ii] = new_cluster;
            
            #pragma omp atomic
            nums[new_cluster]++;
        }
    }

    return (char)changed_arr[0];
}

/*
 * Recalcula centros dos clusters e verifica se houve mudança na partição
 * Retorna 1 se algum ponto mudou de cluster, 0 se convergiu
 * Paralelização: apenas no loop de reatribuição de clusters
 */
/*char check_partition(const double* const x, double* const c,
                     int* const y, int* const nums,
                     const int n, const int m, const int k)
{
    // recompute cluster sums on host (c must be zeroed first)
    memset(c, 0, k * m * sizeof(double));
    int i, j;

    for (i = 0; i < n; ++i) {
        double* const c_yi = c + y[i] * m;
        const double* const x_i = x + (size_t)i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j];
        }
    }

    // compute means (avoid divide by zero) 
    for (i = 0; i < k; ++i) {
        const double f = nums[i] ? (double)nums[i] : 1.0;
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f;
        }
    }

    // prepare for re-assignment on device 
    memset(nums, 0, k * sizeof(int));
    int changed_any = 0;

    // offload re-assignment of n points to GPU 
    #pragma omp target data map(to: x[0:n*m], c[0:k*m]) \
                            map(tofrom: y[0:n]) \
                            map(tofrom: nums[0:k])
    {
        // use reduction to accumulate changed_any (bitwise-or)
        #pragma omp target teams distribute parallel for reduction(|:changed_any)
        for (int ii = 0; ii < n; ++ii) {
            int new_cluster = get_cluster(x + (size_t)ii * m, c, m, k);
            if (y[ii] != new_cluster) {
                changed_any = 1;
            }
            y[ii] = new_cluster;
            #pragma omp atomic
            nums[new_cluster]++;
        }
    }

    return (char)changed_any;
}*/

int* kmeans(const double* const X, const int n, const int m, const int k) {

    printf("[DEBUG] Entrando no kmeans... n=%d m=%d k=%d\n", n, m, k);

    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));

    printf("[DEBUG] Chamando autoscaling...\n");
    autoscaling(x, n, m);
    printf("[DEBUG] autoscaling concluído.\n");

    double *c = (double*)malloc(k * m * sizeof(double));

    printf("[DEBUG] Selecionando centros iniciais...\n");
    det_cores(x, c, n, m, k);
    printf("[DEBUG] Centros iniciais definidos.\n");

    int *nums = (int*)malloc(k * sizeof(int));

    /*Envia 'x' e 'c' para a GPU e mantém lá */
    #pragma omp target enter data map(to: x[0:n*m], c[0:k*m])

    printf("[DEBUG] Calculando partição inicial (GPU)...\n");
    
    int *y = det_start_partition(x, c, nums, n, m, k);
    
    if (!y) {
        // Limpeza de emergência se falhar
        #pragma omp target exit data map(delete: x[0:n*m], c[0:k*m])
        return NULL;
    }

    int iter = 0;
    while (1) {
        
        int changed = check_partition(x, c, y, nums, n, m, k);

        if (!changed) break;
        iter++;
    }

    printf("[DEBUG] Convergiu após %d iterações!\n", iter);

    /*Libera a memória da GPU */
    #pragma omp target exit data map(delete: x[0:n*m], c[0:k*m])

    free(x);
    free(c);
    free(nums);
    return y;
}



/*int* kmeans(const double* const X, const int n, const int m, const int k) {
    printf("[DEBUG] Entrando no kmeans... n=%d m=%d k=%d\n", n, m, k);
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    printf("[DEBUG] Chamando autoscaling...\n");
    autoscaling(x, n, m);
    printf("[DEBUG] autoscaling concluído.\n");
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
}*/

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

/* ----------------- main.c----------------- */
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
    unsigned int seed = 123;          
    srand(seed);
    printf("[INFO] Usando seed fixa para inicialização: %u\n", seed);

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

