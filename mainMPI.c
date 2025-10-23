/*
 * K-Means Clustering - Versão Híbrida MPI/OpenMP
 * 
 * Esta implementação combina MPI para distribuição de dados entre processos
 * e OpenMP para paralelização dentro de cada processo.
 * 
 * TEMPOS DE EXECUÇÃO (dataset covertype: 581012 instâncias, 54 características, 7 clusters):
 *
 * Configuração 1: 1 processo MPI com 4 threads OpenMP
 * Comando: export OMP_NUM_THREADS=4; mpirun -np 1 ./mainMPI ...
 * Resultado:
 *   Precision of k-means clustering = 0.42261
 *   Time for k-means clustering = 3.820257 s
 *   MPI processes = 1, OpenMP threads = 4
 *   real: 8m21.149s  user: 8m21.969s  sys: 0m0.512s
 *
 * Configuração 2: 2 processos MPI com 2 threads OpenMP cada
 * Comando: export OMP_NUM_THREADS=2; mpirun -np 2 ./mainMPI ...
 * Resultado:
 *   Precision of k-means clustering = 0.42516
 *   Time for k-means clustering = 9.600044 s
 *   MPI processes = 2, OpenMP threads = 2
 *   real: 16m57.350s  user: 16m43.485s  sys: 0m17.954s
 *
 * Configuração 3: 4 processos MPI sem threads OpenMP
 * Comando: export OMP_NUM_THREADS=1; mpirun -np 4 ./mainMPI ...
 * Resultado:
 *   [Adicionar resultados aqui após execução]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double p);
void fscanf_splitting(const char *fn, int *y, const int n);
double get_precision(int *x, int *y, const int n);

double get_distance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);
int get_cluster(const double* const x, const double* const c, const int m, int k);
int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k);
char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k);
int* kmeans(const double* const X, const int n, const int m, const int k);

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
    FILE *fl = fopen(fn, "w");
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
    FILE *fl = fopen(fn, "w");
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
 * Atribui cada ponto ao cluster mais próximo (partição inicial)
 * Paralelização híbrida: OpenMP dentro do processo, MPI entre processos
 */
int* det_start_partition(const double* const x, const double* const c,
                         int* const nums, int n, const int m, const int k) {

    int *y = (int*)malloc(n * sizeof(int));
    int *nums_local = (int*)calloc(k, sizeof(int));

    /* OpenMP: paraleliza atribuição de clusters nos dados locais */
    #pragma omp parallel 
    {
        int *nums_thread = (int*)calloc(k, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; ++i) { 
            const int l = get_cluster(x + i * m, c, m, k);
            y[i] = l;
            nums_thread[l]++;
        }

        #pragma omp critical
        {
            for (int j = 0; j < k; ++j) nums_local[j] += nums_thread[j];
        }
        free(nums_thread);
    }

    /* MPI: agrega contadores de todos os processos para obter total global */
    MPI_Allreduce(nums_local, nums, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    free(nums_local);

    return y;
}

/*
 * Recalcula centros dos clusters e verifica se houve mudança na partição
 * Retorna 1 se algum ponto mudou de cluster, 0 se convergiu
 */
char check_partition(const double* const x, double* const c,
                     int* const y, int* const nums,
                     const int n, const int m, const int k)
{
    double *c_local = (double*)calloc(k * m, sizeof(double));
    int i, j;

    /* Soma coordenadas dos pontos de cada cluster (dados locais) */
    for (i = 0; i < n; ++i) {
        double* const c_yi = c_local + y[i] * m;
        const double* const x_i = x + i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j]; 
        }
    }

    /* MPI: agrega somas de todos os processos */
    MPI_Allreduce(c_local, c, k * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(c_local);

    /* Calcula novos centros como média das somas globais */
    for (i = 0; i < k; ++i) {
        const double f = nums[i]; 
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f; 
        }
    }

    int *nums_local = (int*)calloc(k, sizeof(int));
    int changed_any = 0; 

    /* OpenMP: reatribui pontos aos clusters com novos centros */
    #pragma omp parallel
    {
        int changed_local = 0;
        int *nums_thread = (int*)calloc(k, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            const int f = get_cluster(x + (size_t)i * m, c, m, k);
            if (y[i] != f) changed_local = 1; 
            y[i] = f;
            nums_thread[f]++;
        }

        #pragma omp critical
        {
            for (int j = 0; j < k; ++j) nums_local[j] += nums_thread[j];
            if (changed_local) changed_any = 1;
        }
        free(nums_thread);
    }

    /* MPI: agrega contadores globais */
    MPI_Allreduce(nums_local, nums, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    free(nums_local);
    
    /* MPI: verifica se algum processo detectou mudança (critério de parada) */
    int changed_global = 0;
    MPI_Allreduce(&changed_any, &changed_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    return (char)changed_global; 
}

/*
 * Executa algoritmo K-Means nos dados locais de cada processo
 * Sincroniza centros e decisões de convergência entre processos
 */
int* kmeans(const double* const X, const int n, const int m, const int k) {
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    double *c = (double*)malloc(k * m * sizeof(double));

    /* Processo 0 gera centros iniciais aleatórios */
    det_cores(x, c, n, m, k);
    
    /* MPI: sincroniza centros iniciais entre todos os processos */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(c, k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int *nums = (int*)malloc(k * sizeof(int));
    int *y = det_start_partition(x, c, nums, n, m, k);
    if (!y) return NULL;

    /* Loop principal: continua até convergência global */
    while (check_partition(x, c, y, nums, n, m, k));
    
    free(x);
    free(c);
    free(nums);
    return y;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 6) {
        if (rank == 0) puts("Not enough parameters...");
        MPI_Finalize();
        exit(1);
    }
    
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    if (n < 1 || m < 1 || k < 1 || k > n) {
        if (rank == 0) puts("Values of input parameters are incorrect...");
        MPI_Finalize();
        exit(1);
    }
    
    /* Distribui dados balanceadamente entre processos */
    int local_n = n / size;
    if (rank < (n % size)) local_n++;
    
    /* Processo 0: carrega dados completos do arquivo */
    double *x_all = NULL;
    if (rank == 0) {
        x_all = (double*)malloc(n * m * sizeof(double));
        if (!x_all) {
            puts("Memory allocation error...");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf_data(argv[1], x_all, n * m);
    }
    
    double *x_local = (double*)malloc(local_n * m * sizeof(double));
    if (!x_local) {
        if (rank == 0) puts("Memory allocation error...");
        MPI_Finalize();
        exit(1);
    }
    
    /* Prepara vetores para distribuição não uniforme (Scatterv) */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int count = n / size;
            if (i < (n % size)) count++;
            sendcounts[i] = count * m;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    /* MPI: distribui fatias dos dados para cada processo */
    MPI_Scatterv(x_all, sendcounts, displs, MPI_DOUBLE,
                 x_local, local_n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double start_time = MPI_Wtime();
    
    /* Cada processo executa k-means em sua fatia de dados */
    int *y_local = kmeans(x_local, local_n, m, k);
    
    if (!y_local) {
        if (rank == 0) puts("Memory allocation error...");
        free(x_local);
        if (x_all) free(x_all);
        MPI_Finalize();
        exit(1);
    }
    
    /* Prepara vetores para coleta de resultados (Gatherv) */
    int *y_all = NULL;
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0) {
        y_all = (int*)malloc(n * sizeof(int));
        recvcounts = (int*)malloc(size * sizeof(int));
        rdispls = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int count = n / size;
            if (i < (n % size)) count++;
            recvcounts[i] = count;
            rdispls[i] = offset;
            offset += count;
        }
    }
    
    /* MPI: coleta resultados de todos os processos no processo 0 */
    MPI_Gatherv(y_local, local_n, MPI_INT,
                y_all, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    
    /* Processo 0: calcula precisão e grava resultados */
    if (rank == 0) {
        if (argc > 6) {
            int *ideal = (int*)malloc(n * sizeof(int));
            if (!ideal) fprintf_result(argv[5], y_all, n);
            else {
                fscanf_splitting(argv[6], ideal, n);
                const double p = get_precision(ideal, y_all, n);
                printf("Precision of k-means clustering = %.5lf;\n", p);
                fprintf_full_result(argv[5], y_all, n, p);
                free(ideal);
            }
        } else fprintf_result(argv[5], y_all, n);
        
        printf("Time for k-means clustering = %lf s.;\n", end_time - start_time);
        printf("MPI processes = %d, OpenMP threads = %d\n", size, omp_get_max_threads());
        printf("The work of the program is completed...\n");
        
        free(y_all);
        free(recvcounts);
        free(rdispls);
        free(x_all);
        free(sendcounts);
        free(displs);
    }
    
    free(x_local);
    free(y_local);
    
    MPI_Finalize();
    return 0;
}

