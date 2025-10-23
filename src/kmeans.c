// ...existing code...
#include "kmeans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

/* Inicialização aleatória dos k clusters */
void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
    int *nums = (int*)malloc(k * sizeof(int));
    srand((unsigned int)time(NULL));
    int i;
    for (i = 0; i < k; ++i) {
        int val = rand() % n;
        while (constr(nums, val, i)) { /* garante índice não repetido entre já escolhidos */
            val = rand() % n;
        }
        nums[i] = val;
        memcpy(c + i * m, x + val * m, m * sizeof(double)); /* copia ponto como centro */
    }
    free(nums);
}

/* Atribui o ponto x (vetor de dimensão m) ao centro mais próximo entre k centros em c. */
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

/****************************************FUNÇÃO ANTIGA SEQUENCIAL 1 ***********************************************************

	int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k) {

		int *y = (int*)malloc(n * sizeof(int)); 
		memset(nums, 0, k * sizeof(int)); 
		
		while (n > 0) {
			--n;
			const int l = get_cluster(x + n * m, c, m, k);
			y[n] = l;
			++nums[l]; // !!! PROBLEMA NA VERSÃO PARALELA - condição de corrida/dependência !!!
		}
		return y;

	}

/****************************************FUNÇÃO PARALELIZADA 1 ***********************************************************/

int* det_start_partition(const double* const x, const double* const c,
                         int* const nums, int n, const int m, const int k) {

    int *y = (int*)malloc(n * sizeof(int)); //armazena o rótulo de cada cluster
    memset(nums, 0, k * sizeof(int)); 

    #pragma omp parallel 
    {
        // contadores locais: 1 vetor por thread, evita conflito
        int *nums_local = (int*)calloc(k, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; ++i) { 
            const int l = get_cluster(x + i * m, c, m, k);
            y[i] = l;
            nums_local[l]++; // local, por thread 
        }

        //Cada thread entra na seção crítica e adiciona seus contadores locais ao vetor compartilhado nums
        #pragma omp critical
        {
            for (int j = 0; j < k; ++j) nums[j] += nums_local[j];
        }
        free(nums_local);
    }

    return y;
}

/****************************************FUNÇÃO ANTIGA SEQUENCIAL 2 ***********************************************************

char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k) {
    memset(c, 0, k * m * sizeof(double));
    int i, j;

    for (i = 0; i < n; ++i) {
        double* const c_yi = c + y[i] * m;
        const double* const x_i = x + i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j]; // soma componentes dos pontos de cada cluster 
        }
    }

    for (i = 0; i < k; ++i) {
        const double f = nums[i]; // número de pontos no cluster i (pode ser 0) 
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f; // média: atenção a f == 0 (divisão por zero possível) 
        }
    }

    memset(nums, 0, k * sizeof(int));
    char flag = 0; 


    for (i = 0; i < n; ++i) {
        const int f = get_cluster(x + i * m, c, m, k);
        if (y[i] != f) flag = 1; // se algum rótulo mudou marcamos flag 
        y[i] = f;
        ++nums[f];
    }

    return flag;
}

//****************************************FUNÇÃO PARALELIZADA 2 ***********************************************************/

char check_partition(const double* const x, double* const c,
                     int* const y, int* const nums,
                     const int n, const int m, const int k)
{
    memset(c, 0, k * m * sizeof(double));
    int i, j;

    for (i = 0; i < n; ++i) {
        double* const c_yi = c + y[i] * m;
        const double* const x_i = x + i * m;
        for (j = 0; j < m; ++j) {
            c_yi[j] += x_i[j]; 
        }
    }

    for (i = 0; i < k; ++i) {
        const double f = nums[i]; 
        double* const c_i = c + i * m;
        for (j = 0; j < m; ++j) {
            c_i[j] /= f; 
        }
    }

    // reatribuição - PARALELA

    memset(nums, 0, k * sizeof(int));
    int changed_any = 0; 

    #pragma omp parallel
    {
        int changed_local = 0; //variavel local para saber se algum ponto mudou sua classificação
        int *nums_local = (int*)calloc(k, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            const int f = get_cluster(x + (size_t)i * m, c, m, k);
            if (y[i] != f) changed_local = 1; 
            y[i] = f;
            nums_local[f]++;
        }

        // Cada thread combina seus changed_local em changed_any e nums_local em nums dentro de uma região crítica 
        #pragma omp critical
        {
            for (int j = 0; j < k; ++j) nums[j] += nums_local[j];
            if (changed_local) changed_any = 1;
        }
        free(nums_local);
    }

    return (char)changed_any; 
}


int* kmeans(const double* const X, const int n, const int m, const int k) {
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    double *c = (double*)malloc(k * m * sizeof(double));

	//incialização dos cores
    det_cores(x, c, n, m, k);
    int *nums = (int*)malloc(k * sizeof(int));

	//primeira configuração 
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
