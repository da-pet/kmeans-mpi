#!/bin/bash

# ---------------------------------
# CONFIGURAÇÕES DO DATASET
# ---------------------------------

DATASET="./datasets/covertype/data.txt"
OUT1="./datasets/covertype/new_result.txt"
OUT2="./datasets/covertype/res.txt"

N=581012
M=54
K=7

CSV="resultados_mpi_hibrido.csv"

# ---------------------------------
# COMPILAÇÃO
# ---------------------------------
echo "Compilando com MPI + OpenMP..."
mpicc -O3 -fopenmp -o mainMPI mainMPI.c -lm
echo "Compilado."

# ---------------------------------
# CABEÇALHO DO CSV
# ---------------------------------
echo "MPI_processes,OMP_threads,Media_s" > "$CSV"

# ---------------------------------
# LISTA DE CONFIGURAÇÕES PARA TESTE
# ---------------------------------
# Formato:  "np threads"
CONFIGS=(
    "1 4"
    "2 2"
    "4 1"
)

# ---------------------------------
# LOOP PRINCIPAL
# ---------------------------------
for cfg in "${CONFIGS[@]}"; do
    NP=$(echo $cfg | awk '{print $1}')
    TH=$(echo $cfg | awk '{print $2}')

    echo "========================================="
    echo " Testando: $NP processos MPI, $TH threads OMP"
    echo "========================================="

    export OMP_NUM_THREADS=$TH

    SOMA=0.0

    for RUN in 1 2 3 4 5; do
        echo -n "Execução $RUN... "

        TEMPO=$(
            { /usr/bin/time -p mpirun -np $NP ./mainMPI \
                $DATASET $N $M $K $OUT1 $OUT2 >/dev/null; } 2>&1 \
            | grep real | awk '{print $2}'
        )

        echo "$TEMPO s"

        SOMA=$(echo "$SOMA + $TEMPO" | bc -l)
    done

    MEDIA=$(echo "$SOMA / 5" | bc -l)
    echo "MÉDIA FINAL = $MEDIA s"

    echo "$NP,$TH,$MEDIA" >> "$CSV"
done

echo "-----------------------------------------"
echo "TESTES CONCLUÍDOS."
echo "Resultados salvos em: $CSV"
echo "-----------------------------------------"