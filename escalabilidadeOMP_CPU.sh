#!/bin/bash

# ------------------------
# CONFIGURAÇÕES DO TESTE
# ------------------------

DATASET="./datasets/covertype/data.txt"
OUT1="./datasets/covertype/new_result.txt"
OUT2="./datasets/covertype/res.txt"

N=581012
D=54
K=7

CSV="resultados_escalabilidade_OMPcpu.csv"

# ------------------------
# COMPILAÇÃO
# ------------------------
echo "Compilando código..."
gcc -O3 mainOP.c -lm -fopenmp -o omp_normal
echo "Compilado."

# ------------------------
# CABEÇALHO DO CSV
# ------------------------
echo "Threads,Media_s" > "$CSV"

# ------------------------
# TESTES COM 1,2,4,8 THREADS
# ------------------------
for T in 1 2 4 8; do
    echo "======================================"
    echo " Testando com $T threads"
    echo "======================================"

    export OMP_NUM_THREADS=$T

    SOMA=0.0

    for RUN in 1 2 3 4 5; do
        echo -n "Execução $RUN... "

        TEMPO=$(
            { /usr/bin/time -p ./omp_normal $DATASET $N $D $K $OUT1 $OUT2 >/dev/null; } 2>&1 \
            | grep real | awk '{print $2}'
        )

        echo "$TEMPO s"

        SOMA=$(echo "$SOMA + $TEMPO" | bc -l)
    done

    MEDIA=$(echo "$SOMA / 5" | bc -l)
    echo "Média: $MEDIA s"

    echo "$T,$MEDIA" >> "$CSV"
done

echo "Testes concluídos."
echo "Resultados salvos em $CSV"