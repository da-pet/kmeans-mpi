# K-Means Clustering - Implementações Paralelas

Este projeto contém três implementações do algoritmo K-Means em C: uma versão sequencial, uma versão paralela com OpenMP e uma versão híbrida combinando MPI e OpenMP.

## Sobre o K-Means

O algoritmo K-Means é um método iterativo de agrupamento de dados desenvolvido por Stuart Lloyd nos Bell Labs na década de 1950. O algoritmo divide um conjunto de dados em k grupos (clusters) minimizando a distância entre pontos e seus respectivos centros de cluster.

### Entrada
- X = matriz de dados com n instâncias e m características
- k = número de clusters desejados

### Saída
- Y = vetor de rótulos indicando a qual cluster cada instância pertence

### Passos do Algoritmo
1. Normalização dos dados (autoscaling)
2. Inicialização aleatória dos k centros de cluster
3. Atribuição de cada ponto ao cluster mais próximo
4. Recálculo dos centros como média dos pontos de cada cluster
5. Repetição dos passos 3-4 até convergência

## Implementações

### 1. Versão Sequencial (mainseq.c)

Implementação básica sem paralelização, servindo como baseline para comparação de desempenho.

### 2. Versão OpenMP (main2.c)

Primeira paralelização focando nos loops computacionalmente intensivos:

**Decisões de paralelização:**
- Paralelização do loop de atribuição de clusters em `det_start_partition`
- Paralelização do loop de reatribuição em `check_partition`
- Uso de contadores locais por thread para evitar condições de corrida
- Agregação de resultados em seções críticas

**Estratégia de sincronização:**
- Cada thread mantém contadores locais (`nums_local`)
- Seção crítica apenas para agregação final dos contadores
- Minimização de contenção entre threads

### 3. Versão Híbrida MPI + OpenMP (mainMPI.c)

Extensão da versão OpenMP adicionando distribuição de dados entre processos:

**Decisões de paralelização:**
- Distribuição dos dados entre processos MPI usando `MPI_Scatterv`
- Cada processo executa K-Means em seu subconjunto de dados
- OpenMP paraleliza operações dentro de cada processo MPI
- Sincronização global via operações coletivas MPI

**Pontos de comunicação MPI:**
- `MPI_Bcast`: distribui centros iniciais a todos os processos
- `MPI_Allreduce`: agrega contadores de pontos por cluster
- `MPI_Allreduce`: agrega somas para recálculo de centros
- `MPI_Allreduce`: sincroniza flag de convergência
- `MPI_Gatherv`: coleta resultados finais no processo root

**Motivação da abordagem híbrida:**
- MPI permite escalar além de uma única máquina
- OpenMP aproveita memória compartilhada dentro de cada nó
- Combinação adequada para clusters com múltiplos nós multicore

## Compilação e Execução

### Versão Sequencial

Compilar:
```bash
gcc -O3 -std=c99 -Wall mainseq.c -o seq -lm
```

Executar:
```bash
time ./seq ./datasets/covertype/data.txt 581012 54 7 ./datasets/covertype/new_result.txt ./datasets/covertype/res.txt
```

### Versão OpenMP

Configurar variáveis de ambiente:
```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export OMP_NUM_THREADS=16
```

Compilar:
```bash
gcc -O3 -std=c99 -Wall -fopenmp main2.c -o kmeans -lm
```

Executar:
```bash
time ./kmeans ./datasets/covertype/data.txt 581012 54 7 ./datasets/covertype/new_result.txt ./datasets/covertype/res.txt
```

### Versão MPI + OpenMP

Compilar:
```bash
mpicc -O3 -fopenmp -o mainMPI mainMPI.c -lm
```

Executar exemplos:

1 processo com 4 threads:
```bash
export OMP_NUM_THREADS=4
time mpirun -np 1 ./mainMPI ./datasets/covertype/data.txt 581012 54 7 ./datasets/covertype/new_result.txt ./datasets/covertype/res.txt
```

2 processos com 2 threads cada:
```bash
export OMP_NUM_THREADS=2
time mpirun -np 2 ./mainMPI ./datasets/covertype/data.txt 581012 54 7 ./datasets/covertype/new_result.txt ./datasets/covertype/res.txt
```

4 processos sem OpenMP:
```bash
export OMP_NUM_THREADS=1
time mpirun -np 4 ./mainMPI ./datasets/covertype/data.txt 581012 54 7 ./datasets/covertype/new_result.txt ./datasets/covertype/res.txt
```

## Parâmetros de Linha de Comando

```
./programa <arquivo_dados> <num_instancias> <num_caracteristicas> <num_clusters> <arquivo_saida> <arquivo_referencia>
```

- `arquivo_dados`: arquivo com os dados de entrada
- `num_instancias`: número de pontos no dataset
- `num_caracteristicas`: número de dimensões de cada ponto
- `num_clusters`: número de clusters (k)
- `arquivo_saida`: arquivo para salvar os resultados
- `arquivo_referencia`: arquivo com classificação ideal (opcional, para calcular precisão)

## Datasets

O diretório `datasets/` contém diversos conjuntos de dados para testes, incluindo:
- iris: 150 instâncias, 4 características, 3 clusters
- wine: 178 instâncias, 13 características, 3 clusters
- covertype: 581012 instâncias, 54 características, 7 clusters (dataset grande para testes de desempenho)

## Observações sobre Desempenho

- A versão sequencial serve como baseline
- OpenMP oferece speedup significativo em sistemas multicore
- A versão MPI é indicada para datasets muito grandes ou ambientes distribuídos
- O balanceamento entre processos MPI e threads OpenMP depende da arquitetura do sistema

## Limitações e Trabalhos Futuros

- A inicialização aleatória pode levar a resultados diferentes em cada execução
- A qualidade dos clusters depende fortemente da escolha de k
- Possíveis melhorias: implementar K-Means++ para melhor inicialização, adicionar critérios de parada mais sofisticados
