# ==== Projeto ====
TARGET       = k-means-in-C

SRCDIR       = src
OBJDIR       = obj
BINDIR       = bin

# Compilador (padrão: gcc). Pode trocar para clang se quiser.
CC           = gcc

# ==== Flags ====
# -Wall        : avisos
# -std=c90     : padrão C
# -O2          : otimização
# -I./src      : inclui cabeçalhos
# -fopenmp     : habilita OpenMP (compilação)
CFLAGS       = -Wall -std=c99 -O2 -I./$(SRCDIR) -fopenmp

# Link: -fopenmp precisa entrar também aqui para linkar a runtime
LDFLAGS      = -fopenmp -lm

# ==== Fontes/Objetos ====
C_SOURCES    = $(wildcard $(SRCDIR)/*.c)
C_OBJECTS    = $(C_SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# ==== Regras ====
.PHONY: all clean remove windows build_win

all: remove $(BINDIR)/$(TARGET) clean

$(BINDIR)/$(TARGET): $(C_OBJECTS) $(BINDIR)
	$(CC) $(C_OBJECTS) $(LDFLAGS) -o $@

windows: remove build_win clean

build_win: $(C_OBJECTS) $(BINDIR)
	$(CC) $(C_OBJECTS) $(LDFLAGS) -o $(BINDIR)/$(TARGET).exe

$(C_OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

$(BINDIR):
	mkdir -p $@

clean:
	rm -rf $(C_OBJECTS) $(OBJDIR)

remove:
	rm -rf $(C_OBJECTS) $(OBJDIR) $(BINDIR)

