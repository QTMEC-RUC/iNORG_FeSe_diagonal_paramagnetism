PRES_DIR = .
WORK_DIR = .
console_DIR = ./console
# DIRR = ${PRES_DIR}/randomc
DIRG = ${PRES_DIR}/src/gen
DIRS = ${PRES_DIR}/src
norg  = ${WORK_DIR}/FeSe_diagonal_paramagnetism
VPATH = ${DIRG}:${DIRS}

# Bohr use_edmft_py3: MPICH mpicxx wraps Intel Classic icpc.
# Override compiler and flags together when selecting another toolchain:
# make CC=... OPENMP_FLAGS=... MKL_FLAGS=...
CC       = mpicxx
OPENMP_FLAGS ?= -qopenmp
MKL_FLAGS ?= -qmkl=parallel
CPPFLAGS = -std=c++17 -O3
CFLAGS   =
CXXFLAGS = $(CFLAGS)
COMPILE  = $(CC) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) $(MKL_FLAGS) -c
BUILD_DIR ?= .build/hybrid

# Build parallelism is independent of MPI/OpenMP runtime thread counts.
# nproc must ignore OMP_NUM_THREADS/OMP_THREAD_LIMIT set by use_edmft_py3.
# Default: at most 8 compiler jobs within the visible CPU set.
BUILD_JOBS ?= $(shell env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null | awk '{n=$$1} END {if (n<1) n=1; print (n<8 ? n : 8)}')
# Do not create another job pool in recursive builds or an inherited jobserver.
# GNU Make command-line -j/--jobs overrides this default (including -j1).
ifeq ($(MAKELEVEL),0)
ifeq ($(filter -j% --jobs% --jobserver%,$(MAKEFLAGS)),)
MAKEFLAGS += -j$(BUILD_JOBS)
endif
endif

SRCS := $(wildcard ${DIRG}/*.cpp) $(wildcard ${DIRS}/*.cpp)
OBJS := $(patsubst ./%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

HEADERS := $(wildcard ${DIRG}/*.h) $(wildcard ${DIRS}/*.h)

.PHONY: default manpower compile explain clean depend start srun qsub

default: manpower

srun: ${EXE}
	@echo '______________________________________________ sbatch ______________________________________________'
	@echo ''
	@qsub ${console_DIR}/slurm.bscc.t6.txt
	@echo '____________________________________________________________________________________________________'
	@echo ''

qsub: ${norg}
	@echo '_______________________________________________ qsub _______________________________________________'
	@echo ''
	@qsub ${WORK_DIR}/qsub.DPC++CPU.mpi.txt > ${WORK_DIR}/jmwang.job.txt
	@chmod +x ${WORK_DIR}/job.process.DPC++CPU.txt
	@${WORK_DIR}/job.process.DPC++CPU.txt
	@rm -rf ${WORK_DIR}/jmwang.job.txt
	@echo '____________________________________________________________________________________________________'
	@echo ''

${norg}: $(OBJS)
	@echo '_______________________________________________ link _______________________________________________'
	@echo ''
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) $(MKL_FLAGS) $(LDFLAGS) -o ${norg} $(OBJS) $(LIBS)

manpower: ${norg}
	@echo 'Startup check: bash run_hybrid.sh 2 2 ./FeSe_diagonal_paramagnetism --parallel-check'

start:
	@echo '______________________________________________ start ______________________________________________'
	@echo ''

explain:
	@echo '_____________________________________________ explain _____________________________________________'
	@echo ''
	@echo "the following information represents your prgram"
	@echo "build job default: $(BUILD_JOBS); effective make flags: $(MAKEFLAGS)"
	@echo "final norg name: $(norg)"
	@echo "source files: $(SRCS)"
	@echo "object files: $(OBJS)"
	@echo "dependency files: $(DEPS)"

compile:
	@echo '_____________________________________________ compile _____________________________________________'
	@echo ''

# Keep hybrid objects separate from historical source-tree .o/.d files.
# Emit dependencies during compilation; changing this makefile rebuilds objects.
$(BUILD_DIR)/%.o: ./%.cpp makefile
	@mkdir -p "$(@D)"
	@echo "Compiling $<"
	$(COMPILE) -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -o "$@" "$<"

clean:
	rm -f $(OBJS) $(DEPS) $(norg)

clear:
	-rm -rf testing/edmft_back_up
	-rm -rf testing/g0imp.txt
	-rm -rf testing/Gf.out
	-rm -rf testing/h0.txt
	-rm -rf testing/hb_fit.txt
	-rm -rf testing/hb_read.txt
	-rm -rf testing/inorg
	-rm -rf testing/log.norg
	-rm -rf testing/nmat.txt
	-rm -rf testing/nohup.txt
	-rm -rf testing/ose_hop
	-rm -rf testing/Sig.out
	# -rm -rf bi/*.txt
	# -rm -rf bi/*.out
	# -rm -rf io/output.*
	# -rm -rf io/*.txt
	# -rm -rf tso/*
	# -rm -rf jmwang.*
	# -rm -rf *jmw*
	# -rm -rf log.*
deepclean:
	# -rm -rf general/*.cpp
	# -rm -rf general/*.h
	# -rm -rf special/*.cpp
	# -rm -rf special/*.h
	# -rm -rf randomc/*.cpp
	# -rm -rf randomc/*.h
	

depend:$(OBJS)
	@echo "dependencies are now up-to-date."

-include $(DEPS)
