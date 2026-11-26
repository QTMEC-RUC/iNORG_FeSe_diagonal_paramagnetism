#!/usr/bin/env bash
# Single-node launcher. Run from the calculation directory to preserve its inputs.
set -euo pipefail

die() { printf 'run_hybrid: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'EOF'
Usage: bash run_hybrid.sh RANKS THREADS [EXECUTABLE [ARG...]]
Single-node MPI + OpenMP/MKL launch; preserves the current working directory.
Default executable: FeSe_diagonal_paramagnetism beside this script.
Environment: MPIEXEC (launcher path), MPI_FLAVOR=auto|intel|openmpi|mpich,
             HYBRID_CPU_LIMIT (optional stricter allocated CPU count).
Pass --parallel-check to the executable for an input-free startup check.
EOF
}
positive_int() {
    [[ $1 =~ ^[1-9][0-9]*$ ]] && [[ ${#1} -le 10 ]] && (( $1 <= 2147483647 ))
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
ranks=$1
threads=$2
positive_int "$ranks" || die 'RANKS must be an integer in 1..2147483647'
positive_int "$threads" || die 'THREADS must be an integer in 1..2147483647'
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
executable="$script_dir/FeSe_diagonal_paramagnetism"
if [[ $# -gt 0 ]]; then executable=$1; shift; fi
[[ -x "$executable" ]] || die "executable is missing or not executable: $executable"

launcher=${MPIEXEC:-mpiexec}
command -v "$launcher" >/dev/null 2>&1 || die "MPI launcher not found: $launcher"
command -v nproc >/dev/null 2>&1 || die 'nproc is required to check available CPUs'
# Do not let a pre-existing OpenMP setting change nproc's CPU count.
available=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
positive_int "$available" || die 'could not determine the available CPU count'
if [[ -n ${HYBRID_CPU_LIMIT:-} ]]; then
    positive_int "$HYBRID_CPU_LIMIT" || die 'HYBRID_CPU_LIMIT must be a positive integer'
    (( HYBRID_CPU_LIMIT < available )) && available=$HYBRID_CPU_LIMIT
fi
required=$((ranks * threads))
(( required <= available )) || die "requested $required CPUs ($ranks x $threads), available/limited to $available"

flavor=${MPI_FLAVOR:-auto}
if [[ $flavor == auto ]]; then
    version=$("$launcher" --version 2>&1) || die 'cannot detect MPI; set MPI_FLAVOR=intel, openmpi, or mpich'
    case "$version" in
        *Intel*MPI*) flavor=intel ;;
        *Open\ MPI*|*OpenRTE*) flavor=openmpi ;;
        *HYDRA*build*details*|*MPICH*) flavor=mpich ;;
        *) die 'unknown MPI implementation; use its matching launcher and set MPI_FLAVOR' ;;
    esac
fi

export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OMP_THREAD_LIMIT="$threads" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
# These legacy/domain settings can otherwise override the unified policy.
unset MKL_DOMAIN_NUM_THREADS KMP_AFFINITY GOMP_CPU_AFFINITY
# "threads" means hardware execution contexts, matching nproc's CPU units.
export OMP_PLACES=threads OMP_PROC_BIND=close
case "$flavor" in
    intel)
        # Explicit logical-CPU domain size, one disjoint domain per local rank.
        export I_MPI_PIN=1 I_MPI_PIN_DOMAIN="$threads" I_MPI_PIN_CELL=unit
        export I_MPI_PIN_RESPECT_CPUSET=1
        unset I_MPI_PIN_PROCESSOR_LIST
        command=("$launcher" -hosts localhost -n "$ranks" -ppn "$ranks" "$executable" "$@")
        ;;
    openmpi)
        command=("$launcher" --host localhost --use-hwthread-cpus
                 --map-by "slot:PE=$threads" --bind-to hwthread
                 -n "$ranks" "$executable" "$@")
        ;;
    mpich)
        # Single-node UCX default: avoid oversized PMI address exchanges in
        # Bohr's MPICH 4.2.0 Hydra. Explicit UCX_TLS still takes precedence.
        export UCX_TLS="${UCX_TLS:-self,sm}"
        # Local fork avoids SSH; Hydra/hwloc assigns T logical CPUs per rank.
        command=("$launcher" -launcher fork -hosts localhost -n "$ranks"
                 -bind-to "hwthread:$threads" -map-by "hwthread:$threads"
                 "$executable" "$@")
        ;;
    *) die 'MPI_FLAVOR must be auto, intel, openmpi, or mpich' ;;
esac
printf 'run_hybrid: flavor=%s ranks=%s threads=%s cpus=%s cwd=%s\n' \
    "$flavor" "$ranks" "$threads" "$available" "$PWD"
printf 'run_hybrid: command:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
