#pragma once
#include <mpi.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

// Replicate a contiguous partition without a second full-vector staging buffer.
// Only the communicator's root reads configuration; all ranks use identical settings.
// All MPI calls belong on the MPI/FUNNELED thread.
namespace full_vector_gather {
struct Timing {
    double setup = 0., copy = 0., exchange = 0.;
    bool native = false;
};
inline int allgather(const double* local, int local_count, double* full,
                     const int* counts, const int* offsets,
                     MPI_Comm comm, int root, Timing* timing = nullptr)
{
    if(timing) *timing = Timing{};
    const double start = timing ? MPI_Wtime() : 0.;
    int ranks = 0, rank = 0;
    MPI_Comm_size(comm, &ranks);
    MPI_Comm_rank(comm, &rank);
    // mode, chunk length in doubles, configuration error.
    int settings[3] = {0, (4 * 1024 * 1024) / static_cast<int>(sizeof(double)), 0};
    if(rank == root) {
        if(const char* v = std::getenv("NORG_FULL_VECTOR_GATHER")) {
            if(std::strcmp(v, "native") == 0) settings[0] = 1;
            else if(std::strcmp(v, "ring") != 0) settings[2] = 1;
        }
        if(const char* v = std::getenv("NORG_FULL_VECTOR_CHUNK_BYTES")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long bytes = std::strtoull(v, &end, 10);
            if(errno || end == v || *end || *v == '-' || bytes < sizeof(double) ||
               bytes / sizeof(double) > static_cast<unsigned long long>(INT_MAX))
                settings[2] = 1;
            else settings[1] = static_cast<int>(bytes / sizeof(double));
        }
    }
    int rc = MPI_Bcast(settings, 3, MPI_INT, root, comm);
    if(rc != MPI_SUCCESS) return rc;
    if(settings[2]) return MPI_ERR_ARG;
    long long total = 0;
    int bad = local_count < 0, max_count = 0;
    for(int r = 0; r < ranks; ++r) {
        if(counts[r] < 0 || offsets[r] != total) bad = 1;
        total += counts[r];
        max_count = std::max(max_count, counts[r]);
    }
    if(total > INT_MAX || local_count != counts[rank] ||
       (local_count && !local) || (total && !full)) bad = 1;
    int any_bad = 0;
    rc = MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
    if(rc != MPI_SUCCESS) return rc;
    if(any_bad) return MPI_ERR_COUNT;
    if(timing) timing->native = settings[0] != 0;
    double dummy = 0.;
    if(settings[0]) {
        if(timing) timing->setup = MPI_Wtime() - start;
        const double exchange_start = timing ? MPI_Wtime() : 0.;
        rc = MPI_Allgatherv(local_count ? local : &dummy, local_count, MPI_DOUBLE,
                           total ? full : &dummy, counts, offsets, MPI_DOUBLE, comm);
        if(timing) timing->exchange = MPI_Wtime() - exchange_start;
        return rc;
    }
    // A private context prevents matching any unrelated point-to-point traffic.
    MPI_Comm ring = MPI_COMM_NULL;
    if(ranks > 1 && total) {
        rc = MPI_Comm_dup(comm, &ring);
        if(rc != MPI_SUCCESS) return rc;
    }
    if(timing) timing->setup = MPI_Wtime() - start;
    const double copy_start = timing ? MPI_Wtime() : 0.;
    if(local_count)
        std::memmove(full + offsets[rank], local,
                     static_cast<std::size_t>(local_count) * sizeof(double));
    if(timing) timing->copy = MPI_Wtime() - copy_start;
    const double exchange_start = timing ? MPI_Wtime() : 0.;
    if(ring != MPI_COMM_NULL) {
        const int next = (rank + 1) % ranks, prev = (rank + ranks - 1) % ranks;
        const int chunk = settings[1];
        // Every rank executes the same rounds, including empty partitions.
        // Wide offsets avoid overflow at the MPI int-count boundary.
        for(int step = 0; step < ranks - 1 && rc == MPI_SUCCESS; ++step) {
            const int send_owner = (rank - step + ranks) % ranks;
            const int recv_owner = (rank - step - 1 + ranks) % ranks;
            for(long long offset = 0; offset < max_count; offset += chunk) {
                const int ns = static_cast<int>(std::max(0LL,
                    std::min(static_cast<long long>(chunk), counts[send_owner] - offset)));
                const int nr = static_cast<int>(std::max(0LL,
                    std::min(static_cast<long long>(chunk), counts[recv_owner] - offset)));
                rc = MPI_Sendrecv(ns ? full + offsets[send_owner] + offset : &dummy,
                                  ns, MPI_DOUBLE, next, 0,
                                  nr ? full + offsets[recv_owner] + offset : &dummy,
                                  nr, MPI_DOUBLE, prev, 0, ring, MPI_STATUS_IGNORE);
                if(rc != MPI_SUCCESS) break;
            }
        }
    }
    if(timing) timing->exchange = MPI_Wtime() - exchange_start;
    const double free_start = timing ? MPI_Wtime() : 0.;
    if(ring != MPI_COMM_NULL) {
        const int free_rc = MPI_Comm_free(&ring);
        if(rc == MPI_SUCCESS) rc = free_rc;
    }
    if(timing) timing->setup += MPI_Wtime() - free_start;
    return rc;
}
} // namespace full_vector_gather
