#pragma once
#include "myomp.h"
#include <algorithm>
#include <exception>
#include <vector>

// No MPI in workers. Thread counts come from the runtime/user budget, never hardware literals.
namespace stage2_parallel {
inline int workers(std::size_t work) {
#ifdef _OPENMP
    if (!work || omp_in_parallel()) return 1;
    return static_cast<int>(std::min(work, static_cast<std::size_t>(thread_budget())));
#else
    return 1;
#endif
}
class LocalMklSerial {
    int previous_;
public:
    LocalMklSerial(): previous_(mkl_set_num_threads_local(1)) {}
    ~LocalMklSerial() { mkl_set_num_threads_local(previous_); }
    LocalMklSerial(const LocalMklSerial&) = delete;
};
inline std::size_t boundary(std::size_t n, std::size_t i, std::size_t parts) {
    return (n / parts) * i + std::min(i, n % parts);
}
template<class F> void for_each(std::size_t n, F fn) {
    const int nt = workers(n);
    if (nt == 1) {
        LocalMklSerial serial;
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::exception_ptr error;
#pragma omp parallel num_threads(nt)
    {
        LocalMklSerial serial;
#pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < n; ++i) {
            try { fn(i); }
            catch (...) {
#pragma omp critical(norg_stage2_exception)
                { if (!error) error = std::current_exception(); }
            }
        }
    }
    if (error) std::rethrow_exception(error);
}
// Stable contiguous ranges: each range owns its cursor, cache and output.
template<class F> void ranges(std::size_t n, F fn) {
    const std::size_t parts = workers(n);
    for_each(parts, [&](std::size_t part) {
        fn(boundary(n, part, parts), boundary(n, part + 1, parts), part);
    });
}
// Synchronize failures only after leaving all OpenMP regions.
template<class Mpi, class F> void collective(const Mpi& mm, F fn) {
    std::exception_ptr error;
    try { fn(); } catch (...) { error = std::current_exception(); }
    const int failed = error ? 1 : 0;
    if (mm.Allreduce(failed)) {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("Stage-two computation failed on another MPI rank");
    }
}
} // namespace stage2_parallel
