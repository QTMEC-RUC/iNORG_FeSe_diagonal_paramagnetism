/*
code developed by
    Rong-Qiang He (rqhe@ruc.edu.cn, RUC, China)
date 2013 - 2017
*/

#ifndef _MYOMP_H_
#define _MYOMP_H_

#include "stdpfx.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mymkl.h"
#include <stdexcept>

// One positive integer per rank; OpenMP's nested-team list syntax is deliberately
// unsupported. Read once, before any phase changes the runtime's thread count.
inline Int parse_thread_budget(const char* value)
{
	if (value == nullptr) return 1;
	if (*value == '\0') throw std::invalid_argument("OMP_NUM_THREADS must be a positive integer");
	Int result = 0;
	for (const char* digit = value; *digit; ++digit) {
		if (*digit < '0' || *digit > '9' ||
			result > (std::numeric_limits<Int>::max() - (*digit - '0')) / 10)
			throw std::invalid_argument("OMP_NUM_THREADS must be one positive integer within Int range");
		result = result * 10 + (*digit - '0');
	}
	if (result == 0) throw std::invalid_argument("OMP_NUM_THREADS must be positive");
	return result;
}

inline Int thread_budget()
{
	static const Int budget = parse_thread_budget(std::getenv("OMP_NUM_THREADS"));
	return budget;
}

// MKL must not request a team larger than the OpenMP runtime permits.
// In this environment sparse MKL kernels can miscompute if their requested
// team is silently reduced by OMP_THREAD_LIMIT.
inline Int effective_mkl_threads(Int requested)
{
#ifdef _OPENMP
    return requested < omp_get_thread_limit() ? requested : omp_get_thread_limit();
#else
    return requested;
#endif
}

// Settings only: call on the main thread, outside parallel regions.
// Later outer-loop phases can use configure_threads(T, 1), then restore (T, T).
inline void configure_threads(Int omp_threads, Int mkl_threads)
{
	if (omp_threads < 1 || mkl_threads < 1)
		throw std::invalid_argument("OpenMP and MKL thread counts must be positive");
#ifdef _OPENMP
	if (omp_in_parallel()) throw std::logic_error("Configure threads outside OpenMP parallel regions");
	omp_set_dynamic(0);
	omp_set_max_active_levels(1);
	omp_set_num_threads(omp_threads);
#else
	throw std::runtime_error("Hybrid build requires OpenMP; rebuild with OPENMP_FLAGS enabled");
#endif
	mkl_set_dynamic(0);
	mkl_set_num_threads(effective_mkl_threads(mkl_threads));
}

inline Int probe_omp_team_size()
{
	Int actual = 1;
#ifdef _OPENMP
#pragma omp parallel
	{
#pragma omp single
		{
			actual = omp_get_num_threads();
		}
	}
#endif
	return actual;
}

inline void use_omp(Int nthreads = 1)
{
	if (nthreads < 1) throw std::invalid_argument("OpenMP thread count must be positive");
#ifdef _OPENMP
	if (omp_in_parallel()) throw std::logic_error("Configure OpenMP outside parallel regions");
	omp_set_dynamic(0);
	omp_set_max_active_levels(1);
	omp_set_num_threads(nthreads);
#else
	if (nthreads != 1) throw std::runtime_error("OpenMP was not enabled at compile time");
#endif
}

inline void use_mkl(Int nthreads = 1)
{
	if (nthreads < 1) throw std::invalid_argument("MKL thread count must be positive");
#ifdef _OPENMP
	if (omp_in_parallel()) throw std::logic_error("Configure MKL outside parallel regions");
#endif
	mkl_set_dynamic(0);
	mkl_set_num_threads(effective_mkl_threads(nthreads));
}

#endif /* _MYOMP_H_ */
