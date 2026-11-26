#pragma once
#include "vec.h"
#include "miblas.h"
#include "myomp.h"
#include <algorithm>

// Elementwise kernels only; MPI and MKL calls remain on the initializing thread.
namespace lanczos_local {
inline int workers(Idx n) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 1;
#endif
    return n ? static_cast<int>(std::min<Idx>(n, thread_budget())) : 1;
}
template<class T, class A> void axpy(Vec<T>& y, A a, const Vec<T>& x) {
    if(y.size()!=x.size()) throw std::invalid_argument("Lanczos axpy size mismatch");
    const Idx n=y.size(); const int nt=workers(n);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
    for(Idx i=0;i<n;++i) y[i] += a*x[i];
}
template<class T, class A> void scale(Vec<T>& x, A a) {
    const Idx n=x.size(); const int nt=workers(n);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
    for(Idx i=0;i<n;++i) x[i] *= a;
}
// MKL handles real ddot and conjugated complex zdotc; no outer OpenMP team.
template<class T> T dot(const Vec<T>& x,const Vec<T>& y) {
    if(x.size()!=y.size())throw std::invalid_argument("Lanczos dot size mismatch");
    return x.size() ? DOT(x,y) : T(0);
}
// Only use this norm when no subsequent deflation/reorthogonalization changes y.
inline Real axpy_norm(VecReal& y,Real a,const VecReal& x) {
    if(x.size()!=y.size())throw std::invalid_argument("Lanczos fused update size mismatch");
    const Idx n=y.size();const int nt=workers(n);Real sum=0.;
#pragma omp parallel for reduction(+:sum) schedule(static) num_threads(nt) if(nt > 1)
    for(Idx i=0;i<n;++i){y[i]+=a*x[i];sum+=y[i]*y[i];}
    return sum;
}
// Optional performance hint. Generic operators need no plan or new interface.
// Counts describe expected work; only the calling algorithm controls its loops.
template<class H> auto hint_expected_calls_impl(const H& h,unsigned long long calls,int)
    -> decltype(h.hint_expected_calls(calls),void()) { h.hint_expected_calls(calls); }
template<class H> void hint_expected_calls_impl(const H&,unsigned long long,long) {}
template<class H> void hint_expected_calls(const H& h,unsigned long long calls) {
    hint_expected_calls_impl(h,calls,0);
}

// Preserve the generic Lanczos interface for non-SparseMat operators used by tests.
template<class H> auto apply_impl(const H& h,const VecReal& x,VecReal& y,Real beta,int)
    -> decltype(h.apply(x,y,Real(1),beta),void()) { h.apply(x,y,1.,beta); }
template<class H> void apply_impl(const H& h,const VecReal& x,VecReal& y,Real beta,long) {
    VecReal z=h*x;
    if(y.size()!=z.size()) y.reset(z.size(),0.);
    scale(y,beta);axpy(y,1.,z);
}
template<class H> void apply(const H& h,const VecReal& x,VecReal& y,Real beta=0.) {
    apply_impl(h,x,y,beta,0);
}

} // namespace lanczos_local
