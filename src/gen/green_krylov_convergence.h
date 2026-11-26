#pragma once
#include "green_krylov_options.h"
#include "green_krylov_types.h"
#include "green_profile.h"
#include <algorithm>
#include <cmath>
#include <limits>

// Shared physical-frequency stopping policy. No Hamiltonian application, output,
// or collective is added to an ordinary Lanczos step; checks are checkpointed.
namespace green_krylov {
// Vec has a copying constructor but no noexcept move. Grow the descriptor
// array by swapping buffers explicitly, avoiding copies of distributed vectors.
template<class SourceList> void reserve_sources(SourceList& vectors,Idx required) {
    if(required<=vectors.capacity())return;
    SourceList next;next.reserve(std::max(required,std::max(Idx(16),2*vectors.capacity())));
    next.resize(vectors.size());
    for(Idx i=0;i<vectors.size();++i)swapall(next[i],vectors[i]);
    vectors.swap(next);
}
class Convergence {
    const MyMpi& mm_;
    const GreenKrylovOptions& o_;
    const FrequencyGrid& grid_;
    const Real source_norm_;
    VEC<Cmplx> previous_;
    Int next_=0,checked_basis_=0;
public:
    Int checks=0,stable=0;
    Real stability_error=-1.,residual_error=-1.;
    bool converged=false;
    Convergence(const MyMpi& mm, const GreenKrylovOptions& options,
                const FrequencyGrid& grid, Real source_norm)
        : mm_(mm), o_(options), grid_(grid), source_norm_(source_norm), next_(options.convergence_interval) {}
    bool active()const{return o_.lanczos_step==0;}
    bool due(Int basis)const{return active() && basis>=next_;}
    Int checked_basis()const{return checked_basis_;}
    bool observe(const Spectrum& spectrum, const MatReal& eigenvectors, Real beta) {
        const Idx basis = spectrum.energy.size();
        if (!basis || spectrum.amplitude.size() != basis || eigenvectors.ncols() != basis ||
            eigenvectors.nrows() != basis) throw std::invalid_argument("Invalid scalar Ritz projection");
        VecPartition frequencies(mm_.np(), mm_.id(), grid_.z.size());
        VEC<Cmplx> current(frequencies.len());
        const bool have_previous = checks > 0 && previous_.size() == current.size();
        const Real sign = grid_.is_greater ? 1. : -1.;
        double max_change = 0., max_residual = 0.;
        const int nt = lanczos_local::workers(frequencies.len());
#pragma omp parallel for schedule(static) num_threads(nt) if(nt>1) reduction(max:max_change,max_residual)
        for (Int wi = frequencies.bgn(); wi < frequencies.end(); ++wi) {
            Cmplx value(0., 0.), frontier(0., 0.);
            for (Idx j = 0; j < basis; ++j) {
                const Cmplx coordinate = spectrum.amplitude[j] * INV(grid_.z[wi] + sign*(grid_.ge0-spectrum.energy[j]));
                value += spectrum.amplitude[j] * coordinate;
                frontier += (beta * eigenvectors[basis-1][j]) * coordinate;
            }
            const Int local_index = wi - frequencies.bgn();
            current[local_index] = value;
            const Real old_scale = have_previous ? std::abs(previous_[local_index]) : 0.;
            const Real tolerance = o_.convergence_atol + o_.convergence_rtol * std::max(std::abs(value), old_scale);
            const auto scaled = [tolerance](Real error) {
                return tolerance > 0. ? error/tolerance :
                    (error == 0. ? 0. : std::numeric_limits<Real>::infinity());
            };
            const Real residual = scaled(source_norm_ * std::abs(frontier) / std::abs(grid_.z[wi].imag()));
            const Real change = have_previous ? scaled(std::abs(value - previous_[local_index])) : 0.;
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
                !std::isfinite(residual) || !std::isfinite(change)) {
                max_change = max_residual = std::numeric_limits<Real>::infinity();
            } else {
                max_change = std::max(max_change, change);
                max_residual = std::max(max_residual, residual);
            }
        }
        double local[2]={have_previous?max_change:std::numeric_limits<Real>::infinity(),max_residual},global[2];
        if(MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_MAX,mm_.cm())!=MPI_SUCCESS)
            throw std::runtime_error("Green convergence reduction failed");
        stability_error=global[0];residual_error=global[1];++checks;
        stable=(stability_error<=1. && residual_error<=1.)?stable+1:0;
        converged=stable>=o_.convergence_stable_checks;
        previous_.swap(current);checked_basis_=basis;
        next_=basis>std::numeric_limits<Int>::max()-o_.convergence_interval
            ?std::numeric_limits<Int>::max():basis+o_.convergence_interval;
        return converged;
    }
    template<class Spec> void annotate(Spec& s)const {
        s.converged=converged;s.convergence_checks=checks;s.convergence_stable=stable;
        s.stability_error=stability_error;s.residual_error=residual_error;
    }
};
} // namespace green_krylov
