#pragma once
#include "batched_orthogonalization.h"
#include <functional>

// Data containers only; fixed and adaptive iteration engines do not call each other.
namespace green_krylov {
using Sources=VEC<VecReal>;
using Monitor=std::function<void(const char*,Int)>;
struct Spectrum {
    VecReal energy;
    VecReal amplitude; // overlap of the one physical source with each Ritz vector
    bool exhausted=false;
    bool converged=false;
    Int convergence_checks=0,convergence_stable=0;
    Real stability_error=-1.,residual_error=-1.;
    bool valid=true;
    Real error=0.;
    bool finite=true;
    std::array<Real,3> moment_error{{-1.,-1.,-1.}};
    Real orth_error=-1.,projection_error=-1.;
    // Vec/Mat assignment in this codebase requires an unchanged shape.
    // Checkpoints grow; reset their storage explicitly before copying data.
    Spectrum& operator=(const Spectrum& rhs) {
        if(this==&rhs)return *this;
        energy.reset(rhs.energy);amplitude.reset(rhs.amplitude);
        exhausted=rhs.exhausted;converged=rhs.converged;
        convergence_checks=rhs.convergence_checks;convergence_stable=rhs.convergence_stable;
        stability_error=rhs.stability_error;residual_error=rhs.residual_error;
        valid=rhs.valid;error=rhs.error;finite=rhs.finite;moment_error=rhs.moment_error;
        orth_error=rhs.orth_error;projection_error=rhs.projection_error;
        return *this;
    }
};

struct ScalarReference {
    VEC<Real> moments;
    template<class H>
    ScalarReference(const MyMpi& mm, const VecReal& source, const H& h, Int level) {
        moments.push_back(mm.Allreduce(lanczos_local::dot(source, source)));
        if (level >= 2) {
            VecReal hsource;
            lanczos_local::apply(h, source, hsource);
            moments.push_back(mm.Allreduce(lanczos_local::dot(source, hsource)));
            moments.push_back(mm.Allreduce(lanczos_local::dot(hsource, hsource)));
        }
        for (Real moment : moments)
            if (!std::isfinite(moment)) throw std::runtime_error("Nonfinite scalar source moment");
    }
};

template<class Options>
void audit_scalar_spectrum(Spectrum& spectrum, const ScalarReference& reference, const Options& o) {
    for (Idx j = 0; j < spectrum.energy.size(); ++j) {
        if (!std::isfinite(spectrum.energy[j]) || !std::isfinite(spectrum.amplitude[j]))
            spectrum.finite = spectrum.valid = false;
    }
    if (!o.check_level) return;
    for (Idx order = 0; order < reference.moments.size(); ++order) {
        Real moment = 0.;
        for (Idx j = 0; j < spectrum.energy.size(); ++j) {
            const Real e = spectrum.energy[j], a = spectrum.amplitude[j];
            moment += a*a*(order == 0 ? 1. : order == 1 ? e : e*e);
        }
        const Real exact = reference.moments[order];
        const Real error = std::abs(moment-exact) / std::max(Real(1.),std::abs(exact));
        spectrum.moment_error[order] = error;
        if (!std::isfinite(error) || error > o.check_tol) spectrum.valid = false;
        spectrum.error = std::max(spectrum.error, error);
    }
}
struct FrequencyGrid {
    VecCmplx z;
    Real ge0=0.;
    bool is_greater=true;
};
} // namespace green_krylov
