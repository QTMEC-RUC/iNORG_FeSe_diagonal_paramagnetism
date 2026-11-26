#pragma once
#include "green_krylov_options.h"
#include "green_krylov_types.h"
#include "green_krylov_fixed.h"
#include "green_krylov_adaptive.h"

namespace green_krylov {
// Copy only fixed-engine inputs. Adaptive tolerances/checkpoints never enter it.
inline GreenKrylovFixedOptions fixed_options(const GreenKrylovOptions& o) {
    GreenKrylovFixedOptions result;
    result.solver=o.solver;result.max_basis=o.lanczos_step;
    result.check_level=o.check_level;result.fallback=o.fallback;
    result.orth_method=o.orth_method;result.check_tol=o.check_tol;
    result.deflation_tol=o.deflation_tol;result.zero_tol=o.zero_tol;
    return result;
}

// Synchronize BEFORE choosing an engine so all MPI ranks take the same branch.
// The selection occurs once per solve, outside every Krylov iteration.
template<class H>
Spectrum solve(const MyMpi& mm,const VecReal& f,const H& h,BasisIndex dim,Int default_limit,
    bool reorth,GreenKrylovOptions o,const Str& context="",const Monitor& monitor=Monitor(),
    const FrequencyGrid& grid=FrequencyGrid()) {
    o.synchronize(mm);
    if(o.lanczos_step>0)
        return fixed::solve(mm,f,h,dim,default_limit,reorth,fixed_options(o),context,monitor);
    return adaptive::solve(mm,f,h,dim,default_limit,reorth,o,context,monitor,grid);
}
// Physical frequency assembly shared with CrrltFun; no additional matrix scans.
template<class GreenType>
void accumulate_frequency(const MyMpi& mm,const Spectrum& spectrum,GreenType& g0,
    Real ge0,bool is_greater,green_profile::Profile* profile,std::ostringstream& checks) {
    green_profile::Scope frequency_timer(profile,green_profile::Frequency);
    VecPartition split_w(mm.np(),mm.id(),g0.nomgs);
    const Real sign=is_greater?1.:-1.;
    VecCmplx local(split_w.len(), Cmplx(0.,0.));
    const int nt=lanczos_local::workers(split_w.len());
#pragma omp parallel for schedule(static) num_threads(nt) if(nt>1)
    for(Int wi=split_w.bgn();wi<split_w.end();++wi) {
        Cmplx value(0., 0.);
        for (Idx j = 0; j < spectrum.energy.size(); ++j) {
            const Real amplitude = spectrum.amplitude[j];
            value += (amplitude * amplitude) * INV(g0.z(wi) + sign * (ge0 - spectrum.energy[j]));
        }
        local[wi - split_w.bgn()] = value;
    }
    frequency_timer.stop();
    green_profile::Scope finite_timer(profile,green_profile::FiniteCheck);
    Int finite_local=1;
    for(Idx i=0;i<local.size();++i)
        if(!std::isfinite(local[i].real()) || !std::isfinite(local[i].imag()))finite_local=0;
    const bool finite=mm.Allreduce(finite_local)==mm.np();finite_timer.stop();
    checks<<"GF_CHECK "<<(profile?profile->context:Str())<<" stage=frequency_finite_check status="
        <<(finite?"PASS":"FAIL")<<" frequencies="<<g0.nomgs
        <<" physical_causality=NOT_CHECKED frequency_convergence=SEE_GF_CONVERGENCE\n";
    if(!finite){if(profile)profile->finish(mm,checks.str());throw std::runtime_error("Nonfinite Green frequency result");}
    green_profile::Scope gather_timer(profile,green_profile::Gather);
    const VecCmplx gathered = mm.Allgatherv(local, split_w);
    for_Int(wi, 0, g0.nomgs) g0[wi][0][0] += gathered[wi];
    checks<<"GF_CHECK "<<(profile?profile->context:Str())
        <<" stage=frequency_gather_commit status=COMPLETE test=finite_before_commit\n";
}
} // namespace green_krylov
