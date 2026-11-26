#pragma once
#include "green_krylov_types.h"
#include "milapacke.h"
#include "green_profile.h"
#include "green_krylov_options.h"
#include "green_krylov_convergence.h"

// Preserve the current adaptive kernels and convergence mathematics.
namespace green_krylov { namespace adaptive {
inline Real norm(const MyMpi& mm,const VecReal& v) {
    const Real n=mm.Allreduce(lanczos_local::dot(v,v));
    if(!std::isfinite(n) || n<0.) throw std::runtime_error("Nonfinite/negative Krylov norm");
    return std::sqrt(n);
}
inline MatReal overlaps(const MyMpi& mm,const Sources& left,const Sources& right) {
    const Idx size=left.size()*right.size();
    if(size>static_cast<Idx>(std::numeric_limits<Int>::max()))
        throw std::overflow_error("Krylov overlap exceeds MPI count range");
    VecReal local(size),global(size);
    for(Idx i=0;i<left.size();++i)for(Idx j=0;j<right.size();++j)
        local[i*right.size()+j]=lanczos_local::dot(left[i],right[j]);
    mm.Allreduce_into(local,global);
    MatReal m(left.size(),right.size(),0.);
    for(Idx i=0;i<left.size();++i)for(Idx j=0;j<right.size();++j)m[i][j]=global[i*right.size()+j];
    return m;
}
struct ProjectionAudit { Real orth=0.,projection=-1.; };
template<class H>
inline Real audit_projection(const MyMpi& mm,const Sources& q,const H& h,const MatReal& t,ProjectionAudit* detail=nullptr) {
    ProjectionAudit result;
    MatReal gram=overlaps(mm,q,q);
    Real error=0.;
    for(Idx i=0;i<q.size();++i)for(Idx j=0;j<q.size();++j) {
        if(!std::isfinite(gram[i][j])) {
            result.orth=std::numeric_limits<Real>::infinity();if(detail)*detail=result;return result.orth;
        }
        error=std::max(error,std::abs(gram[i][j]-(i==j?1.:0.)));
        result.orth=error;
    }
    result.projection=0.;
    VecReal local(q.size()),global(q.size());
    for(Idx j=0;j<q.size();++j) {
        VecReal w;lanczos_local::apply(h,q[j],w);
        for(Idx i=0;i<q.size();++i)local[i]=lanczos_local::dot(q[i],w);
        mm.Allreduce_into(local,global);
        for(Idx i=0;i<q.size();++i) {
            const Real d=std::abs(global[i]-t[i][j])/std::max(Real(1.),std::abs(global[i]));
            if(!std::isfinite(d)) {
                result.projection=std::numeric_limits<Real>::infinity();if(detail)*detail=result;return result.projection;
            }
            result.projection=std::max(result.projection,d);
            error=std::max(error,d);
        }
    }
    if(detail)*detail=result;
    return error;
}
inline Spectrum scalar_spectrum(const VEC<Real>& diagonal,const VEC<Real>& off,
    Real initial,green_profile::Profile* profile=nullptr,MatReal* eigenvectors=nullptr) {
    Spectrum s;
    green_profile::Scope diagonal_timer(profile,green_profile::Diagonalize);
    s.energy.reset(diagonal);VecReal beta(off);MatReal eig(diagonal.size(),diagonal.size(),0.);
    if(diagonal.size()==1)eig[0][0]=1.;
    else if(trd_heevr_vd(s.energy,beta,eig)!=0)throw std::runtime_error("Scalar Green eigensolver failed");
    diagonal_timer.stop();
    green_profile::Scope amplitude_timer(profile,green_profile::Amplitudes);
    s.amplitude.reset(diagonal.size());
    for (Idx j = 0; j < diagonal.size(); ++j) s.amplitude[j] = initial * eig[0][j];
    if(eigenvectors)swapall(*eigenvectors,eig);
    return s;
}
template<class H>
Spectrum scalar(const MyMpi& mm,const VecReal& f,const H& h,Int limit,
                bool reorth,const GreenKrylovOptions& o,const Monitor& monitor=Monitor(),
                green_profile::Profile* profile=nullptr,const FrequencyGrid& grid=FrequencyGrid()) {
    green_profile::Scope seed_timer(profile,green_profile::Seed);
    Spectrum s;
    const Real initial=norm(mm,f);
    if(initial<=o.zero_tol){if(monitor)monitor("LANCZOS_SKIPPED_ZERO_SOURCE",0);s.exhausted=true;return s;}
    Convergence convergence(mm,o,grid,initial);
    seed_timer.stop();
    if(monitor)monitor("LANCZOS_BEFORE",0);
    green_profile::Scope krylov_timer(profile,green_profile::Krylov);
    VEC<Real> diagonal,off;Sources q;
    VecReal v(f);lanczos_local::scale(v,1./initial);
    VecReal w;green_profile::apply(profile,h,v,w);
    Real previous=0.;VecReal local(2,0.),global(2);
    lanczos_local::ProjectionWorkspace<Real> projection;VEC<const VecReal*> views;
    Spectrum checked;bool exhausted=false;
    for(Int k=0;k<limit;++k) {
        if(reorth || o.check_level>=3){reserve_sources(q,q.size()+1);q.push_back(v);}
        local[0]=lanczos_local::dot(w,v);local[1]=lanczos_local::dot(w,w);
        mm.Allreduce_into(local,global);
        const Real a=global[0];
        if(!std::isfinite(a))throw std::runtime_error("Nonfinite scalar Lanczos diagonal");
        diagonal.push_back(a);
        const Real scale=std::sqrt(global[1])+std::abs(previous);
        Real b;
        if(reorth) {
            lanczos_local::axpy(w,-a,v);
            green_profile::Scope timer(profile,green_profile::Orth);
            views.clear();for(const auto& basis:q)views.push_back(&basis);
            lanczos_local::project_out_views(mm,w,views,projection,o.orth_method);b=norm(mm,w);
        } else {
            Real n=mm.Allreduce(lanczos_local::axpy_norm(w,-a,v));
            if(!std::isfinite(n)||n<0.)throw std::runtime_error("Nonfinite scalar residual");
            b=std::sqrt(n);
        }
        if(b<=o.deflation_tol*std::max(Real(1.),scale)){exhausted=true;break;}
        if(convergence.due(k+1) || (convergence.active() && k+1==limit)) {
            green_profile::Scope timer(profile,green_profile::Convergence);
            MatReal eig;
            checked=scalar_spectrum(diagonal,off,initial,nullptr,&eig);
            if(convergence.observe(checked,eig,b))break;
        }
        if(k+1==limit)break;
        off.push_back(b);lanczos_local::scale(w,1./b);swapall(v,w);
        green_profile::apply(profile,h,v,w,-b);previous=b;
    }
    if(diagonal.size()!=off.size()+1)throw std::runtime_error("Invalid scalar tridiagonal");
    krylov_timer.stop();
    ProjectionAudit detail;Real error=0.;
    if(o.check_level>=3) {
        green_profile::Scope audit_timer(profile,green_profile::ProjectionCheck);
        MatReal t(diagonal.size(),diagonal.size(),0.);
        for(Idx i=0;i<diagonal.size();++i) {
            t[i][i]=diagonal[i];
            if(i<off.size())t[i][i+1]=t[i+1][i]=off[i];
        }
        error=audit_projection(mm,q,h,t,&detail);
    }
    if(monitor)monitor("LANCZOS_AFTER",0);
    Sources().swap(q);v.reset();w.reset();
    if(monitor){monitor("KRYLOV_RELEASED",0);monitor("SPECTRAL_BEFORE",0);}
    s=convergence.checked_basis()==static_cast<Int>(diagonal.size())?checked:
        scalar_spectrum(diagonal,off,initial,profile);
    s.exhausted=exhausted;s.error=error;s.valid=error<=o.check_tol;
    if(o.check_level>=3){s.orth_error=detail.orth;s.projection_error=detail.projection;}
    convergence.annotate(s);
    if(monitor)monitor("SPECTRAL_AFTER",0);
    return s;
}
template<class H>
Spectrum solve(const MyMpi& mm,const VecReal& f,const H& h,BasisIndex dim,Int default_limit,
                   bool reorth,GreenKrylovOptions o,const Str& context="",const Monitor& monitor=Monitor(),const FrequencyGrid& grid=FrequencyGrid()) {
    if(o.lanczos_step!=0)throw std::invalid_argument("Adaptive Green engine requires g_mode_lanczos_step=0");
    green_profile::Profile profile(mm,context,"solve",o.solver);
    green_profile::Scope setup_timer(&profile,green_profile::Setup);
    Int orth_flag=reorth?1:0;mm.Bcast(orth_flag);reorth=orth_flag!=0;
    const bool adaptive=o.lanczos_step==0;
    // In floating point a short recurrence may need more than dim steps:
    // loss of global orthogonality creates repeated Ritz directions. Only
    // full-history methods have dim as their natural numerical bound.
    const bool full_history=reorth;
    const Int representable_limit=std::numeric_limits<Int>::max()-2;
    const Int adaptive_limit=o.max_basis?o.max_basis:(full_history?std::min<BasisIndex>(dim,representable_limit):representable_limit);
    const Int limit=adaptive?(full_history?std::min<BasisIndex>(dim,adaptive_limit):adaptive_limit):
        std::min<BasisIndex>(dim,o.lanczos_step);
    if(dim<0 || limit<0)throw std::invalid_argument("Invalid Green Krylov dimension");
    if(limit>representable_limit)throw std::overflow_error("Green Krylov budget plus source frontier exceeds index range");
    VecPartition partition(mm.np(),mm.id(),dim);
    if(f.size()!=static_cast<Idx>(partition.len()))
        throw std::invalid_argument("Green source does not match equal-row partition");
    if(adaptive && dim>0) {
        Int invalid=!grid.z.size() || !std::isfinite(grid.ge0);
        for(Idx i=0;i<grid.z.size();++i)if(!std::isfinite(grid.z[i].real()) ||
            !std::isfinite(grid.z[i].imag()) || grid.z[i].imag()==0.)invalid=1;
        if(mm.Allreduce(invalid))throw std::invalid_argument(
            "Adaptive Green convergence requires the actual finite frequency grid with nonzero imaginary shifts");
    }
    // The old default is solely a tuning-cost estimate, never an adaptive cap.
    // Include all scalar source chains when amortizing a shared scalar H plan.
    const unsigned long long expected=static_cast<unsigned long long>(std::max(0,
        adaptive?std::min(limit,std::max(1,default_limit)):limit));
    if(expected)lanczos_local::hint_expected_calls(h,expected);
    std::ostringstream log;log<<std::scientific<<std::setprecision(9);
    log<<"GF_OPTIONS "<<context<<" solver="<<o.solver<<" max_basis="<<limit
        <<" reorth="<<reorth<<" orth_method="<<o.orth_method<<" check_level="<<o.check_level
        <<" check_tol="<<o.check_tol<<" fallback="<<o.fallback
        <<" implementation=adaptive_current iteration_policy="<<(adaptive?"adaptive":"fixed")<<" requested_steps="<<o.lanczos_step
        <<" convergence_rtol="<<o.convergence_rtol<<" convergence_atol="<<o.convergence_atol
        <<" convergence_interval="<<o.convergence_interval<<" convergence_stable_checks="<<o.convergence_stable_checks
        <<" residual_bound="<<"source_norm_linear"
        <<" hv_expected_calls="<<expected<<" hv_hint_is_iteration_cap=0"
        <<" adaptive_cap_kind="<<(o.max_basis?"user":full_history?"orthogonal_dimension":"index_range")<<'\n';
    setup_timer.stop();
    green_profile::Scope reference_timer(&profile,green_profile::Reference);
    ScalarReference ref(mm,f,h,o.check_level);
    reference_timer.stop();
    log<<"GF_CHECK "<<context<<" stage=reference_moments status=PASS test=dimensions_and_finite_reference"
        <<" moments="<<ref.moments.size()<<" physical_source_mapping=NOT_CHECKED\n";
    if(dim==0){log<<"GF_CHECK "<<context<<" stage=solve status=SKIPPED reason=empty_input\n";profile.finish(mm,log.str());return Spectrum();}
    Int attempts=0,fallbacks=0;
    const auto verdict=[&](Real error){return error<0.?"NOT_CHECKED":(std::isfinite(error)&&error<=o.check_tol?"PASS":"FAIL");};
    const Monitor measured_monitor=monitor?Monitor([&](const char* stage,Int column){
        green_profile::Scope timer(&profile,green_profile::Monitor);monitor(stage,column);
    }):Monitor();
    auto validate=[&](Spectrum& spec,Int mode) {
        green_profile::Scope audit_timer(&profile,green_profile::SpectralCheck);
        audit_scalar_spectrum(spec,ref,o);
        const bool stopped=spec.exhausted || spec.converged || !adaptive;
        spec.valid=spec.valid && stopped;
        Int ok=spec.valid?1:0;spec.valid=mm.Allreduce(ok)==mm.np();
        audit_timer.stop();++attempts;
        log<<"GF_KRYLOV "<<context<<" solver="<<mode<<" column=0"
            <<" basis="<<spec.energy.size()<<" stop="<<(spec.exhausted?"closed":spec.converged?"converged":adaptive?"adaptive_cap_unconverged":
                (o.lanczos_step>dim?"dimension_limit":"fixed_steps"))
            <<" check="<<(spec.valid?"PASS":"FAIL")<<" error="<<spec.error<<'\n';
        log<<"GF_CONVERGENCE "<<context<<" solver="<<mode<<" column=0"
            <<" policy="<<(adaptive?"adaptive":"fixed")<<" requested_steps="<<o.lanczos_step
            <<" basis="<<spec.energy.size()<<" checks="<<spec.convergence_checks
            <<" stable_checks="<<spec.convergence_stable<<" stability_scaled="<<spec.stability_error
            <<" residual_scaled="<<spec.residual_error<<" rtol="<<o.convergence_rtol<<" atol="<<o.convergence_atol
            <<" frequencies="<<grid.z.size()<<" status="<<(adaptive?(stopped?"PASS":"FAIL"):"NOT_REQUESTED")
            <<" criterion=all_entries_stability_and_residual_guard residual_bound_assumes=accurate_recurrence"
            <<" residual_bound="<<"source_norm_linear"
            <<" cap="<<limit<<'\n';
        log<<"GF_CHECK "<<context<<" solver="<<mode<<" column=0"
            <<" stage=projection_check orth_status="<<verdict(spec.orth_error)<<" orth_error="<<spec.orth_error
            <<" projection_status="<<verdict(spec.projection_error)<<" projection_error="<<spec.projection_error<<'\n';
        log<<"GF_CHECK "<<context<<" solver="<<mode<<" column=0"
            <<" stage=diagonalization status="<<(spec.energy.size()?(spec.finite?"PASS":"FAIL"):"SKIPPED")
            <<" test=lapack_return_and_finite_spectrum eigenpair_residual=NOT_CHECKED\n";
        for(Int k=0;k<3;++k)log<<"GF_CHECK "<<context<<" solver="<<mode<<" column=0"
            <<" stage=spectral_check moment="<<k<<" status="<<verdict(spec.moment_error[k])
            <<" error="<<spec.moment_error[k]<<" tolerance="<<o.check_tol<<'\n';
    };
    auto finish=[&](bool committed) {
        log<<"GF_CHECK "<<context<<" stage=solve status="<<(committed?"PASS":"FAIL")
            <<" attempts="<<attempts<<" fallbacks="<<fallbacks<<" timings_include_retries=1"
            <<" frequency_convergence="<<(adaptive?"RESIDUAL_GUARDED_STABILITY":"NOT_REQUESTED")<<'\n';
        profile.finish(mm,log.str());
    };
    Spectrum spec=scalar(mm,f,h,limit,reorth,o,measured_monitor,&profile,grid);validate(spec,-1);
    if(!spec.valid && o.fallback && !reorth) {
        ++fallbacks;log<<"GF_FALLBACK scalar -> scalar with reorthogonalization\n";
        spec=scalar(mm,f,h,std::min<BasisIndex>(dim,limit),true,o,measured_monitor,&profile,grid);validate(spec,-1);
    }
    if(!spec.valid){finish(false);throw std::runtime_error(adaptive && !spec.exhausted && !spec.converged?
        "Scalar Green adaptive convergence not reached at the configured cap/dimension; result not committed":
        "Scalar Green self-check failed; result not committed");}
    finish(true);
    return spec;
}
} } // namespace green_krylov::adaptive
