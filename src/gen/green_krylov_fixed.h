#pragma once
#include "green_krylov_types.h"
#include "milapacke.h"
#include "green_profile.h"
#include "green_krylov_fixed_options.h"

// Fixed recurrences copied from the designated optimization_2 source backup.
namespace green_krylov { namespace fixed {
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
template<class H>
Spectrum scalar(const MyMpi& mm,const VecReal& f,const H& h,Int limit,
                bool reorth,const GreenKrylovFixedOptions& o,const Monitor& monitor=Monitor(),green_profile::Profile* profile=nullptr) {
    green_profile::Scope seed_timer(profile,green_profile::Seed);
    Spectrum s;
    const Real initial=norm(mm,f);
    if(initial<=o.zero_tol){if(monitor)monitor("LANCZOS_SKIPPED_ZERO_SOURCE",0);s.exhausted=true;return s;}
    seed_timer.stop();
    if(monitor)monitor("LANCZOS_BEFORE",0);
    green_profile::Scope krylov_timer(profile,green_profile::Krylov);
    VEC<Real> diagonal,off;Sources q;
    VecReal v(f);lanczos_local::scale(v,1./initial);
    VecReal w;green_profile::apply(profile,h,v,w);
    Real previous=0.;VecReal local(2,0.),global(2);
    for(Int k=0;k<limit;++k) {
        if(reorth || o.check_level>=3)q.push_back(v);
        local[0]=lanczos_local::dot(w,v);local[1]=lanczos_local::dot(w,w);
        mm.Allreduce_into(local,global);
        const Real a=global[0];
        if(!std::isfinite(a))throw std::runtime_error("Nonfinite scalar Lanczos diagonal");
        diagonal.push_back(a);
        const Real scale=std::sqrt(global[1])+std::abs(previous);
        Real b;
        if(reorth) {
            lanczos_local::axpy(w,-a,v);
            green_profile::project(profile,mm,w,q,q.size(),o.orth_method);b=norm(mm,w);
        } else {
            Real n=mm.Allreduce(lanczos_local::axpy_norm(w,-a,v));
            if(!std::isfinite(n)||n<0.)throw std::runtime_error("Nonfinite scalar residual");
            b=std::sqrt(n);
        }
        if(b<=o.deflation_tol*std::max(Real(1.),scale)){s.exhausted=true;break;}
        if(k+1==limit)break;
        off.push_back(b);lanczos_local::scale(w,1./b);swapall(v,w);
        green_profile::apply(profile,h,v,w,-b);previous=b;
    }
    if(diagonal.size()!=off.size()+1)throw std::runtime_error("Invalid scalar tridiagonal");
    krylov_timer.stop();
    if(o.check_level>=3) {
        green_profile::Scope audit_timer(profile,green_profile::ProjectionCheck);
        ProjectionAudit detail;
        MatReal t(diagonal.size(),diagonal.size(),0.);
        for(Idx i=0;i<diagonal.size();++i) {
            t[i][i]=diagonal[i];
            if(i<off.size())t[i][i+1]=t[i+1][i]=off[i];
        }
        s.error=audit_projection(mm,q,h,t,&detail);s.valid=s.error<=o.check_tol;
        s.orth_error=detail.orth;s.projection_error=detail.projection;
    }
    if(monitor)monitor("LANCZOS_AFTER",0);
    Sources().swap(q);v.reset();w.reset(); // release large vectors before diagonalization
    if(monitor){monitor("KRYLOV_RELEASED",0);monitor("SPECTRAL_BEFORE",0);}
    green_profile::Scope diagonal_timer(profile,green_profile::Diagonalize);
    s.energy.reset(diagonal);VecReal beta(off);MatReal eig(diagonal.size(),diagonal.size(),0.);
    if(diagonal.size()==1)eig[0][0]=1.;
    else if(trd_heevr_vd(s.energy,beta,eig)!=0)throw std::runtime_error("Scalar Green eigensolver failed");
    diagonal_timer.stop();
    green_profile::Scope amplitude_timer(profile,green_profile::Amplitudes);
    s.amplitude.reset(diagonal.size());
    for (Idx j = 0; j < diagonal.size(); ++j) s.amplitude[j] = initial * eig[0][j];
    amplitude_timer.stop();
    if(monitor)monitor("SPECTRAL_AFTER",0);
    return s;
}
template<class H>
Spectrum solve(const MyMpi& mm,const VecReal& f,const H& h,BasisIndex dim,Int default_limit,
                   bool reorth,GreenKrylovFixedOptions o,const Str& context="",const Monitor& monitor=Monitor()) {
    o.validate();
    if(o.max_basis<=0)throw std::invalid_argument("Fixed Green engine requires a positive basis budget");
    green_profile::Profile profile(mm,context,"solve",o.solver);
    green_profile::Scope setup_timer(&profile,green_profile::Setup);
    Int orth_flag=reorth?1:0;mm.Bcast(orth_flag);reorth=orth_flag!=0;
    const Int limit=std::min<BasisIndex>(dim,o.max_basis?o.max_basis:default_limit);
    if(dim<0 || limit<0)throw std::invalid_argument("Invalid Green Krylov dimension");
    VecPartition partition(mm.np(),mm.id(),dim);
    if(f.size()!=static_cast<Idx>(partition.len()))
        throw std::invalid_argument("Green source does not match equal-row partition");
    std::ostringstream log;log<<std::scientific<<std::setprecision(9);
    log<<"GF_OPTIONS "<<context<<" solver="<<o.solver<<" max_basis="<<limit
        <<" reorth="<<reorth<<" orth_method="<<o.orth_method<<" check_level="<<o.check_level
        <<" check_tol="<<o.check_tol<<" fallback="<<o.fallback
        <<" implementation=fixed_backup iteration_policy=fixed requested_steps="<<o.max_basis
        <<" hv_expected_calls="<<0<<'\n';
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
        Int ok=spec.valid?1:0;spec.valid=mm.Allreduce(ok)==mm.np();
        audit_timer.stop();++attempts;
        log<<"GF_KRYLOV "<<context<<" solver="<<mode<<" column=0"
            <<" basis="<<spec.energy.size()<<" stop="<<(spec.exhausted?"closed":(o.max_basis>dim?"dimension_limit":"fixed_steps"))
            <<" check="<<(spec.valid?"PASS":"FAIL")<<" error="<<spec.error<<'\n';
        log<<"GF_CONVERGENCE "<<context<<" solver="<<mode<<" column=0"
            <<" policy=fixed requested_steps="<<o.max_basis<<" basis="<<spec.energy.size()
            <<" checks=0 stable_checks=0 status=NOT_REQUESTED cap="<<limit<<'\n';
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
            <<" frequency_convergence=NOT_REQUESTED\n";
        profile.finish(mm,log.str());
    };
    Spectrum spec=scalar(mm,f,h,limit,reorth,o,measured_monitor,&profile);validate(spec,-1);
    if(!spec.valid && o.fallback && !reorth) {
        ++fallbacks;log<<"GF_FALLBACK scalar -> scalar with reorthogonalization\n";
        spec=scalar(mm,f,h,limit,true,o,measured_monitor,&profile);validate(spec,-1);
    }
    if(!spec.valid){finish(false);throw std::runtime_error("Scalar Green self-check failed; result not committed");}
    finish(true);
    return spec;
}
} } // namespace green_krylov::fixed
