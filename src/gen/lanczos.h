#pragma once

/*
code developed by
    Jia-Ming Wang (jmw@ruc.edu.cn, RUC, China) date 2020 - 2024
*/


#include "miblas.h"
#include "milapacke.h"		// to get the trd_heevr_vd(diagonalize Tridiagonal matrix function)
#include "stdpfx.h"
#include "vec.h"
#include "mat.h"
#include "random.h"
#include "mympi.h"
#include "lanczos_timing.h"
#include "parallel_lanczos.h"
#include "parallel_preparation.h"
#include "batched_orthogonalization.h"
#include "lowest_tridiagonal.h"

template<typename T>
bool residual_method_for_b(const VEC<T>& ltd,const VEC<T>& lt_sd,Real b,Int k,
    bool fast_mode=false,bool full=false)
{
    const Real tolerance=fast_mode?1.E-8:1.E-9;
    if(!full) {
        static thread_local LowestTridiagonal workspace;
        try { return ABS(b*workspace.last_component(ltd,lt_sd,k))<tolerance; }
        catch(const std::runtime_error&) { /* Reliable full-spectrum fallback below. */ }
    }
    Mat<T> ev(ltd.size(),ltd.size(),0.);
    Vec<T> va(ltd),vb(lt_sd);
    const Int info=trd_heevr_qr(va,vb,ev);
    if(info!=0)ERR("Residual tridiagonal eigensolver failed."+NAV(info));
    return ABS(b*ev[k][0])<tolerance;
}

template<typename T>
inline Real compare_error(T& pre, T& lst) { return (2 * ABS(lst - pre)) / (ABS(pre) + ABS(lst) + 1.); }

template<typename T>
void orthogonalization_one_to_multiple(const MyMpi& mm, Vec<T>& v, const VEC<Vec<T>>& m, Int e, Int method=1)
//	orthogonalization v to m[0] through m[e - 1], which are assumed to be orthonormal
{
    if(e==0) return;
    if(method==1) { lanczos_local::project_out(mm,v,m,e,method); return; }
    if(method!=0) throw std::invalid_argument("Invalid Lanczos orthogonalization method");
    Real mag0, mag1;
    mag1 = sqrt(mm.Allreduce(real(lanczos_local::dot(v, v))));		// mag1 != 0 is supposed
    mag0 = mag1 * 1.E+16;
    while (mag1 / mag0 < 0.98) {
        lanczos_timing::count(lanczos_timing::OrthPasses);
        lanczos_timing::count(lanczos_timing::OrthProjections, e);
        for_Int(i, 0, e) lanczos_local::axpy(v,-mm.Allreduce(lanczos_local::dot(m[i], v)),m[i]);
        mag0 = mag1;
        mag1 = sqrt(mm.Allreduce(real(lanczos_local::dot(v, v))));		// mag1 != 0 is supposed
    }
}

template<typename T, typename F>
VecInt lanczos(VecReal& evals, Mat<T>& evecs, Int& gs_dgcy, Idx n, Int evals_size, F& H,
    VecReal& inital_state, const MyMpi& mm, bool fast_mode = false, Int max_krylov = 9999,
    Int reorthogonalization = 0, Int orth_method = 1, bool local_initial = false)
{
    using namespace std;
    namespace lt = lanczos_timing;
    if(reorthogonalization != 0 && reorthogonalization != 1)
        ERR("gs_lanczos_reorthogonalization must be 0 or 1.");
    if(n == 0 || n > static_cast<Idx>(std::numeric_limits<BasisIndex>::max()) ||
        inital_state.size() != (local_initial ? static_cast<Idx>(VecPartition(mm.np(),mm.id(),static_cast<BasisIndex>(n)).len()) : n) || evals_size <= 0 || max_krylov < 0)
        ERR("Invalid ground-state Lanczos dimensions.");
    mm.Bcast(orth_method);
    if(orth_method!=0 && orth_method!=1) throw std::invalid_argument("Invalid lanczos_orth_method");
    int residual_full=0;
    if(mm) {
        const char* value=std::getenv("NORG_RESIDUAL_SOLVER");
        if(value && std::string(value)=="full")residual_full=1;
        else if(value && std::string(value)!="lowest")residual_full=-1;
    }
    mm.Bcast(residual_full);
    if(residual_full<0)throw std::invalid_argument("NORG_RESIDUAL_SOLVER=lowest|full");
    const bool reorthogonalize = reorthogonalization == 1;
    // Read the switch on the master, so differing rank environments cannot hang
    // the final gather. No new collectives are added inside either iteration pass.
    int timing_enabled = mm ? static_cast<int>(lt::requested()) : 0;
    mm.Bcast(timing_enabled);
    lt::ProgressOptions progress;
    if(mm)progress=lt::ProgressOptions::read(mm.np());
    mm.Bcast(progress.seconds);mm.Bcast(progress.rank);
    lt::Session timing(timing_enabled != 0 || progress.selects(mm.id(),mm.ms()));
    static unsigned long long timing_solve = 0;
    const unsigned long long solve_id = ++timing_solve;
    timing.configure_progress(progress,mm.id(),mm.ms(),solve_id,n);
    // Keep the historical 20-step allowance, but never exceed the Hilbert dimension.
    const Idx max_basis = std::min<Idx>(n, static_cast<Idx>(max_krylov) + 21);
    const Real tiny = 64. * eps_Real;
    Int e = 0;
    gs_dgcy = 1;
    VEC<VecReal> evec; // converged eigenstates, NOT the historical Krylov basis
    VEC<Real> eval;
    VEC<Int> Krylovsize;
    VecPartition vp(mm.np(), mm.id(), static_cast<BasisIndex>(n));
    while(e < evals_size && static_cast<Idx>(e) < n) {
        timing.begin_state();
        const char* stop_reason = "max_basis";
        lt::Scope initial_timer(lt::Initial);
        VEC<Real> ltd, lt_sd;
        ltd.reserve(max_basis);
        lt_sd.reserve(max_basis - 1);
        VecReal temp(vp.len());
        if(e == 0) {
            stage2_parallel::ranges(vp.len(),[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto i=first;i<last;++i)temp[i]=inital_state[i+(local_initial?0:vp.bgn())];
            });
        } else {
            // Reusing the first start can miss an orthogonal partner in a degenerate
            // eigenspace. Generate only this rank's slice of a fresh reproducible start.
            preparation_parallel::fill_start(temp,vp.bgn(),e+1);
        }
        if(e > 0) {
            lt::Scope deflation_timer(lt::Deflation);
            orthogonalization_one_to_multiple(mm, temp, evec, e, orth_method);
        }
        Real norm2 = mm.Allreduce(lanczos_local::dot(temp, temp));
        if(!std::isfinite(norm2)) ERR("Non-finite ground-state Lanczos initial norm.");
        // A supplied start can lie entirely in the already found eigenspace.
        // At least one of e+1 coordinate vectors lies outside its e-dimensional span.
        for(Int seed = 0; norm2 <= tiny * tiny && seed <= e; ++seed) {
            temp = 0.;
            if(seed >= vp.bgn() && seed < vp.end()) temp[seed - vp.bgn()] = 1.;
            if(e > 0) {
                lt::Scope deflation_timer(lt::Deflation);
                orthogonalization_one_to_multiple(mm, temp, evec, e, orth_method);
            }
            norm2 = mm.Allreduce(lanczos_local::dot(temp, temp));
        }
        if(!std::isfinite(norm2) || norm2 <= tiny * tiny)
            ERR("Cannot construct a nonzero ground-state Lanczos start.");
        lanczos_local::scale(temp,INV(SQRT(norm2)));

        VEC<VecReal> basis;
        if(reorthogonalize) basis.reserve(max_basis);
        VecReal v0(temp);
        initial_timer.stop();
        timing.iteration(0,"recurrence");
        lt::Scope recurrence_timer(lt::Recurrence);
        VecReal v1(H * v0);
        Real previous_b = 0.;
        VecReal local_moments(2),moments(2);
        for_Idx(k, 0, max_basis) {
            timing.iteration(k,"recurrence");
            if(reorthogonalize) {
                lt::Scope basis_timer(lt::BasisStore);
                basis.push_back(v0);
            }
            {lt::Scope local_timer(lt::LocalDot);
            local_moments[0]=lanczos_local::dot(v1,v0);
            local_moments[1]=lanczos_local::dot(v1,v1);}
            mm.Allreduce_into(local_moments,moments);
            const Real a=moments[0];
            if(!std::isfinite(a)) ERR("Non-finite ground-state Lanczos diagonal.");
            ltd.push_back(a);
            const Real scale = SQRT(moments[1]) + ABS(previous_b);
            Real fused_norm=0.;
            {lt::Scope local_timer(lt::LocalUpdate);
            if(e==0 && !reorthogonalize)fused_norm=lanczos_local::axpy_norm(v1,-a,v0);
            else lanczos_local::axpy(v1,-a,v0);}
            // Deflation against already solved eigenstates is required in BOTH modes.
            if(e > 0) {
                lt::Scope deflation_timer(lt::Deflation);
                orthogonalization_one_to_multiple(mm, v1, evec, e, orth_method);
            }
            if(reorthogonalize) {
                lt::Scope reorth_timer(lt::Reorthogonalization);
                orthogonalization_one_to_multiple(mm, v1, basis, static_cast<Int>(basis.size()), orth_method);
            }
            const Real b = SQRT(mm.Allreduce(e==0 && !reorthogonalize ? fused_norm : lanczos_local::dot(v1,v1)));
            if(!std::isfinite(b)) ERR("Non-finite ground-state Lanczos residual.");
            if(b <= tiny * std::max(Real(1.), scale)) { stop_reason = "breakdown"; break; }
            // b is the trailing residual, not the preceding off-diagonal coefficient.
            if(k >= 60) {
                lt::Scope check_timer(lt::ResidualCheck);
                if(residual_method_for_b(ltd, lt_sd, b, static_cast<Int>(k), fast_mode,residual_full!=0)) {
                    stop_reason = "residual";
                    break;
                }
            }
            if(k == static_cast<Idx>(max_krylov) && mm)
                WRN("Ground-state Lanczos reached max_krylov: " + NAV2(e, k));
            if(k + 1 == max_basis) break;

            lt_sd.push_back(b);
            lanczos_local::scale(v1,INV(b));
            swapall(v0, v1); // exchange buffers without a full-vector temporary
            lanczos_local::apply(H,v0,v1,-b);
            previous_b = b;
        }

        recurrence_timer.stop();
        timing.iteration(ltd.size(),"diagonalize");
        lt::Scope tridiagonal_timer(lt::FinalTridiagonal);
        VecReal va(ltd), vb(lt_sd);
        if(va.size() != vb.size() + 1) ERR("Invalid ground-state tridiagonal dimensions.");
        MatReal ev(va.size(), va.size(), 0.);
        if(va.size() == 1) ev[0][0] = 1.;
        else {
            const Int info = trd_heevr_qr(va, vb, ev);
            if(info != 0) ERR("Ground-state tridiagonal eigensolver failed: " + NAV(info));
        }
        tridiagonal_timer.stop();
        timing.iteration(0,"reconstruction");
        lt::Scope reconstruction_timer(lt::Reconstruction);
        VecReal gs(vp.len(), 0.);
        if(reorthogonalize) {
            // Use the actual reorthogonalized basis; do not replay a different recurrence.
            for_Idx(j, 0, ltd.size()) {
                timing.iteration(j,"reconstruction");
                lt::Scope update_timer(lt::LocalUpdate);
                lanczos_local::axpy(gs,ev[j][0],basis[j]);
            }
            VEC<VecReal>().swap(basis);
        } else {
            // Memory-light second pass. Include EVERY component, including the last one.
            v0 = temp;
            lanczos_local::apply(H,v0,v1);
            lanczos_local::axpy(gs,ev[0][0],v0);
            for_Idx(j, 1, ltd.size()) {
                timing.iteration(j,"reconstruction");
                lanczos_local::axpy(v1,-ltd[j-1],v0);
                if(e > 0) {
                    lt::Scope deflation_timer(lt::Deflation);
                    orthogonalization_one_to_multiple(mm, v1, evec, e, orth_method);
                }
                lanczos_local::scale(v1,INV(lt_sd[j - 1]));
                swapall(v0, v1);
                lanczos_local::axpy(gs,ev[j][0],v0);
                if(j + 1 < ltd.size()) {
                    lanczos_local::apply(H,v0,v1,-lt_sd[j-1]);
                }
            }
        }
        reconstruction_timer.stop();
        lt::Scope normalize_timer(lt::Normalize);
        if(e > 0) {
            lt::Scope deflation_timer(lt::Deflation);
            orthogonalization_one_to_multiple(mm, gs, evec, e, orth_method);
        }
        const Real gs_norm2 = mm.Allreduce(lanczos_local::dot(gs, gs));
        if(!std::isfinite(gs_norm2) || gs_norm2 <= tiny * tiny)
            ERR("Invalid reconstructed ground-state norm.");
        lanczos_local::scale(gs,INV(SQRT(gs_norm2)));
        normalize_timer.stop();
        lt::Scope store_timer(lt::EigenpairStore);
        Krylovsize.push_back(static_cast<Int>(ltd.size()));
        evec.push_back(gs);
        eval.push_back(va[0]);
        ++e;
        store_timer.stop();
        timing.end_state(ltd.size(), stop_reason);
        if(fast_mode) break;
        if(e == evals_size && static_cast<Idx>(e) < n) {
            Int ndeg = 1;
            for_Int(i, 1, static_cast<Int>(eval.size()))
                if(compare_error(eval[0], eval[i]) < 2E-6) ++ndeg;
            if(evals_size == ndeg) ++evals_size;
        }
    }
    timing.iteration(0,"output");
    lt::Scope output_timer(lt::Output);
    evecs.reset(evec.size(), n);
    for_Int(i, 0, static_cast<Int>(evec.size())) evecs[i] = mm.Allgatherv_full(evec[i], vp);
    evals.reset(eval);
    slctsort(evals, evecs);
    for_Int(i, 1, static_cast<Int>(evals.size()))
        if(compare_error(evals[0], evals[i]) < 1E-6) ++gs_dgcy;
    output_timer.stop();
    timing.finish();
    if(timing_enabled) {
        const auto packed = timing.pack();
        VecReal local(packed);
        const VecReal gathered = mm.Gather(local);
        if(mm) cout << timing.report(gathered.p(), mm.np(), n,
            reorthogonalization, fast_mode, solve_id);
    }
    return VecInt(Krylovsize);
}
