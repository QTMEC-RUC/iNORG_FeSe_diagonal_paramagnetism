#pragma once
#include "mympi.h"
#include "batched_orthogonalization.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// Aggregate only: no output, allocations or MPI calls in timed iteration scopes.
namespace green_profile {
using Clock=std::chrono::steady_clock;
enum Stage { Setup, Source, Reference, Krylov, Seed, Hv, Orth, ProjectionCheck,
    Diagonalize, Amplitudes, SpectralCheck, Frequency, FiniteCheck, Gather,
    Nocspace, Table, Coefficients, Hamiltonian, Release, StatePrep, Ensemble,
    Monitor, Convergence, Count };
inline const char* name(Stage stage) {
    static const char* names[]={"setup","source_projection","reference_moments","krylov",
        "scalar_seed","h_apply","orthogonalization","projection_check","diagonalization",
        "spectral_amplitudes","spectral_check","frequency_sum","frequency_finite_check",
        "frequency_gather_commit","nocspace","operator_table","h_coefficients","hamiltonian",
        "table_release","state_preparation","ensemble_accumulate","legacy_monitor",
        "convergence_check"};
    static_assert(sizeof(names)/sizeof(names[0])==Count,"GF stage/name mismatch");
    return names[stage];
}
// Once per process, outside every measured algorithm. This is a cost estimate,
// not a correction to measured stage times or a hardware performance promise.
struct ProbeCalibration {
    double pair_accumulate_seconds;
    ProbeCalibration():pair_accumulate_seconds(0.) {
        constexpr Int samples=2048;
        volatile double accumulated=0.;
        const auto begin=Clock::now();
        for(Int i=0;i<samples;++i) {
            const auto first=Clock::now();
            const double seconds=std::chrono::duration<double>(Clock::now()-first).count();
            accumulated=accumulated+seconds;
        }
        pair_accumulate_seconds=std::chrono::duration<double>(Clock::now()-begin).count()/samples;
        (void)accumulated;
    }
};
inline const ProbeCalibration& probe_calibration() {
    static const ProbeCalibration value;return value;
}
class Profile {
    bool enabled_;
    Clock::time_point start_;
    std::array<double,Count> seconds_{},calls_{};
    double initialization_seconds_=0.,probe_pair_seconds_=0.,finish_io_seconds_=0.;
public:
    const Str context;
    const char* scope;
    Int solver;
    Profile(const MyMpi& mm,const Str& ctx,const char* kind,Int mode)
        : enabled_(false),context(ctx),scope(kind),solver(mode) {
        const auto init_start=Clock::now();
        Int flag=1;
        if(mm) {
            const char* value=std::getenv("NORG_GF_PROFILE");
            if(value && std::string(value)=="0")flag=0;
            else if(value && std::string(value)!="1")flag=-1;
        }
        mm.Bcast(flag);
        if(flag<0)throw std::invalid_argument("NORG_GF_PROFILE must be 0 or 1");
        enabled_=flag!=0;
        if(enabled_) {
            probe_pair_seconds_=probe_calibration().pair_accumulate_seconds;
            start_=Clock::now();
            initialization_seconds_=std::chrono::duration<double>(start_-init_start).count();
        }
    }
    bool enabled() const {return enabled_;}
    void add(Stage stage,double seconds) {seconds_[stage]+=seconds;calls_[stage]+=1.;}
    // Available after finish; includes the final diagnostic line and its flush.
    // The value is local to the rank. Timing it needs no additional collective.
    double finish_io_seconds() const {return finish_io_seconds_;}
    double stage_seconds(Stage stage) const {return seconds_[stage];}
    double stage_calls(Stage stage) const {return calls_[stage];}
    void finish(const MyMpi& mm,const std::string& checks="") {
        if(!enabled_) {if(mm && !checks.empty())std::cout<<checks;return;}
        const auto summary_start=Clock::now();
        // One fixed-size gather per completed scope, never per iteration/stage.
        constexpr Int header=3,fields=header+2*Count;
        std::array<double,fields> local{};
        local[0]=std::chrono::duration<double>(summary_start-start_).count();
        local[1]=initialization_seconds_;local[2]=probe_pair_seconds_;
        for(Int i=0;i<Count;++i){local[header+2*i]=seconds_[i];local[header+2*i+1]=calls_[i];}
        VEC<double> all(mm?fields*mm.np():0);
        MPI_Gather(local.data(),fields,MPI_DOUBLE,mm?all.data():nullptr,fields,MPI_DOUBLE,mm.ms(),mm.cm());
        if(!mm) {finish_io_seconds_=std::chrono::duration<double>(Clock::now()-summary_start).count();return;}
        std::ostringstream out;out<<std::scientific<<std::setprecision(9);
        double total_max=0.,init_max=0.,probe_estimate_max=0.,probe_pairs_max=0.,pair_max=0.;
        for(Int rank=0;rank<mm.np();++rank) {
            const double* row=all.data()+rank*fields;
            total_max=std::max(total_max,row[0]);init_max=std::max(init_max,row[1]);
            double pairs=0.;for(Int stage=0;stage<Count;++stage)pairs+=row[header+2*stage+1];
            probe_pairs_max=std::max(probe_pairs_max,pairs);
            pair_max=std::max(pair_max,row[2]);
            probe_estimate_max=std::max(probe_estimate_max,pairs*row[2]);
        }
        out<<"GF_PROFILE scope="<<scope<<" "<<context<<" requested_solver="<<solver
            <<" ranks="<<mm.np()<<" wall_max_s="<<total_max<<" initialization_max_s="<<init_max
            <<" clock=steady_wall inclusive=1 summary_excluded=1 initialization_excluded=1\n";
        for(Int stage=0;stage<Count;++stage) {
            double lo=std::numeric_limits<double>::infinity(),hi=0.,sum=0.,calls=0.;
            for(Int rank=0;rank<mm.np();++rank) {
                const double value=all[rank*fields+header+2*stage];
                lo=std::min(lo,value);hi=std::max(hi,value);sum+=value;
                calls=std::max(calls,all[rank*fields+header+2*stage+1]);
            }
            if(calls==0.)continue;
            out<<"GF_STAGE scope="<<scope<<" "<<context<<" requested_solver="<<solver
                <<" stage="<<name(static_cast<Stage>(stage))<<" min_s="<<lo
                <<" mean_s="<<sum/mm.np()<<" max_s="<<hi<<" calls_max="<<calls
                <<" validation="<<((stage==Reference || stage==Krylov || stage==ProjectionCheck
                    || stage==Diagonalize || stage==Amplitudes || stage==SpectralCheck
                    || stage==Source || stage==Frequency || stage==FiniteCheck || stage==Gather)
                    ?"SEE_GF_CHECK":"NOT_CHECKED")<<'\n';
        }
        if(std::string(scope)=="solve") {
            double algorithm=0.,audit=0.;
            for(Int rank=0;rank<mm.np();++rank) {
                const double* row=all.data()+rank*fields+header;
                algorithm=std::max(algorithm,row[2*Krylov]+row[2*Diagonalize]+row[2*Amplitudes]+row[2*Seed]);
                audit=std::max(audit,row[2*Reference]+row[2*ProjectionCheck]+row[2*SpectralCheck]);
            }
            out<<"GF_COST "<<context<<" requested_solver="<<solver<<" algorithm_max_s="<<algorithm
                <<" selfcheck_max_s="<<audit<<" clock_pairs_max="<<probe_pairs_max
                <<" algorithm_stages=krylov+diagonalization+spectral_amplitudes+scalar_seed\n";
        }
        out<<checks;
        const std::string text=out.str();std::cout.write(text.data(),text.size());std::cout.flush();
        const double summary_s=std::chrono::duration<double>(Clock::now()-summary_start).count();
        std::ostringstream cost;cost<<std::scientific<<std::setprecision(9)
            <<"GF_PROFILE_IO scope="<<scope<<" "<<context<<" root_summary_s="<<summary_s
            <<" fraction_of_wall_max="<<(total_max>0.?summary_s/total_max:0.)
            <<" clock_pairs_max="<<probe_pairs_max<<" clock_pair_estimate_max_s="<<pair_max
            <<" clock_probe_estimate_max_s="<<probe_estimate_max
            <<" clock_probe_fraction_estimate="<<(total_max>0.?probe_estimate_max/total_max:0.)
            <<" initialization_max_s="<<init_max
            <<" includes=gather_format_write_flush excludes=this_cost_line"
            <<" probe_cost=calibrated_estimate full_output_cost=end_to_end_benchmark\n";
        const std::string cost_text=cost.str();std::cout.write(cost_text.data(),cost_text.size());std::cout.flush();
        finish_io_seconds_=std::chrono::duration<double>(Clock::now()-summary_start).count();
    }
};
class Scope {
    Profile* profile_;Stage stage_;Clock::time_point start_;
public:
    Scope(Profile* p,Stage s):profile_(p && p->enabled()?p:nullptr),stage_(s) {if(profile_)start_=Clock::now();}
    ~Scope(){stop();}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
    void stop(){if(profile_){profile_->add(stage_,std::chrono::duration<double>(Clock::now()-start_).count());profile_=nullptr;}}
};
template<class H> void apply(Profile* p,const H& h,const VecReal& x,VecReal& y,Real beta=0.) {
    Scope timer(p,Hv);lanczos_local::apply(h,x,y,beta);
}
inline VecReal project(Profile* p,const MyMpi& mm,VecReal& v,const VEC<VecReal>& q,Int n,Int method) {
    Scope timer(p,Orth);return lanczos_local::project_out(mm,v,q,n,method);
}
} // namespace green_profile
