#include "matrix_build_profile.h"
#include "mympi.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
namespace matrix_build_profile {
Options options(const MyMpi& mm,const char* enabled_env,const char* sample_env) {
    VecInt values(2);Options result;
    if(mm) {
        const char* on=std::getenv(enabled_env);
        result.enabled=!on || std::string(on)!="0";
        if(const char* text=std::getenv(sample_env)) {
            char* end=nullptr;errno=0;const long parsed=std::strtol(text,&end,10);
            if(*text && !errno && end && !*end && parsed>=0 && parsed<=std::numeric_limits<int>::max())
                result.sample_every=static_cast<int>(parsed);
        }
        values[0]=result.enabled;values[1]=result.sample_every;
    }
    mm.Bcast(values);result.enabled=values[0]!=0;result.sample_every=values[1];return result;
}
Report::Report(const MyMpi& mm,bool on):enabled(on) {
    // Exclude prior diagnostics on every rank; no MPI inside worker timers.
    mm.barrier();begin_date=std::chrono::system_clock::now();begin=last=Clock::now();
}
void Report::mark(Phase phase) {
    if(!active)return;
    const auto now=Clock::now();wall[phase]+=std::chrono::duration<double>(now-last).count();
    used[phase]=true;last=now;
}
void Report::finish() {
    if(!active)return;
    mark(Other);wall[Total]=std::chrono::duration<double>(last-begin).count();used[Total]=true;
    end_date=std::chrono::system_clock::now();active=false;
}
static std::string timestamp(std::chrono::system_clock::time_point point) {
    const auto whole=std::chrono::time_point_cast<std::chrono::seconds>(point);
    const auto us=std::chrono::duration_cast<std::chrono::microseconds>(point-whole).count();
    const auto raw=std::chrono::system_clock::to_time_t(whole);std::tm local{};localtime_r(&raw,&local);
    std::ostringstream out;out<<std::put_time(&local,"%Y-%m-%dT%H:%M:%S")<<'.'
        <<std::setw(6)<<std::setfill('0')<<us<<std::put_time(&local,"%z");return out.str();
}
void emit(const MyMpi& mm,const Report& r,const char* prefix,const char* path,
    std::size_t dim,std::size_t rows,std::size_t entries,std::uint64_t build) {
    if(!r.enabled)return;
    if(r.active)throw std::logic_error("Stop matrix build clock before reporting");
    constexpr int width=2*PhaseCount+2;
    VecReal local(width,0.);
    for(int s=0;s<PhaseCount;++s){local[s]=r.wall[s];local[PhaseCount+s]=r.used[s];}
    local[2*PhaseCount]=rows;local[2*PhaseCount+1]=entries;
    const auto gathered=mm.Gather(local);if(!mm)return;
    static const char* names[]={"setup","validation","diagonal","row_counts","csr_row_allocation",
        "csr_offsets","csr_values_allocation","csr_legacy_release","csr_fill","collective_sync",
        "state_prepare","density_setup","diagonal","offdiagonal","thread_reduce","mpi_allreduce",
        "result_unpack","check","cleanup","other","total"};
    double row_sum=0,entry_sum=0;
    for(int rank=0;rank<mm.np();++rank){row_sum+=gathered[rank*width+2*PhaseCount];entry_sum+=gathered[rank*width+2*PhaseCount+1];}
    std::ostringstream out;out<<std::fixed<<std::setprecision(6);
    out<<prefix<<": build="<<build<<" path="<<path<<" dim="<<dim<<" ranks="<<mm.np()
        <<" rows_sum="<<row_sum<<" tab_entries_sum="<<entry_sum
        <<" clock=steady_wall reporting_excluded=1 instrumentation_overhead_included=1"
        <<" build_begin_time="<<timestamp(r.begin_date)<<" build_end_time="<<timestamp(r.end_date)<<'\n';
    for(int s=0;s<PhaseCount;++s) {
        double lo=gathered[s],hi=lo,sum=0.;int max_rank=0;bool used=false;
        for(int rank=0;rank<mm.np();++rank) {
            const double v=gathered[rank*width+s];sum+=v;lo=std::min(lo,v);
            if(v>hi){hi=v;max_rank=rank;}used=used || gathered[rank*width+PhaseCount+s]!=0.;
        }
        if(!used)continue;
        out<<prefix<<": build="<<build<<" stage="<<names[s]<<" measurement=rank_wall parent="
            <<(s==Total?"none":"total")<<" min_s="<<lo<<" mean_s="<<sum/mm.np()
            <<" max_s="<<hi<<" max_rank="<<max_rank<<'\n';
    }
    std::cout<<out.str()<<std::flush;
}
void emit_samples(const MyMpi& mm,const std::vector<Worker>& workers,std::uint64_t build,int interval) {
    constexpr int width=2*DetailCount+3;
    VecReal local(width,0.);local[width-1]=1.;
    for(const auto& worker:workers) {
        local[width-3]+=worker.rows;local[width-2]+=worker.hits;
        if(worker.rows && !worker.hits)local[width-1]=0.;
        for(int d=0;d<DetailCount;++d) {
            local[DetailCount+d]+=worker.seconds[d];
            if(worker.hits)local[d]+=worker.seconds[d]*double(worker.rows)/worker.hits;
        }
    }
    const auto all=mm.Gather(local);if(!mm)return;
    static const char* names[]={"sample_row","group_lookup","state_decode","one_body_diagonal","two_body_diagonal","row_store"};
    std::ostringstream out;out<<std::fixed<<std::setprecision(6);
    for(int d=0;d<DetailCount;++d) {
        double lo=std::numeric_limits<double>::infinity(),hi=-1,sum=0,sampled=0,hits=0;int valid=0;
        for(int rank=0;rank<mm.np();++rank) {
            const auto base=rank*width;sampled+=all[base+DetailCount+d];hits+=all[base+width-2];
            if(!all[base+width-1])continue;
            const auto v=all[base+d];lo=std::min(lo,v);hi=std::max(hi,v);sum+=v;++valid;
        }
        out<<"H_DETAIL: build="<<build<<" stage="<<names[d]
            <<" measurement=sampled_thread_work not_wall_time=1 parent="<<(d==Row?"diagonal":"sample_row")
            <<" sample_every="<<interval<<" valid_ranks="<<valid<<'/'<<mm.np()<<" min_s=";
        if(valid)out<<lo;else out<<"NA";
        out<<" mean_valid_s=";if(valid)out<<sum/valid;else out<<"NA";
        out<<" max_s=";if(valid)out<<hi;else out<<"NA";
        out<<" sampled_seconds_sum="<<sampled<<" sampled_rows_sum="<<hits<<'\n';
    }
    std::cout<<out.str()<<std::flush;
}
} // namespace matrix_build_profile
