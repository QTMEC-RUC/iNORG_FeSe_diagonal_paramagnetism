#include "nocspace_profile.h"
#include "nocspace_timing.h"
#include "mympi.h"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace nocspace_profile {
Report::Report(const MyMpi& mm)
{
    enabled = nocspace_timing::enabled_for(true);
    // Also exclude another rank's preceding diagnostic output.
    mm.barrier();
    begin_date = std::chrono::system_clock::now();
    begin = last = Clock::now();
    active = true;
}
void Report::mark(Phase phase)
{
    if (!active) return;
    const auto now = Clock::now();
    wall[phase] += std::chrono::duration<double>(now-last).count();
    last = now;
}
void Report::finish()
{
    if (!active) return;
    mark(Other);
    wall[Total] = std::chrono::duration<double>(last-begin).count();
    end_date = std::chrono::system_clock::now();
    active = false;
}
static std::string timestamp(std::chrono::system_clock::time_point point)
{
    const auto whole=std::chrono::time_point_cast<std::chrono::seconds>(point);
    const auto fraction=std::chrono::duration_cast<std::chrono::microseconds>(point-whole).count();
    const std::time_t raw=std::chrono::system_clock::to_time_t(whole);
    std::tm local{};localtime_r(&raw,&local);
    std::ostringstream out;
    out<<std::put_time(&local,"%Y-%m-%dT%H:%M:%S")<<'.'<<std::setw(6)<<std::setfill('0')<<fraction<<std::put_time(&local,"%z");
    return out.str();
}
void emit(const MyMpi& mm, const Report& r, std::size_t dimension,
    std::size_t groups, const char* source)
{
    // Everything here, including MPI aggregation, happens after finish().
    const int enabled=mm.Allreduce(r.enabled ? 1:0);
    if (!enabled) return;
    if (enabled!=mm.np()) throw std::runtime_error("NORG_NOC_TIMING differs between ranks");
    const int width=PhaseCount+1;
    VecReal local(width,0.);
    for (int p=0;p<PhaseCount;++p)local[p]=r.wall[p];
    local[PhaseCount]=static_cast<double>(r.raw_candidates);
    const auto values=mm.Gather(local);
    if (!mm) return;
    static std::uint64_t build=0;++build;
    static const char* names[]={"member_initialization","set_control","columns",
        "pruning_setup","worker_setup","candidate_enumeration","group_offsets",
        "storage_allocation","group_copy_index_prepare","index_routing","index_build","append_cleanup",
        "worker_cleanup","collective_sync","outer_cleanup","legacy_enumeration","other","total"};
    std::ostringstream out;out<<std::fixed<<std::setprecision(6);
    out<<"NOCSPACE_PROFILE: build="<<build<<" source="<<source<<" dim="<<dimension
       <<" groups="<<groups<<" ranks="<<mm.np()<<" clock=steady_wall reporting_excluded=1"
       <<" instrumentation_overhead_included=1 index_backend=sharded_hash build_begin_time="<<timestamp(r.begin_date)
       <<" build_end_time="<<timestamp(r.end_date)<<'\n';
    for (int p=0;p<PhaseCount;++p) {
        double lo=values[p],hi=lo,sum=0.;int max_rank=0;
        for(int rank=0;rank<mm.np();++rank){
            const double v=values[rank*width+p];sum+=v;lo=std::min(lo,v);
            if(v>hi){hi=v;max_rank=rank;}
        }
        out<<"NOCSPACE_PROFILE: build="<<build<<" stage="<<names[p]
           <<" measurement=rank_wall parent="<<(p==Total ? "none":"total")
           <<" min_s="<<lo<<" mean_s="<<sum/mm.np()<<" max_s="<<hi<<" max_rank="<<max_rank<<'\n';
    }
    std::cout<<out.str()<<std::flush;
}
} // namespace nocspace_profile
