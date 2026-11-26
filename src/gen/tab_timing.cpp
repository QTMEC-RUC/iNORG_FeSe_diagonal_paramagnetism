#include "tab_timing.h"
#include "mympi.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <ctime>

namespace tab_timing {
static std::string timestamp(std::chrono::system_clock::time_point point)
{
    const auto whole=std::chrono::time_point_cast<std::chrono::seconds>(point);
    const auto fraction=std::chrono::duration_cast<std::chrono::microseconds>(point-whole).count();
    const std::time_t raw=std::chrono::system_clock::to_time_t(whole);
    std::tm local{};
    localtime_r(&raw,&local);
    std::ostringstream out;
    out<<std::put_time(&local,"%Y-%m-%dT%H:%M:%S")<<'.'<<std::setw(6)<<std::setfill('0')<<fraction
       <<std::put_time(&local,"%z");
    return out.str();
}
void emit(const MyMpi& mm, const Report& r, BasisIndex dim,
    const char* scope, const std::string& context, bool two_fermion)
{
    // Called only AFTER the build timer and the existing TAB_AFTER event.
    // Gather, formatting, output and flushing all stay outside the measured build.
    const int active = mm.Allreduce(r.enabled ? 1 : 0);
    if (!active) return;
    if (active != mm.np()) throw std::runtime_error("NORG_TAB_TIMING differs between MPI ranks");
    enum Phase { Total, IndexBuild, Workers, MergeTotal, MergeAllocation,
        MergeCopyRelease, IndexDestroy, Other, PhaseCount };
    const char* names[] = {"total", "index_build", "parallel_build", "merge_total",
        "merge_allocation", "merge_copy_release", "index_destroy", "other"};
    enum Count { Groups, SampleGroups, Rows, SampleRows, Stride, Fallback, CountCount };
    const int width = PhaseCount + CountCount + 2 * DetailCount;
    VecReal local(width, 0.);
    local[Total]=r.total_s; local[IndexBuild]=r.index_build_s; local[Workers]=r.workers_s;
    local[MergeTotal]=r.merge_s; local[MergeAllocation]=r.merge.allocation_s;
    local[MergeCopyRelease]=r.merge.copy_release_s; local[IndexDestroy]=r.index_destroy_s;
    local[Other]=std::max(0.,r.total_s-r.index_build_s-r.workers_s-r.merge_s-r.index_destroy_s);
    local[PhaseCount+Stride]=static_cast<double>(r.sample_every);
    local[PhaseCount+Fallback]=r.config_fallback ? 1. : 0.;
    for(const auto& worker:r.workers) {
        local[PhaseCount+Groups]+=worker.groups;
        local[PhaseCount+SampleGroups]+=worker.sampled_groups;
        local[PhaseCount+Rows]+=worker.rows;
        local[PhaseCount+SampleRows]+=worker.sampled_rows;
        for(int d=0;d<DetailCount;++d) {
            local[PhaseCount+CountCount+2*d]+=worker.seconds[d];
            local[PhaseCount+CountCount+2*d+1]+=worker.estimate(static_cast<Detail>(d));
        }
    }
    const auto values=mm.Gather(local);
    if(!mm) return;
    static std::uint64_t next_build=0;
    const auto build=++next_build;
    const auto sum=[&](int column) {
        double value=0.;for(int rank=0;rank<mm.np();++rank)value+=values[rank*width+column];return value;
    };
    const auto minimum=[&](int column) {
        double value=values[column];for(int rank=1;rank<mm.np();++rank)value=std::min(value,values[rank*width+column]);return value;
    };
    const auto maximum=[&](int column) {
        double value=values[column];for(int rank=1;rank<mm.np();++rank)value=std::max(value,values[rank*width+column]);return value;
    };
    std::ostringstream out;
    out<<std::fixed<<std::setprecision(6);
    out<<"TAB_TIMING: build="<<build<<" scope="<<scope<<" dim="<<dim<<" ranks="<<mm.np()
       <<" path="<<(two_fermion ? "two_fermion" : "rotated_fallback")
       <<" clock=steady_wall reporting_excluded=1 instrumentation_overhead_included=1"
       <<" build_begin_time="<<timestamp(r.wall_begin)<<" build_end_time="<<timestamp(r.wall_end)
       <<" sample_every_min="<<minimum(PhaseCount+Stride)<<" sample_every_max="<<maximum(PhaseCount+Stride)
       <<" config_fallback_ranks="<<sum(PhaseCount+Fallback)
       <<(context.empty() ? "" : " ")<<context<<'\n';
    for(int phase=0;phase<PhaseCount;++phase) {
        const char* parent=(phase==MergeAllocation || phase==MergeCopyRelease) ? "merge_total" : (phase==Total ? "none" : "total");
        out<<"TAB_TIMING: build="<<build<<" stage="<<names[phase]<<" measurement=rank_wall"
           <<" parent="<<parent<<" min_s="<<minimum(phase)<<" mean_s="<<sum(phase)/mm.np()
           <<" max_s="<<maximum(phase)<<'\n';
    }
    if(two_fermion) for(int d=0;d<DetailCount;++d) {
        const auto detail=static_cast<Detail>(d);
        const bool grouped=group_detail(detail);
        const char* parent=(detail==Constraints || detail==TargetLookup) ? "group_refresh" : "parallel_build";
        out<<"TAB_TIMING: build="<<build<<" stage="<<detail_name(detail)
           <<" measurement=sampled_thread_work parent="<<parent
           <<" sample_unit="<<(grouped ? "group_refresh" : "basis_row")
           <<" population="<<sum(PhaseCount+(grouped ? Groups : Rows))
           <<" samples="<<sum(PhaseCount+(grouped ? SampleGroups : SampleRows))
           <<" measured_thread_work_s="<<sum(PhaseCount+CountCount+2*d)
           <<" estimated_thread_work_s="<<sum(PhaseCount+CountCount+2*d+1)
           <<" not_wall_time=1\n";
    }
    std::cout<<out.str()<<std::flush;
}
} // namespace tab_timing
