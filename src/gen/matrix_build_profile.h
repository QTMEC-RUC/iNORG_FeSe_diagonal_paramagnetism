#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
class MyMpi;
namespace matrix_build_profile {
using Clock=std::chrono::steady_clock;
enum Phase { Setup, Validation, Diagonal, RowCounts, CsrRowAllocation,
    CsrOffsets, CsrValuesAllocation, CsrLegacyRelease, CsrFill, CollectiveSync,
    StatePrepare, DensitySetup, DensityDiagonal, DensityOffDiagonal,
    LocalReduce, Allreduce, Unpack, Check, Cleanup, Other, Total, PhaseCount };
struct Options { bool enabled=true; int sample_every=4096; };
Options options(const MyMpi&,const char* enabled_env,const char* sample_env);
struct Report {
    bool enabled;
    bool active=true;
    std::array<double,PhaseCount> wall{};
    std::array<bool,PhaseCount> used{};
    Clock::time_point begin{},last{};
    std::chrono::system_clock::time_point begin_date{},end_date{};
    explicit Report(const MyMpi&,bool enabled);
    void mark(Phase);
    void finish();
    double total() const { return wall[Total]; }
};
void emit(const MyMpi&,const Report&,const char* prefix,const char* path,
    std::size_t dim,std::size_t rows,std::size_t entries,std::uint64_t build);
// Samples are per-worker elapsed work, NOT wall times of the parallel phase.
enum Detail { Row, GroupLookup, StateDecode, OneBody, TwoBody, RowStore, DetailCount };
struct Worker {
    bool enabled;
    int every;
    std::uint64_t random;
    std::size_t rows=0,hits=0;
    std::array<double,DetailCount> seconds{};
    Worker(bool on,int interval,std::uint64_t seed):enabled(on),every(interval),
        random(seed ^ UINT64_C(0x9e3779b97f4a7c15)) { if(!random)random=1; }
    std::uint64_t next() {random^=random<<13;random^=random>>7;random^=random<<17;return random;}
    template<class F> void traverse(std::size_t count,F visit) {
        rows=count;
        if(!enabled || !every) {
            for(std::size_t i=0;i<count;++i)visit(i,std::false_type{});
            return;
        }
        std::size_t sampled=next()%every,i=0;
        while(i<count) {
            for(;i<count && i<sampled;++i)visit(i,std::false_type{});
            if(i==count)break;
            ++hits;visit(i++,std::true_type{});
            sampled=i+next()%(2ULL*every-1);
        }
    }
};
template<bool Selected> class Sample {
public:
    Sample(Worker&,Detail) {}
    void stop() {}
};
template<> class Sample<true> {
    double* value_;
    Clock::time_point begin_;
public:
    Sample(Worker& worker,Detail detail):value_(&worker.seconds[detail]),begin_(Clock::now()){}
    Sample(const Sample&)=delete;
    ~Sample(){stop();}
    void stop(){if(value_){*value_+=std::chrono::duration<double>(Clock::now()-begin_).count();value_=nullptr;}}
};
void emit_samples(const MyMpi&,const std::vector<Worker>&,std::uint64_t build,int interval);
} // namespace matrix_build_profile
