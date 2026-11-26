#pragma once
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lanczos_timing {
struct ProgressOptions {
    double seconds=30.;
    int rank=-2; // -2: communicator master; -1: every rank; otherwise selected rank
    bool selects(int id,int master) const {
        return seconds>0. && (rank==-1 || id==(rank==-2?master:rank));
    }
    static ProgressOptions read(int ranks) {
        ProgressOptions o;
        if(const char* value=std::getenv("NORG_LANCZOS_PROGRESS_SECONDS")) {
            char* end=nullptr;errno=0;o.seconds=std::strtod(value,&end);
            if(errno || end==value || *end || !std::isfinite(o.seconds) || o.seconds<0.)
                throw std::invalid_argument("NORG_LANCZOS_PROGRESS_SECONDS must be finite and nonnegative; 0 disables live output");
        }
        if(const char* value=std::getenv("NORG_LANCZOS_PROGRESS_RANK")) {
            const std::string text(value);
            if(text=="master")o.rank=-2;
            else if(text=="all")o.rank=-1;
            else {
                char* end=nullptr;errno=0;long rank=std::strtol(value,&end,10);
                if(errno || end==value || *end || rank<0 || rank>=ranks || rank>INT_MAX)
                    throw std::invalid_argument("NORG_LANCZOS_PROGRESS_RANK must be master, all, or a valid communicator rank");
                o.rank=static_cast<int>(rank);
            }
        }
        return o;
    }
};
// Called once per selected rank/solve, never in the iteration hot path.
// A main thread restricted to one CPU can be normal OpenMP binding; report facts,
// not an automatic diagnosis. omp_places helps distinguish a single available place.
inline std::string affinity_description() {
    std::string cpus="unavailable";
#ifdef __linux__
    std::ifstream file("/proc/self/status");std::string line;
    while(std::getline(file,line))if(line.compare(0,18,"Cpus_allowed_list:")==0) {
        cpus=line.substr(18);
        cpus.erase(0,cpus.find_first_not_of(" \t"));break;
    }
#endif
    std::ostringstream out;out<<" main_cpu_list="<<cpus;
#ifdef _OPENMP
    out<<" omp_max_threads="<<omp_get_max_threads()<<" omp_places="<<omp_get_num_places();
#endif
    return out.str();
}

class ProgressWriter {
    using Clock=std::chrono::steady_clock;
    bool enabled_=false;
    int rank_=0;
    unsigned long long solve_=0;
    double interval_=0.;
    Clock::time_point start_{},last_report_{};
public:
    bool enabled() const {return enabled_;}
    void configure(const ProgressOptions& o,int rank,int master,unsigned long long solve) {
        enabled_=o.selects(rank,master);rank_=rank;solve_=solve;interval_=o.seconds;
        if(enabled_) {start_=Clock::now();reschedule(start_);}
    }
    void reschedule(Clock::time_point now) {
        // Compare elapsed double seconds; avoid overflow converting user seconds
        // into the clock's integral duration.
        last_report_=now;
    }
    bool interval_due(Clock::time_point now) const {
        return enabled_ && std::chrono::duration<double>(now-last_report_).count()>=interval_;
    }
    double elapsed(Clock::time_point now) const {return std::chrono::duration<double>(now-start_).count();}
    void event(const char* event,int state,std::size_t step,const char* pass,
               const char* stage,double stage_seconds,Clock::time_point now,const std::string& extra="") const {
        if(!enabled_)return;
        std::ostringstream out;
        out<<std::fixed<<std::setprecision(6)<<"LANCZOS_PROGRESS: solve="<<solve_<<" rank="<<rank_
           <<" state="<<state<<" k="<<step<<" pass="<<pass<<" event="<<event<<" stage="<<stage
           <<" wall_s="<<elapsed(now)<<" stage_s="<<stage_seconds
           <<" local_only=1"<<extra<<'\n';
        std::cout<<out.str()<<std::flush;
    }
    void snapshot(const std::string& text) const {if(enabled_)std::cout<<text<<std::flush;}
    int rank() const{return rank_;}
    unsigned long long solve() const{return solve_;}
};
} // namespace lanczos_timing
