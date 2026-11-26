#pragma once
#include "basis_index.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <string>
class MyMpi;

// No output, shared counters, or MPI calls in a timed worker region.
namespace tab_timing {
using Clock = std::chrono::steady_clock;
using Point = Clock::time_point;
inline Point tick(bool enabled) { return enabled ? Clock::now() : Point{}; }
inline double since(Point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
class Span {
    double* seconds_;
    Point start_;
public:
    explicit Span(double* seconds): seconds_(seconds), start_(tick(seconds != nullptr)) {}
    ~Span() { if(seconds_) *seconds_ += since(start_); }
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
};
enum Detail { GroupRefresh, Constraints, TargetLookup, Reserve, StateDecode,
    EntryGenerate, StateDestroy, DetailCount };
inline const char* detail_name(Detail d) {
    static const char* names[] = {"group_refresh", "candidate_constraints", "target_lookup",
        "capacity_reserve", "state_decode", "entry_generate", "state_destroy"};
    return names[d];
}
inline bool group_detail(Detail d) { return d <= Reserve; }
struct Worker {
    std::uint64_t stride = 1, groups = 0, sampled_groups = 0, rows = 0, sampled_rows = 0;
    std::array<double, DetailCount> seconds{};
    bool sample_group() {
        const bool selected = groups++ % stride == 0;
        sampled_groups += selected;
        return selected;
    }
    bool sample_row() {
        const bool selected = rows++ % stride == 0;
        sampled_rows += selected;
        return selected;
    }
    double estimate(Detail d) const {
        const auto population = group_detail(d) ? groups : rows;
        const auto samples = group_detail(d) ? sampled_groups : sampled_rows;
        return samples ? seconds[d] * static_cast<double>(population) / samples : 0.;
    }
};
struct Merge { double allocation_s = 0., copy_release_s = 0.; };
struct Report {
    bool enabled = true, config_fallback = false;
    std::uint64_t sample_every = 1024; // Configurable sampling cadence, not a hardware size.
    double total_s = 0., index_build_s = 0., workers_s = 0., merge_s = 0., index_destroy_s = 0.;
    std::chrono::system_clock::time_point wall_begin{}, wall_end{};
    Merge merge;
    std::vector<Worker> workers;
    Report() {
        if(const char* value = std::getenv("NORG_TAB_TIMING")) enabled = *value != '0';
        if(const char* value = std::getenv("NORG_TAB_SAMPLE_EVERY")) {
            char* end = nullptr; errno = 0;
            const auto parsed = std::strtoull(value, &end, 10);
            if(*value && *value != '-' && end && !*end && !errno && parsed) sample_every = parsed;
            else config_fallback = true;
        }
    }
};
void emit(const MyMpi& mm, const Report& report, BasisIndex dim,
    const char* scope, const std::string& context, bool two_fermion);
} // namespace tab_timing
