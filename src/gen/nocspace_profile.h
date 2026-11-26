#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
class MyMpi;
namespace nocspace_profile {
using Clock = std::chrono::steady_clock;
enum Phase { MemberInit, SetControl, Columns, PruningSetup, WorkerSetup,
    Enumeration, GroupOffsets, StorageAllocation, GroupCopyIndexPrepare,
    IndexRouting, IndexBuild, AppendCleanup, WorkerCleanup, CollectiveSync, OuterCleanup,
    LegacyEnumeration, Other, Total, PhaseCount };
struct Report {
    bool enabled = false;
    bool active = false;
    std::array<double, PhaseCount> wall{};
    std::uint64_t raw_candidates = 0;
    std::chrono::system_clock::time_point begin_date{}, end_date{};
    Clock::time_point begin{}, last{};
    Report() = default;
    explicit Report(const MyMpi& mm);
    void mark(Phase phase);
    void finish();
    double total() const { return wall[Total]; }
};
void emit(const MyMpi& mm, const Report& report, std::size_t dimension,
    std::size_t groups, const char* source);
} // namespace nocspace_profile
