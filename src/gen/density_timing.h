#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Exact outer phases, sampled hot-loop details. No MPI or I/O in this helper.
namespace density_timing {
using Clock = std::chrono::steady_clock;
enum Stage {
    Total, StatePrepare, Setup, Diagonal, OffDiagonal, Reduce, ReduceCall, Check,
    Row, RowBuffers, GroupLookup, StateDecode, DiagonalAdd,
    Entry, Filter, OrbitalDecode, OffDiagonalAdd, Count
};
inline const char* name(int s)
{
    static const char* names[] = {"build", "state_prepare", "setup", "diagonal",
        "offdiagonal", "reduce_D", "allreduce_D_call", "check_D", "sample_row", "row_buffers",
        "group_lookup", "state_decode", "diagonal_add", "sample_entry",
        "table_filter", "orbital_decode", "offdiagonal_match_add"};
    return names[s];
}
inline bool requested()
{
    const char* s = std::getenv("NORG_DENSITY_TIMING");
    return !s || std::string(s) != "0";
}
inline int sample_interval()
{
    const char* s = std::getenv("NORG_DENSITY_TIMING_SAMPLE_EVERY");
    if(!s || !*s) return 4096;
    int n = 0;
    for(; *s; ++s) {
        if(*s < '0' || *s > '9' || n > (1000000000 - (*s - '0')) / 10) return 4096;
        n = n * 10 + (*s - '0');
    }
    return n; // 0: exact outer phases only; 1: every event (diagnostic mode).
}
struct Bucket { double seconds = 0., hits = 0.; };
class Timing {
    bool enabled_;
    int every_;
    std::uint64_t random_;
    std::array<std::size_t, 2> next_{};
    Clock::time_point start_{};
    std::uint64_t random()
    {
        random_ ^= random_ << 13; random_ ^= random_ >> 7; random_ ^= random_ << 17;
        return random_;
    }
public:
    std::array<Bucket, Count> buckets{};
    std::array<double, Count> worker_estimates{};
    double rows = 0., entries = 0.;
    Timing(bool enabled, int every, int rank, std::uint64_t build)
        : enabled_(enabled), every_(every),
          random_(0x9e3779b97f4a7c15ULL ^ (std::uint64_t(rank + 1) * 0xbf58476d1ce4e5b9ULL) ^ build)
    {
        if(!random_) random_ = 1;
        if(enabled_) {
            start_ = Clock::now();
            if(every_ > 0) for(auto& n : next_) n = random() % every_;
        }
    }
    bool enabled() const { return enabled_; }
    bool sample(std::size_t index, int stream)
    {
        if(!enabled_ || every_ == 0 || index != next_[stream]) return false;
        // Random gaps avoid locking onto a periodic orbital/table layout.
        next_[stream] = index + 1 + random() % (2ULL * every_ - 1);
        return true;
    }
    // Unmeasured runs have a compile-time false tag: no clock, scope branches,
    // or per-entry sampling test survives in their optimized inner loop.
    template<class F>
    void traverse(std::size_t size, int stream, F&& visit)
    {
        if(!enabled_ || every_ == 0) {
            for(std::size_t i = 0; i < size; ++i) visit(i, std::false_type{});
            return;
        }
        std::size_t i = 0;
        while(i < size) {
            const std::size_t end = next_[stream] < size ? next_[stream] : size;
            for(; i < end; ++i) visit(i, std::false_type{});
            if(i == size) break;
            sample(i, stream);
            visit(i++, std::true_type{});
        }
    }
    class Scope {
        Bucket* bucket_;
        Clock::time_point start_{};
    public:
        Scope(Timing& t, Stage s, bool selected = true)
            : bucket_(t.enabled_ && selected ? &t.buckets[s] : nullptr)
        { if(bucket_) start_ = Clock::now(); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { stop(); }
        void stop()
        {
            if(!bucket_) return;
            bucket_->seconds += std::chrono::duration<double>(Clock::now() - start_).count();
            ++bucket_->hits; bucket_ = nullptr;
        }
    };
    // Merge after workers join. Sample durations are thread work, not phase wall time.
    void merge_samples(const Timing& worker)
    {
        for(int stage = Row; stage < Count; ++stage) {
            buckets[stage].seconds += worker.buckets[stage].seconds;
            buckets[stage].hits += worker.buckets[stage].hits;
            const bool row=stage<Entry;
            const double population=row ? worker.rows : worker.entries;
            const double samples=worker.buckets[row ? Row : Entry].hits;
            worker_estimates[stage] += population==0. ? 0. : (samples
                ? worker.buckets[stage].seconds*population/samples
                : std::numeric_limits<double>::quiet_NaN());
        }
    }
    void finish()
    {
        if(enabled_) {
            buckets[Total].seconds = std::chrono::duration<double>(Clock::now() - start_).count();
            buckets[Total].hits = 1.;
        }
    }
    std::vector<double> pack() const
    {
        std::vector<double> out{rows, entries};
        for(int stage=0;stage<Count;++stage) {
            out.push_back(buckets[stage].seconds); out.push_back(buckets[stage].hits);
            out.push_back(worker_estimates[stage]);
        }
        return out;
    }
    // Returns NaN when a nonempty rank has no samples. Empty ranks cost zero.
    static double estimate(const double* rank, int stage)
    {
        return stage<Row ? rank[2+3*stage] : rank[2+3*stage+2];
    }
    std::string report(const double* all, int ranks, std::size_t dim,
        std::size_t groups, int states, int sets, std::uint64_t build) const
    {
        constexpr int stride = 2 + 3 * Count;
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        double rows_sum = 0., entries_sum = 0., rows_min = all[0], rows_max = all[0];
        double entries_min = all[1], entries_max = all[1];
        for(int r = 0; r < ranks; ++r) {
            const auto p = all + r * stride;
            rows_sum += p[0]; entries_sum += p[1];
            if(p[0] < rows_min) rows_min = p[0]; if(p[0] > rows_max) rows_max = p[0];
            if(p[1] < entries_min) entries_min = p[1]; if(p[1] > entries_max) entries_max = p[1];
        }
        out << "DENSITY_TIMING: build=" << build << " dim=" << dim << " groups=" << groups
            << " input_states=" << states << " sets=" << sets << " ranks=" << ranks
            << " sample_every=" << every_ << " lookup=private_group_cursor clock=steady_wall sample_scope=thread_work"
            << " rows_sum=" << rows_sum << " rows_min=" << rows_min << " rows_max=" << rows_max
            << " table_entries_sum=" << entries_sum << " entries_min=" << entries_min
            << " entries_max=" << entries_max << " summary_excluded=1 extrapolation=per_worker"
            << " exact_phase_source=DENSITY_PROFILE\n";
        for(int s = 0; s < Count; ++s) {
            double lo = std::numeric_limits<double>::infinity(), hi = -1., sum = 0.;
            double hits = 0., sampled_s = 0.;
            int valid = 0, max_rank = -1;
            for(int r = 0; r < ranks; ++r) {
                const auto p = all + r * stride;
                hits += p[2 + 3 * s + 1]; sampled_s += p[2 + 3 * s];
                const double t = estimate(p, s);
                if(!std::isfinite(t)) continue;
                ++valid; sum += t;
                if(t < lo) lo = t;
                if(t > hi) { hi = t; max_rank = r; }
            }
            out << "DENSITY_TIMING: build=" << build << " stage=" << name(s)
                << " measurement=" << (s < Row ? "exact_wall" : "sample_thread_work_estimate")
                << " not_wall_time=" << (s>=Row ? 1:0)
                << " parent=" << (s==Total ? "none" : s==ReduceCall ? "reduce_D"
                    : s<Row ? "build" : s==Row ? "diagonal" : s<Entry ? "sample_row"
                    : s==Entry ? "offdiagonal" : "sample_entry")
                << " valid_ranks=" << valid << '/' << ranks << " min_s=";
            if(valid) out << lo; else out << "NA";
            out << " mean_valid_s="; if(valid) out << sum / valid; else out << "NA";
            out << " max_s="; if(valid) out << hi; else out << "NA";
            out << " max_rank=" << max_rank << " measured_calls_sum=" << hits
                << " measured_seconds_sum=" << sampled_s << '\n';
        }
        return out.str();
    }
};
// Empty specialization guarantees no timer object or runtime timer checks in
// the unsampled path, even when a compiler declines to inline Scope itself.
template<bool Selected> class SampleScope {
public:
    SampleScope(Timing&, Stage) {}
    void stop() {}
};
template<> class SampleScope<true> : public Timing::Scope {
public:
    SampleScope(Timing& timing, Stage stage) : Timing::Scope(timing, stage) {}
};
} // namespace density_timing
