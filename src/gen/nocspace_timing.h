#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Local master-rank diagnostics only: no MPI calls and no I/O in hot loops.
// NORG_NOC_TIMING=0 disables diagnostics. NORG_NOC_TIMING_SAMPLE_EVERY sets
// the mean sampling interval (default 1024); 0 keeps only counts/coarse times,
// and 1 times every event. Inner-stage estimates are NOT additive exact times.
namespace nocspace_timing {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline bool enabled_for(bool master)
{
    if (!master) return false;
    const char* value = std::getenv("NORG_NOC_TIMING");
    return !value || std::string(value) != "0";
}

inline std::size_t sample_interval()
{
    constexpr std::size_t fallback = 1024;
    const char* value = std::getenv("NORG_NOC_TIMING_SAMPLE_EVERY");
    if (!value || !*value) return fallback;
    std::size_t result = 0;
    const auto limit = std::numeric_limits<std::size_t>::max() / 2;
    for (const char* c = value; *c; ++c) {
        if (*c < '0' || *c > '9') return fallback;
        const auto digit = std::size_t(*c - '0');
        if (result > (limit - digit) / 10) return fallback;
        result = result * 10 + digit;
    }
    return result;
}

inline double seconds(TimePoint begin, TimePoint end)
{
    return std::chrono::duration<double>(end - begin).count();
}

enum Stage { Decode, Judge, StarChoices, PrepareFixed, Assemble, Store, StarAdvance, Dimension, Compact, Key, StageCount };

struct Bucket {
    std::size_t calls = 0, samples = 0, until_sample = 0;
    double sampled_s = 0.;
};

class BuildTiming {
public:
    class Sample {
    public:
        Sample(BuildTiming& owner, Stage stage): bucket_(nullptr)
        {
            if (!owner.enabled_) return;
            Bucket& bucket = owner.buckets_[stage];
            ++bucket.calls;
            if (!owner.every_) return;
            if (bucket.until_sample) { --bucket.until_sample; return; }
            // Jitter avoids phase-locking to repeating mixed-radix patterns.
            owner.random_ ^= owner.random_ << 13;
            owner.random_ ^= owner.random_ >> 7;
            owner.random_ ^= owner.random_ << 17;
            bucket.until_sample = owner.random_ % (2 * owner.every_ - 1);
            bucket_ = &bucket;
            start_ = Clock::now();
        }
        Sample(const Sample&) = delete;
        Sample& operator=(const Sample&) = delete;
        ~Sample() { stop(); }
        void stop()
        {
            if (!bucket_) return;
            const auto end = Clock::now();
            bucket_->sampled_s += seconds(start_, end);
            ++bucket_->samples;
            bucket_ = nullptr;
        }
    private:
        Bucket* bucket_;
        TimePoint start_;
    };

    explicit BuildTiming(bool master): enabled_(enabled_for(master)),
        every_(enabled_ ? sample_interval() : 0), start_(stamp()) {}

    bool enabled() const { return enabled_; }
    TimePoint stamp() const { return enabled_ ? Clock::now() : TimePoint{}; }
    Sample sample(Stage stage) { return Sample(*this, stage); }

    void column(int position, bool paired, std::size_t raw, std::size_t kept, TimePoint begin)
    {
        if (enabled_) columns_.push_back({position, paired, raw, kept, seconds(begin, Clock::now())});
    }
    void finish_columns() { columns_end_ = stamp(); }
    void begin_enumeration(std::size_t raw)
    {
        raw_ = raw;
        enumeration_start_ = stamp();
    }
    void reject(std::size_t skipped)
    {
        if (enabled_) { ++rejected_blocks_; skipped_ += skipped; }
    }
    // Merge worker counters after the join; sampled times are sums of worker time.
    void merge_worker(const BuildTiming& worker) {
        if (!enabled_) return;
        merged_workers_ = true;
        rejected_blocks_ += worker.rejected_blocks_; skipped_ += worker.skipped_;
        for (std::size_t i = 0; i < buckets_.size(); ++i) {
            const auto& source = worker.buckets_[i];
            if (source.samples) estimates_[i] += source.sampled_s * (double(source.calls) / source.samples);
            buckets_[i].calls += worker.buckets_[i].calls;
            buckets_[i].samples += worker.buckets_[i].samples;
            buckets_[i].sampled_s += worker.buckets_[i].sampled_s;
        }
    }
    void finish_enumeration() { enumeration_end_ = stamp(); }
    void finish_build() { build_end_ = stamp(); }

    std::string report(const std::string& mode, const std::string& particles,
        std::size_t dimension, std::size_t stored, int rank, int ranks) const
    {
        if (!enabled_) return {};
        // finish_build() is called after builder temporaries are destroyed. Formatting/output is
        // excluded from the measured total; exact phases include sample overhead.
        const auto end = build_end_;
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        out << "NOCSPACE_TIMING: stage=build mode=" << mode << " nppso=" << particles
            << " rank=" << rank << " ranks=" << ranks << " timing_scope=local_master"
            << " dim=" << dimension << " stored=" << stored << " raw_candidates=" << raw_
            // Only fully generated candidates reach Decode. Skipped counts
            // whole excluded label ranges, so visited + skipped = raw_candidates.
            << " visited=" << buckets_[Decode].calls << " skipped=" << skipped_
            << " rejected_blocks=" << rejected_blocks_ << " prefilter_pass=" << buckets_[Decode].calls
            << " mode_pass=" << buckets_[StarChoices].calls << " sample_every=" << every_
            << " columns_s=" << seconds(start_, columns_end_)
            << " setup_s=" << seconds(columns_end_, enumeration_start_)
            << " enumerate_s=" << seconds(enumeration_start_, enumeration_end_)
            << " cleanup_s=" << seconds(enumeration_end_, end)
            << " total_s=" << seconds(start_, end);
        for (const auto& column : columns_)
            out << "\nNOCSPACE_TIMING: stage=column position=" << column.position
                << " paired=" << column.paired << " raw=" << column.raw << " kept=" << column.kept
                << " elapsed_s=" << column.elapsed_s;
        static const char* names[] = {"decode_matrix", "suit_NOOC",
            "star_choices", "prepare_fixed", "assemble_matrix", "store_and_dimension", "star_advance",
            "group_dimension", "compact_occupations", "target_key"};
        for (std::size_t i = 0; i < buckets_.size(); ++i) {
            const auto& b = buckets_[i];
            out << "\nNOCSPACE_TIMING: stage=" << names[i] << " calls=" << b.calls
                << " samples=" << b.samples << " sampled_s=" << b.sampled_s << " estimated_s=";
            if (b.samples) out << (merged_workers_ ? estimates_[i] : b.sampled_s * (double(b.calls) / double(b.samples)));
            else out << "NA";
            out << " measurement=sampled_thread_work not_wall_time=1 parent="
                << (i >= Dimension ? "store_and_dimension" : "candidate_enumeration");
        }
        return out.str();
    }

private:
    struct Column {
        int position;
        bool paired;
        std::size_t raw, kept;
        double elapsed_s;
    };
    bool enabled_;
    std::size_t every_, raw_ = 0, rejected_blocks_ = 0, skipped_ = 0;
    std::uint64_t random_ = 0x9e3779b97f4a7c15ULL;
    std::array<Bucket, StageCount> buckets_{};
    std::array<double, StageCount> estimates_{};
    bool merged_workers_ = false;
    std::vector<Column> columns_;
    TimePoint start_, columns_end_, enumeration_start_, enumeration_end_, build_end_;
};

} // namespace nocspace_timing
