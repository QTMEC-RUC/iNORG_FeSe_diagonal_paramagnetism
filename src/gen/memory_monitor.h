#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

class MyMpi;
// Collective stage logger: all communicator ranks must call in the same order.
void log_computation_memory(const MyMpi& mm, const char* stage, std::size_t dimension,
    const char* scope, const std::string& context, bool enabled = true,
    double measured_stage_s = -1.);
// Shared GS logger, implemented in operator.cpp; all communicator ranks participate.
void log_ground_state_memory(const MyMpi& mm, const char* stage, std::size_t dimension,
    bool enabled = true, double measured_stage_s = -1.);

namespace memory_monitor {

constexpr std::size_t unknown_dimension = std::numeric_limits<std::size_t>::max();

struct Timing {
    double elapsed_s = 0.;       // Since the first monitored event, not process startup.
    double since_previous_s = 0.;
    double stage_elapsed_s = -1.; // NA until a matching BEFORE/BEGIN has been seen.
};

// Pair stages by suffix; repeated or nested phases do not share a stale start.
class StageTimer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    Timing record(const std::string& stage, TimePoint now = Clock::now())
    {
        if(!initialized_) {
            first_ = previous_ = now;
            initialized_ = true;
        }
        Timing result;
        result.elapsed_s = std::chrono::duration<double>(now - first_).count();
        result.since_previous_s = std::chrono::duration<double>(now - previous_).count();
        previous_ = now;
        const auto separator = stage.rfind('_');
        if(separator == std::string::npos) return result;
        const std::string phase = stage.substr(0, separator);
        const std::string suffix = stage.substr(separator + 1);
        const bool begins = suffix == "BEFORE" || suffix == "BEGIN";
        const bool ends = suffix == "AFTER" || suffix == "END";
        if(!begins && !ends) return result;
        const std::string key = phase + (suffix == "BEFORE" || suffix == "AFTER"
            ? "_BEFORE" : "_BEGIN");
        if(begins) starts_[key].push_back(now);
        else {
            const auto found = starts_.find(key);
            if(found != starts_.end()) {
                result.stage_elapsed_s = std::chrono::duration<double>(
                    now - found->second.back()).count();
                found->second.pop_back();
                if(found->second.empty()) starts_.erase(found);
            }
        }
        return result;
    }

private:
    bool initialized_ = false;
    TimePoint first_, previous_;
    std::map<std::string, std::vector<TimePoint>> starts_;
};

inline std::string timestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _MSC_VER
    if(localtime_s(&local, &now) != 0) return "NA";
#else
    if(!localtime_r(&now, &local)) return "NA";
#endif
    char value[64];
    if(!std::strftime(value, sizeof(value), "%Y-%m-%dT%H:%M:%S%z", &local)) return "NA";
    return value;
}

inline std::string format_timing(const Timing& timing, const std::string& time)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
        << "time=" << time << " elapsed_s=" << timing.elapsed_s
        << " since_previous_s=" << timing.since_previous_s << " stage_elapsed_s=";
    if(timing.stage_elapsed_s < 0.) output << "NA";
    else output << timing.stage_elapsed_s;
    return output.str();
}

// /proc reports kB in units of 1024 bytes. Negative values mean unavailable.
struct Snapshot {
    double rss_kib = -1.;
    double hwm_kib = -1.;
};

inline Snapshot read_status(std::istream& input)
{
    Snapshot result;
    std::string line;
    while(std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        double value;
        if(!(fields >> key) || (key != "VmRSS:" && key != "VmHWM:")) continue;
        if(!(fields >> value >> unit) || unit != "kB" || !std::isfinite(value) || value < 0.)
            continue;
        if(key == "VmRSS:") result.rss_kib = value;
        else result.hwm_kib = value;
    }
    return result;
}

inline Snapshot read_self()
{
    std::ifstream status("/proc/self/status");
    return read_status(status); // Missing /proc is reported as NA, never as zero usage.
}

struct Aggregate {
    double rss_sum_kib = 0.;
    double rss_max_kib = -1.;
    double hwm_max_kib = -1.;
    int rss_max_rank = -1;
    int hwm_max_rank = -1;
    int rss_valid = 0;
    int hwm_valid = 0;

    void add(const Snapshot& value, int rank)
    {
        if(std::isfinite(value.rss_kib) && value.rss_kib >= 0.) {
            ++rss_valid;
            rss_sum_kib += value.rss_kib;
            if(value.rss_kib > rss_max_kib) {
                rss_max_kib = value.rss_kib;
                rss_max_rank = rank;
            }
        }
        if(std::isfinite(value.hwm_kib) && value.hwm_kib >= 0.) {
            ++hwm_valid;
            if(value.hwm_kib > hwm_max_kib) {
                hwm_max_kib = value.hwm_kib;
                hwm_max_rank = rank;
            }
        }
    }
};

inline std::string gib(double kib)
{
    if(kib < 0.) return "NA";
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << kib / (1024. * 1024.);
    return output.str();
}

inline std::string format(const char* stage, std::size_t dimension, int ranks,
    const Aggregate& value, const char* scope = "GS")
{
    std::ostringstream output;
    output << "MEM: scope=" << scope << " stage=" << stage << " dim=";
    if(dimension == unknown_dimension) output << "NA";
    else output << dimension;
    output << " ranks=" << ranks
        << " rss_sum_GiB=" << gib(value.rss_valid == ranks ? value.rss_sum_kib : -1.)
        << " rss_max_GiB=" << gib(value.rss_max_kib) << " rss_max_rank=" << value.rss_max_rank
        << " hwm_max_GiB=" << gib(value.hwm_max_kib) << " hwm_max_rank=" << value.hwm_max_rank
        << " rss_valid=" << value.rss_valid << '/' << ranks
        << " hwm_valid=" << value.hwm_valid << '/' << ranks;
    return output.str();
}

} // namespace memory_monitor
