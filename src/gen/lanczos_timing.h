#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "lanczos_progress.h"

// No added MPI/vector scans. Live output is first-occurrence plus time-throttled;
// existing completed-solve statistics and GF behavior are preserved.
namespace lanczos_timing {
enum Stage {
    Total, Initial, Recurrence, BasisStore, Deflation, Reorthogonalization,
    ResidualCheck, FinalTridiagonal, Reconstruction, Normalize, EigenpairStore,
    Output, Hv, HvBuffers, HvGather, HvSparse, CsrCopy, MklCreate, MklOptimize,
    MklMultiply, MklDestroy, ScalarAllreduce, VectorAllgatherv,
    OrthPasses, OrthProjections, LocalDot, LocalUpdate, HvSetup, HvTune,
    HaloPack, HaloPost, HaloWait, HaloInterior, HaloBoundary, HaloScatter,
    FullVectorSetup, FullVectorCopy, FullVectorNative, FullVectorRing, Count
};
inline const char* name(Stage s)
{
    static const char* names[] = {
        "solve", "initial", "recurrence", "basis_store", "deflation", "reorthogonalization",
        "residual_check", "final_tridiagonal", "reconstruction", "normalize", "eigenpair_store",
        "output_gather_sort", "hv", "hv_buffers", "hv_exchange_exposed", "hv_sparse",
        "csr_index_copy", "mkl_create", "mkl_optimize", "mkl_mv", "mkl_destroy",
        "mpi_scalar_allreduce", "mpi_vector_allgatherv", "orth_passes", "orth_projections",
        "local_dot", "local_update", "hv_setup", "hv_autotune", "halo_pack",
        "halo_post", "halo_wait", "halo_interior_mv", "halo_boundary_mv", "halo_scatter",
        "full_vector_setup", "full_vector_local_copy",
        "full_vector_native_exchange", "full_vector_ring_exchange"
    };
    return names[static_cast<int>(s)];
}
struct Record {
    std::array<double, Count> seconds{};
    std::array<double, Count> calls{};
};
struct State {
    Record record;
    std::size_t k = 0;
    const char* stop = "max_basis";
};
class Session;
inline thread_local Session* active = nullptr;
using Clock = std::chrono::steady_clock;

inline bool requested()
{
    const char* value = std::getenv("NORG_LANCZOS_TIMING");
    return !value || std::string(value) != "0";
}

class Session {
    Session* previous_;
    bool enabled_;
    Clock::time_point start_{};
    bool matrix_seen_ = false;
    ProgressWriter progress_;
    std::array<bool,Count> entered_{},left_{};
    Record last_progress_;
    std::size_t iteration_=0;
    const char* pass_="initial";
    void print_progress(Stage stage,Clock::time_point now) {
        progress_.event("progress",state,iteration_,pass_,name(stage),0.,now);
        std::ostringstream out;out<<std::fixed<<std::setprecision(6);
        for(int s=0;s<Count;++s)if(total.calls[s]!=0.) {
            out<<"LANCZOS_STAGE: solve="<<progress_.solve()<<" rank="<<progress_.rank()
               <<" state="<<state<<" k="<<iteration_<<" pass="<<pass_
               <<" stage="<<name(static_cast<Stage>(s))<<" cumulative_s="<<total.seconds[s]
               <<" interval_s="<<total.seconds[s]-last_progress_.seconds[s]
               <<" calls="<<total.calls[s]<<" interval_calls="<<total.calls[s]-last_progress_.calls[s]
               <<" scope=solve local_only=1 inclusive=1\n";
        }
        progress_.snapshot(out.str());last_progress_=total;progress_.reschedule(now);
    }

public:
    Record total;
    std::vector<State> states;
    int state = -1;
    double rows = 0., nnz = 0.;
    Session(bool enabled) : previous_(active), enabled_(enabled)
    {
        active = enabled ? this : nullptr;
        if(enabled) start_ = Clock::now();
    }
    ~Session() { active = previous_; }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void configure_progress(const ProgressOptions& options,int rank,int master,
                            unsigned long long solve,std::size_t dim) {
        progress_.configure(options,rank,master,solve);
        if(progress_.enabled())progress_.event("solve_begin",state,0,pass_,"solve",0.,Clock::now(),
            " dim="+std::to_string(dim)+affinity_description());
    }
    // Just assignments: no clock, allocation, MPI or printing per iteration.
    void iteration(std::size_t step,const char* pass) {iteration_=step;pass_=pass;}
    void enter(Stage s,Clock::time_point now) {
        if(!progress_.enabled())return;
        if(!entered_[s]) {
            entered_[s]=true;progress_.event("begin",state,iteration_,pass_,name(s),0.,now);
        }
        if(progress_.interval_due(now))print_progress(s,now);
    }
    void leave(Stage s,double seconds,Clock::time_point now) {
        if(!progress_.enabled())return;
        if(!left_[s]) {
            left_[s]=true;progress_.event("end",state,iteration_,pass_,name(s),seconds,now);
        }
        if(progress_.interval_due(now))print_progress(s,now);
    }
    void begin_state()
    {
        if(!enabled_) return;
        states.emplace_back();
        state = static_cast<int>(states.size()) - 1;
        entered_.fill(false);left_.fill(false);iteration_=0;pass_="initial";
        if(progress_.enabled())progress_.event("state_begin",state,0,pass_,"initial",0.,Clock::now());
    }
    void end_state(std::size_t k, const char* reason)
    {
        if(!enabled_) return;
        states[state].k = k;
        states[state].stop = reason;
        if(progress_.enabled())progress_.event("state_end",state,k,pass_,"eigenpair_store",0.,Clock::now(),
            " stop="+std::string(reason));
        state = -1;
    }
    void add(Stage s, int index, double seconds, double count = 1.)
    {
        total.seconds[s] += seconds;
        total.calls[s] += count;
        if(index >= 0) {
            states[index].record.seconds[s] += seconds;
            states[index].record.calls[s] += count;
        }
    }
    void matrix(std::size_t local_rows, std::size_t local_nnz)
    {
        if(matrix_seen_) return;
        rows = static_cast<double>(local_rows);
        nnz = static_cast<double>(local_nnz);
        matrix_seen_ = true;
    }
    void finish()
    {
        if(!enabled_) return;
        total.seconds[Total] = std::chrono::duration<double>(Clock::now() - start_).count();
        total.calls[Total] = 1.;
        if(progress_.enabled()){
            const auto now=Clock::now();print_progress(Total,now);
            progress_.event("solve_end",-1,iteration_,"output","solve",total.seconds[Total],now);
        }
        active = previous_; // Exclude summary communication and formatting.
    }
    std::vector<double> pack() const
    {
        std::vector<double> values{rows, nnz, matrix_seen_ ? 1. : 0.};
        auto append = [&](const Record& r) {
            for(int s = 0; s < Count; ++s) {
                values.push_back(r.seconds[s]);
                values.push_back(r.calls[s]);
            }
        };
        append(total);
        for(const auto& item : states) append(item.record);
        return values;
    }
    // all[] is rank-major and obtained by ONE gather after the solve.
    std::string report(const double* all, int ranks, std::size_t dim,
        int reorth, bool fast, unsigned long long solve) const
    {
        const std::size_t stride = 3 + (states.size() + 1) * 2 * Count;
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        double nnz_sum = 0., nnz_max = 0., rows_min = all[0], rows_max = all[0];
        int nnz_rank = 0, matrix_ranks = 0;
        for(int r = 0; r < ranks; ++r) {
            const auto p = all + r * stride;
            nnz_sum += p[1];
            if(p[1] > nnz_max) { nnz_max = p[1]; nnz_rank = r; }
            if(p[0] < rows_min) rows_min = p[0];
            if(p[0] > rows_max) rows_max = p[0];
            matrix_ranks += p[2] != 0.;
        }
        out << "LANCZOS_TIMING: solve=" << solve << " dim=" << dim << " ranks=" << ranks
            << " reorth=" << reorth << " fast=" << fast << " states=" << states.size()
            << " nnz_sum=" << nnz_sum << " nnz_max=" << nnz_max << " nnz_max_rank=" << nnz_rank
            << " matrix_ranks=" << matrix_ranks << " rows_min=" << rows_min << " rows_max=" << rows_max
            << " clock=steady_wall inclusive=1 summary_excluded=1\n";
        for(std::size_t block = 0; block <= states.size(); ++block) {
            if(block) out << "LANCZOS_TIMING: solve=" << solve << " state=" << block - 1
                << " k=" << states[block - 1].k << " stop=" << states[block - 1].stop << '\n';
            for(int s = 0; s < Count; ++s) {
                const std::size_t offset = 3 + block * 2 * Count + 2 * s;
                double lo = all[offset], hi = lo, sum = 0.;
                double clo = all[offset + 1], chi = clo, csum = 0.;
                int max_rank = 0;
                for(int r = 0; r < ranks; ++r) {
                    const double t = all[r * stride + offset], c = all[r * stride + offset + 1];
                    if(t < lo) lo = t;
                    if(t > hi) { hi = t; max_rank = r; }
                    if(c < clo) clo = c;
                    if(c > chi) chi = c;
                    sum += t; csum += c;
                }
                if(csum == 0.) continue;
                out << "LANCZOS_TIMING: solve=" << solve << " state=";
                if(block) out << block - 1; else out << "all";
                out << " stage=" << name(static_cast<Stage>(s))
                    << " min_s=" << lo << " mean_s=" << sum / ranks << " max_s=" << hi
                    << " max_rank=" << max_rank << " calls_min=" << clo << " calls_max=" << chi
                    << " mean_s_per_call=" << sum / csum << '\n';
            }
        }
        return out.str();
    }
};

class Scope {
    Session* session_;
    Stage stage_;
    int state_;
    Clock::time_point start_{};
public:
    explicit Scope(Stage s) : session_(active), stage_(s), state_(active ? active->state : -1)
    { if(session_) {start_ = Clock::now();session_->enter(stage_,start_);} }
    ~Scope() { stop(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void stop()
    {
        if(!session_) return;
        const auto now=Clock::now();
        const double seconds=std::chrono::duration<double>(now-start_).count();
        session_->add(stage_,state_,seconds);
        session_->leave(stage_,seconds,now);
        session_ = nullptr;
    }
};
inline void count(Stage s, std::size_t n = 1)
{ if(active) active->add(s, active->state, 0., static_cast<double>(n)); }
} // namespace lanczos_timing
