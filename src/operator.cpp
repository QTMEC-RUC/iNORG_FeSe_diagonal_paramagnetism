/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "operator.h"
#include "gen/operator_index.h"
#include "gen/tab_io.h"
#include "gen/memory_monitor.h"
#include <thread>
#include <chrono>

using namespace std;

// H builders run on the MPI initializing thread.
static std::uint64_t next_h_profile_build() { static std::uint64_t id=0; return ++id; }

// All ranks must enter together; only the communicator's master prints.
void log_ground_state_memory(const MyMpi& mm, const char* stage, Idx dimension,
    bool enabled, double measured_stage_s)
{
    log_computation_memory(mm, stage, dimension, "GS", "", enabled, measured_stage_s);
}

void log_computation_memory(const MyMpi& mm, const char* stage, Idx dimension,
    const char* scope, const std::string& context, bool enabled, double measured_stage_s)
{
    if(!enabled) return;
    mm.barrier();
    // Record after synchronization, before collecting memory statistics.
    // Only the master keeps timing state; separate MyMpi objects have separate timelines.
    memory_monitor::Timing timing;
    std::string event_time;
    if(mm) {
        static std::map<const MyMpi*, memory_monitor::StageTimer> timers;
        // Share the elapsed timeline, but never pair stages from different sectors,
        // ground states, source columns or GS/GF scopes.
        timing = timers[&mm].record(std::string(scope) + " " + context + " " + stage);
        if(measured_stage_s >= 0.) timing.stage_elapsed_s = measured_stage_s;
        event_time = memory_monitor::timestamp();
    }
    const memory_monitor::Snapshot memory = memory_monitor::read_self();
    VecReal local(2);
    local[0] = memory.rss_kib;
    local[1] = memory.hwm_kib;
    const VecReal gathered = mm.Gather(local);
    if(mm) {
        memory_monitor::Aggregate total;
        for(Int rank = 0; rank < mm.np(); ++rank) {
            memory_monitor::Snapshot sample;
            sample.rss_kib = gathered[2 * rank];
            sample.hwm_kib = gathered[2 * rank + 1];
            total.add(sample, rank);
        }
        std::cout << memory_monitor::format(stage, dimension, mm.np(), total, scope)
            << (context.empty() ? "" : " ") << context
            << ' ' << memory_monitor::format_timing(timing, event_time)
            << (measured_stage_s >= 0. ? (std::string(stage) == "NOCSPACE_AFTER"
                ? " stage_elapsed_source=nocspace_build_clock reporting_excluded=1"
                : (std::string(stage) == "H_AFTER"
                    ? " stage_elapsed_source=h_build_clock reporting_excluded=1"
                    : " stage_elapsed_source=tab_build_clock reporting_excluded=1")) : "")
            << std::endl;
    }
}

Operator::Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i,
	bool monitor_table_memory, const char* monitor_scope, const Str& monitor_context):
	mm(mm_i), p(prmtr_i), scsp(s_i), dim(s_i.dim),
	// Keep the original construction order and avoid a wrapper that returns/copies Tab.
	table((log_computation_memory(mm, "TAB_BEFORE", dim,
		monitor_scope, monitor_context, monitor_table_memory),
		find_fullH_offdiagonal_idx(&tab_timing_))),
	// The table is initialized and builder temporaries are gone; oper_value is not allocated yet.
	oper_value((log_computation_memory(mm, "TAB_AFTER", dim,
		monitor_scope, monitor_context, monitor_table_memory, tab_timing_.total_s),
        tab_timing::emit(mm, tab_timing_, dim, monitor_scope, monitor_context, !p.if_norg_imp),
		pow(p.norbit, 4) + pow(p.norbit, 2) + 1))
	// mm(mm_i), p(prmtr_i), scsp(s_i), table(find_fullH_idx()), dim(s_i.dim), oper_value(pow(p.norbit, 4) + pow(p.norbit, 2) + 1)
{
}

Operator::Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i,const Tab &per_table):
	mm(mm_i), p(prmtr_i), scsp(s_i), dim(s_i.dim), table(per_table), oper_value(pow(p.norbit, 4) + pow(p.norbit, 2) + 1)
{
    VecPartition rows(mm.np(), mm.id(), dim);
    table.require_layout(dim, rows.bgn(), rows.len());
}

Operator::Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i, Str tab_name):
	mm(mm_i), p(prmtr_i), scsp(s_i), dim(s_i.dim), table(read_the_Tab(tab_name)), oper_value(pow(p.norbit, 4) + pow(p.norbit, 2) + 1)
{
}

Tab Operator::find_fullH_idx()
{
    VecPartition rank_rows(mm.np(), mm.id(), dim);
    Tab result;
    stage2_parallel::collective(mm, [&] {
        const std::size_t parts = stage2_parallel::workers(rank_rows.len());
        std::vector<Tab> shards(parts);
        stage2_parallel::ranges(rank_rows.len(), [&](std::size_t first, std::size_t last, std::size_t part) {
            struct Rows {
                BasisIndex first, count;
                BasisIndex bgn() const { return first; }
                BasisIndex end() const { return first + count; }
                BasisIndex len() const { return count; }
            } row_H{rank_rows.bgn() + static_cast<BasisIndex>(first), static_cast<BasisIndex>(last - first)};
            const auto build = [&]() -> Tab {
	NocSpace::GroupCursor group_cursor(scsp);
	TabBuilder h_idxs(dim, row_H.bgn(), row_H.len());
	MatInt mat_hop_pos(scsp.hopint.nrows(),scsp.hopint.ncols());
	for_Int(i, 0, mat_hop_pos.nrows()) for_Int(j, 0, mat_hop_pos.ncols()) mat_hop_pos[i][j] = i * mat_hop_pos.ncols() + j;
	for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i) {
		// To save as sparse matrix, [0]: row number;[1]: colum number;[2]: idx.
		const auto sparse_idx = static_cast<std::size_t>(h_i - row_H.bgn());
		//WRN("wherein_NocSpace" + NAV(h_i - scsp.idx_div[scsp.wherein_NocSpace(h_i)]));

		StateStatistics a(h_i, group_cursor.locate(h_i), scsp);
		
		//! Diagonal term. n_i
		for (const auto &x : a.filled_spinless) {
			for (const auto &i : x) {
				h_idxs.append(sparse_idx, h_i, mat_hop_pos[i][i] + 1);
			}
		}
/*
		//! Diagonal term. n_i n_j
		for_Int(b1, 0, p.nband) {
			if( a.cfg.cf[(b1 * 2) * ndiv].isocc(0) && a.cfg.cf[(b1 * 2 + 1) * ndiv].isocc(0)){
				Int interact_pos = mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, SUM_0toX(p.nO2sets, (b1 * 2)), SUM_0toX(p.nO2sets, (b1 * 2) + 1), SUM_0toX(p.nO2sets, (b1 * 2) + 1), SUM_0toX(p.nO2sets, (b1 * 2)));
				h_idxs.append(sparse_idx, h_i, interact_pos);
			}
			for_Int(b2, 0, p.nband) {
				if (b1 != b2) {
					if ((a.cfg.cf[(b1 * 2) * ndiv].isocc(0) && a.cfg.cf[(b2 * 2 + 1) * ndiv].isocc(0))) {
						Int interact_pos = mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, SUM_0toX(p.nO2sets, (b1 * 2)), SUM_0toX(p.nO2sets, (b2 * 2) + 1), SUM_0toX(p.nO2sets, (b2 * 2) + 1), SUM_0toX(p.nO2sets, (b1 * 2)));
						h_idxs.append(sparse_idx, h_i, interact_pos);
					}
					if ((a.cfg.cf[(b1 * 2) * ndiv].isocc(0) && a.cfg.cf[(b2 * 2) * ndiv].isocc(0))) {
						Int interact_pos = mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, SUM_0toX(p.nO2sets, (b1 * 2)), SUM_0toX(p.nO2sets, (b2 * 2)), SUM_0toX(p.nO2sets, (b2 * 2)), SUM_0toX(p.nO2sets, (b1 * 2)));
						h_idxs.append(sparse_idx, h_i, interact_pos);
					}
					if ((a.cfg.cf[(b1 * 2 + 1) * ndiv].isocc(0) && a.cfg.cf[(b2 * 2 + 1) * ndiv].isocc(0))) {
						Int interact_pos = mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, SUM_0toX(p.nO2sets, (b1 * 2 + 1)), SUM_0toX(p.nO2sets, (b2 * 2) + 1), SUM_0toX(p.nO2sets, (b2 * 2) + 1), SUM_0toX(p.nO2sets, (b1 * 2 + 1)));
						h_idxs.append(sparse_idx, h_i, interact_pos);
					}
				}
			}
		}
*/
		/*
		//! Diagonal term. n_i n_j
		for_Int(set_i, 0, p.norg_sets) {
			for_Int(set_j, 0, p.norg_sets) {
				VEC<Int> N_veci(a.filled_spinless[set_i]), N_vecj(a.filled_spinless[set_j]);
				Int offset_i = 0; for_Int(k, 0, set_i) offset_i += p.nO2sets[k];
				Int offset_j = 0; for_Int(l, 0, set_j) offset_j += p.nO2sets[l];
				for (const auto& N_i : N_veci) for (const auto& N_j : N_vecj) {
					// if (N_i != N_j) {
						if (N_i < offset_i + p.control_divs[set_i+1][0] && N_j < offset_j + p.control_divs[set_j+1][0]) {
							h_idxs.append(sparse_idx, h_i, int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_i, N_j, N_j, N_i)));
							// ? if this term(↓) is necessary? only can mute this term when the interaction happen in different sets.
							// h_idxs.append(sparse_idx, h_i,-int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_i, N_j, N_i, N_j)));
						}
					// }
				}
			}
		}
		*/
		
		//! Diagonal term. n_i n_j
		for_Int(set_i, 0, p.norg_sets) {
			for_Int(set_j, 0, p.norg_sets) {
				VEC<Int> N_veci(a.filled_spinless[set_i]), N_vecj(a.filled_spinless[set_j]);
				for (const auto& N_i : N_veci) for (const auto& N_j : N_vecj) {
					// if (N_i != N_j) {
						h_idxs.append(sparse_idx, h_i, int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_i, N_j, N_j, N_i)));
						// ? if this term(↓) is necessary? only can mute this term when the interaction happen in different sets.
						// h_idxs.append(sparse_idx, h_i,-int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_i, N_j, N_i, N_j)));
					// }
				}
			}
		}

		//! off-Diagonal term. C\power-index{i|†}C\index{j}|| C\power-index{i|†}C\power-index{j|†}C\power-index{k}C\index{l}
		for_Int(sets_i, 0, p.norg_sets) {
			{
				// off_diagonal_term Two-Fermi
				// i[0]:annihilation orbit's position; i[1]:creation orbit's positon; i[2]:Colum idx(i); i[3]:sign(anticommutativity)
				const auto off_dt_next = a.find_each_spiless_group_off_diagonal_term(a.divocchop_ingroup(a.div_idx, sets_i), sets_i);
				for (const auto &i : off_dt_next){
					h_idxs.append(sparse_idx, i[2], i[3] * (mat_hop_pos[i[1]][i[0]] + 1));

					//! Diagonal+off-Diagonal term. n_sC_e^+C_f
					if (p.if_norg_imp) for (const auto& x : a.filled_spinless) {
						for (const auto& N_s : x) if (N_s != i[0] && N_s != i[1]) {
							//																		[i   ][j  ][k  ][l   ]
							h_idxs.append(sparse_idx, i[2],-i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_s, i[1], N_s, i[0])));
							h_idxs.append(sparse_idx, i[2], i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_s, i[1], i[0], N_s)));
							h_idxs.append(sparse_idx, i[2], i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[1], N_s, N_s, i[0])));
							h_idxs.append(sparse_idx, i[2],-i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[1], N_s, i[0], N_s)));
						}
					}
				}
			}

			if (p.if_norg_imp) {
				// for_Int(sets_j, 0, p.norg_sets) {
				for_Int(sets_j, 0, p.norg_sets) if (sets_j != sets_i) {
					// off_diagonal_term Four-Fermi
					// [0]~[3] i-j-k-l orbit's position(C^+_i C^+_j C_k C_l); [4]:Colum idx(i);[5]:sign(fermion anticommutativity)
					const auto off_dt_next = a.find_off_diagonal_term_fourFermi(a.divs_change_fourFermi(a.div_idx, sets_i, sets_j), sets_i, sets_j);
					for (const auto& i : off_dt_next) {
						h_idxs.append(sparse_idx, i[4], i[5] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[0], i[1], i[2], i[3])));
					}
				}
			}
		}
	}
	// TIME_END("t_find_hmlt_table" + NAV(mm.id()), t_find_hmlt_table);

	return h_idxs.finish();

            };
            shards[part] = build();
        });
        result = Tab::concatenate(shards);
    });
    return result;
}

SparseMatReal Operator::find_hmlt(const Tab& h_idx) const
{
    namespace hp=matrix_build_profile;
    const auto options=hp::options(mm,"NORG_H_TIMING","NORG_H_TIMING_SAMPLE_EVERY");
    const auto build=next_h_profile_build();
    hp::Report profile(mm,options.enabled);
    VecPartition row_H(mm.np(), mm.id(), dim);
    SparseMatReal result(row_H.len(), dim, mm);
    {
    std::vector<std::size_t> counts;
    profile.mark(hp::Setup);
    stage2_parallel::collective(mm, [&] {
        h_idx.require_layout(dim, row_H.bgn(), row_H.len());
        profile.mark(hp::Validation);
        counts.assign(h_idx.nrows(),1);
        stage2_parallel::for_each(h_idx.nrows(), [&](std::size_t row) {
            for (Tab::Offset pos = h_idx.row_ptr()[row]; pos < h_idx.row_ptr()[row + 1]; ++pos)
                if (h_idx.cols()[pos] != row_H.bgn() + static_cast<BasisIndex>(row)) ++counts[row];
        });
        profile.mark(hp::RowCounts);
        result.prepare_csr(counts,&profile);
        stage2_parallel::for_each(h_idx.nrows(), [&](std::size_t row) {
            Real diagonal = 0.;
            for (Tab::Offset pos = h_idx.row_ptr()[row]; pos < h_idx.row_ptr()[row + 1]; ++pos)
                if (h_idx.cols()[pos] == row_H.bgn() + static_cast<BasisIndex>(row)) {
                    const Int op = h_idx.ops()[pos];
                    diagonal += (op > 0 ? 1 : -1) * oper_value[abs(op)];
                }
            auto out = result.csr_row_begin(row);
            result.set_csr_entry(out++, row_H.bgn() + static_cast<BasisIndex>(row), diagonal);
            for (Tab::Offset pos = h_idx.row_ptr()[row]; pos < h_idx.row_ptr()[row + 1]; ++pos)
                if (h_idx.cols()[pos] != row_H.bgn() + static_cast<BasisIndex>(row)) {
                    const Int op = h_idx.ops()[pos];
                    result.set_csr_entry(out++, h_idx.cols()[pos], (op > 0 ? 1 : -1) * oper_value[abs(op)]);
                }
        });
        profile.mark(hp::CsrFill);
    });
    profile.mark(hp::CollectiveSync);
    }
    profile.mark(hp::Cleanup);profile.finish();h_build_seconds_=profile.total();
    hp::emit(mm,profile,"H_PROFILE","full_tab",dim,row_H.len(),h_idx.nnz(),build);
    return result;
}

Tab Operator::find_fullH_offdiagonal_idx(tab_timing::Report* timing)
{
    // Other ranks must not start while the master is still printing TAB_BEFORE.
    // This synchronization is intentionally outside the build clock.
    if(timing) mm.barrier();
    if(timing) timing->wall_begin = std::chrono::system_clock::now();
    const auto begin = tab_timing::tick(timing != nullptr);
    const bool detailed = timing && timing->enabled;
    if(timing) {
        timing->index_build_s = timing->workers_s = timing->merge_s = timing->index_destroy_s = 0.;
        timing->merge = {};
        timing->workers.clear();
    }
    VecPartition rank_rows(mm.np(), mm.id(), dim);
    Tab result;
    stage2_parallel::collective(mm, [&] {
        // Build before entering OpenMP; worker caches only perform const lookups.
        std::unique_ptr<TwoFermionGroupIndex> group_index;
        {
            tab_timing::Span phase(detailed ? &timing->index_build_s : nullptr);
            if (!p.if_norg_imp) group_index.reset(new TwoFermionGroupIndex(scsp));
        }
        const std::size_t parts = stage2_parallel::workers(rank_rows.len());
        std::vector<Tab> shards(parts);
        if(detailed) {
            timing->workers.resize(parts);
            for(auto& worker:timing->workers) worker.stride = timing->sample_every;
        }
        {
        tab_timing::Span phase(detailed ? &timing->workers_s : nullptr);
        stage2_parallel::ranges(rank_rows.len(), [&](std::size_t first, std::size_t last, std::size_t part) {
            struct Rows {
                BasisIndex first, count;
                BasisIndex bgn() const { return first; }
                BasisIndex end() const { return first + count; }
                BasisIndex len() const { return count; }
            } row_H{rank_rows.bgn() + static_cast<BasisIndex>(first), static_cast<BasisIndex>(last - first)};
            const auto build = [&]() -> Tab {
	NocSpace::GroupCursor group_cursor(scsp);
	TabBuilder h_idxs(dim, row_H.bgn(), row_H.len());
	if (!p.if_norg_imp) {
            tab_timing::Worker* worker = detailed ? &timing->workers[part] : nullptr;
            TwoFermionGroupCache cache(*group_index, worker);
            BasisIndex h_i = row_H.bgn();
            while (h_i < row_H.end()) {
                const BasisIndex group = group_cursor.locate(h_i);
                const BasisIndex end = std::min(row_H.end(), scsp.idx_div[group + 1]);
                cache.prepare(group, end - h_i, h_idxs);
                for (; h_i < end; ++h_i) {
                    const bool sampled = worker && worker->sample_row();
                    if (cache.single_orbital()) {
                        // Occupations already uniquely identify this basis state.
                        // No ComDivs decoding, per-orbital arrays or state destructor.
                        tab_timing::Span generate(sampled ? &worker->seconds[tab_timing::EntryGenerate] : nullptr);
                        cache.append_single_orbital(h_i - row_H.bgn(), h_idxs);
                        continue;
                    }
                    const auto decode_begin = tab_timing::tick(sampled);
                    tab_timing::Point destroy_begin;
                    {
                        StateStatistics state(h_i, group, scsp,
                            StateStatistics::OccupationStatistics::skip);
                        if(sampled) worker->seconds[tab_timing::StateDecode] += tab_timing::since(decode_begin);
                        {
                            tab_timing::Span generate(sampled ? &worker->seconds[tab_timing::EntryGenerate] : nullptr);
                            cache.append(state, h_i - row_H.bgn(), h_idxs);
                        }
                        destroy_begin = tab_timing::tick(sampled);
                    }
                    if(sampled) worker->seconds[tab_timing::StateDestroy] += tab_timing::since(destroy_begin);
                }
            }

		return h_idxs.finish();
	}

	MatInt mat_hop_pos(scsp.hopint.nrows(),scsp.hopint.ncols());
	for_Int(i, 0, mat_hop_pos.nrows()) for_Int(j, 0, mat_hop_pos.ncols()) mat_hop_pos[i][j] = i * mat_hop_pos.ncols() + j;
	for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i) {
		// To save as sparse matrix, [0]: row number;[1]: colum number;[2]: idx.
		const auto sparse_idx = static_cast<std::size_t>(h_i - row_H.bgn());
		//WRN("wherein_NocSpace" + NAV(h_i - scsp.idx_div[scsp.wherein_NocSpace(h_i)]));

		StateStatistics a(h_i, group_cursor.locate(h_i), scsp);
		
		//! off-Diagonal term. C\power-index{i|†}C\index{j}|| C\power-index{i|†}C\power-index{j|†}C\power-index{k}C\index{l}
		for_Int(sets_i, 0, p.norg_sets) {
			{
				// off_diagonal_term Two-Fermi
				// i[0]:annihilation orbit's position; i[1]:creation orbit's positon; i[2]:Colum idx(i); i[3]:sign(anticommutativity)
				const auto off_dt_next = a.find_each_spiless_group_off_diagonal_term(a.divocchop_ingroup(a.div_idx, sets_i), sets_i);
				for (const auto &i : off_dt_next){
					h_idxs.append(sparse_idx, i[2], i[3] * (mat_hop_pos[i[1]][i[0]] + 1));

					//! Diagonal+off-Diagonal term. n_sC_e^+C_f
					if (p.if_norg_imp) for (const auto& x : a.filled_spinless) {
						for (const auto& N_s : x) if (N_s != i[0] && N_s != i[1]) {
							//																		[i   ][j  ][k  ][l   ]
							h_idxs.append(sparse_idx, i[2],-i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_s, i[1], N_s, i[0])));
							h_idxs.append(sparse_idx, i[2], i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_s, i[1], i[0], N_s)));
							h_idxs.append(sparse_idx, i[2], i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[1], N_s, N_s, i[0])));
							h_idxs.append(sparse_idx, i[2],-i[3] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[1], N_s, i[0], N_s)));
						}
					}
				}
			}

			if (p.if_norg_imp) {
				for_Int(sets_j, 0, p.norg_sets) if (sets_j != sets_i) {
					// off_diagonal_term Four-Fermi
					// [0]~[3] i-j-k-l orbit's position(C^+_i C^+_j C_k C_l); [4]:Colum idx(i);[5]:sign(fermion anticommutativity)
					const auto off_dt_next = a.find_off_diagonal_term_fourFermi(a.divs_change_fourFermi(a.div_idx, sets_i, sets_j), sets_i, sets_j);
					for (const auto& i : off_dt_next) {
						h_idxs.append(sparse_idx, i[4], i[5] * int(mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, i[0], i[1], i[2], i[3])));
					}
				}
			}
		}
	}
	// TIME_END("t_find_hmlt_table" + NAV(mm.id()), t_find_hmlt_table);

	return h_idxs.finish();

            };
            shards[part] = build();
        });
        }
        {
            tab_timing::Span phase(detailed ? &timing->merge_s : nullptr);
            result = Tab::concatenate(shards, detailed ? &timing->merge : nullptr);
        }
        {
            tab_timing::Span phase(detailed ? &timing->index_destroy_s : nullptr);
            group_index.reset();
        }
    });
    // End BEFORE diagnostics aggregation, formatting, printing or flushing.
    if(timing) {
        timing->total_s = tab_timing::since(begin);
        timing->wall_end = std::chrono::system_clock::now();
    }
    return result;
}

SparseMatReal Operator::find_hmlt_V2(const Tab& h_table) const
{
    namespace hp=matrix_build_profile;
    const auto options=hp::options(mm,"NORG_H_TIMING","NORG_H_TIMING_SAMPLE_EVERY");
    const auto build=next_h_profile_build();
    hp::Report profile(mm,options.enabled);
    std::vector<hp::Worker> worker_timing;
	// clock_t t_find_hmlt;
	// TIME_BGN("find_hmlt" + NAV(mm.id()), t_find_hmlt);
	VecPartition row_H(mm.np(), mm.id(), dim);


	SparseMatReal hmlt_splited(row_H.len(), dim, mm);
	{
    MatInt mat_hop_pos(scsp.hopint.nrows(),scsp.hopint.ncols());
	for_Int(i, 0, mat_hop_pos.nrows()) for_Int(j, 0, mat_hop_pos.ncols()) mat_hop_pos[i][j] = i * mat_hop_pos.ncols() + j;
	VEC<Real> diagonals(row_H.len());
	std::vector<Int> set_begin(p.norg_sets, 0);
	for_Int(set, 1, p.norg_sets) set_begin[set] = set_begin[set - 1] + p.nO2sets[set - 1];
    std::vector<std::size_t> counts;
    const auto parts=stage2_parallel::workers(row_H.len());
    worker_timing.reserve(parts);
    for(int part=0;part<parts;++part)worker_timing.emplace_back(options.enabled,options.sample_every,
        build+static_cast<std::uint64_t>(part)+static_cast<std::uint64_t>(mm.id())*parts);
    profile.mark(hp::Setup);
    stage2_parallel::collective(mm, [&] {
        h_table.require_layout(dim, row_H.bgn(), row_H.len());
        profile.mark(hp::Validation);
        stage2_parallel::ranges(row_H.len(), [&](std::size_t first, std::size_t last, std::size_t part) {
            NocSpace::GroupCursor group_cursor(scsp);
            OccupationDecoder occupation_decoder(scsp);
            std::vector<std::vector<Int>> impurity_occupied(p.norg_sets);
            for_Int(set, 0, p.norg_sets)
                impurity_occupied[set].reserve(p.control_divs[set + 1][0]);
	auto& sample=worker_timing[part];
    sample.traverse(last-first,[&](std::size_t offset,auto selected) {
        constexpr bool measured=decltype(selected)::value;
        const BasisIndex h_i = row_H.bgn() + static_cast<BasisIndex>(first + offset);
        hp::Sample<measured> row_timer(sample,hp::Row);
		// To save as sparse matrix, [0]: row number;[1]: colum number;[2]: idx.
		Real diagonal(0.);
		const auto sparse_idx = static_cast<std::size_t>(h_i - row_H.bgn());
		hp::Sample<measured> lookup_timer(sample,hp::GroupLookup);
            const auto group=group_cursor.locate(h_i);lookup_timer.stop();
            hp::Sample<measured> decode_timer(sample,hp::StateDecode);
            const auto& filled = occupation_decoder.decode(h_i, group);decode_timer.stop();
            hp::Sample<measured> one_body_timer(sample,hp::OneBody);

		//! Diagonal term. n_i
		for (const auto& x : filled) {
			for (const auto& i : x) {
				diagonal += oper_value[mat_hop_pos[i][i] + 1];
			}
		}
		one_body_timer.stop();
            hp::Sample<measured> two_body_timer(sample,hp::TwoBody);
            // Filter once per set, preserving order and all original pair terms.
            for_Int(set, 0, p.norg_sets) {
                auto& orbitals = impurity_occupied[set];
                orbitals.clear();
                for (const Int orbital : filled[set])
                    if (orbital < set_begin[set] + p.control_divs[set + 1][0])
                        orbitals.push_back(orbital);
            }
            //! Diagonal term. n_i n_j
		for_Int(set_i, 0, p.norg_sets) {
			for_Int(set_j, 0, p.norg_sets) {
				const auto& N_veci = impurity_occupied[set_i];
				const auto& N_vecj = impurity_occupied[set_j];
				const Int offset_i = set_begin[set_i];
				const Int offset_j = set_begin[set_j];
				for (const auto& N_i : N_veci) for (const auto& N_j : N_vecj) {
					// if (N_i != N_j) {
						if (N_i < offset_i + p.control_divs[set_i+1][0] && N_j < offset_j + p.control_divs[set_j+1][0]) {
							const Idx op = mat_hop_pos.size() + 1 + four_fermion_index(p.norbit, N_i, N_j, N_j, N_i);
							diagonal += oper_value[op];
							// ? if this term(↓) is necessary? only can mute this term when the interaction happen in different sets.
						}
					// }
				}
			}
		}
		/*
		{
			Str cfg_str;
			for (const auto& onb : a.cfg.cf) { ostringstream oss; oss << onb; cfg_str += oss.str(); }
			//if (cfg_str == "10001000101001010101")
			cout << "PIO: diag h_i=" << h_i << " cfg=" << cfg_str << "ni diag=" << iofmt("sci") << diagonal << endl;
		}
		*/
		// hmlt_splited.addelement(diagonal, h_i, sparse_idx);
		two_body_timer.stop();
            hp::Sample<measured> store_timer(sample,hp::RowStore);
            diagonals[sparse_idx] = diagonal;store_timer.stop();
		/*
		{
			Str cfg_str;
			for (const auto& onb : a.cfg.cf) { ostringstream oss; oss << onb; cfg_str += oss.str(); }
			if (cfg_str == "10001000101001010101")
				cout << "PIO: diag h_i=" << h_i << " cfg=" << cfg_str << " diag=" << iofmt("sci") << diagonal << endl;
		}
		*/
    });
        });
        profile.mark(hp::Diagonal);
        counts.resize(h_table.nrows());
        stage2_parallel::ranges(counts.size(),[&](std::size_t first,std::size_t last,std::size_t) {
            for(auto row=first;row<last;++row)
                counts[row]=1+h_table.row_ptr()[row+1]-h_table.row_ptr()[row];
        });
        profile.mark(hp::RowCounts);
        hmlt_splited.prepare_csr(counts,&profile);
        stage2_parallel::for_each(h_table.nrows(), [&](std::size_t row) {
            auto out = hmlt_splited.csr_row_begin(row);
            hmlt_splited.set_csr_entry(out++, row_H.bgn() + static_cast<BasisIndex>(row), diagonals[row]);
            for (Tab::Offset pos = h_table.row_ptr()[row]; pos < h_table.row_ptr()[row + 1]; ++pos) {
                const Int op = h_table.ops()[pos];
                hmlt_splited.set_csr_entry(out++, h_table.cols()[pos], (op > 0 ? 1 : -1) * oper_value[abs(op)]);
            }
        });
        profile.mark(hp::CsrFill);
    });
    profile.mark(hp::CollectiveSync);
    }
    profile.mark(hp::Cleanup);profile.finish();h_build_seconds_=profile.total();
    hp::emit(mm,profile,"H_PROFILE","offdiagonal_tab",dim,row_H.len(),h_table.nnz(),build);
    if(options.enabled)hp::emit_samples(mm,worker_timing,build,options.sample_every);
    return hmlt_splited;
}

// Try to find the Ground state.
MatReal Operator::lowest_eigpairs(const BasisIndex n, bool if_need_fast, Int wish_nev)
//	n      ( input) is the dimension of the matrix
//	wish_nev    ( input) is the expected maximum eigenstates to be found
//	actual number of eigenpairs is returned
//	nev_eff is the number of eigenvectors corresponding to complete degenerate levels reached
{
	UURandomSFMT uur;
	VecReal eval(wish_nev, 0.);				// is the eigenvalues
	// VEC < VecReal> eigenvec_i;			// a VEC to contain all the eigenvector.
	MatReal eigenvec_i(wish_nev, n);		// a VEC to contain all the eigenvector.
	Int gs_dgcy(1);							// cotain ground state degeneracy.
	// if (mm)PIO("find_hmlt BEGIN :::");
	// SparseMatReal sep_hmltoperator = find_hmlt(table);
	log_ground_state_memory(mm, "H_BEFORE", dim);
	SparseMatReal sep_hmltoperator = find_hmlt_V2(table);
	log_ground_state_memory(mm, "H_AFTER", dim, true, h_build_seconds());
	// if(mm) WRN("finished fiding the hmlt")
//#ifdef _ASSERTION_
//	if (mm.np() == 1) {
//		WRN("test begin:::");
//		if (sep_hmltoperator.if_hermitian() == false)ERR("This sparse matrix is not Hermitian matrix!!!!");
//		WRN("test end  !!!");
//	}
//#endif
    const VecPartition initial_rows(mm.np(),mm.id(),n);
    VecReal inital_state(initial_rows.len());
    preparation_parallel::fill_start(inital_state,initial_rows.bgn(),0);
	
	if(!if_need_fast){
		if(p.if_eigenstate_number){
			wish_nev = p.eigenstate_number;
		}
	}

	if(mm) WRN(NAV3(p.if_eigenstate_number,p.eigenstate_number,wish_nev));
	log_ground_state_memory(mm, "LANCZOS_BEFORE", dim);
	VecInt krylov_space_size = lanczos(eval, eigenvec_i, gs_dgcy, n, wish_nev,
		sep_hmltoperator, inital_state, mm, if_need_fast, 1000,
		p.gs_lanczos_reorthogonalization, p.green_krylov.orth_method, true);
	log_ground_state_memory(mm, "LANCZOS_AFTER", dim);
	if(mm) {cout <<"PIO: krylov_space_size = ";cout <<iofmt("sci"); for_Int(i, 0, krylov_space_size.size()) cout << eval[i] << ","<< krylov_space_size[i] << "; "; cout << std::endl;}
	// if(mm) std::cout << "The eigenvalue" << iofmt("sci") << eval << std::endl;
	groundstate_energy = eval[0];
	if(p.degel < gs_dgcy) {p.degel = gs_dgcy; if(mm) WRN(NAV(p.degel));}

	if(!if_need_fast){
		if(p.if_eigenstate_number){
			p.degel = p.eigenstate_number;
			if(mm) WRN(NAV2(p.degel,eigenvec_i.size()));
		}
	}

	// MatReal eigenvec(eigenvec_i.size(),n);
	// for_Int(i, 0, eigenvec_i.size()) eigenvec[i] = eigenvec_i[i];
	// ground_state = eigenvec_i[0];
	ground_state.reset(eigenvec_i.truncate_row(0, p.degel));

	/*
	// Compute <n_i n_j> for impurity div-0 orbitals only.
	// global_off[k]: global spinless orbit start of set k (= sum_{m<k} nO2sets[m])
	// imp_size[k] = control_divs[k+1][0]: number of impurity (div-0) orbitals in set k
	// imp_off[k]: continuous impurity index start of set k (= sum_{m<k} imp_size[m])
	// n_imp = total impurity orbitals = one impurity per spin-orbital set
	// Algorithm: for each many-body state h_i, build ngs[r][li] = gs[h_i] if orbital r is occupied,
	//            then ninj_mat[r][c] += dot(ngs[r], ngs[c]) / degel
	{
		VecInt global_off(p.norg_sets, 0), imp_off(p.norg_sets, 0), imp_size(p.norg_sets, 0);
		for_Int(k, 0, p.norg_sets) imp_size[k] = p.control_divs[k+1][0];
		for_Int(k, 1, p.norg_sets) { global_off[k] = global_off[k-1] + p.nO2sets[k-1]; imp_off[k] = imp_off[k-1] + imp_size[k-1]; }
		Int n_imp = 0; for_Int(k, 0, p.norg_sets) n_imp += imp_size[k];
		ninj_mat.reset(n_imp, n_imp, 0.);
		VecPartition row_H(mm.np(), mm.id(), dim);
		MatReal ninj_loc(n_imp, n_imp, 0.);
		for_Int(egs_idx, 0, p.degel) {
			VecReal gs = ground_state[egs_idx]; gs.normalize();
			VecReal gs_part = gs.truncate(row_H.bgn(), row_H.end());
			MatReal ngs(n_imp, row_H.len(), 0.);
			for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i) {
				Int li = h_i - row_H.bgn();
				StateStatistics a(h_i, scsp.wherein_NocSpace(h_i), scsp);
				for_Int(k, 0, p.norg_sets)
					for (const auto& N : a.filled_spinless[k])
						if (N >= global_off[k] && N < global_off[k] + imp_size[k])
							ngs[imp_off[k] + (N - global_off[k])][li] = gs_part[li];
			}
			for_Int(r, 0, n_imp) for_Int(c, 0, n_imp)
				ninj_loc[r][c] += DOT(ngs[r], ngs[c]) / p.degel;
		}
		ninj_mat = mm.Allreduce(ninj_loc);

		if(mm) WRN(NAV(ninj_mat));
	}
	*/

	/*
	MatReal cicj_mat;

	// ============================================================
	// 3. Compute cicj_mat
	//
	//     cicj_mat[i][j]
	//       = <GS|c_i^+ c_j|GS>
	//       = <c_i GS|c_j GS>
	//
	// This implementation does NOT use table.
	//
	// Only impurity orbitals in div 0 are included.
	//
	// The impurity-orbital order is exactly the same as ninj_mat:
	//
	//     all impurity orbitals of set 0,
	//     followed by all impurity orbitals of set 1,
	//     ...
	// ============================================================

	{
		VecInt global_off(p.norg_sets, 0);
		VecInt imp_off(p.norg_sets, 0);
		VecInt imp_size(p.norg_sets, 0);

		// --------------------------------------------------------
		// 3.1 Identify div-0 impurity orbitals
		// --------------------------------------------------------

		for_Int(k, 0, p.norg_sets) {

			imp_size[k] =
				p.control_divs[k + 1][0];

			if (
				imp_size[k] >
				scsp.sit_mat[k][0]
			) {
				ERR(
					"lowest_eigpairs: "
					"impurity size exceeds div-0 size: "
					+ NAV3(
						k,
						imp_size[k],
						scsp.sit_mat[k][0]
					)
				);
			}
		}

		for_Int(k, 1, p.norg_sets) {

			global_off[k] =
				global_off[k - 1]
				+ p.nO2sets[k - 1];

			imp_off[k] =
				imp_off[k - 1]
				+ imp_size[k - 1];
		}

		Int n_imp = 0;

		for_Int(k, 0, p.norg_sets) {
			n_imp += imp_size[k];
		}

		cicj_mat.reset(
			n_imp,
			n_imp,
			0.0
		);

		MatReal cicj_loc(
			n_imp,
			n_imp,
			0.0
		);

		VecPartition row_H(
			mm.np(),
			mm.id(),
			dim
		);

		
		for_Int(set_i, 0, p.norg_sets) {

			if (imp_size[set_i] == 0) {
				continue;
			}

			
			if (p.npartical[set_i] <= 0) {
				continue;
			}

			// ----------------------------------------------------
			// 3.2 Construct the corresponding N-1 space
			// ----------------------------------------------------

			VecInt nppso_m1(p.npartical);

			--nppso_m1[set_i];

			NocSpace scsp_m1(
				mm,
				p,
				nppso_m1
			);

			VecPartition row_m1(
				mm.np(),
				mm.id(),
				scsp_m1.dim
			);

			// ----------------------------------------------------
			// 3.3 Apply one impurity annihilation operator
			//
			//     result = c_{set_i,imp_pos}|state>
			//
			// state_full:
			//     full N-particle vector available on every rank
			//
			// return:
			//     local segment of the N-1 result vector
			// ----------------------------------------------------

			auto apply_imp_ann =
				[&](
					const VecReal& state_full,
					const Int imp_pos
				) -> VecReal {

				if (
					imp_pos < 0 ||
					imp_pos >= imp_size[set_i]
				) {
					ERR(
						"lowest_eigpairs/apply_imp_ann: "
						"wrong impurity position: "
						+ NAV3(
							set_i,
							imp_pos,
							imp_size[set_i]
						)
					);
				}

				
				VecReal result_part(
					scsp_m1.dim,
					0.0
				);

				
				const Int cf_idx =
					set_i * scsp.ndivs;

				
				const Int global_orb =
					global_off[set_i]
					+ imp_pos;

				for_Int(
					h_i,
					row_H.bgn(),
					row_H.end()
				) {
					const Real amp =
						state_full[h_i];

					if (ABS(amp) < 1E-30) {
						continue;
					}

					const Int subnosp =
						scsp.wherein_NocSpace(h_i);

					StateStatistics stat_old(
						h_i,
						subnosp,
						scsp
					);

					
					if (
						!stat_old.cfg.cf[cf_idx]
							.isocc(imp_pos)
					) {
						continue;
					}

					
					MatInt new_nospdiv =
						scsp.div[subnosp];

					--new_nospdiv[set_i][0];

					if (
						!scsp_m1.ifin_NocSpace(
							new_nospdiv,
							scsp_m1.nppso
						)
					) {
						continue;
					}

					
					const Int sign =
						stat_old.cfg.sgn(
							-1,
							global_orb
						);

					
					VecOnb new_cf =
						stat_old.cfg.cf;

					new_cf[cf_idx] =
						new_cf[cf_idx]
							.ann(imp_pos);

					
					const ComDivs new_group(
						new_cf,
						new_nospdiv,
						scsp_m1.sit_mat
					);

					const Str key =
						new_nospdiv
							.vec()
							.string();

					auto it =
						scsp_m1.divs_to_idx.find(
							key
						);

					if (
						it ==
						scsp_m1.divs_to_idx.end()
					) {
						ERR(
							"lowest_eigpairs/"
							"apply_imp_ann: "
							"target div sector "
							"not found: "
							+ key
						);
					}

					const Int final_idx =
						it->second
						+ new_group.idx;

					if (
						final_idx < 0 ||
						final_idx >= scsp_m1.dim
					) {
						ERR(
							"lowest_eigpairs/"
							"apply_imp_ann: "
							"final index out of range: "
							+ NAV4(
								final_idx,
								it->second,
								new_group.idx,
								scsp_m1.dim
							)
						);
					}

					result_part[final_idx] +=
						Real(sign) * amp;
				}

				
				VecReal result_full =
					mm.Allreduce(result_part);

				
				VecReal result_local(
					row_m1.len(),
					0.0
				);

				for_Int(
					li,
					0,
					row_m1.len()
				) {
					result_local[li] =
						result_full[
							row_m1.bgn() + li
						];
				}

				return result_local;
			};

			// ----------------------------------------------------
			// 3.4 Calculate <c_i GS|c_j GS>
			// ----------------------------------------------------

			for_Int(
				egs_idx,
				0,
				p.degel
			) {
				VecReal gs =
					ground_state[egs_idx];

				gs.normalize();

				
				VEC<VecReal> ann_state(
					imp_size[set_i]
				);

				for_Int(
					a,
					0,
					imp_size[set_i]
				) {
					ann_state[a] =
						apply_imp_ann(
							gs,
							a
						);
				}

				
				for_Int(
					i,
					0,
					imp_size[set_i]
				) {
					const Int gi =
						imp_off[set_i] + i;

					for_Int(
						j,
						0,
						imp_size[set_i]
					) {
						const Int gj =
							imp_off[set_i] + j;

						cicj_loc[gi][gj] +=
							DOT(
								ann_state[i],
								ann_state[j]
							)
							/ Real(p.degel);
					}
				}
			}
		}

		
		cicj_mat =
			mm.Allreduce(cicj_loc);

		// --------------------------------------------------------
		// 3.5 Output and consistency checks
		// --------------------------------------------------------

		if (mm) {
			WRN(
				"impurity one-body density matrix "
				"<c_i^+ c_j>:"
			);

			WRN(NAV(cicj_mat));

			
			Real max_diag_err = 0.0;

			Real max_sym_err = 0.0;

			for_Int(i, 0, n_imp) {

				const Real diag_err =
					ABS(
						cicj_mat[i][i]
						- ninj_mat[i][i]
					);

				if (
					diag_err >
					max_diag_err
				) {
					max_diag_err =
						diag_err;
				}

				for_Int(j, 0, n_imp) {

					const Real sym_err =
						ABS(
							cicj_mat[i][j]
							- cicj_mat[j][i]
						);

					if (
						sym_err >
						max_sym_err
					) {
						max_sym_err =
							sym_err;
					}
				}
			}

			WRN(
				"check max_i "
				"|<c_i^+c_i>-<n_i>| = "
				+ STR(max_diag_err)
			);

			WRN(
				"check max_ij "
				"|rho_ij-rho_ji| = "
				+ STR(max_sym_err)
			);
		}
	}
	*/


	// if(mm) WRN("finished lanczos.")



#ifdef _ASSERTION_
	//WRN("TEST_Lanczos is right:::" + NAV(mm.np()));
	//VecReal test_a(eigenvec[0]);
	//VecReal test_b(sep_hmltoperator * test_a);
	//WRN("TEST_Lanczos[0] state" + NAV3(test_a.isnormalized(), eval[0], test_b.avg_abs_elem_diff(eval[0] * test_a)));
#endif
	// if(mm) WRN(NAV(groundstate_energy));
	return eigenvec_i;
}


VecReal Operator::sn_prtcl_ex_state(const Int imp_div, const VecReal ground_state, const Int crtann) const
{
	VecPartition row_H(mm.np(), mm.id(), dim);
	NocSpace::GroupCursor group_cursor(scsp);
	VecReal ex_state_part(row_H.len(), 0.);
	for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i)
	{
		const BasisIndex subscsp = group_cursor.locate(h_i);
		ComDivs cfg(h_i - scsp.idx_div[subscsp], (scsp.div[subscsp]), (scsp.sit_mat), true);
		if (crtann == +1)if (cfg.cf[imp_div * scsp.ndivs].isuno(0))ex_state_part[h_i - row_H.bgn()] = 1.;
		if (crtann == -1)if (cfg.cf[imp_div * scsp.ndivs].isocc(0))ex_state_part[h_i - row_H.bgn()] = 1.;
	}
	ex_state_part *= ground_state.truncate(row_H.bgn(), row_H.end());
	VecReal ex_state(dim, 0.);
	ex_state = mm.Allgatherv(ex_state_part, row_H);
	return std::move(ex_state);
}

//------------------------------------------------------------------ io ------------------------------------------------------------------

void Operator::save_the_Tab(const Tab& tab, Str name) const {
    Int success = 1;
    try {
        VecPartition rows(mm.np(), mm.id(), dim);
        tab.require_layout(dim, rows.bgn(), rows.len());
        tab_io::save_shard(name, mm.id(), tab);
    } catch (const std::exception& e) {
        success = 0;
        std::cerr << "Tab write rank " << mm.id() << ": " << e.what() << '\n';
    }
    if (mm.Allreduce(success) != mm.np())
        throw std::runtime_error("Failed to save one or more Tab shards");
    if (mm) {
        try { tab_io::write_info(name, dim, mm.np()); }
        catch (const std::exception& e) { success = 0; std::cerr << e.what() << '\n'; }
    }
    mm.Bcast(success);
    if (!success) throw std::runtime_error("Failed to save Tab metadata");
}

void Operator::write_the_multiTab(Str name) const {
    OFS ofs(name + to_string(mm.id()) + ".txt");
    ofs << "\t" << setw(w_Real) << "row_idx"
        << "\t" << setw(w_Real) << "col_idx"
        << "\t" << setw(w_Real) << "hmt_idx" << '\n';
    for_Idx(row, 0, table.nrows())
        for (Tab::Offset pos = table.row_ptr()[row]; pos < table.row_ptr()[row + 1]; ++pos)
            ofs << "\t" << setw(w_Real) << row
                << "\t" << setw(w_Real) << table.cols()[pos]
                << "\t" << setw(w_Real) << table.ops()[pos] << '\n';
}

Tab Operator::read_the_Tab(Str name) const {
    Tab tab;
    Int success = 1;
    try {
        VecPartition rows(mm.np(), mm.id(), dim);
        const auto info = tab_io::read_info(name);
        tab = tab_io::load(name, info, mm.np(), mm.id(), dim, rows.bgn(), rows.len());
    } catch (const std::exception& e) {
        success = 0;
        std::cerr << "Tab read rank " << mm.id() << ": " << e.what() << '\n';
    }
    if (mm.Allreduce(success) != mm.np())
        throw std::runtime_error("Failed to read one or more Tab shards");
    return tab;
}

// ! only suit for the two orbital cases.
MatReal Operator::local_multiplets_state(const MatReal& g_state) const {
	MatReal local_multiplets_state(3, 6, 0.);
	VecReal gs_p = g_state[0]; gs_p.normalize();

	//set the first line for N;
	// {0, 1, 2,..., 2*n}
	local_multiplets_state[0] = VecReal{ 0,  1,  2,  2,  3,  4 };
	//set the second line for Sz;
	// {0, 1/2, 2/2,..., n/2}
	local_multiplets_state[1] = VecReal{ 0,0.5,  0,  1,0.5,  0 };

	NocSpace::GroupCursor group_cursor(scsp);
	for (BasisIndex h_i = 0; h_i < dim; ++h_i) {
		StateStatistics a(h_i, group_cursor.locate(h_i), scsp);
		VecInt imp(4, 0);// the last one left for check the right.
		for_Int(i, 0, p.norbs) imp[i] = a.cfg.cf[(i)*scsp.ndivs].isocc(0) ? 1 : 0;
		switch (SUM(imp)) {
		case 0:
			local_multiplets_state[2][0] += gs_p[h_i] * gs_p[h_i];
			break;
		case 1:
			local_multiplets_state[2][1] += gs_p[h_i] * gs_p[h_i];
			break;
		case 2:
			if (SUM(imp.mat(2, 2).tr()[0]) == 1) local_multiplets_state[2][2] += gs_p[h_i] * gs_p[h_i];
			else local_multiplets_state[2][3] += gs_p[h_i] * gs_p[h_i];
			break;
		case 3:
			local_multiplets_state[2][4] += gs_p[h_i] * gs_p[h_i];
			break;
		case 4:
			local_multiplets_state[2][5] += gs_p[h_i] * gs_p[h_i];
			break;
		default:
			cout << "Input is not within the specified range" << endl;
		}
	}
	return local_multiplets_state;
}

