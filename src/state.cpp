/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "state.h"

using namespace std;
StateStatistics::StateStatistics(BasisIndex h_i, BasisIndex comdiv_idx, const NocSpace& scsp, OccupationStatistics stats) :
	space(scsp), div_idx(comdiv_idx), idx(h_i - scsp.idx_div[comdiv_idx]),
	occ_n(scsp.div[comdiv_idx]), cfg(cfgdetail()), filled_spinless(stats == OccupationStatistics::collect ? scsp.sit_mat.nrows() : 0)
	// cfgup(cfg.cf.truncate(0, scsp.ndivs)), cfgdw(cfg.cf.truncate(scsp.ndivs, scsp.ndivs + scsp.ndivs))
{
	//DBG("statistics BEGIN: " + NAV(idx));
	if (stats == OccupationStatistics::collect) statistics();
}

TwoFermionGroupIndex::TwoFermionGroupIndex(const NocSpace& space) : space_(space)
{
    if(space.div.empty())return;
    namespace par=stage2_parallel;
    const std::size_t shards=par::workers(space.div.size());
    slots_.resize(shards);
    std::vector<std::size_t> hashes(space.div.size());
    std::vector<std::vector<std::vector<Idx>>> routed(shards,
        std::vector<std::vector<Idx>>(shards));
    par::ranges(space.div.size(),[&](std::size_t first,std::size_t last,std::size_t part){
        for(auto g=first;g<last;++g) {
            const auto& occ=space.div[g];
            const auto hash=std::hash<std::string_view>{}(
                std::string_view(reinterpret_cast<const char*>(occ.p()),occ.size()));
            hashes[g]=hash;routed[part][hash%shards].push_back(g);
        }
    });
    // Each shard has exactly one writer; group insertion order remains deterministic.
    par::for_each(shards,[&](std::size_t shard){
        auto& slots=slots_[shard];std::size_t count=0;
        for(const auto& part:routed)count+=part[shard].size();
        if(count>slots.max_size()/2)throw std::overflow_error("Group index too large");
        std::size_t capacity=1;
        while(capacity<count*2) {
            if(capacity>slots.max_size()/2)throw std::overflow_error("Group index capacity overflow");
            capacity*=2;
        }
        slots.assign(capacity,empty_);
        for(const auto& part:routed)for(const auto g:part[shard]) {
            std::size_t slot=(hashes[g]/shards)&(capacity-1);
            while(slots[slot]!=empty_)slot=(slot+1)&(capacity-1);
            slots[slot]=g;
        }
    });
}

BasisIndex TwoFermionGroupIndex::at(std::string_view key) const
{
    if(!slots_.empty()) {
        const auto hash=std::hash<std::string_view>{}(key);
        const auto& slots=slots_[hash%slots_.size()];
        const auto mask=slots.size()-1;
        std::size_t slot=(hash/slots_.size())&mask;
        while(slots[slot]!=empty_) {
            const Idx group=slots[slot];const auto& occ=space_.div[group];
            if(key.size()==occ.size() && std::memcmp(key.data(),occ.p(),key.size())==0)
                return space_.idx_div[group];
            slot=(slot+1)&mask;
        }
    }
    throw std::out_of_range("Two-fermion target group missing");
}

TwoFermionGroupCache::TwoFermionGroupCache(const TwoFermionGroupIndex& index, tab_timing::Worker* timing)
    : space_(index.space()), index_(index), orbital_begin_(space_.sit_mat.size()),
      local_rank_(space_.sit_mat.size()),
      target_(space_.sit_mat.nrows(), space_.ndivs), counts_(space_.ndivs, 0),
      key_(space_.sit_mat.size(), '\0'),
      occupation_prefix_(space_.sit_mat.size() + 1, 0), timing_(timing)
{
    const std::size_t columns = space_.ndivs;
    if (columns && columns > allowed_.max_size() / columns)
        throw std::overflow_error("Two-fermion constraint cache too large");
    allowed_.resize(columns * columns);
    Int offset = 0;
    for (Idx d = 0; d < orbital_begin_.size(); ++d) {
        orbital_begin_[d] = offset;
        offset += space_.sit_mat.p()[d];
        if (space_.sit_mat.p()[d] > 1) single_orbital_ = false;
    }
}

void TwoFermionGroupCache::refresh(BasisIndex group)
{
    sample_group_ = timing_ && timing_->sample_group();
    tab_timing::Span refresh_time(sample_group_ ? &timing_->seconds[tab_timing::GroupRefresh] : nullptr);
    active_rules_ = 0;
    entries_per_row_ = 0;
    needs_rank_ = false;
    // One editable matrix per group, rather than one per candidate per basis.
    MatInt& target = target_;
    VecInt& counts = counts_;
    std::string& key = key_;
    std::fill(counts.p(), counts.p() + counts.size(), 0);
    std::fill(allowed_.begin(), allowed_.end(), -1);
    for (Idx d = 0; d < target.size(); ++d) {
        target.p()[d] = space_.div[group][d];
        counts[d % space_.ndivs] += target.p()[d];
        key[d] = static_cast<char>(target.p()[d]);
        occupation_prefix_[d + 1] = occupation_prefix_[d] + target.p()[d];
    }
    // Validate invariant source conditions once per group, not per candidate.
    if (!space_.ifin_NocSpace(target, space_.nppso))
        throw std::logic_error("Invalid source occupation group");
    // Column constraints depend on (a,b), not on the spin/orbital set.
    // Cache only after a candidate passes its own cell-capacity checks.
    const Int ndiv = static_cast<Int>(space_.ndivs);
    for (Int set = 0; set < space_.p.norg_sets; ++set) {
        for (Int a = 0; a < ndiv; ++a) {
            for (Int b = 0; b < ndiv; ++b) {
                // Discard rules that produce no concrete occupied-to-empty hop.
                // This also removes single-orbital intra-division candidates.
                if (target[set][a] == 0
                    || target[set][b] == space_.sit_mat[set][b]) continue;
                --target[set][a];
                ++target[set][b];
                --counts[a]; ++counts[b];
                bool allowed;
                {
                    tab_timing::Span check_time(sample_group_ ? &timing_->seconds[tab_timing::Constraints] : nullptr);
                    signed char& cached = allowed_[static_cast<std::size_t>(a) * ndiv + b];
                    if (cached < 0)
                        cached = space_.two_fermion_columns_allowed(counts, a, b) ? 1 : 0;
                    allowed = cached != 0;
                }
                if (allowed) {
                    if (active_rules_ == rules_.size()) rules_.emplace_back();
                    Rule& rule = rules_[active_rules_++];
                    rule.ann_div = set * ndiv + a;
                    rule.crt_div = set * ndiv + b;
                    if (target[set][a] > std::numeric_limits<std::uint8_t>::max()
                        || target[set][b] > std::numeric_limits<std::uint8_t>::max())
                        throw std::overflow_error("Target occupation exceeds compact key range");
                    {
                    tab_timing::Span lookup_time(sample_group_ ? &timing_->seconds[tab_timing::TargetLookup] : nullptr);
                    const char old_a = key[rule.ann_div], old_b = key[rule.crt_div];
                    key[rule.ann_div] = static_cast<char>(target[set][a]);
                    key[rule.crt_div] = static_cast<char>(target[set][b]);
                    rule.target_begin = index_.at(key);
                    key[rule.ann_div] = old_a; key[rule.crt_div] = old_b;
                    }
                    rule.single_to_single = space_.sit_mat[set][a] == 1
                        && space_.sit_mat[set][b] == 1;
                    if (!rule.single_to_single) {
                        // Same mixed-radix weights as ComDivs::find_cf_idx.
                        // Cache them once; local Onb::idx() retains legacy ordering.
                        rule.target_weights.resize(target.size());
                        BasisIndex weight = 1;
                        for (Idx d = 0; d < target.size(); ++d) {
                            rule.target_weights[d] = weight;
                            const BasisIndex radix = BnmlFast{}(space_.sit_mat.p()[d], target.p()[d]);
                            weight = basis_count_product(weight, radix);
                        }
                        needs_rank_ = true;
                    }
                    const std::size_t occupied = space_.div[group][rule.ann_div];
                    const std::size_t empty = space_.sit_mat.p()[rule.crt_div]
                        - space_.div[group][rule.crt_div];
                    const auto limit = std::numeric_limits<std::size_t>::max();
                    if (occupied && empty > (limit - entries_per_row_) / occupied)
                        throw std::overflow_error("Two-fermion row size overflow");
                    entries_per_row_ += occupied * empty;
                }
                --counts[b]; ++counts[a];
                --target[set][b];
                ++target[set][a];
            }
        }
    }
    group_ = group;
}

void TwoFermionGroupCache::prepare(BasisIndex group, std::size_t rows, TabBuilder& table)
{
    if (group_ != group) refresh(group);
    if (entries_per_row_ && rows > std::numeric_limits<std::size_t>::max() / entries_per_row_)
        throw std::overflow_error("Two-fermion group size overflow");
    tab_timing::Span reserve_time(sample_group_ ? &timing_->seconds[tab_timing::Reserve] : nullptr);
    table.reserve_additional(rows * entries_per_row_);
}

void TwoFermionGroupCache::append_single_orbital(BasisIndex local_row, TabBuilder& table) const
{
    if (!single_orbital_ || group_ < 0)
        throw std::logic_error("Unprepared single-orbital Tab builder");
    for (std::size_t k = 0; k < active_rules_; ++k) {
        const Rule& rule = rules_[k];
        const Int a = rule.ann_div, b = rule.crt_div;
        const Int lo = std::min(a, b), hi = std::max(a, b);
        const Int sign = 1 - 2 * ((occupation_prefix_[hi] - occupation_prefix_[lo + 1]) & 1);
        const BasisIndex col = rule.target_begin;
        if (col < 0 || col >= space_.dim)
            throw std::out_of_range("Two-fermion target column outside basis/index range");
        table.append(local_row, col,
            sign * (orbital_begin_[b] * space_.p.norbit + orbital_begin_[a] + 1));
    }
}

void TwoFermionGroupCache::append(const StateStatistics& state, BasisIndex local_row, TabBuilder& table)
{
    if (&state.space != &space_ || space_.p.if_norg_imp)
        ERR("TwoFermionGroupCache requires its own space and if_norg_imp=false");
    if (group_ != state.div_idx) refresh(state.div_idx);
    if (needs_rank_)
        for (Idx d = 0; d < local_rank_.size(); ++d) local_rank_[d] = state.cfg.cf[d].idx();

    for (std::size_t k = 0; k < active_rules_; ++k) {
        const Rule& rule = rules_[k];
        const Int a = rule.ann_div, b = rule.crt_div;
        BasisIndex unchanged = 0;
        if (!rule.single_to_single) {
            for (Idx d = 0; d < local_rank_.size(); ++d)
                if (d != static_cast<Idx>(a) && d != static_cast<Idx>(b))
                    unchanged += rule.target_weights[d] * local_rank_[d];
        }
        // Keep the original set/division/orbital traversal and summation order.
        for (const Int q_local : state.cfg.div_orb_e[a]) {
            const Int q = orbital_begin_[a] + q_local;
            const Onb removed = state.cfg.cf[a].ann(q_local);
            const BasisIndex ann_rank = rule.single_to_single || a == b ? 0 : removed.idx();
            for (const Int p_local : state.cfg.div_orb_c[b]) {
                const Int p = orbital_begin_[b] + p_local;
                BasisIndex target_rank;
                if (rule.single_to_single) {
                    target_rank = state.idx;
                } else if (a == b) {
                    target_rank = unchanged + rule.target_weights[a]
                        * removed.crt(p_local).idx();
                } else {
                    target_rank = unchanged + rule.target_weights[a] * ann_rank
                        + rule.target_weights[b] * state.cfg.cf[b].crt(p_local).idx();
                }
                const BasisIndex col = rule.target_begin + target_rank;
                if (col < 0 || col >= space_.dim)
                    ERR("Two-fermion target column outside basis/index range");
                table.append(local_row, col,
                    state.cfg.sgn(q, p) * (p * space_.p.norbit + q + 1));
            }
        }
    }
}

vector<array<BasisIndex, 4>> StateStatistics::find_each_spiless_group_off_diagonal_term(const hopdata &hopspi, const Int sets_n)
{
	vector<array<BasisIndex, 4>> off_diagonal_term;
	{
		array<BasisIndex, 4> ODT_i{};// [0]\([1]):annihilation\(creation) orbit's position;[2]:Colum idx(i);[i][3]:sign.
		//off-diagonal, two-fermion operator terms for the Up spin C^+_i C_j
		for_Int(i, 0, hopspi.size()) {
			BasisIndex idx_div_i(find_newdiv_idx(get<2>(hopspi[i])));
			Int div_pst_ann(get<0>(hopspi[i]) + sets_n * space.ndivs), div_pst_crt(get<1>(hopspi[i]) + sets_n * space.ndivs);

			for_Int(c, 0, cfg.div_orb_e[div_pst_ann].size()) {
				for_Int(cp, 0, cfg.div_orb_c[div_pst_crt].size()) {
					// for_Int(k, 0, sets_n){ ODT_i[0] += SUM(space.sit_mat[k]); ODT_i[1] += SUM(space.sit_mat[k]); }
					ODT_i[0] = (SUM_0toX(space.sit_mat, sets_n, get<0>(hopspi[i])) + cfg.div_orb_e[div_pst_ann][c]);
					ODT_i[1] = (SUM_0toX(space.sit_mat, sets_n, get<1>(hopspi[i])) + cfg.div_orb_c[div_pst_crt][cp]);
					VecOnb newcf(cfg.cf);
					#ifdef _ASSERTION_
						if (newcf[div_pst_ann].isuno(cfg.div_orb_e[div_pst_ann][c]))ERR("This position can't ann");
						if (newcf[div_pst_crt].isocc(cfg.div_orb_c[div_pst_crt][cp]))ERR("This position can't crt");
					#endif
					newcf[div_pst_ann] = newcf[div_pst_ann].ann(cfg.div_orb_e[div_pst_ann][c]);
					newcf[div_pst_crt] = newcf[div_pst_crt].crt(cfg.div_orb_c[div_pst_crt][cp]);
					
					ComDivs b(newcf, (get<2>(hopspi[i])), (space.sit_mat));
					ODT_i[2] = idx_div_i + b.idx;
					if (ODT_i[2] >= space.dim) ERR(STR("Hmlt Off-Diag Elements IHTL > IHM ") + NAV2(ODT_i[2], space.dim));
					ODT_i[3] = cfg.sgn(ODT_i[0], ODT_i[1]);
					#ifdef _ASSERTION_
						if (ODT_i[0] == ODT_i[1]) ERR("A impossible thing happened:ODT_i[0] == ODT_i[1]" + NAV(ODT_i[0]));
					#endif
					off_diagonal_term.push_back(ODT_i);
				}
			}
		}
	}
	return off_diagonal_term;
}

VEC<array<BasisIndex, 6>> StateStatistics::find_off_diagonal_term_fourFermi(const furfrm &furfrm, const Int sets_i, const Int sets_j)
{
	VEC<array<BasisIndex, 6>> off_diagonal_term;
	{
		// [0]~[3] i-j-k-l orbit's position; [4]:Colum idx(i);[5]:sign.
		//off-diagonal, four-fermion operator terms for C^+_i C^+_j C_k C_l
		array<BasisIndex, 6> ODT_i = {0,0,0,0,0,0};
		
		for_Int(i, 0, furfrm.size()) {
			BasisIndex idx_div_i(find_newdiv_idx(furfrm[i].second));
			Int div_pst_crt_i(furfrm[i].first[0] + sets_i * space.ndivs), div_pst_crt_j(furfrm[i].first[1] + sets_j * space.ndivs), 
				div_pst_ann_k(furfrm[i].first[2] + sets_j * space.ndivs), div_pst_ann_l(furfrm[i].first[3] + sets_i * space.ndivs);

			for_Int(cp_i, 0, cfg.div_orb_c[div_pst_crt_i].size()) {
			for_Int(cp_j, 0, cfg.div_orb_c[div_pst_crt_j].size()) {
			for_Int(c__k, 0, cfg.div_orb_e[div_pst_ann_k].size()) {
			for_Int(c__l, 0, cfg.div_orb_e[div_pst_ann_l].size()) {

				ODT_i[0] = (SUM_0toX(space.sit_mat, sets_i, furfrm[i].first[0]) + cfg.div_orb_c[div_pst_crt_i][cp_i]);
				ODT_i[1] = (SUM_0toX(space.sit_mat, sets_j, furfrm[i].first[1]) + cfg.div_orb_c[div_pst_crt_j][cp_j]);
				ODT_i[2] = (SUM_0toX(space.sit_mat, sets_j, furfrm[i].first[2]) + cfg.div_orb_e[div_pst_ann_k][c__k]);
				ODT_i[3] = (SUM_0toX(space.sit_mat, sets_i, furfrm[i].first[3]) + cfg.div_orb_e[div_pst_ann_l][c__l]);
				if(ODT_i[0] != ODT_i[1] && ODT_i[2] != ODT_i[3]) {
					VecOnb newcf(cfg.cf);
					
					newcf[div_pst_crt_i] = newcf[div_pst_crt_i].crt(cfg.div_orb_c[div_pst_crt_i][cp_i]);
					newcf[div_pst_crt_j] = newcf[div_pst_crt_j].crt(cfg.div_orb_c[div_pst_crt_j][cp_j]);
					newcf[div_pst_ann_k] = newcf[div_pst_ann_k].ann(cfg.div_orb_e[div_pst_ann_k][c__k]);
					newcf[div_pst_ann_l] = newcf[div_pst_ann_l].ann(cfg.div_orb_e[div_pst_ann_l][c__l]);
					
					ComDivs b(newcf, furfrm[i].second, space.sit_mat);
					ODT_i[4] = idx_div_i + b.idx;
					if (ODT_i[4] >= space.dim) ERR(STR("Hmlt Off-Diag Elements IHTL > IHM ") + NAV2(ODT_i[2], space.dim));
					ODT_i[5] = cfg.sgn(ODT_i[0], ODT_i[3]);	// sign for C^+_i C_l
					ODT_i[5] *= cfg.sgn(ODT_i[1], ODT_i[2]);// sign for C^+_j C_k
					auto [min, max] = minmax(ODT_i[0], ODT_i[3]);
					if (ODT_i[1] == myclamp(ODT_i[1], min, max)) ODT_i[5] *= -1;
					if (ODT_i[2] == myclamp(ODT_i[2], min, max)) ODT_i[5] *= -1;
					off_diagonal_term.push_back(ODT_i);
				}
			}
			}
			}
			}
		}
	}
	return off_diagonal_term;
}

VEC<array<BasisIndex, 4>> StateStatistics::off_diagonal_soc_term(const VEC<MatInt> &hop_soc)
{
	VEC<array<BasisIndex, 4>> off_diagonal_term;
	{
		array<BasisIndex, 4> ODT_i{};// [0]\([1]):annihilation\(creation) orbit's position;[2]:Colum idx(i);[i][3]:sign.

		//off-diagonal, SOC operator terms
		for_Int(i, 0, hop_soc.size()) {
			BasisIndex idx_div_i(find_newdiv_idx(hop_soc[i]));
			VecOnb newcf(cfg.cf);
			
			for_Int(orb, 0, space.sit_mat.nrows()) {
				Int a = Int(occ_n[orb * space.ndivs]) - hop_soc[i][orb][0];
				if(a = 1)			newcf[orb * space.ndivs] = newcf[orb * space.ndivs].ann(0);
				else if(a = -1)		newcf[orb * space.ndivs] = newcf[orb * space.ndivs].crt(0);
				else ERR("This position can't ann");
			}
			
			ComDivs b(newcf, hop_soc[i], space.sit_mat);
			ODT_i[2] = idx_div_i + b.idx;
			if (ODT_i[2] >= space.dim) ERR(STR("Hmlt Off-Diag Elements IHTL > IHM ") + NAV2(ODT_i[2], space.dim));
			ODT_i[3] = cfg.sgn(ODT_i[0], ODT_i[1]);
			#ifdef _ASSERTION_
				if (ODT_i[0] == ODT_i[1]) ERR("A impossible thing happened:ODT_i[0] == ODT_i[1]" + NAV(ODT_i[0]));
			#endif
			off_diagonal_term.push_back(ODT_i);
		}
	}
	return off_diagonal_term;
}

// This  function is Deprecation!
/* vector<VecInt> StateStatistics::find_off_diagonal_term(const hopdata& hopup, const hopdata& hopdw)
{
	vector<VecInt> off_diagonal_term;
	// UP term.
	{
		VecInt ODT_i(4, 0);// [0]\([1]):annihilation\(creation) orbit's position;[2]:Colum idx(i);[i][3]:sign.
		//off-diagonal, two-fermion operator terms for the Up spin C^+_i C_j
		for_Int(i, 0, hopup.size()) {
			BasisIndex idx_div_i(find_newdiv_idx(get<2>(hopup[i])));
			for_Int(c, 0, cfg.div_orb_e[get<0>(hopup[i])].size()) {
				for_Int(cp, 0, cfg.div_orb_c[get<1>(hopup[i])].size()) {
					ODT_i[0] = (SUM_0toX(sit_n, get<0>(hopup[i])) + cfg.div_orb_e[get<0>(hopup[i])][c]);
					ODT_i[1] = (SUM_0toX(sit_n, get<1>(hopup[i])) + cfg.div_orb_c[get<1>(hopup[i])][cp]);
					VecOnb newcf(cfg.cf);
#ifdef _ASSERTION_
					if (newcf[get<0>(hopup[i])].isuno(cfg.div_orb_e[get<0>(hopup[i])][c]))ERR("This position can't ann");
					if (newcf[get<1>(hopup[i])].isocc(cfg.div_orb_c[get<1>(hopup[i])][cp]))ERR("This position can't crt");
#endif
					newcf[get<0>(hopup[i])] = newcf[get<0>(hopup[i])].ann(cfg.div_orb_e[get<0>(hopup[i])][c]);
					newcf[get<1>(hopup[i])] = newcf[get<1>(hopup[i])].crt(cfg.div_orb_c[get<1>(hopup[i])][cp]);
#ifdef _ASSERTION_
					VecInt ne_b(occ_n);
					--ne_b[get<0>(hopup[i])];
					++ne_b[get<1>(hopup[i])];
					if (ne_b != get<2>(hopup[i])) ERR("NOT equal" + NAV4(ne_b, get<0>(hopup[i]), get<1>(hopup[i]), get<2>(hopup[i])));
#endif
					ComDivs b(newcf, get<2>(hopup[i]), sit_n);
					ODT_i[2] = idx_div_i + b.idx;
					if (ODT_i[2] > space.dim) ERR(STR("Hmlt Off-Diag Elements IHTL > IHM ") + NAV2(ODT_i[2], space.dim));
					ODT_i[3] = cfg.sgn(ODT_i[0], ODT_i[1]);
#ifdef _ASSERTION_
					if (ODT_i[0] == ODT_i[1]) ERR("A impossible thing happened:ODT_i[0] == ODT_i[1]" + NAV(ODT_i[0]));
#endif
					off_diagonal_term.push_back(ODT_i);
				}
			}
		}
	}

	// DOWN term.
	{
		VecInt ODT_i(4, 0);// [0]/([1]):annihilation/(creation) orbit's position;[2]:Colum idx(i);[i][3]:sign.
		//off-diagonal, two-fermion operator terms for the down spin C^+_i C_j
		for_Int(i, 0, hopdw.size()) {
			BasisIndex idx_div_i(find_newdiv_idx(get<2>(hopdw[i])));
			for_Int(c, 0, cfg.div_orb_e[get<0>(hopdw[i])].size()) {
				for_Int(cp, 0, cfg.div_orb_c[get<1>(hopdw[i])].size()) {
					ODT_i[0] = (SUM_0toX(sit_n, get<0>(hopdw[i])) + cfg.div_orb_e[get<0>(hopdw[i])][c]);
					ODT_i[1] = (SUM_0toX(sit_n, get<1>(hopdw[i])) + cfg.div_orb_c[get<1>(hopdw[i])][cp]);
					VecOnb newcf(cfg.cf);
#ifdef _ASSERTION_
					if (newcf[get<0>(hopdw[i])].isuno(cfg.div_orb_e[get<0>(hopdw[i])][c]))ERR("This position can't ann");
					if (newcf[get<1>(hopdw[i])].isocc(cfg.div_orb_c[get<1>(hopdw[i])][cp]))ERR("This position can't crt");
#endif
					newcf[get<0>(hopdw[i])] = newcf[get<0>(hopdw[i])].ann(cfg.div_orb_e[get<0>(hopdw[i])][c]);
					newcf[get<1>(hopdw[i])] = newcf[get<1>(hopdw[i])].crt(cfg.div_orb_c[get<1>(hopdw[i])][cp]);
#ifdef _ASSERTION_
					VecInt ne_b(occ_n);
					--ne_b[get<0>(hopdw[i])];
					++ne_b[get<1>(hopdw[i])];
					if (ne_b != get<2>(hopdw[i])) ERR("NOT equal" + NAV4(ne_b, get<0>(hopdw[i]), get<1>(hopdw[i]), get<2>(hopdw[i])));
#endif
					ComDivs b(newcf, get<2>(hopdw[i]), sit_n);
					ODT_i[2] = idx_div_i + b.idx;
					if (ODT_i[2] > space.dim) ERR(STR("Hmlt Off-Diag Elements IHTL > IHM ") + NAV2(ODT_i[2], space.dim));
					ODT_i[3] = cfg.sgn(ODT_i[0], ODT_i[1]);
#ifdef _ASSERTION_
					if (ODT_i[0] == ODT_i[1]) ERR("A impossible thing happened:ODT_i[0] == ODT_i[1]" + NAV(ODT_i[0]));
#endif
					off_diagonal_term.push_back(ODT_i);
				}
			}
		}
	}
	return off_diagonal_term;
} */

BasisIndex StateStatistics::find_newdiv_idx(const MatInt& newdiv)
{

	return space.divs_to_idx.at(newdiv.vec().string());

	ERR("NOT found the div in the short cut space.");
}

StateStatistics::hopdata StateStatistics::divocchop_ingroup(BasisIndex ComDiv, Idx sets_n)
{
	hopdata hop;
	// "c": means annihilation, and "cp" mean creation. Which is act on all div.
	// For the spinless orbits.
	for_Int(c, 0, space.ndivs){				//Here "1" refer to div_idx, according to the "set_control()" function.
		for_Int(cp, 0, space.ndivs){		//Here "1" refer to div_idx, according to the "set_control()" function.
			MatInt occupy = expand_occupations(space.div[ComDiv], space.sit_mat.nrows(), space.ndivs);
			--occupy[sets_n][c];
			++occupy[sets_n][cp];
			// if (space.ifin_NocSpace(occupy)) {
			if (space.ifin_NocSpace(occupy, space.nppso)) {
				tuple<Int, Int, MatInt> tup1(make_tuple(c, cp, occupy));
				hop.push_back(tup1);
			}
		}
	}
	return hop;
}

StateStatistics::furfrm StateStatistics::divs_change_fourFermi(BasisIndex ComDiv, Idx sets_i, Idx sets_j)
{
	furfrm hop;
	// "c": means annihilation, and "cp" mean creation. Which is act on all div.
	// For the spinless orbits.
	for_Int(cp_i, 0, space.ndivs) {
	for_Int(cp_j, 0, space.ndivs) {
		for_Int(c_k, 0, space.ndivs) {
		for_Int(c_l, 0, space.ndivs) {
			MatInt occupy = expand_occupations(space.div[ComDiv], space.sit_mat.nrows(), space.ndivs);
			++occupy[sets_i][cp_i]; ++occupy[sets_j][cp_j]; 
			--occupy[sets_j][c_k]; --occupy[sets_i][c_l]; 
			if (space.ifin_NocSpace(occupy, space.nppso)) {
				pair<array<int, 4>, MatInt> tup1(make_pair(array<int, 4>{cp_i, cp_j, c_k, c_l}, occupy));
				hop.push_back(tup1);
			}
		}
		}
	}
	}
	return hop;
}

VEC<MatInt> StateStatistics::interation_soc_hop(BasisIndex ComDiv)
{
	VEC<MatInt> hop_soc;
	// For the spinless orbits.
	
	MatInt occupy = expand_occupations(space.div[ComDiv], space.sit_mat.nrows(), space.ndivs);
	WRN(NAV(occupy));
	for_Int(i, 0, space.p.nband) {
		Int cnt(occupy[i*2][0] + occupy[i*2+1][0]);
		if(cnt == 0) {
			for_Int(j, 0, space.p.nband) if(i != j) {
				Int cnt_j(occupy[j*2][0] + occupy[j*2+1][0]);
				if(cnt_j == 2) {
					occupy[i*2][0] = occupy[i*2+1][0] = 1;
					occupy[j*2][0] = occupy[j*2+1][0] = 0;
					if (space.ifin_NocSpace(occupy, space.nppso)) hop_soc.push_back(occupy);
					WRN(NAV(occupy));
				}
			}
		}

		if(cnt == 2) {
			for_Int(j, 0, space.p.nband) if(i != j) {
				Int cnt_j(occupy[j*2][0] + occupy[j*2+1][0]);
				if(cnt_j == 0) {
					occupy[i*2][0] = occupy[i*2+1][0] = 0;
					occupy[j*2][0] = occupy[j*2+1][0] = 1;
					if (space.ifin_NocSpace(occupy, space.nppso)) hop_soc.push_back(occupy);
					WRN(NAV(occupy));
				}
			}
		}

		if(cnt == 1) {
			for_Int(j, 0, space.p.nband) if(i != j) {
				Int cnt_j(occupy[j*2][0] + occupy[j*2+1][0]);
				if(cnt == 1 && occupy[j*2][0] != occupy[i*2][0]) {
					SWAP(occupy[i*2][0], occupy[i*2+1][0]);
					SWAP(occupy[j*2][0], occupy[j*2+1][0]);
					if (space.ifin_NocSpace(occupy, space.nppso)) hop_soc.push_back(occupy);
					WRN(NAV(occupy));
				}
			}
		}
	}

	return hop_soc;
}

Mat<Char> StateStatistics::show_cfg() {
	const VecOnb& cf(cfg.cf);
	Str cfig;
	for_Int(i, 0, cf.size()){
		for_Int(j, 0, cf[i].get_ns()){
			cfig += cf[i][j] ? '1':'0';
		}
	}
	Vec<Char> ci(cfig.length());
	for_Int(i, 0, ci.size()) ci[i] = cfig[i];
	// if(space.mm) WRN(NAV(ci.mat(space.p.norbs,space.p.nI2B[0]+1)));
	return ci.mat(space.p.norbs,space.p.nI2B[0]+1);
}

/* version 1
array<UInt,6> StateStatistics::cfg2nums() {
	// UInt nums[occ_n.nrows()];
	array<UInt,6> nums;
	const VecOnb& cf(cfg.cf);
	for_Idx(i, 0, cf.size()){
		Str cfig;
		for_Int(j, 0, cf[i].get_ns()) cfig += cf[i][j];
		const UInt num {stoul(cfig, nullptr, 2)};
		nums[i] = num;
	}
	return move(nums);
}
*/

// version 2
VecBool StateStatistics::cfg2vecbool() {
	const VecOnb cf(cfg.cf);
	
	Str cfig;
	for_Int(i, 0, cf.size()){
		for_Int(j, 0, cf[i].get_ns()){
			cfig += cf[i][j] ? '1':'0';
		}
	}
	VecBool ci(cfig.length());
	for_Int(i, 0, ci.size()) ci[i] = cfig[i] == '1' ? true : false;

	return move(ci);
}

// version 2
VecBool StateStatistics::cfgex2vecbool(Int ex_pos) {
	
	MatOnb cf_temp(cfg.cf.mat(space.sit_mat.nrows(), space.ndivs));
	Idx pos(ABS(ex_pos) - 1);
	cf_temp[pos][0] = ex_pos > 0 ? cf_temp[pos][0].crt(0) : cf_temp[pos][0].ann(0);
	
	const VecOnb cf(cf_temp.vec());
	
	Str cfig;
	for_Int(i, 0, cf.size()){
		for_Int(j, 0, cf[i].get_ns()){
			cfig += cf[i][j] ? '1':'0';
		}
	}
	VecBool ci(cfig.length());
	for_Int(i, 0, ci.size()) ci[i] = cfig[i] == '1' ? true : false;

	return move(ci);
}

/*  version 1
array<UInt,6> StateStatistics::cfg2ex2nums(Int ex_pos) {
	// UInt nums[occ_n.nrows()];
	array<UInt,6> nums;
	MatOnb cf_temp(cfg.cf.mat(occ_n.nrows(), occ_n.ncols()));
	Idx pos(ABS(ex_pos) - 1);
	// if(ex_pos > 0) cf_temp = cf_temp[pos][0].crt(0);
	// if(ex_pos < 0) cf_temp = cf_temp[pos][0].ann(0);
	cf_temp[pos][0] = ex_pos > 0 ? cf_temp[pos][0].crt(0) : cf_temp[pos][0].ann(0);
		
	const VecOnb cf(cf_temp.vec());
	for_Idx(i, 0, cf.size()){
		Str cfig;
		for_Int(j, 0, cf[i].get_ns()) cfig += cf[i][j];
		const UInt num {stoul(cfig, nullptr, 2)};
		nums[i] = num;
	}
	return move(nums);
}
*/

//---------------------------------------------Private function---------------------------------------------

//For the assertion that the imp is in the first division[0].
//Int StateStatistics::anticommu(const Int& pi, const Int& pj)
//{
//	Int gap;
//	gap = ABS(pj - pi - 1);
////	if (pi == 0 || pj == 0)	gap = ABS(pj - pi - 1) - 1;//Here discriminate the imp and bath
////	else gap = ABS(pj - pi - 1);
////#ifdef _ASSERTION_
////	if (pj == 0 && pi == 0) ERR("There are some thing wrong with the IDX" + NAV3(gap, pi, pj));
////#endif
//	if ((gap % 2) == 0) return +1;
//	return -1;
//}

ComDivs StateStatistics::cfgdetail()
{
	return ComDivs(idx, (occ_n), (space.sit_mat));
}

void StateStatistics::statistics()
{
	Int sit_spinless(0);
	// spinless orbits
	for_Int(n, 0, filled_spinless.size())
	{
		for_Int(i, n * space.ndivs, (n + 1) * space.ndivs)
		{
			for_Int(j, 0, cfg.cf[i].get_ns())
			{
				if (cfg.cf[i].isocc(j))
					filled_spinless[n].push_back(sit_spinless);
				sit_spinless++;
			}
		}
	}
	// // down site
	// for_Int(i, space.ndivs, space.ndivs + space.ndivs) {
	// 	for_Int(j, 0, cfg.cf[i].get_ns()) {
	// 		if (cfg.cf[i].isocc(j)) filled_dw.push_back(sit_spinless);
	// 		sit_spinless++;
	// 	}
	// }
//#ifdef _ASSERTION_
//	if (space.nspa != filled_up.size() + filled_dw.size()) {
//		ERR("TOTAL partical counting wrong   " + NAV3(space.nspa, filled_up.size(), filled_dw.size()));
//	}
//#endif
}



OccupationDecoder::OccupationDecoder(const NocSpace& space)
    : space_(space), capacity_(space.sit_mat.size()),
      begin_(capacity_.size()), set_(capacity_.size()),
      radix_(capacity_.size()), filled_(space.sit_mat.nrows())
{
    Int orbital = 0;
    for (Int set = 0; set < space.sit_mat.nrows(); ++set) {
        Int count = 0;
        for (Int div = 0; div < space.ndivs; ++div) {
            const Int cell = set * space.ndivs + div;
            capacity_[cell] = space.sit_mat[set][div];
            begin_[cell] = orbital;
            set_[cell] = set;
            count += capacity_[cell];
            orbital += capacity_[cell];
        }
        filled_[set].reserve(count);
    }
}

void OccupationDecoder::refresh(BasisIndex group)
{
    static const BnmlFast binomial;
    const auto& occupation = space_.div[group];
    unique_ = true;
    for (std::size_t cell = 0; cell < capacity_.size(); ++cell) {
        const Int ne = occupation[cell], ns = capacity_[cell];
        // Empty/full cells have exactly one configuration, including ns == 0.
        radix_[cell] = (ne == 0 || ne == ns) ? 1 : binomial(ns, ne);
        unique_ = unique_ && radix_[cell] == 1;
    }
    group_ = group;
}

const std::vector<std::vector<Int>>& OccupationDecoder::decode(BasisIndex row, BasisIndex group)
{
    const bool changed = group != group_;
    if (changed) refresh(group);
    // A group with only empty/full cells has a single basis configuration.
    if (!changed && unique_) return filled_;
    for (auto& orbitals : filled_) orbitals.clear();
    const auto& occupation = space_.div[group];
    BasisIndex rank = row - space_.idx_div[group];
    for (std::size_t cell = 0; cell < capacity_.size(); ++cell) {
        const Int ne = occupation[cell], ns = capacity_[cell];
        const BasisIndex local_rank = rank % radix_[cell];
        rank /= radix_[cell];
        auto& orbitals = filled_[set_[cell]];
        if (ne == 0) continue;
        if (ne == ns) {
            for (Int site = 0; site < ns; ++site)
                orbitals.push_back(begin_[cell] + site);
        } else {
            // Same mixed-radix order and same local basis numbering as ComDivs.
            const Onb configuration(ne, ns, local_rank);
            for (Int site = 0; site < ns; ++site)
                if (configuration.isocc(site)) orbitals.push_back(begin_[cell] + site);
        }
    }
    return filled_;
}
