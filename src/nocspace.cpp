/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "nocspace.h"
#include "gen/parallel_preparation.h"
#include "gen/tab_io.h"
#include "gen/storage_monitor.h"
#include "gen/nocspace_timing.h"
#include "gen/nocspace_pruning.h"
#include "gen/nocspace_cache.h"
#define nip p.norg_sets
#define i2b p.nI2B
#define nob p.norbit

namespace {
// Fill the two unrestricted divisions in ascending impurity-occupation order.
VEC<VEC<VecInt>> star_occupations(const VecInt& particles, const MatInt& restricted,
    const MatInt& capacities, Idx active)
{
    VEC<VEC<VecInt>> choices(particles.size());
    for (Idx set = 0; set < particles.size(); ++set) {
        const Int remaining = particles[set] - SUM(restricted[set]);
        const Int first = std::max(0, remaining - capacities[set][active]);
        const Int last = std::min(remaining, capacities[set][0]);
        for (Int impurity = first; impurity <= last; ++impurity)
            choices[set].push_back(Vec{impurity, remaining - impurity});
    }
    return choices;
}

struct NocGroupResult {
    VecOcc occupations;
    std::string key;
    BasisIndex dimension;
    NocGroupResult(VecOcc&& occ,std::string&& label,BasisIndex size)
        :key(std::move(label)),dimension(size){swapall(occupations,occ);}
    NocGroupResult(NocGroupResult&& other) noexcept
        :key(std::move(other.key)),dimension(other.dimension){swapall(occupations,other.occupations);}
    NocGroupResult& operator=(NocGroupResult&& other) noexcept {
        swapall(occupations,other.occupations);key=std::move(other.key);
        dimension=other.dimension;return *this;
    }
    NocGroupResult(const NocGroupResult&)=delete;
    NocGroupResult& operator=(const NocGroupResult&)=delete;
};
void append_noc_groups(NocSpace& space, std::vector<std::vector<NocGroupResult>>& chunks,
    nocspace_profile::Report& profile) {
    using namespace nocspace_profile;
    {
    namespace par=stage2_parallel;
    namespace pp=preparation_parallel;
    const std::size_t old_count=space.div.size();
    const BasisIndex old_dim=space.dim;
    std::vector<std::size_t> positions(chunks.size()+1);
    pp::offsets(chunks.size(),[&](std::size_t i){return chunks[i].size();},
        [&](std::size_t i,std::size_t offset){positions[i]=offset;});
    std::vector<BasisIndex> dimensions(chunks.size()+1);
    par::for_each(chunks.size(),[&](std::size_t i) {
        BasisIndex total=0;
        for(const auto& group:chunks[i]) total=basis_count_sum(total,group.dimension);
        dimensions[i+1]=total;
    });
    // There is one chunk per worker. Accumulate their global offsets once.
    dimensions[0]=old_dim;
    for(std::size_t i=0;i<chunks.size();++i)
        dimensions[i+1]=basis_count_sum(dimensions[i],dimensions[i+1]);
    profile.mark(GroupOffsets);
    space.div.resize(old_count+positions.back());
    space.idx_div.resize(old_count+positions.back()+1);
    using Pending = OccupationIndex<BasisIndex>::Pending;
    std::vector<std::vector<Pending>> entries(chunks.size());
    profile.mark(StorageAllocation);
    par::for_each(chunks.size(),[&](std::size_t part) {
        entries[part].reserve(chunks[part].size());
        BasisIndex dim=dimensions[part];
        std::size_t out=old_count+positions[part];
        for(auto& group:chunks[part]) {
            swapall(space.div[out],group.occupations);
            space.idx_div[out]=dim;
            entries[part].emplace_back(std::move(group.key),dim);
            dim+=group.dimension;++out;
        }
        std::vector<NocGroupResult>().swap(chunks[part]);
    });
    profile.mark(GroupCopyIndexPrepare);
    space.dim=dimensions.back();
    space.idx_div.back()=space.dim;
    // Route keys in parallel, then build each shard with one exclusive writer.
    // Chunk/group ordering is retained for duplicate keys and basis offsets.
    space.divs_to_idx.append_chunks(entries,[&] { profile.mark(IndexRouting); });
    profile.mark(IndexBuild);
    par::for_each(entries.size(),[&](std::size_t part) {
        std::vector<Pending>().swap(entries[part]);
    });
    }
    profile.mark(AppendCleanup);
}
}


/*
// the NocSpace, is start frome the shortcutspace.
NocSpace::NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const Int& NumberSpa) :
	mm(mm_i), p(prmtr_i), ndivs(prmtr_i.ndiv), nspa(NumberSpa), sit_mat(p.norg_sets, prmtr_i.ndiv, 0),
	hopint(nob, nob, 0.),
	control_divs(p.norg_sets + 1, prmtr_i.ndiv, 0), orbital_divcnt(prmtr_i.ndiv, 0), dim(0)
{
	set_control();
	find_combined_number_subspaces();
	//WRN("Finish finding the subspace and the Dim is:" + NAV(dim));
}
*/

NocSpace::NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const Int& NumberSpa) :
    construction_profile_(mm_i), mm(mm_i), p(prmtr_i), ndivs(prmtr_i.ndiv), nspa(NumberSpa), sit_mat(p.norg_sets, prmtr_i.ndiv, 0),
	hopint(nob, nob, 0.), coefficient(int(std::pow(nob,4) + hopint.size() + 1), 0.),
 	control_divs(p.norg_sets + 1, prmtr_i.ndiv, 0), orbital_divcnt(prmtr_i.ndiv, 0), dim(0)
{
    construction_profile_.mark(nocspace_profile::MemberInit);
    set_control();
    construction_profile_.mark(nocspace_profile::SetControl);
 	find_combined_number_subspaces();
	finish_construction("enumerated_number");
}

// nppso mean: number of partical per spin orbital.
NocSpace::NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i) :
    construction_profile_(mm_i), mm(mm_i), p(prmtr_i), ndivs(prmtr_i.ndiv), nspa(SUM(nppso_i)), sit_mat(p.norg_sets, prmtr_i.ndiv, 0),
	hopint(nob, nob, 0.), coefficient(int(std::pow(nob,4) + hopint.size() + 1), 0.), nppso(nppso_i),
	control_divs(p.norg_sets + 1, prmtr_i.ndiv, 0), orbital_divcnt(prmtr_i.ndiv, 0), dim(0)
{
    construction_profile_.mark(nocspace_profile::MemberInit);
    set_control();
    construction_profile_.mark(nocspace_profile::SetControl);
	// find_combined_number_subspaces(1);
	// find_all_noc_subspaces();
	// find_all_noc_subspaces_by_row();

	// find_all_noc_subspaces();// ! tested for the YBCO-CDMFT version.
	// find_thought_noc_subspaces();
    basis_cache_hit_=nocspace_cache::restore(*this);
    if(!basis_cache_hit_) {
        find_combined_number_subspaces_no_active_orbital();
        nocspace_cache::remember(*this);
    }

    finish_construction(basis_cache_hit_?"cached_nppso":"enumerated_nppso");
}

// nppso mean: number of partical per spin orbital.
NocSpace::NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i, Str tab_name) :
    NocSpace(mm_i, prmtr_i, nppso_i)
{
    // Restore occupation groups and their numbering as well as the dimension.
    const auto info = tab_io::read_info(tab_name);
    if (info.dim != dim || info.ranks != mm.np())
        throw std::invalid_argument("Saved Tab dimension/MPI partition mismatch");
}

NocSpace::NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i, const Tab& tab) :
    NocSpace(mm_i, prmtr_i, nppso_i)
{
    VecPartition rows(mm.np(), mm.id(), dim);
    tab.require_layout(dim, rows.bgn(), rows.len());
}

void NocSpace::finish_construction(const char* source)
{
    // Stop before any counter aggregation, formatting, reporting or storage scan.
    construction_profile_.finish();
    nocspace_profile::emit(mm, construction_profile_, dim, div.size(), source);
    if (construction_detail_.enabled()) {
        for (const auto& worker : construction_workers_)
            construction_detail_.merge_worker(*worker);
        std::ostringstream particles;
        particles << '[';
        for_Idx(r, 0, nppso.size()) { if (r) particles << ','; particles << nppso[r]; }
        particles << ']';
        std::cout << construction_detail_.report(p.nooc_mode, particles.str(), dim,
            div.size(), mm.id(), mm.np()) << std::endl;
    }
    construction_workers_.clear();
    if (mm) PIO("End NocSpace construction source=" + STR(source) + " " + NAV(dim) + "   " + present());
    log_storage_after_construction(source);
}

// Called after enumeration returns: builder temporaries are already destroyed.
// GS, G and restored Tab spaces share this report.
void NocSpace::log_storage_after_construction(const char* source) const
{
	const bool time_storage = nocspace_timing::enabled_for(mm ? true : false);
	const auto storage_start = time_storage ? nocspace_timing::Clock::now() : nocspace_timing::TimePoint{};
	using namespace storage_monitor;
	std::vector<Entry> entries;
	entries.reserve(14);
	Entry occupations = vectors("div", div);
	std::ostringstream shape;
	shape << " first_rows=" << (div.empty() ? 0 : sit_mat.nrows())
		<< " first_cols=" << (div.empty() ? 0 : ndivs)
		<< " row_views_bytes=0";
	occupations.detail = shape.str();
	entries.push_back(occupations);
	entries.push_back(index_map("divs_to_idx", divs_to_idx));
	entries.push_back(std_vector("idx_div", idx_div));
	entries.push_back(matrix("control_divs", control_divs));
	entries.push_back(matrix("sit_mat", sit_mat));
	entries.push_back(storage_monitor::vector("orbital_divcnt", orbital_divcnt));
	entries.push_back(storage_monitor::vector("nppso", nppso));
	entries.push_back(matrix("hopint", hopint));
	entries.push_back(storage_monitor::vector("coefficient", coefficient));
	entries.push_back(storage_monitor::vector("mu", mu));
	entries.push_back(vectors("t_ose", t_ose));
	entries.push_back(vectors("t_hyb", t_hyb));
	Entry binomial("bf");
	binomial.count = binomial.capacity = binomial.scalar_count = double(bf.storage_elements());
	binomial.element_bytes = binomial.scalar_bytes = sizeof(BasisIndex);
	binomial.object_bytes = sizeof(bf);
	binomial.payload_bytes = binomial.scalar_count * sizeof(BasisIndex);
	binomial.dynamic_bytes = double(bf.storage_dynamic_bytes());
	entries.push_back(binomial);
	entries.push_back(total(entries, sizeof(*this)));
	report(mm, source, dim, entries);
	if (time_storage) {
		const double elapsed = nocspace_timing::seconds(storage_start, nocspace_timing::Clock::now());
		std::ostringstream out;
		out << std::fixed << std::setprecision(6)
			<< "NOCSPACE_TIMING: stage=storage_report source=" << source
			<< " dim=" << dim << " rank=" << mm.id() << " ranks=" << mm.np()
			<< " elapsed_s=" << elapsed << " includes_existing_collectives=1";
		std::cout << out.str() << std::endl;
	}
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> judge function begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// here when return true, mean satisfy the PHSs rule.
bool NocSpace::check_correlated_column(const Int& col_pos, const VecInt& div_colsum) const {
	Int change_nhole(0), change_nelec(0);
	if (p.if_norg_imp) {
		change_nhole = orbital_divcnt[col_pos] - div_colsum[col_pos];
		change_nelec = div_colsum[div_colsum.size() - col_pos - 1];
		if (change_nhole + change_nelec <= control_divs[0][div_colsum.size() - col_pos - 1]) return true;
	}
	else {
		change_nhole = orbital_divcnt[col_pos] - div_colsum[col_pos];
		change_nelec = div_colsum[div_colsum.size() - col_pos];
		if (change_nhole + change_nelec <= control_divs[0][div_colsum.size() - col_pos]) return true;
	}
	return false;
}

// here when return true, mean satisfy the PHSs version.2's rule.
bool NocSpace::check_if_PHSs_v2(const VecInt& div_colsum) const {
	// bool thought_NOOC(true);
	if (p.if_norg_imp) {
		ERR("Since this part was not in used, FOR here I have not test!");
		return true;
	}
	else {
		// In this part of code we consider the PHSs rule with out the "*" terms.
		Int length = div_colsum.size() / 2 - 1;
		VecInt temp_hole_chages(length, 0), temp_elec_chages(length, 0), sum_hole_chages(length, 0), sum_elec_chages(length, 0);
		/*
		// count the division from inside to outside.
		for_Int(col_idx, 0, length) {
			temp_hole_chages[col_idx] = orbital_divcnt[div_colsum.size() / 2 - 1 - col_idx] - div_colsum[div_colsum.size() / 2 - 1 - col_idx];
			temp_elec_chages[col_idx] = div_colsum[div_colsum.size() / 2 + 1 + col_idx];
		}
		*/
		// count the division from outside to inside.
		for_Int(col_idx, 0, length) {
			temp_hole_chages[col_idx] = orbital_divcnt[col_idx + 1] - div_colsum[col_idx + 1];
			temp_elec_chages[col_idx] = div_colsum[div_colsum.size() - col_idx - 1];
		}
		for_Int(col_idx, 0, length) {
			sum_hole_chages[col_idx] = SUM_0toX(temp_hole_chages, length - col_idx);
			sum_elec_chages[col_idx] = SUM_0toX(temp_elec_chages, length - col_idx);
		}

		for_Int(col_pos, 0, length) if((sum_hole_chages[col_pos] + sum_elec_chages[col_pos]) > control_divs[0][(div_colsum.size() / 2) + 1 + col_pos]) return false;
	}
	return true;
}


// here when return true, mean satisfy the PHSs rule. //!Only suit for the 8 divisions.
bool NocSpace::check_if_PHSs(const VecInt& div_colsum) const {
	bool thought_NOOC(true);
	if (p.if_norg_imp) {
		ERR("Since this part was not in used, FOR here I have not test!");
		return true;
	}
	else {
		// In this part of code we consider the PHSs rule with out the "*" terms.
		// VecInt sum_hole_chages(div_colsum.size() / 2 - 1, 0), sum_elec_chages(div_colsum.size() / 2 - 1, 0);
		VecInt temp_hole_chages(div_colsum.size() / 2 - 1, 0), temp_elec_chages(div_colsum.size() / 2 - 1, 0);

		Int length = div_colsum.size() / 2 - 1;
		for_Int(col_idx, 0, length) {
			temp_hole_chages[col_idx] = orbital_divcnt[col_idx + 1] - div_colsum[col_idx + 1];
			temp_elec_chages[col_idx] = div_colsum[div_colsum.size() - col_idx - 1];
		}
		// for_Int(col_idx, 0, length) {
		// 	sum_hole_chages[col_idx] = SUM_0toX(temp_hole_chages.reverse(), col_idx);
		// 	sum_elec_chages[col_idx] = SUM_0toX(temp_elec_chages.reverse(), col_idx);
		// }
		// if((SUM(temp_hole_chages) + SUM(temp_elec_chages)) == 1) thought_NOOC = true;
		if((SUM(temp_hole_chages) + SUM(temp_elec_chages)) <= control_divs[0][div_colsum.size() - 2]) return true; // Only suit for the 8 divisions.

		for_Int(col_pos, 0, length) if((temp_hole_chages[col_pos] + temp_elec_chages[col_pos]) > control_divs[0][div_colsum.size() - col_pos - 1]) return false;

		for_Int(col_pos, 0, length)
			if (SUM(temp_hole_chages.truncate(col_pos, length)) + SUM(temp_elec_chages.truncate(col_pos, length)) > 0 \
				&& (SUM_0toX(temp_hole_chages, col_pos) != 0 || SUM_0toX(temp_elec_chages, col_pos) != 0)) return false;
	}
	return thought_NOOC;
}


Vec<MatInt> NocSpace::multi_judger_with_return(VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const {
	VEC<MatInt> split_spilss_divs;
	
	if (p.if_norg_imp) {
		VecPartition div_dim(mm.np(), mm.id(), s.size()); VecInt splited_judge(div_dim.len(), 0), judge(s.size());
		for_Idx(i, div_dim.bgn(), div_dim.end()) {
			MatInt spilss_div(read_from_col_lable(s[i], a).mat(control_divs.ncols(), control_divs.nrows() - 1).tr());
			if (ifin_NocSpace(spilss_div, nppso)) splited_judge[i - div_dim.bgn()] = 1;
		}
		judge = mm.Allgatherv(splited_judge, div_dim);

		for_Idx(i, 0, s.size()) if(judge[i]) {
			MatInt spilss_div(read_from_col_lable(s[i], a).mat(control_divs.ncols(), control_divs.nrows() - 1).tr());
			if (ifin_NocSpace(spilss_div, nppso)) split_spilss_divs.push_back(spilss_div);
		}
	} else for_Idx(i, 0, s.size()) {
		MatInt spilss_div(read_from_col_lable(s[i], a).mat(control_divs.ncols() - 2, control_divs.nrows() - 1).tr());
		if (suit_NOOC(spilss_div, nppso)) { //!! here need to chage the restraint.
			// if (mm) WRN(NAV(spilss_div));
            const auto left_ii = star_occupations(nppso, spilss_div, sit_mat, ndivs/2);
			if (!p.if_norg_imp) for (const auto& x : cart_product(left_ii)) {
				MatInt unnooc(p.norg_sets, 2);
				for_Int(j, 0, x.size()) unnooc[j] = x[j];
				MatInt div(concat(concat(unnooc.tr()[0],spilss_div.tr().truncate_row(0,spilss_div.ncols()/2).vec()), \
						   concat(unnooc.tr()[1],spilss_div.tr().truncate_row(spilss_div.ncols()/2,spilss_div.ncols()).vec())).\
						   mat(sit_mat.ncols(),sit_mat.nrows()).tr());				
				split_spilss_divs.push_back(div);
			}
		}
	}

	Vec<MatInt> out = Vec(split_spilss_divs);
	return out;
}

VecInt NocSpace::multi_judger(const VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const{
	VecPartition div_dim(mm.np(), mm.id(), s.size());
	// MatInt div_dim()
	VecInt splited_out(div_dim.len(), 0);
	// VecInt temp_splited_out()
	for_Idx(i, div_dim.bgn(), div_dim.end()){
		// VEC<Int> x(s[i]);
		// VecInt spilss_div_v(read_from_col_lable(x,a));
		// MatInt spilss_div_tr(spilss_div_v.mat(control_divs.ncols(), control_divs.nrows() - 1));
		// MatInt spilss_div(spilss_div_tr.tr());
		MatInt spilss_div(read_from_col_lable(s[i],a).mat(control_divs.ncols(), control_divs.nrows() - 1).tr());
		if (ifin_NocSpace(spilss_div, nppso)) splited_out[i-div_dim.bgn()] = 1;
		// if (ifin_NocSpace_judge_by_nppso(spilss_div, nppso)) splited_out[i-div_dim.bgn()] = 1;
	}
	VecInt out(s.size());
	out = mm.Allgatherv(splited_out, div_dim);
	return out;
}

VecInt NocSpace::multi_judger_by_row(const VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const{
	VecPartition div_dim(mm.np(), mm.id(), s.size());
	VecInt splited_out(div_dim.len(), 0);
	for_Idx(i, div_dim.bgn(), div_dim.end()){
		MatInt spilss_div(read_from_col_lable(s[i],a).mat( control_divs.nrows() - 1, control_divs.ncols()));
		if (ifin_NocSpace(spilss_div, nppso)) splited_out[i-div_dim.bgn()] = 1;
	}
	VecInt out(s.size());
	out = mm.Allgatherv(splited_out, div_dim);
	return out;
}

bool NocSpace::ifin_NocSpace(MatInt& spilss_div) const
{
	if (SUM(spilss_div) == nspa) {
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) if (spilss_div[i - 1][j] < 0 || spilss_div[i - 1][j] > control_divs[i][j]) return false;
		VecInt spilsdiv_countvec(ndivs, 0);
		for_Int(j, 0, spilss_div.ncols())for_Int(i, 0, spilss_div.nrows()) spilsdiv_countvec[j] += spilss_div[i][j];
		if (spilsdiv_countvec[1] >= orbital_divcnt[1] + control_divs[0][1] && \
			spilsdiv_countvec[1] <= orbital_divcnt[1] && \
			spilsdiv_countvec[2] >= orbital_divcnt[2] + control_divs[0][2] && \
			spilsdiv_countvec[2] <= orbital_divcnt[2] && \
			spilsdiv_countvec[4] >= 0 && \
			spilsdiv_countvec[4] <= control_divs[0][4] && \
			spilsdiv_countvec[5] >= 0 && \
			spilsdiv_countvec[5] <= control_divs[0][5]) return true;
	}
	return false;
}

bool NocSpace::ifin_NocSpace(MatInt& spilss_div, const VecInt& nppso) const
{
	if(ndivs%2) ERR("ndivs is not a even number!");
	for_Int(i, 0, p.norg_sets)	if(SUM(spilss_div[i]) != nppso[i]) return false;
	if (SUM(spilss_div) == nspa) {
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) if (spilss_div[i - 1][j] < 0 || spilss_div[i - 1][j] > control_divs[i][j]) return false;
		VecInt spilsdiv_countvec(ndivs, 0);
		for_Int(j, 0, spilss_div.ncols()) for_Int(i, 0, spilss_div.nrows()) spilsdiv_countvec[j] += spilss_div[i][j];

		Int start = p.if_norg_imp ? 0 : 1;
		for_Int(i, start, ndivs) if(!check_each_column(i, spilsdiv_countvec)) return false;
		if(p.nooc_mode == STR("nooc")) return true;
		else if(p.nooc_mode == STR("cpnooc")) {for_Int(i, start, ndivs/2 - 1) if(!check_correlated_column(i, spilsdiv_countvec)) return false;}		// Here we set middle of one were "nooc", and others were "cnooc".
		else if(p.nooc_mode == STR("cnooc")) {for_Int(i, start, ndivs/2) if(!check_correlated_column(i, spilsdiv_countvec)) return false;}
		else if (p.nooc_mode == STR("phess")) {return check_if_PHSs(spilsdiv_countvec);}
		else if (p.nooc_mode == STR("phss_v2")) {return check_if_PHSs_v2(spilsdiv_countvec);}
		else ERR("nooc_mode in put was wrong!");
		return true;
	}
	else return false;
}

bool NocSpace::two_fermion_columns_allowed(const VecInt& counts, Int ann, Int crt) const
{
    // Row particle counts and the total are invariant under a same-set hop.
    if (ann == crt) return true;
    const Int start = p.if_norg_imp ? 0 : 1;
    if (ann >= start && !check_each_column(ann, counts)) return false;
    if (crt >= start && !check_each_column(crt, counts)) return false;
    if (p.nooc_mode == STR("nooc")) return true;
    if (p.nooc_mode == STR("cpnooc") || p.nooc_mode == STR("cnooc")) {
        const Int stop = ndivs / 2 - (p.nooc_mode == STR("cpnooc") ? 1 : 0);
        // Each correlated condition involves a column and its mirrored partner.
        const auto pair_column = [&](Int c) {
            return c < ndivs / 2 ? c : static_cast<Int>(ndivs) - c - (p.if_norg_imp ? 1 : 0);
        };
        const Int a = pair_column(ann), b = pair_column(crt);
        if (a >= start && a < stop && !check_correlated_column(a, counts)) return false;
        if (b != a && b >= start && b < stop && !check_correlated_column(b, counts)) return false;
        return true;
    }
    // Coupled constraints also depend on unchanged columns. Preserve every
    // original inequality rather than assuming two independent bounds.
    if (p.nooc_mode == STR("phess")) return check_if_PHSs(counts);
    if (p.nooc_mode == STR("phss_v2")) {
        if (p.if_norg_imp) return check_if_PHSs_v2(counts);
        // Same prefix inequalities as check_if_PHSs_v2, without four temporary
        // vectors or repeated prefix summation. Keep the full predicate as oracle.
        const Int length = counts.size() / 2 - 1;
        Int excitation = 0;
        for (Int k = 0; k < length; ++k) {
            excitation += orbital_divcnt[k + 1] - counts[k + 1]
                + counts[counts.size() - k - 1];
            if (excitation > control_divs[0][counts.size() - k - 1]) return false;
        }
        return true;
    }
    ERR("nooc_mode in put was wrong!");
    return false;
}

// In here we only consider the impurity not rotate.
bool NocSpace::suit_NOOC(MatInt& spilss_div, const VecInt& nppso) const
{
	if(ndivs%2) ERR("ndivs is not a even number!");
	for_Int(i, 0, p.norg_sets)	if( sit_mat[i][0] + sit_mat[i][ndivs/2] < nppso[i] - SUM(spilss_div[i]) || nppso[i] - SUM(spilss_div[i]) < 0) return false;

	// if(mm) WRN(NAV(spilss_div));
	MatInt tmp_cntrl_divs = concat(control_divs.tr().truncate_row(1, ndivs/2).vec(), \
	control_divs.tr().truncate_row(ndivs/2+1, ndivs).vec()).mat(spilss_div.ncols(),spilss_div.nrows()+1).tr();
	// if(mm) WRN(NAV(tmp_cntrl_divs));
	for_Int(i, 1, tmp_cntrl_divs.nrows()) for_Int(j, 0, ndivs - 2) if (spilss_div[i - 1][j] < 0 || spilss_div[i - 1][j] > tmp_cntrl_divs[i][j]) return false;
	VecInt spils_c(spilss_div.ncols(), 0), zero(1,0);
	for_Int(j, 0, spilss_div.ncols()) for_Int(i, 0, spilss_div.nrows()) spils_c[j] += spilss_div[i][j];
	VecInt countvec = concat(concat(zero,spils_c.truncate(0, spils_c.size()/2)),concat(zero,spils_c.truncate(spils_c.size()/2, spils_c.size())));
	// if(mm) WRN(NAV(countvec));
	Int countvec_length = countvec.size();
	for_Int(i, 1, countvec_length) if (!check_each_column(i, countvec)) return false;
	if (p.nooc_mode == STR("nooc")) return true;
	else if (p.nooc_mode == STR("cpnooc")) { for_Int(i, 1, countvec_length / 2 - 1) if (!check_correlated_column(i, countvec)) return false; }
	else if (p.nooc_mode == STR("cnooc")) { for_Int(i, 1, countvec_length / 2) if (!check_correlated_column(i, countvec)) return false; }
	else if (p.nooc_mode == STR("phess")) {return check_if_PHSs(countvec);}
	else if (p.nooc_mode == STR("phss_v2")) {return check_if_PHSs_v2(countvec);}
	else ERR("nooc_mode in put was wrong!");
	
	// if (mm) WRN(NAV2(spilss_div, countvec));
	return true;
	// else ERR("nooc_mode in put was wrong!");
}

/*
bool NocSpace::ifin_NocSpace(const VecBool new_cfig, const VecInt& nppso) const //! The input is VecBool type.
{
	MatInt spilss_div(sit_mat.nrows(), sit_mat.ncols(), 0);
	{//chnage new_cfig to spilss_div
		VecInt sit_vec(sit_mat.vec());
		for_Int(i, 0, sit_vec.size()) for_Int(j, SUM_0toX(sit_vec, i), SUM_0toX(sit_vec, i+1)) spilss_div.vec()[i] += new_cfig[j] ? 1 : 0;
	}


	if(ndivs%2) ERR("ndivs is not a even number!");
	for_Int(i, 0, p.norg_sets)	if( sit_mat[i][0] + sit_mat[i][ndivs/2] < nppso[i] - SUM(spilss_div[i]) || nppso[i] - SUM(spilss_div[i]) < 0) return false;

	// if(mm) WRN(NAV(spilss_div));
	MatInt tmp_cntrl_divs = concat(control_divs.tr().truncate_row(1, ndivs/2).vec(), \
	control_divs.tr().truncate_row(ndivs/2+1, ndivs).vec()).mat(spilss_div.ncols(),spilss_div.nrows()+1).tr();
	// if(mm) WRN(NAV(tmp_cntrl_divs));
	for_Int(i, 1, tmp_cntrl_divs.nrows()) for_Int(j, 0, ndivs - 2) if (spilss_div[i - 1][j] < 0 || spilss_div[i - 1][j] > tmp_cntrl_divs[i][j]) return false;
	VecInt spils_c(spilss_div.ncols(), 0), zero(1,0);
	for_Int(j, 0, spilss_div.ncols()) for_Int(i, 0, spilss_div.nrows()) spils_c[j] += spilss_div[i][j];
	VecInt countvec = concat(concat(zero,spils_c.truncate(0, spils_c.size()/2)),concat(zero,spils_c.truncate(spils_c.size()/2, spils_c.size())));
	// if(mm) WRN(NAV(countvec));
	for_Int(i, 1, ndivs) if(!check_each_column(i, countvec)) return false;
	if(p.nooc_mode == STR("nooc")) return true;
	else if(p.nooc_mode == STR("cpnooc")) {for_Int(i, 1, ndivs/2 - 1) if(!check_correlated_column(i, countvec)) return false;}
	else if(p.nooc_mode == STR("cnooc")) {for_Int(i, 1, ndivs/2) if(!check_correlated_column(i, countvec)) return false;}
	else if (p.nooc_mode == STR("phess")) {return check_if_PHSs(countvec);}
	else ERR("nooc_mode in put was wrong!");
	// else ERR("nooc_mode in put was wrong!");
}
*/

bool NocSpace::ifin_NocSpace_more_strict(MatInt& spilss_div, const VecInt& nppso) const
{
	for_Int(i, 0, p.norg_sets)	if(SUM(spilss_div[i]) != nppso[i]) return false;
	if (SUM(spilss_div) == nspa) {
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) if (spilss_div[i - 1][j] < 0 || spilss_div[i - 1][j] > control_divs[i][j]) return false;
		VecInt spilsdiv_countvec(ndivs, 0);
		for_Int(j, 0, spilss_div.ncols())for_Int(i, 0, spilss_div.nrows()) spilsdiv_countvec[j] += spilss_div[i][j];
		/*
		if (spilsdiv_countvec[1] >= orbital_divcnt[1] + control_divs[0][1] && \
			spilsdiv_countvec[1] <= orbital_divcnt[1] && \
			spilsdiv_countvec[2] >= orbital_divcnt[2] + control_divs[0][2] && \
			spilsdiv_countvec[2] <= orbital_divcnt[2] && \
			spilsdiv_countvec[4] >= 0 && \
			spilsdiv_countvec[4] <= control_divs[0][4] && \
			spilsdiv_countvec[5] >= 0 && \
			spilsdiv_countvec[5] <= control_divs[0][5] && \
			orbital_divcnt[1] - spilsdiv_countvec[1] + spilsdiv_countvec[5] <= control_divs[0][5]) return true;
		*/
		for_Int(i, 1, ndivs) if(check_each_column(i, spilsdiv_countvec)) return false;
	//  && orbital_divcnt[1] - spilsdiv_countvec[1] + spilsdiv_countvec[5] <= control_divs[0][5]
		return true;
	}
	else return false;
}

bool NocSpace::ifin_NocSpace_judge_by_nppso(const MatInt& spilss_div, const VecInt& nppso) const
{
	for_Int(i, 0, p.norg_sets)	if(SUM(spilss_div[i]) != nppso[i]) return false;

	return true;
}

bool NocSpace::if_div_in_restraint(const VecInt& restraint, const Int position, const Int max, const Int now) const
{
	if(p.if_norg_imp){
		if (position < ndivs / 2 && now >= (max + restraint[position])) return true;
		if (position >= ndivs / 2 && now <= restraint[position]) return true;
	} else {
		if (position == 0 || position == ndivs / 2) return true;
		if (position < ndivs / 2 && now >= (max + restraint[position])) return true;
		if (position > ndivs / 2 && now <= restraint[position]) return true;
	}
	return false;
}

bool NocSpace::if_col_divs_in_restraint(const Int& restraint, const VEC<Int>& divcol_i, Int col_idx) const
{
	VecInt divcol(divcol_i);	
	if(p.if_norg_imp){
		if(col_idx < ndivs / 2 && SUM(divcol) >= (orbital_divcnt[col_idx] + restraint)) return true;
		if(col_idx >= ndivs / 2 && SUM(divcol) <= restraint ) return true;
	} else {
		if (col_idx == 0 || col_idx == ndivs / 2) return true;
		if(col_idx < ndivs / 2 && SUM(divcol) >= (orbital_divcnt[col_idx] + restraint)) return true;
		if(col_idx > ndivs / 2 && SUM(divcol) <= restraint ) return true;
	}
	return false;
}

bool NocSpace::if_row_divs_in_restraint(const Int& restraint, const VEC<Int>& divrow_i, VecInt count_sit_mat) const
{
	if(SUM(Vec(divrow_i)) != restraint) return false;
	// VecInt countvec(sit_mat.ncols(), 0);
	for_Int(col_idx, 1, divrow_i.size()){
		// if(!(col_idx < ndivs / 2 && col_idx >= (divrow_i[col_idx] + control_divs[0][col_idx]))) return false;
		// if(!(col_idx > ndivs / 2 && col_idx <= control_divs[0][col_idx] )) return false;
		// if (col_pos == ndivs / 2) return true;
		if (col_idx < ndivs / 2 && divrow_i[col_idx] < count_sit_mat[col_idx] + control_divs[0][col_idx] ) return false;
		if (col_idx > ndivs / 2 && divrow_i[col_idx] > control_divs[0][col_idx]) return false;
	}
	return true;
}


//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< judge function end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> space function (before judge) begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>



void NocSpace::find_all_possible_state(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const {
	VEC<VEC<Int> > a_lable;
	Int counter(0);
	for_Int(col, 0, control_divs.ncols()) {
		VEC<VEC<Int>> temp_a;
		VEC<VEC<Int> > a_rol_temp;
		for_Int(row, 1, control_divs.nrows()) {
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1) {
				if (if_div_in_restraint(control_divs[0], col, control_divs[row][col], spl))
					one_div.push_back(spl);
			}
			a_rol_temp.push_back(one_div);
		}
		temp_a = cart_product(a_rol_temp);
		for (const auto& one_divs : temp_a) {
			if (if_col_divs_in_restraint(control_divs[0][col], one_divs, col))
				a.push_back(one_divs);
		}
		VEC<Int> a_lable_i;
		for_Int(i, counter, a.size()) {
			a_lable_i.push_back(i);
			counter++;
		}
		a_lable.push_back(a_lable_i);
	}
	s = cart_product(a_lable);
}

void NocSpace::find_all_possible_state_by_col(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const
{
	VEC<VEC<Int> > a_lable;
	Int counter(0);
	for_Int(col, 0, control_divs.ncols())
	{
		VEC<VEC<Int>> temp_a, a_rol_temp;
		for_Int(row, 1, control_divs.nrows())
		{
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1)
			{
				if (if_div_in_restraint(control_divs[0], col, control_divs[row][col], spl))
					one_div.push_back(spl);
			}
			a_rol_temp.push_back(one_div);
		}
		temp_a = cart_product(a_rol_temp);
		for (const auto &one_divs : temp_a)	{
			if (if_col_divs_in_restraint(control_divs[0][col], one_divs, col))
				a.push_back(one_divs);
		}
		VEC<Int> a_lable_i;
		for_Int(i, counter, a.size()){
			a_lable_i.push_back(i);
			counter++;
		}
		a_lable.push_back(a_lable_i);
		// if(mm) WRN(NAV2( col,a.size()));
	}

	s = cart_product_monitor_col(a_lable, a);
	// s = cart_product(a_lable);
}

void NocSpace::find_all_possible_state_by_row(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const
{
	VEC<VEC<Int> > a_lable;
	Int counter(0);
	for_Int(row, 1, control_divs.nrows())
	{
		VEC<VEC<Int>> temp_a, a_row_temp;
		for_Int(col, 0, control_divs.ncols())
		{
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1)
			{
				if (if_div_in_restraint(control_divs[0], col, control_divs[row][col], spl))
					one_div.push_back(spl);
			}
			a_row_temp.push_back(one_div);
		}
		temp_a = cart_product(a_row_temp);
		for (const auto &one_divs : temp_a)	{
			if (if_row_divs_in_restraint(nppso[row-1], one_divs, sit_mat[row-1])) a.push_back(one_divs);
		}
		VEC<Int> a_lable_i;
		for_Int(i, counter, a.size()){
			a_lable_i.push_back(i);
			counter++;
		}
		a_lable.push_back(a_lable_i);
		// if(mm) WRN(NAV2( row,a.size()));
	}

	s = cart_product_monitor_row(a_lable, a);
	// s = cart_product(a_lable);
}

void NocSpace::find_all_possible_state_by_nooc(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const
{
	VEC<VEC<Int> > a_lable;
	Int counter(0);
	for_Int(col, 0, control_divs.ncols()) if (p.if_norg_imp || (col != 0 && col != ndivs / 2))
	{
		VEC<VEC<Int>> temp_a, a_rol_temp;
		for_Int(row, 1, control_divs.nrows())
		{
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1)
			{
				if (if_div_in_restraint(control_divs[0], col, control_divs[row][col], spl))
					one_div.push_back(spl);
			}
			a_rol_temp.push_back(one_div);
		}
		temp_a = cart_product(a_rol_temp);
		for (const auto &one_divs : temp_a)	{
			if (if_col_divs_in_restraint(control_divs[0][col], one_divs, col))
				{a.push_back(one_divs);
				// if(mm)WRN(NAV(Vec(one_divs)));
				}
		}
		VEC<Int> a_lable_i;
		for_Int(i, counter, a.size()){
			a_lable_i.push_back(i);
			counter++;
		}
		a_lable.push_back(a_lable_i);
		// if(mm) WRN(NAV2( col,a.size()));
	}

	// s = cart_product_monitor_col(a_lable, a);
	s = cart_product(a_lable);
}

//! not test yet.
void NocSpace::find_all_possible_state_suit_for_PHSs(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const {

	VEC<VEC<Int> > a_lable;
	Int counter(0);
	if(p.if_norg_imp) {ERR("Since this part was not in used, FOR here I have not test yet!");}
	else for_Int(col, 0, control_divs.ncols()) if (col != 0 && col != ndivs / 2)
	{
		VEC<VEC<Int>> temp_a, a_rol_temp;
		for_Int(row, 1, control_divs.nrows())
		{
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1)
			{
				if (if_div_in_restraint(control_divs[0], col, control_divs[row][col], spl))
					one_div.push_back(spl);
			}
			a_rol_temp.push_back(one_div);
		}
		temp_a = cart_product(a_rol_temp);
		for (const auto &one_divs : temp_a)	{
			if (if_col_divs_in_restraint(control_divs[0][col], one_divs, col))
				{a.push_back(one_divs);
				// if(mm)WRN(NAV(Vec(one_divs)));
				}
		}
		VEC<Int> a_lable_i;
		for_Int(i, counter, a.size()){
			a_lable_i.push_back(i);
			counter++;
		}
		a_lable.push_back(a_lable_i);
		// if(mm) WRN(NAV2( col,a.size()));
	}
	// s = cart_product(a_lable);
	s = cart_product_monitor_PHS(a_lable, a);
}

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< space function (before judge) end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> space function (after judge) begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

void NocSpace::find_thought_noc_subspaces()
{
	BasisIndex length_ECNS;						// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	VEC<VEC<Int> > a, s;

	if(p.if_norg_imp) find_all_possible_state_by_col(a, s);
	// else if(p.nooc_mode == STR("phss_v2")) find_all_possible_state_suit_for_PHSs(a, s); //!! TSSTING
	else find_all_possible_state_by_nooc(a, s);
	// if(mm) WRN(NAV(s.size()));
	Vec<MatInt> spilss_divs = multi_judger_with_return(s, a);
	// if(mm) WRN(NAV(spilss_divs.size()));
	for_Idx(k, 0, spilss_divs.size()) {
		const MatInt & spilss_div(spilss_divs[k]);
		div.push_back(compact_occupations(spilss_div));
		divs_to_idx.insert(std::pair<std::string, BasisIndex>(spilss_div.vec().string(), dim));
		length_ECNS = 1;
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) length_ECNS = basis_count_product(length_ECNS, bf(control_divs[i][j], spilss_div[i - 1][j]));
		dim = basis_count_sum(dim, length_ECNS);
		idx_div.push_back(dim);
	}
}

void NocSpace::find_all_noc_subspaces()
{
	BasisIndex length_ECNS;						// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	VEC<VEC<Int> > a, s;


	find_all_possible_state_by_col(a, s);
	// if(mm) WRN(NAV(s.size()));
	for (const auto& x : s) {
		VecInt spilss_div_v(read_from_col_lable(x,a));
		MatInt spilss_div_tr(spilss_div_v.mat(control_divs.ncols(), control_divs.nrows() - 1));
		MatInt spilss_div(spilss_div_tr.tr());
		if (ifin_NocSpace(spilss_div, nppso)) {
			// if (mm) WRN(NAV(spilss_div));
			div.push_back(compact_occupations(spilss_div));
			divs_to_idx.insert(std::pair<std::string, BasisIndex>(spilss_div.vec().string(), dim));
			length_ECNS = 1;
			for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) length_ECNS = basis_count_product(length_ECNS, bf(control_divs[i][j], spilss_div[i - 1][j]));
			dim = basis_count_sum(dim, length_ECNS);
			idx_div.push_back(dim);
		}
	}
			// WRN(NAV(dim));
}

// find all the possible state speed up with MPI.
void NocSpace::find_all_noc_subspaces_multi() {
	BasisIndex length_ECNS;						// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	VEC<VEC<Int> > a, s;

	find_all_possible_state_by_col(a, s);
	if(mm) PIO(NAV(s.size())+"   "+present());
	IFS ifs(STR("judger"+nppso_str()+ ".bdat"));
	VecInt judger_out(s.size());
	if(!ifs) judger_out = multi_judger(s, a);
	else biread(ifs, CharP(judger_out.p()), judger_out.szof());
	// for (const auto &x : out)
	for_Idx(ii, 0, s.size()) if(judger_out[ii]) {
		VEC<Int>& x = s[ii];
		VecInt spilss_div_v(read_from_col_lable(x,a));
		MatInt spilss_div_tr(spilss_div_v.mat(control_divs.ncols(), control_divs.nrows() - 1));
		MatInt spilss_div(spilss_div_tr.tr());
		// if (ifin_NocSpace(spilss_div, nppso)) 
		div.push_back(compact_occupations(spilss_div));
		divs_to_idx.insert(std::pair<std::string, BasisIndex>(spilss_div.vec().string(), dim));
		length_ECNS = 1;
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) length_ECNS = basis_count_product(length_ECNS, bf(control_divs[i][j], spilss_div[i - 1][j]));
		dim = basis_count_sum(dim, length_ECNS);
		idx_div.push_back(dim);
	}
}

void NocSpace::find_all_noc_subspaces_by_row()
{
	BasisIndex length_ECNS;						// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	VEC<VEC<Int> > a, s;

	find_all_possible_state_by_row(a, s);
	if(mm) PIO(NAV(s.size())+"   "+present());
	IFS ifs(STR("judger"+nppso_str()+ ".bdat"));
	VecInt judger_out(s.size());
	if(!ifs) judger_out = multi_judger_by_row(s, a);
	else biread(ifs, CharP(judger_out.p()), judger_out.szof());
	// for (const auto &x : out)
	for_Idx(ii, 0, s.size()) if(judger_out[ii]) {
		MatInt spilss_div(read_from_col_lable(s[ii],a).mat( control_divs.nrows() - 1, control_divs.ncols()));
		div.push_back(compact_occupations(spilss_div));
		divs_to_idx.insert(std::pair<std::string, BasisIndex>(spilss_div.vec().string(), dim));
		length_ECNS = 1;
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) length_ECNS = basis_count_product(length_ECNS, bf(control_divs[i][j], spilss_div[i - 1][j]));
		dim = basis_count_sum(dim, length_ECNS);
		idx_div.push_back(dim);
	}
	// if(mm) {
	// 	OFS ofs;	ofs.open("judger"+nppso_str()+ ".bdat");
	// 	biwrite(ofs, CharP(judger_out.p()), judger_out.szof());
	// 	ofs.close();
	// }
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< space function (after judge) end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (old/stable)space function begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// find all the combined number subspaces OR find all the combined number subspaces with special partical control.
void NocSpace::find_combined_number_subspaces(const Int mode)
{
	BasisIndex length_ECNS;			// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);

	VEC<VEC<Int> > a;
	for_Int(row, 1, control_divs.nrows()) {
		for_Int(col, 0, control_divs.ncols()) {
			VEC<Int> one_div;
			for_Int(spl, 0, control_divs[row][col] + 1) {
				if(if_div_in_restraint(control_divs[0],col, control_divs[row][col],spl)) one_div.push_back(spl);
			}
			a.push_back(one_div);
		}
	}
	const Idx total_posibile = free_div_combination_count(a);
    construction_profile_.raw_candidates = total_posibile;

    stage2_parallel::collective(mm, [&] {
        std::vector<std::vector<NocGroupResult>> chunks(stage2_parallel::workers(total_posibile));
        stage2_parallel::ranges(total_posibile, [&](Idx first, Idx last, Idx part) {
            auto& output = chunks[part];
            for (Idx x = first; x < last; ++x) {
                VecInt decoded(free_div_base_decode(x, a));
                MatInt occupations(decoded.mat(control_divs.nrows() - 1, control_divs.ncols()));
                const bool accepted = mode == 1 ? ifin_NocSpace(occupations, nppso)
                    : (mode == 0 && ifin_NocSpace(occupations));
                if (!accepted) continue;
                BasisIndex size = 1;
                for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) {
                    size = basis_count_product(size, bf(control_divs[i][j], occupations[i - 1][j]));
                }
                output.push_back({compact_occupations(occupations), occupations.vec().string(), size});
            }
        });
        construction_profile_.mark(nocspace_profile::LegacyEnumeration);
        append_noc_groups(*this, chunks, construction_profile_);
    });
    construction_profile_.mark(nocspace_profile::CollectiveSync);
}

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< (old/stable)space function end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> space function upgrade for 5 band begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// The code is desigened specital for the no rotate impurity orbital case.
namespace {
// Column pruning helpers. Choices are ascending occupations; an empty
// hole_capacity means the choices already represent nonnegative excitations.
// Visit high mixed-radix digits first so the emitted labels stay ascending,
// exactly as in the original exhaustive decode/filter loop.
void visit_bounded_column(const VEC<VEC<Int>>& choices,
	const VEC<Int>& hole_capacity, const VEC<Idx>& stride,
	Idx digits, LLInt remaining, Idx label, VEC<Idx>& labels)
{
	if (digits == 0) {
		labels.push_back(label);
		return;
	}
	const Idx k = digits - 1;
	const auto& values = choices[k];
	auto first = values.begin();
	auto last = values.end();
	if (hole_capacity.empty())
		last = std::upper_bound(first, last, remaining);
	else
		first = std::lower_bound(first, last, LLInt(hole_capacity[k]) - remaining);
	for (auto value = first; value != last; ++value) {
		const LLInt cost = hole_capacity.empty() ? LLInt(*value)
			: LLInt(hole_capacity[k]) - *value;
		const Idx digit = static_cast<Idx>(value - values.begin());
		visit_bounded_column(choices, hole_capacity, stride, k,
			remaining - cost, label + digit * stride[k], labels);
	}
}

void generate_bounded_column(const VEC<VEC<Int>>& choices,
	const VEC<Int>& hole_capacity, LLInt budget, VEC<Idx>& labels)
{
	if (budget < 0) return;
	for (const auto& values : choices) if (values.empty()) return;
	// The caller has already checked the full product for Idx overflow.
	VEC<Idx> stride(choices.size(), 1);
	for (Idx k = 1; k < choices.size(); ++k)
		stride[k] = stride[k - 1] * choices[k - 1].size();
	visit_bounded_column(choices, hole_capacity, stride, choices.size(),
		budget, 0, labels);
}
} // namespace (column pruning helpers)

void NocSpace::find_combined_number_subspaces_no_active_orbital()
{
    construction_detail_ = nocspace_timing::BuildTiming(mm ? true : false);
    auto& timing = construction_detail_;
    auto& profile = construction_profile_;
    using namespace nocspace_profile;

	{ // Measure temporary-buffer destruction separately from enumeration.
	BasisIndex length_ECNS;			// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	// if(mm) WRN(NAV(dim));
	VEC<VEC<VEC<Int>>> right_col_div;
	VEC<VEC<Idx>> right_cols;
	MatInt new_restratin_divs(p.norg_sets * 2 + 1, (control_divs.ncols() - 2) / 2, 0);

	// if(1) {
	if(p.nooc_mode != STR("phss_v2")) {
		for_Int(col, 0, control_divs.ncols()) if (p.if_norg_imp || (col != 0 && col != ndivs / 2)) {
			const auto column_start = timing.stamp();
			VEC<VEC<Int> > col_data;
			VEC<Idx> right_col;
			for_Int(row, 1, control_divs.nrows()) {
				VEC<Int> one_div;
				for_Int(spl, 0, control_divs[row][col] + 1) {
					if(if_div_in_restraint(control_divs[0],col, control_divs[row][col],spl)) one_div.push_back(spl);
				}
				col_data.push_back(one_div);
			}
			const Idx total_one_noc_col_posibile = free_div_combination_count(col_data);
			VEC<Int> hole_capacity;
			LLInt budget = control_divs[0][col];
			if (col < ndivs / 2) {
				// Sum(capacity - occupation) <= -restraint. Keep occupations
				// in their original ascending order, including on the hole side.
				budget = -budget;
				for_Int(row, 1, control_divs.nrows())
					hole_capacity.push_back(control_divs[row][col]);
			}
			generate_bounded_column(col_data, hole_capacity, budget, right_col);
			// if(mm) WRN(NAV2(col,VecInt(right_col))+"   "+present());
			right_cols.push_back(right_col);
			right_col_div.push_back(col_data);
			timing.column(col, false, total_one_noc_col_posibile, right_col.size(), column_start);
		}
	} else if(p.nooc_mode == STR("phss_v2")){
		for_Int(i, 0, new_restratin_divs.nrows()) for_Int(j, 0, new_restratin_divs.ncols()) {
			if (i < p.norg_sets+1)
				new_restratin_divs[i][j] = control_divs[i][j + control_divs.ncols() - new_restratin_divs.ncols()];
			else
				new_restratin_divs[i][j] = control_divs[i - p.norg_sets][ndivs / 2 - 1 - j];
		}
		// if(mm) WRN(NAV2(new_restratin_divs, control_divs));
		for_Int(col, 0, new_restratin_divs.ncols()) {
			const auto column_start = timing.stamp();
			VEC<VEC<Int> > col_data;
			VEC<Idx> right_col;
			for_Int(row, 1, new_restratin_divs.nrows()) {
				VEC<Int> one_div;
				for_Int(spl, 0, new_restratin_divs[row][col] + 1) {
					if(spl <= new_restratin_divs[0][col]) one_div.push_back(spl);
				}
				col_data.push_back(one_div);
			}
			const Idx total_one_noc_col_posibile = free_div_combination_count(col_data);
			// These choices already contain electrons followed by holes.
			generate_bounded_column(col_data, VEC<Int>(),
				new_restratin_divs[0][col], right_col);
			right_cols.push_back(right_col);
			right_col_div.push_back(col_data);
			timing.column(ndivs / 2 + 1 + col, true, total_one_noc_col_posibile, right_col.size(), column_start);
			// if(mm) WRN(NAV(right_col.size())+"   "+present());
		}
	}

	timing.finish_columns();
    profile.mark(Columns);
	const Idx total_noc_posibile = free_div_combination_count(right_cols);
	// VEC<VEC<Int> > combine_right_cols = cart_product(right_cols);
    profile.raw_candidates = total_noc_posibile;

	// Cache column occupations once; generate only combinations compatible with
	// excitation budgets and per-set particle counts, in the original order.
	const bool prune_combinations = !p.if_norg_imp && ndivs >= 4 && ndivs % 2 == 0
		&& (p.nooc_mode == "nooc" || p.nooc_mode == "cnooc" || p.nooc_mode == "cpnooc"
			|| p.nooc_mode == "phess" || p.nooc_mode == "phss_v2");
	const bool paired_columns = p.nooc_mode == STR("phss_v2");
	std::vector<nocspace_pruning::Column> columns(right_cols.size());
	std::vector<nocspace_pruning::Count> limits;
	std::vector<int> target, star_capacity;
	if (prune_combinations && total_noc_posibile != 0) {
		for_Int(c, ndivs / 2 + 1, ndivs) limits.push_back(control_divs[0][c]);
		for_Int(r, 0, p.norg_sets) {
			target.push_back(nppso[r]);
			star_capacity.push_back(sit_mat[r][0] + sit_mat[r][ndivs / 2]);
		}
		for_Idx(k, 0, right_cols.size()) {
			const Int col = paired_columns ? ndivs / 2 + 1 + k
				: (k < (ndivs - 2) / 2 ? k + 1 : k + 2);
			columns[k].layer = col < ndivs / 2 ? ndivs / 2 - 1 - col : col - ndivs / 2 - 1;
			columns[k].choices.reserve(right_cols[k].size());
			for (Idx label : right_cols[k]) {
				const VecInt counts(free_div_base_decode(label, right_col_div[k]));
				nocspace_pruning::Choice choice;
				choice.particles.reserve(p.norg_sets);
				for_Int(r, 0, p.norg_sets) {
					choice.particles.push_back(paired_columns
						? counts[r] + sit_mat[r][ndivs - col] - counts[r + p.norg_sets] : counts[r]);
					choice.excitations += paired_columns ? counts[r] + counts[r + p.norg_sets]
						: (col < ndivs / 2 ? sit_mat[r][col] - counts[r] : counts[r]);
				}
				columns[k].choices.push_back(std::move(choice));
			}
		}
	}
	// Capacity rows only: the first control row contains excitation limits.
	bool unit_subspace_dimension = true;
	for_Int(r, 1, control_divs.nrows()) for_Int(c, 0, ndivs) {
		if (control_divs[r][c] < 0 || control_divs[r][c] > 1)
			unit_subspace_dimension = false;
	}
    timing.begin_enumeration(total_noc_posibile);
    profile.mark(PruningSetup);
    stage2_parallel::collective(mm, [&] {
        {
        const Idx parts = stage2_parallel::workers(total_noc_posibile);
        std::vector<std::vector<NocGroupResult>> chunks(parts);
        std::vector<std::unique_ptr<nocspace_timing::BuildTiming>> worker_times;
        for (Idx i = 0; i < parts; ++i)
            worker_times.emplace_back(new nocspace_timing::BuildTiming(mm ? true : false));
        profile.mark(WorkerSetup);
        stage2_parallel::ranges(total_noc_posibile, [&](Idx first, Idx last, Idx part) {
            auto& timing = *worker_times[part];
            auto& output = chunks[part];
            MatInt spilss_div(p.norg_sets, ndivs, 0);
            VecIdx star_choice(p.norg_sets, 0);
            BasisIndex length_ECNS;
	const auto store_candidate = [&](Idx sigle_posibile_idx) {
		auto decode_sample = timing.sample(nocspace_timing::Decode);
		MatInt spilss_noc_div(control_divs.nrows() - 1, control_divs.ncols() - 2, 0);
		// if (1) {
		if (p.nooc_mode != STR("phss_v2")) {
			// VecInt col_idx(free_div_base_decode(sigle_posibile_idx, right_cols));
			// if(mm) WRN(NAV2(sigle_posibile_idx,col_idx)+"   "+present());
			VecIdx one_combine_right_cols(free_div_base_decode(sigle_posibile_idx, right_cols));
			MatInt spilss_noc_div_tr(control_divs.ncols() - 2, control_divs.nrows() - 1);
			for_Idx(i, 0, control_divs.ncols() - 2) spilss_noc_div_tr[i] = free_div_base_decode(one_combine_right_cols[i], right_col_div[i]);
			// MatInt spilss_noc_div(free_div_base_decode(sigle_posibile_idx, ？？？).mat(control_divs.nrows() - 1, control_divs.ncols() - 2));
			// MatInt spilss_noc_div(spilss_noc_div_tr.tr());
			spilss_noc_div = spilss_noc_div_tr.tr();

		// if (mm && sigle_posibile_idx < 1) WRN(NAV2(sigle_posibile_idx, spilss_noc_div));
		}
		else if(p.nooc_mode == STR("phss_v2")) {
			VecIdx one_combine_right_cols(free_div_base_decode(sigle_posibile_idx, right_cols));
			MatInt spilss_noc_div_tr(new_restratin_divs.ncols(), new_restratin_divs.nrows() - 1);
			for_Idx(i, 0, new_restratin_divs.ncols()) spilss_noc_div_tr[i] = free_div_base_decode(one_combine_right_cols[i], right_col_div[i]);
			// MatInt spilss_noc_div(spilss_noc_div_tr.tr());
			// spilss_noc_div = spilss_noc_div_tr.tr();
			MatInt temp_spilss_noc_div(spilss_noc_div_tr.tr());
			MatInt temp_control_divs(control_divs.truncate_row(1, control_divs.nrows()));
			// if (mm) WRN(NAV2(temp_spilss_noc_div, temp_control_divs));
			for_Int(i, 0, spilss_noc_div.nrows()) for_Int(j, 0, spilss_noc_div.ncols()) {
				if (j < spilss_noc_div.ncols()/2)
					spilss_noc_div[i][j] = temp_control_divs[i][j + 1] - temp_spilss_noc_div[i + p.norg_sets][temp_spilss_noc_div.ncols() - 1 - j];
				else
					spilss_noc_div[i][j] = temp_spilss_noc_div[i][j - temp_spilss_noc_div.ncols()];
			}
			// if (mm) WRN(NAV2(temp_spilss_noc_div, spilss_noc_div))
			// if (mm) WRN(NAV3(spilss_noc_div, temp_control_divs[5][0 + 1], temp_spilss_noc_div[5 + p.norg_sets][spilss_noc_div.ncols() - 0]))
		}
		decode_sample.stop();
		auto judge_sample = timing.sample(nocspace_timing::Judge);
		const bool accepted = suit_NOOC(spilss_noc_div, nppso);
		judge_sample.stop();
		if (accepted) {
			auto choices_sample = timing.sample(nocspace_timing::StarChoices);
			// if (mm) WRN("This is Right!!: "+ NAV(spilss_noc_div));
            const auto left_ii = star_occupations(nppso, spilss_noc_div, sit_mat, ndivs/2);
			choices_sample.stop();
			if (!p.if_norg_imp) {
				auto fixed_sample = timing.sample(nocspace_timing::PrepareFixed);
				BasisIndex restricted_dimension = 1;
				bool have_star_combination = std::all_of(left_ii.begin(), left_ii.end(),
                    [](const auto& choices) { return !choices.empty(); });
				for_Int(r, 0, p.norg_sets) {
					star_choice[r] = 0;
					// Insert * columns at 0 and ndivs/2. All other occupations
					// are fixed for every expansion of this candidate.
					for_Int(c, 0, spilss_noc_div.ncols()) {
						const Int full_col = c < (ndivs - 2) / 2 ? c + 1 : c + 2;
						spilss_div[r][full_col] = spilss_noc_div[r][c];
						if (have_star_combination && !unit_subspace_dimension)
							restricted_dimension = basis_count_product(restricted_dimension, bf(control_divs[r + 1][full_col], spilss_noc_div[r][c]));
					}
				}
				fixed_sample.stop();
				while (have_star_combination) {
					auto assemble_sample = timing.sample(nocspace_timing::Assemble);
					for_Int(r, 0, p.norg_sets) {
						const VecInt& choice = left_ii[r][star_choice[r]];
						spilss_div[r][0] = choice[0];
						spilss_div[r][ndivs / 2] = choice[1];
					}
					assemble_sample.stop();
					auto store_sample = timing.sample(nocspace_timing::Store);

                    auto dimension_sample = timing.sample(nocspace_timing::Dimension);
					length_ECNS = restricted_dimension; // Equals 1 on the 0/1-capacity path.
					if (!unit_subspace_dimension) for_Int(r, 0, p.norg_sets) {
						length_ECNS = basis_count_product(length_ECNS, bf(control_divs[r + 1][0], spilss_div[r][0]));
						length_ECNS = basis_count_product(length_ECNS, bf(control_divs[r + 1][ndivs / 2], spilss_div[r][ndivs / 2]));
					}
                    dimension_sample.stop();
                    output.push_back({
                        [&]() { auto sample = timing.sample(nocspace_timing::Compact);
                            return compact_occupations(spilss_div); }(),
                        [&]() { auto sample = timing.sample(nocspace_timing::Key);
                            return spilss_div.vec().string(); }(), length_ECNS});
					store_sample.stop();
					auto advance_sample = timing.sample(nocspace_timing::StarAdvance);
					// Match cart_product(left_ii): the last group changes fastest.
					// No product-sized temporary or total-product counter is needed.
					have_star_combination = false;
					for (Idx r = star_choice.size(); r > 0;) {
						--r;
						if (++star_choice[r] < left_ii[r].size()) {
							have_star_combination = true;
							break;
						}
						star_choice[r] = 0;
					}
				}
			}
		}
	};

            if (prune_combinations && total_noc_posibile != 0) {
                nocspace_pruning::enumerate(columns, p.nooc_mode, limits, target, star_capacity,
                    control_divs[0][ndivs - 2], store_candidate,
                    [&](Idx skipped) { timing.reject(skipped); }, first, last);
            } else {
                for (Idx label = first; label < last; ++label) store_candidate(label);
            }
        });
        profile.mark(Enumeration);
        append_noc_groups(*this, chunks, profile);
        construction_workers_ = std::move(worker_times);
        }
        profile.mark(WorkerCleanup);
    });
    profile.mark(CollectiveSync);
	timing.finish_enumeration();
	}
    profile.mark(OuterCleanup);
	timing.finish_build();
	// if(mm) WRN(NAV3(div.size(), idx_div.size(), divs_to_idx.size()))
}



/*
void NocSpace::nestedLoops(int depth, int n, MatInt control_divs, std::vector<int>& current) {
    if (depth == n) {
        for (int num : current) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        return;
    }

    for (int i = 0; i < control_divs; i++) {
        current[depth] = i;
        nestedLoops(depth + 1, n, control_divs, current);
    }
}

void NocSpace::find_thought_noc_subspaces_nestedloops()
{
	BasisIndex length_ECNS;						// length for each combined number subspaces(ECNS).
	idx_div.push_back(dim);
	VEC<VEC<Int> > a, s;

	if(p.if_norg_imp) find_all_possible_state_by_col(a, s);
	else find_all_possible_state_by_nooc(a, s);
	// if(mm) WRN(NAV(s.size()));
	Vec<MatInt> spilss_divs = multi_judger_with_return(s, a);
	// if(mm) WRN(NAV(spilss_divs.size()));
	for_Idx(k, 0, spilss_divs.size()) {
		const MatInt & spilss_div(spilss_divs[k]);
		div.push_back(spilss_div);
		divs_to_idx.insert(std::pair<std::string, BasisIndex>(spilss_div.vec().string(), dim));
		length_ECNS = 1;
		for_Int(i, 1, control_divs.nrows()) for_Int(j, 0, ndivs) length_ECNS = basis_count_product(length_ECNS, bf(control_divs[i][j], spilss_div[i - 1][j]));
		dim = basis_count_sum(dim, length_ECNS);
		idx_div.push_back(dim);
	}
}
*/

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< space function upgrade for 5 band end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> basic function begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

void NocSpace::set_control()
{
	control_divs = p.control_divs;
	sit_mat = control_divs.truncate_row(1, control_divs.nrows());
	for_Int(j, 0, control_divs.ncols())for_Int(i, 1, control_divs.nrows()) orbital_divcnt[j] += control_divs[i][j];
	// sit_mat.reset(control_divs.truncate_row(1, control_divs.nrows()));
}

VEC<VEC<Int> > NocSpace::cart_product_monitor_col (const VEC<VEC<Int> >& v, const VEC<VEC<Int> >& a)const
{
    VEC<VEC<Int> > s = {{}};
    for (const auto& u : v) {
		VEC<VEC<Int> > r;
		for (const auto& x : s) {
			// Int sum(0);
			// for (const auto x_i : x) sum += SUM(VecInt(a[x_i]));
			// if (sum <= nspa) {
			VecInt sum(ndivs,0); bool judge(true);
			for (const auto x_i : x) sum += VecInt(a[x_i]);
			for_Int(i, 0, ndivs)	if(sum[i] > nppso[i]) judge = false;
			// if (s[0].size() == sit_mat.size()) for_Int(i, 0, ndivs)	if(sum[i] != nppso[i]) judge = false;
			if (judge) {
				for (const auto y : u) {
					r.push_back(x);
					r.back().push_back(y);
				}
			}
		}
		s = move(r);
    }
    return s;
}

VEC<VEC<Int> > NocSpace::cart_product_monitor_row (const VEC<VEC<Int> >& v, const VEC<VEC<Int> >& a)const
{
    VEC<VEC<Int> > s = {{}};
    for (const auto& u : v) {
        VEC<VEC<Int> > r;
        for (const auto& x : s) {
			bool judge(true); 
			VecInt m_row(ndivs, 0), count_sit_mat(ndivs, 0);
			for_Int(i, 0, x.size()) {count_sit_mat += sit_mat[i]; m_row += Vec(a[x[i]]);}
			if (if_row_divs_in_restraint(SUM(nppso.truncate(0, x.size())), m_row.stdvec(), count_sit_mat))
				for (const auto y : u) {
					r.push_back(x);
					r.back().push_back(y);
				}
		}
        s = move(r);
    }
    return s;
}

VEC<VEC<Int> > NocSpace::cart_product_monitor_PHS (const VEC<VEC<Int> >& v, const VEC<VEC<Int> >& a)const
{
    VEC<VEC<Int>> result = {{}};
    for (const auto& u : v) {
		// bool judge(true);
		VEC<VEC<Int>> temp;
		for (const auto& x : result) {
			// if(x.size() == (ndivs - 2)){
			// 	VecInt change_sum(control_divs.nrows(), 0); judge = true;
			// 	for_Int(i, 0, (ndivs - 2) / 2) change_sum += sit_mat.tr()[i + 1] - VecInt(a[i]);
			// 	for_Int(i, (ndivs - 2) / 2, (ndivs - 2)) change_sum += VecInt(a[i]);
			// 	if (SUM(change_sum) > control_divs[0][ndivs/2]) judge = false;
			// }
			for (const auto y : u) {
				auto new_combination = x; 					// Create a new combination
				new_combination.push_back(y); 				// Add current element
				if (x.size() == (ndivs - 2)) {
					VecInt change_sum(control_divs.nrows(), 0);
					for_Int(i, 0, (ndivs - 2) / 2) change_sum += sit_mat.tr()[i + 1] - VecInt(a[i]);
					for_Int(i, (ndivs - 2) / 2, (ndivs - 2)) change_sum += VecInt(a[i]);
					if (SUM(change_sum) > control_divs[0][ndivs / 2])		temp.push_back(std::move(new_combination)); // Move to temp
				}
				else {
					temp.push_back(std::move(new_combination)); // Move to temp
				}
			}
			
		}
		// Move temp to result
		result = std::move(temp);
	}
    return result;
}

VecInt NocSpace::read_from_col_lable(const VEC<Int> x, const VEC<VEC<Int> > a) const {
	VecInt ren;
	for (const auto& lable : x) {
		VecInt ren_i(ren);
		VecInt col(a[lable]);
		ren.reset(concat(ren_i, col));
	}
	return ren;
}

BasisIndex NocSpace::wherein_NocSpace(BasisIndex h_i)const
{
	// idx_div contains cumulative boundaries, including the final end sentinel.
	// Invalid indices must not become an out-of-range group number.
	if (idx_div.size() < 2 || h_i < 0 ||
		h_i < idx_div.front() || h_i >= idx_div.back())
		ERR("wherein_NocSpace: basis index outside group boundaries.");
	const auto upper = std::upper_bound(idx_div.begin(), idx_div.end(), h_i);
	return upper - idx_div.begin() - 1;
}

BasisIndex NocSpace::GroupCursor::locate(BasisIndex h_i)
{
	// Lazy initialization also makes empty MPI partitions safe: no lookup occurs.
	// Backward queries start a fresh binary search rather than reusing a stale group.
	if (!initialized_ || h_i < previous_h_) {
		group_ = space_.wherein_NocSpace(h_i);
		initialized_ = true;
	}
	else {
		const auto& boundaries = space_.idx_div;
		if (h_i >= boundaries.back())
			ERR("GroupCursor: basis index outside group boundaries.");
		// Strict upper boundary: handles skipped groups and zero-sized groups too.
		while (h_i >= boundaries[group_ + 1]) ++group_;
	}
	previous_h_ = h_i;
	return group_;
}

void NocSpace::print(std::ostream& os) const {
#define nocspace_print(var, comment) print(os, NAME(var), STR(var), comment)

	using namespace std;
	Str NOOC = p.nooc_mode;
	Str nppsos = nppso_str();
	Int norbits = SUM(sit_mat);
	Str rotation_imp = p.if_norg_imp ? "Yes" : "NO";


	os << "// NORG setting" << endl;

	// nocspace_print(ndivs, "The amount of divisons's number. ");
	nocspace_print(dim, "the dimension of the Shortcut space.");
	nocspace_print(nppsos, "The number of partical per spin orbital. ");
	nocspace_print(nspa, "The amount of partical's number.");
	nocspace_print(norbits, "The amount of orbital(imp+bath) number.");
	nocspace_print(NOOC, "Correlation nature orbital occupation constraint.");
	nocspace_print(rotation_imp, "if rotation impurity orbital is used. ");
	// nocspace_print(ndivs, "The amount of divisons's number. ");
	nocspace_print(control_divs, "to set the number of division and the shortcut restratin.");
	// nocspace_print(h0, "transformed hopping integral");
	// nocspace_print(mu, "-mu");
	nocspace_print(p.U, "The Hubbard term U.");
	nocspace_print(p.Uprm, "The U^' term");
	nocspace_print(p.jz, "The hund coupling");
	nocspace_print(p.mu, "The hund coupling");
    nocspace_print(p.fit_max_omg, "The fitting max omg");
    if(!p.fit_points.empty()) nocspace_print(VecInt(p.fit_points).mat(1, p.fit_points.size()), "to set the fix fitting points.");
    nocspace_print(p.fit_num_omg, "The max number of fitting points.");
	// u_hbd, p.U12, p.U12, p.U12

	os << "// prmtr print end  " << present() << endl;

#undef nocspace_print
}
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<< basic function end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
