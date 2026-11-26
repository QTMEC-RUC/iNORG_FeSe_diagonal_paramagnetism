/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "occler.h"
using namespace std;

Occler::Occler(const MyMpi& mm_i, Prmtr& prmtr_i):
    mm(mm_i), tp(prmtr_i), sub_energy({1., 0., 0.}), search_rotation(prmtr_i.rotationU)
{
    // initial_uormat();
}


NORG Occler::find_ground_state_adaptive(
    const Impdata &impH_i,
    const VecInt& or_deg,
    Int determinacy_threshold)
{
    reset_search_state();

    // Evaluate the supplied center first, including a restored EDMFT subspace.
    calculate_ground_state_energy(impH_i, tp.npartical);

    VEC<VEC<Int>> super_blocks = build_super_blocks(or_deg);

    AllBlocksScanResult scan1 = scan_all_blocks_once(impH_i, tp.npartical, or_deg, determinacy_threshold);
    if(mm) {
        WRN(NAV2(scan1.determined_blocks.size(), scan1.uncertain_blocks.size()));
        PIO("Determined super-blocks (sorted by D_value desc):");
        for_Int(si, 0, (Int)scan1.determined_blocks.size()) {
            const auto& sb   = scan1.determined_blocks[si];
            Int          sb_idx = (Int)(std::find(super_blocks.begin(), super_blocks.end(), sb) - super_blocks.begin());
            const auto&  info   = scan1.block_infos[sb_idx];
            Str members; for(Int idx : sb) members += STR(idx)+" ";
            PIO("  super-block ["+members+"]: optimal_n="+STR(info.optimal_n)
                +", D_value="+STR(info.D_value)
                +", D_less="+STR(info.D_less)
                +", D_more="+STR(info.D_more));
        }
        PIO("Uncertain super-blocks:");
        for_Int(si, 0, (Int)scan1.uncertain_blocks.size()) {
            const auto& sb     = scan1.uncertain_blocks[si];
            Int          sb_idx = (Int)(std::find(super_blocks.begin(), super_blocks.end(), sb) - super_blocks.begin());
            const auto&  info   = scan1.block_infos[sb_idx];
            Str members; for(Int idx : sb) members += STR(idx)+" ";
            Str cands;   for(Int c : info.candidate_n) cands += STR(c)+" ";
            PIO("  super-block ["+members+"]: optimal_n="+STR(info.optimal_n)
                +", D_value="+STR(info.D_value)
                +", D_less="+STR(info.D_less)
                +", D_more="+STR(info.D_more)
                +", candidates=["+cands+"]");
        }
    }

    VecInt optimal_nppso = scan1.current_best_nppso;

    if(!scan1.uncertain_blocks.empty()) {
        AllBlocksScanResult* use = &scan1;
        AllBlocksScanResult scan2;
        if((Int)scan1.determined_blocks.size() < determinacy_threshold) {
            scan2 = scan_all_blocks_once(impH_i, scan1.current_best_nppso, or_deg, determinacy_threshold);
            use = &scan2;
            if(mm) WRN(NAV2(scan2.determined_blocks.size(), scan2.uncertain_blocks.size()));
        }

        VEC<VEC<Int>> use_super_blocks = build_super_blocks(or_deg);
        Int nunc = (Int)use->uncertain_blocks.size();
        VEC<VEC<Int>> candidates(nunc);
        for_Int(i, 0, nunc) {
            const auto& sb = use->uncertain_blocks[i];
            Int sb_idx = (Int)(std::find(use_super_blocks.begin(), use_super_blocks.end(), sb) - use_super_blocks.begin());
            candidates[i] = use->block_infos[sb_idx].candidate_n;
        }

        optimal_nppso = search_uncertain_combinations(impH_i, use->current_best_nppso, use->uncertain_blocks, candidates);
    }

    // The block-wise optimum can be a newly assembled combination which has
    // not yet been evaluated as a complete particle-number subspace.
    calculate_ground_state_energy(impH_i, optimal_nppso);
    collect_degenerate_N_subspaces();

    tp.according_nppso(tp.npartical = optimal_nppso);
    prioritize_current_N_subspace();
    if(mm) WRN(NAV2(tp.control_divs,tp.npartical));
    
    NORG a(mm, tp);
    initialize_refinement_rotation(a);
    a.up_date_h0_to_solve(impH_i, 1);
    const std::string key = a.scsp.nppso.string();
    occupationnumber_cache[key] = a.oneedm.occupationnumber;
    rotationU_cache[key] = a.uormat;
    if(!occupationnumber_N_subspaces_.empty()
        && !rotationU_N_subspaces_.empty()
        && degenerate_N_subspaces_[0] == tp.npartical
        && control_divs_degenerate_N_subspaces_[0] == tp.control_divs) {
        occupationnumber_N_subspaces_[0] = a.oneedm.occupationnumber;
        tp.occupationnumber_N_subspaces[0] = a.oneedm.occupationnumber;
        rotationU_N_subspaces_[0] = a.uormat;
        tp.rotationU_N_subspaces[0] = a.uormat;
        tp.rotationU = a.uormat;
    } else if(mm) {
        WRN("Cannot update the final occupationnumber and rotationU because the current tp "
            "is not the first degenerate N-subspace.");
    }
    return a;
}

// ! tested.
NORG Occler::find_ground_state_partical(const Impdata &impH_i, const VecInt& or_deg)
{
    Int counter_norg(0);
    reset_search_state();
    // tp.templet_restrain[1] = -1; tp.templet_restrain[tp.ndiv - 1] = 1;  // ! abandoned on 2024-10-18
    while(1){
            Int counter(0);
            VEC<VecInt> nppsos = list_all_posible_nppsos(tp.npartical, or_deg);
            if(nppsos.empty() || !(nppsos[0] == tp.npartical)) {
                ERR("Search starting subspace is incompatible with the orbital grouping")
            }
            sub_energy.reset(nppsos.size(),0.); sub_energy = 0.;
        for(const auto& nppso: nppsos) {
            if(MIN(nppso)==0) break;
            for_Int(i, 0, nppso.size()) if(nppso[i] >= tp.nO2sets[i]) break;
            if (mm) std::cout << "The " << ++counter_norg << "-th NORG begin "<< present() << std::endl; // norg counter

            tp.according_nppso(tp.npartical = nppso);
            if(mm) WRN(NAV2(tp.control_divs,tp.npartical));

            const std::string key = nppso.string();
            auto iter = energy_cache.find(key);
            if (iter != energy_cache.end()) {
                use_cached_search_rotation(key);
                sub_energy[counter] = iter->second;
                if(mm) PIO("The subspace: "+nppso_str(nppso)+", and energy: "+STR(sub_energy[counter]));
            }
            else {
                std::streambuf* old = std::cout.rdbuf();
                std::cout.rdbuf(NULL); // redirect std::cout to nowhere
                
                Prmtr search_p = make_search_parameters();
                NORG a(mm, search_p);
                std::cout.rdbuf(old);
                initialize_rotation(a, search_rotation, "candidate search");
                a.up_date_h0_to_solve(impH_i, sub_energy.truncate(0, counter)); sub_energy[counter] = a.groune_lst;
                remember_search_result(a);
                if(mm) PIO("The subspace: "+nppso_str(nppso)+", and energy: "+STR(sub_energy[counter]));
            }

            if (mm) {
                // OFS ofs_a;
                // ofs_a.open("ru" + nppso_str(a.scsp.nppso) + ".bi"); 
                // for_Int(i, 0, a.uormat.size()) biwrite(ofs_a, CharP(a.uormat[i].p()), a.uormat[i].szof());
            }
            // if(mm) WRN(NAV4(counter_norg, counter, a.groune_lst, a.scsp.nppso.mat(1,nppsos[0].size())));
            counter++;
        }

        if(sub_energy.idx_min() == 0) {

            tp.according_nppso(tp.npartical = nppsos[0]);
            collect_degenerate_N_subspaces();
            prioritize_current_N_subspace();
            if(mm) WRN(NAV2(tp.control_divs,tp.npartical));

            NORG a(mm, tp);
            initialize_refinement_rotation(a);
            a.up_date_h0_to_solve(impH_i, 1);
            const std::string key = a.scsp.nppso.string();
            occupationnumber_cache[key] = a.oneedm.occupationnumber;
            rotationU_cache[key] = a.uormat;
            if(!occupationnumber_N_subspaces_.empty()
                && !rotationU_N_subspaces_.empty()
                && degenerate_N_subspaces_[0] == tp.npartical
                && control_divs_degenerate_N_subspaces_[0] == tp.control_divs) {
                occupationnumber_N_subspaces_[0] = a.oneedm.occupationnumber;
                tp.occupationnumber_N_subspaces[0] = a.oneedm.occupationnumber;
                rotationU_N_subspaces_[0] = a.uormat;
                tp.rotationU_N_subspaces[0] = a.uormat;
                tp.rotationU = a.uormat;
            } else if(mm) {
                WRN("Cannot update the final occupationnumber and rotationU because the current tp "
                    "is not the first degenerate N-subspace.");
            }
            return a;
        } else {
            // nppsos.clear();
            // nppsos = list_all_posible_nppsos(nppsos[sub_energy.idx_min()], or_deg);
            // if(mm) WRN(NAV(sub_energy.stdvec().at(MIN(sub_energy))));
            // tp.npartical = nppsos[sub_energy.stdvec().at(MIN(sub_energy))];
            // if(mm) WRN(NAV2(sub_energy.idx_min(),nppsos[sub_energy.idx_min()].mat(1,10)));
            tp.npartical = nppsos[sub_energy.idx_min()];
            // break;
        }
    }
}

/*
VecInt Occler::find_gs_nppso(const Impdata &impH_i, const VecInt& or_deg)
{
    Int counter_norg(0);
    VEC<MatReal> u_temp;
    while(1){
            Int counter(0);
            VEC<VecInt> nppsos = list_all_posible_nppsos(tp.npartical, or_deg);
            sub_energy.reset(nppsos.size(),0.); sub_energy = 0.;
        for(const auto& nppso: nppsos) {
            if(MIN(nppso)==0 || MAX(nppso)==tp.nO2sets[0]) break;
            if (mm) std::cout << "The " << ++counter_norg << "-th NORG begin" << present() << std::endl; // norg counter

            tp.according_nppso(tp.npartical = nppso);
            NORG a(mm, tp);
            // IFS ifs_a("ru" + nppso_str(a.scsp.nppso) + ".bi");
            // if (ifs_a) for_Int(i, 0, a.uormat.size()) biread(ifs_a, CharP(a.uormat[i].p()), a.uormat[i].szof());
            a.up_date_h0_to_solve(impH_i, sub_energy.truncate(0, counter)); sub_energy[counter] = a.groune_lst;
            if (mm) {
                // OFS ofs_a;
                // ofs_a.open("ru" + nppso_str(a.scsp.nppso) + ".bi"); 
                // for_Int(i, 0, a.uormat.size()) biwrite(ofs_a, CharP(a.uormat[i].p()), a.uormat[i].szof());
            }
            // if(mm) WRN(NAV4(counter_norg, counter, a.groune_lst, a.scsp.nppso.mat(1,nppsos[0].size())));
            counter++;
        }

        if (sub_energy.idx_min() == 0) {
            return nppsos[0];
        }
        else {
            tp.npartical = nppsos[sub_energy.idx_min()];
        }
    }
}
*/
//-----------------------------------------private-----------------------------------------


// void Occler::initial_uormat(){
// 	for_Int(i, 0, tp.nI2B.size()) {
// 		MatReal temp(dmat(tp.nI2B[i], 1.));
// 		uormat.push_back(std::move(temp));
// 	}
// }


// bool Occler::if_ground_state(VecReal sub_energy) {
// }

VEC<VecInt> Occler::list_all_posible_nppsos(const VecInt& nppso_i, const VecInt& or_deg) const {
    if(nppso_i.size() != or_deg.size()) ERR("The or_deg's number should same as the nppso.")
	VEC<Int> idx(MAX(or_deg),0);
    if(mm) WRN(NAV(nppso_i)); // temporary@2024-11-28

	// Build idx: for each independent variable (1 to MAX(or_deg)), find the first orbital with that or_deg value
	for_Int(deg_val, 1, MAX(or_deg)+1) {
		for_Int(i, 0, or_deg.size()) {
			if(or_deg[i] == deg_val) {
				idx[deg_val-1] = i;
				break;
			}
		}
	}

    VEC<VEC<Int>> nppsos_ii(MAX(or_deg)), nppsos_idx;
    for_Int(i, 0, idx.size()){
        Int idx_m(nppso_i[idx[i]]-1), idx_p(nppso_i[idx[i]]+1);
        nppsos_ii[i].push_back(nppso_i[idx[i]]);
        if(idx_m > 0) nppsos_ii[i].push_back(idx_m);
        if(idx_p > 0) nppsos_ii[i].push_back(idx_p);
        if(MIN(VecInt(nppsos_ii[i])) < 0) ERR("nppsos_ii's element should not has the minus.")
    }
    // if(mm) WRN(NAV2(nppsos_ii[0].size(),nppsos_ii[1].size()));
    nppsos_idx = cart_product(nppsos_ii);
    VEC<VecInt> nppsos(nppsos_idx.size(), tp.npartical);
    // for_Int(i, 0, nppsos_idx.size()) for_Int(j, 0, or_deg.size()*2) nppsos[i][j] = nppsos_idx[i][or_deg[j/2] - 1]; // for spin inversion symmetry
    for_Int(i, 0, nppsos_idx.size()) for_Int(j, 0, or_deg.size()) nppsos[i][j] = nppsos_idx[i][or_deg[j] - 1];        // list all the possible
    return nppsos;
}

VecInt Occler::update_nppso_for_block(const VecInt& base_nppso, const VEC<Int>& super_block, Int new_n) {
    VecInt result = base_nppso;
    for(Int idx : super_block) result[idx] = new_n;
    return result;
}

// Build super-blocks from or_deg.
// Returns a list of groups; each group is a list of block indices sharing the same or_deg value.
// Groups are ordered by first appearance of their or_deg value in or_deg.
// Example: or_deg = [1,2,3,4,5,6,5,6,7,8]
//   → [ {0}, {1}, {2}, {3}, {4,6}, {5,7}, {8}, {9} ]
VEC<VEC<Int>> Occler::build_super_blocks(const VecInt& or_deg) const {
    std::map<Int, Int> val_to_group;
    VEC<VEC<Int>> super_blocks;

    for_Int(i, 0, or_deg.size()) {
        Int v = or_deg[i];
        auto it = val_to_group.find(v);
        if (it == val_to_group.end()) {
            val_to_group[v] = (Int)super_blocks.size();
            super_blocks.push_back({i});
        } else {
            super_blocks[it->second].push_back(i);
        }
    }
    return super_blocks;
}

Real Occler::calculate_ground_state_energy(const Impdata &impH_i, const VecInt& nppso) {
    const std::string key = nppso.string();
    auto iter = energy_cache.find(key);
    if (iter != energy_cache.end()) {
        use_cached_search_rotation(key);
        if(mm) PIO("The subspace: "+nppso_str(nppso)+", energy (cached): "+STR(iter->second));
        return iter->second;
    }

    tp.according_nppso(tp.npartical = nppso);
    if(mm) WRN(NAV2(tp.control_divs,tp.npartical));

    std::streambuf* old = std::cout.rdbuf();
    std::cout.rdbuf(NULL);

    Prmtr search_p = make_search_parameters();
    NORG a(mm, search_p);

    initialize_rotation(a, search_rotation, "candidate search");
    a.up_date_h0_to_solve(impH_i,0.00001);

    std::cout.rdbuf(old);

    remember_search_result(a);
    if(mm) PIO("The subspace: "+nppso_str(nppso)+", energy: "+STR(a.groune_lst));
    return a.groune_lst;
}

Prmtr Occler::make_search_parameters() const
{
    // Keep tp in the configured mode for refinement and returned solver data.
    // The local copy outlives its candidate solver and cannot leak cnooc to tp.
    Prmtr search_p(tp);
    search_p.nooc_mode = STR("cnooc");
    return search_p;
}

void Occler::initialize_refinement_rotation(NORG& solver) const
{
    const std::string key = solver.scsp.nppso.string();
    const auto rotation = rotationU_cache.find(key);
    const auto particles = nppso_cache.find(key);
    const auto control = control_divs_cache.find(key);
    if(rotation == rotationU_cache.end() || particles == nppso_cache.end()
        || control == control_divs_cache.end()) {
        ERR("Missing search result for refinement of N-subspace " + key)
    }
    if(!(particles->second == solver.scsp.nppso)
        || !(solver.p.npartical == solver.scsp.nppso)
        || !(control->second == solver.p.control_divs)) {
        ERR("Search and refinement particle counts or constraints do not match: " + key)
    }
    initialize_rotation(solver, rotation->second, "refinement of " + key);
    if(mm) PIO("Refinement uses the cached search rotation for N-subspace " + key);
}

void Occler::reset_search_state()
{
    search_rotation = tp.rotationU;
    energy_cache.clear();
    nppso_cache.clear();
    control_divs_cache.clear();
    occupationnumber_cache.clear();
    rotationU_cache.clear();
    degenerate_N_subspaces_.clear();
    control_divs_degenerate_N_subspaces_.clear();
    occupationnumber_N_subspaces_.clear();
    rotationU_N_subspaces_.clear();
    tp.degenerate_N_subspaces.clear();
    tp.control_divs_degenerate_N_subspaces.clear();
    tp.occupationnumber_N_subspaces.clear();
    tp.rotationU_N_subspaces.clear();
}

void Occler::remember_search_result(const NORG& solver)
{
    const std::string key = solver.scsp.nppso.string();
    energy_cache[key] = solver.groune_lst;
    nppso_cache[key] = solver.scsp.nppso;
    control_divs_cache[key] = solver.p.control_divs;
    occupationnumber_cache[key] = solver.oneedm.occupationnumber;
    rotationU_cache[key] = solver.uormat;
    search_rotation = solver.uormat;
}

void Occler::use_cached_search_rotation(const std::string& key)
{
    const auto found = rotationU_cache.find(key);
    if(found == rotationU_cache.end()) ERR("Missing cached search rotation: " + key)
    // A cache hit is also a visited subspace; the next new solve starts here.
    search_rotation = found->second;
}

void Occler::initialize_rotation(NORG& solver,
    const VEC<MatReal>& initial_rotation, const Str& context) const
{
    const Int nsets = solver.p.norg_sets;
    if(nsets <= 0 || initial_rotation.size() != static_cast<Idx>(nsets)
        || solver.p.nO2sets.size() != static_cast<Idx>(nsets)
        || solver.uormat.size() != static_cast<Idx>(nsets)) {
        ERR("Rotation set count does not match solver: " + context)
    }
    for_Int(i, 0, nsets) {
        const Int size = solver.p.nO2sets[i];
        if(size <= 0 || initial_rotation[i].nrows() != static_cast<Idx>(size)
            || initial_rotation[i].ncols() != static_cast<Idx>(size)
            || solver.uormat[i].nrows() != static_cast<Idx>(size)
            || solver.uormat[i].ncols() != static_cast<Idx>(size)) {
            ERR("Rotation dimensions do not match solver: " + context
                + ", set " + STR(i))
        }
    }
    // Copy only after validation; do not replace the seed with an NTR file.
    solver.uormat = initial_rotation;
    solver.p.rotationU = initial_rotation;
}

void Occler::collect_degenerate_N_subspaces(Real tolerance) {
    degenerate_N_subspaces_.clear();
    control_divs_degenerate_N_subspaces_.clear();
    occupationnumber_N_subspaces_.clear();
    rotationU_N_subspaces_.clear();
    if(energy_cache.empty()) return;

    Real lowest_energy = std::numeric_limits<Real>::max();
    for(const auto& item : energy_cache)
        lowest_energy = std::min(lowest_energy, item.second);

    for(const auto& item : energy_cache) {
        const Real relative_error = 2. * ABS(item.second - lowest_energy)
            / (ABS(item.second) + ABS(lowest_energy) + 1.);
        if(relative_error < tolerance) {
            degenerate_N_subspaces_.push_back(nppso_cache.at(item.first));
            control_divs_degenerate_N_subspaces_.push_back(control_divs_cache.at(item.first));
            occupationnumber_N_subspaces_.push_back(occupationnumber_cache.at(item.first));
            rotationU_N_subspaces_.push_back(rotationU_cache.at(item.first));
        }
    }
}

void Occler::prioritize_current_N_subspace() {
    // NORG::p refers to this tp, so make the collected subspaces visible to NORG.
    tp.degenerate_N_subspaces = degenerate_N_subspaces_;
    tp.control_divs_degenerate_N_subspaces = control_divs_degenerate_N_subspaces_;
    tp.occupationnumber_N_subspaces = occupationnumber_N_subspaces_;
    tp.rotationU_N_subspaces = rotationU_N_subspaces_;

    const Int n_subspaces = static_cast<Int>(tp.degenerate_N_subspaces.size());
    if(n_subspaces != static_cast<Int>(tp.control_divs_degenerate_N_subspaces.size())
        || n_subspaces != static_cast<Int>(tp.occupationnumber_N_subspaces.size())
        || n_subspaces != static_cast<Int>(tp.rotationU_N_subspaces.size())) {
        if(mm) {
            WRN("The sizes of the degenerate N-subspace data do not match. "
                + NAV4(tp.degenerate_N_subspaces.size(),
                       tp.control_divs_degenerate_N_subspaces.size(),
                       tp.occupationnumber_N_subspaces.size(),
                       tp.rotationU_N_subspaces.size()));
        }
        return;
    }

    Int found_idx = n_subspaces;
    for_Int(i, 0, n_subspaces) {
        if(tp.degenerate_N_subspaces[i] == tp.npartical
            && tp.control_divs_degenerate_N_subspaces[i] == tp.control_divs) {
            found_idx = i;
            break;
        }
    }

    if(found_idx == n_subspaces) {
        if(mm) {
            WRN("The current tp is absent from the degenerate N-subspace lists. "
                + NAV2(tp.npartical, tp.control_divs));
        }
        return;
    }

    if(found_idx != 0) {
        std::swap(tp.degenerate_N_subspaces[0],
                  tp.degenerate_N_subspaces[found_idx]);
        std::swap(tp.control_divs_degenerate_N_subspaces[0],
                  tp.control_divs_degenerate_N_subspaces[found_idx]);
        std::swap(tp.occupationnumber_N_subspaces[0],
                  tp.occupationnumber_N_subspaces[found_idx]);
        std::swap(tp.rotationU_N_subspaces[0],
                  tp.rotationU_N_subspaces[found_idx]);
    }

    // Keep the values returned by Occler getters in the same order as NORG::p.
    degenerate_N_subspaces_ = tp.degenerate_N_subspaces;
    control_divs_degenerate_N_subspaces_ = tp.control_divs_degenerate_N_subspaces;
    occupationnumber_N_subspaces_ = tp.occupationnumber_N_subspaces;
    rotationU_N_subspaces_ = tp.rotationU_N_subspaces;
}

DeterminacyInfo Occler::calculate_block_determinacy(const BlockEnergyResult& energies) {
    DeterminacyInfo info;

    info.D_less = energies.E_minus - energies.E_center;
    info.D_more = energies.E_plus  - energies.E_center;

    VEC<std::pair<Real, Int>> en_list;
    en_list.push_back({energies.E_center, energies.n_center});
    if(energies.n_minus != -1) en_list.push_back({energies.E_minus, energies.n_minus});
    if(energies.n_plus  != -1) en_list.push_back({energies.E_plus,  energies.n_plus});
    std::sort(en_list.begin(), en_list.end());

    info.optimal_n     = en_list[0].second;
    info.lowest_energy = en_list[0].first;
    Int ncand = std::min((Int)en_list.size(), 2);
    for(Int i = 0; i < ncand; i++) info.candidate_n.push_back(en_list[i].second);

    bool has_minus = (energies.n_minus != -1);
    bool has_plus  = (energies.n_plus  != -1);

    if(!has_minus && !has_plus) {
        // isolated point (only one valid n): trivially determined
        info.D_value       = 1.0;
        info.is_determined = true;
    } else if(!has_minus) {
        // lower boundary: D_less is meaningless, judge by D_more only
        info.D_value       = (info.D_more > 0.) ? 1.0 : 0.0;
        info.is_determined = (info.D_more >= 0.);
    } else if(!has_plus) {
        // upper boundary: D_more is meaningless, judge by D_less only
        info.D_value       = (info.D_less > 0.) ? 1.0 : 0.0;
        info.is_determined = (info.D_less >= 0.);
    } else {
        // both neighbors valid: full D calculation
        Real D_max = std::max(info.D_less, info.D_more);
        Real D_min = std::min(info.D_less, info.D_more);
        info.D_value       = (D_max != 0.) ? D_min / D_max : 0.;
        // both neighbors lower than center: very rare, always mark uncertain
        info.is_determined = !(info.D_less < 0. && info.D_more < 0.);
    }

    return info;
}

BlockEnergyResult Occler::scan_orbital_block_energy(
    const Impdata &impH_i,
    const VecInt& current_nppso,
    const VEC<Int>& super_block)
{
    BlockEnergyResult res;
    Int rep = super_block[0];  // all members share the same n; use first as representative
    Int n = current_nppso[rep];
    res.n_center = n;
    res.E_center = calculate_ground_state_energy(impH_i, current_nppso);

    Int n_minus = n - 1;
    if(n_minus > 0) {
        res.n_minus = n_minus;
        res.E_minus = calculate_ground_state_energy(impH_i, update_nppso_for_block(current_nppso, super_block, n_minus));
    } else {
        res.n_minus = -1;
        res.E_minus = std::numeric_limits<Real>::max();
    }

    Int n_plus = n + 1;
    if(n_plus < tp.nO2sets[rep]) {
        res.n_plus = n_plus;
        res.E_plus = calculate_ground_state_energy(impH_i, update_nppso_for_block(current_nppso, super_block, n_plus));
    } else {
        res.n_plus = -1;
        res.E_plus = std::numeric_limits<Real>::max();
    }

    return res;
}

AllBlocksScanResult Occler::scan_all_blocks_once(
    const Impdata &impH_i,
    const VecInt& current_nppso,
    const VecInt& or_deg,
    Int determinacy_threshold)
{
    AllBlocksScanResult result;
    VecInt center_nppso = current_nppso;

    VEC<VEC<Int>> super_blocks = build_super_blocks(or_deg);
    Int nsb = (Int)super_blocks.size();

    result.block_infos.resize(nsb);
    result.current_best_nppso = center_nppso;

    for_Int(sb, 0, nsb) {
        result.block_infos[sb] = calculate_block_determinacy(
            scan_orbital_block_energy(impH_i, center_nppso, super_blocks[sb])
        );
        Int opt_n = result.block_infos[sb].optimal_n;
        for(Int idx : super_blocks[sb]) result.current_best_nppso[idx] = opt_n;
    }

    VEC<std::pair<Real, Int>> d_rank;

    for_Int(sb, 0, nsb) {
        if(!result.block_infos[sb].is_determined) {
            result.uncertain_blocks.push_back(super_blocks[sb]);
        } else {
            d_rank.push_back({result.block_infos[sb].D_value, sb});
        }
    }

    std::sort(d_rank.begin(), d_rank.end(),
        [](const std::pair<Real,Int>& a, const std::pair<Real,Int>& b) {
            return a.first > b.first;
        }
    );

    Int n_det = (Int)d_rank.size();

    if(n_det <= determinacy_threshold) {
        for(const auto& p : d_rank)
            result.determined_blocks.push_back(super_blocks[p.second]);
    } else {
        for(Int i = 0; i < determinacy_threshold; i++)
            result.determined_blocks.push_back(super_blocks[d_rank[i].second]);
        for(Int i = determinacy_threshold; i < n_det; i++)
            result.uncertain_blocks.push_back(super_blocks[d_rank[i].second]);
    }

    return result;
}

VecInt Occler::search_uncertain_combinations(
    const Impdata &impH_i,
    const VecInt& base_nppso,
    const VEC<VEC<Int>>& uncertain_super_blocks,
    const VEC<VEC<Int>>& candidates)
{
    VEC<VEC<Int>> combos = cart_product(candidates);

    VecInt best_nppso = base_nppso;
    Real best_energy = std::numeric_limits<Real>::max();

    for(const auto& combo : combos) {
        VecInt nppso = base_nppso;
        for_Int(i, 0, (Int)uncertain_super_blocks.size())
            nppso = update_nppso_for_block(nppso, uncertain_super_blocks[i], combo[i]);
        Real e = calculate_ground_state_energy(impH_i, nppso);
        if(e < best_energy) { best_energy = e; best_nppso = nppso; }
    }

    if(mm) PIO("Best uncertain combination: "+nppso_str(best_nppso)+", energy: "+STR(best_energy));
    return best_nppso;
}
