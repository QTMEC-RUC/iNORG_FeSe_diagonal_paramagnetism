#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "operator.h"
#include "specs.h"
#include "prmtr.h"
#include "densitymatrix.h"
#include "nocspace.h"
#include "norg.h"

struct BlockEnergyResult {
    Real E_minus, E_center, E_plus;
    Int  n_minus, n_center, n_plus;
};

struct DeterminacyInfo {
    Real D_value, D_less, D_more;
    Int  optimal_n;
    VEC<Int> candidate_n;
    bool is_determined;
    Real lowest_energy;
};

struct AllBlocksScanResult {
    VEC<DeterminacyInfo> block_infos;       // indexed by super-block
    VEC<VEC<Int>> determined_blocks;        // each entry is a super-block (list of block indices)
    VEC<VEC<Int>> uncertain_blocks;         // each entry is a super-block (list of block indices)
    VecInt current_best_nppso;
};

//The class for the Occler(occupancy controler)

class Occler{
	const MyMpi &mm;						// parameters
	// Prmtr &p;							    // parameters
	Prmtr tp;							    // test parameters
    // VecInt &nparticals;                     // temporyly storge the partical number
    // Real sub_energy[0], sub_energy[2], sub_energy[1];
	VecReal sub_energy;
	std::map<std::string, Real> energy_cache;
	std::map<std::string, VecInt> nppso_cache;
	std::map<std::string, MatInt> control_divs_cache;
	std::map<std::string, VEC<VecReal>> occupationnumber_cache;
	std::map<std::string, VEC<MatReal>> rotationU_cache;
    VEC<MatReal> search_rotation; // Initial seed, then the previous visited result.
	VEC<VecInt> degenerate_N_subspaces_;
	VEC<MatInt> control_divs_degenerate_N_subspaces_;
	VEC<VEC<VecReal>> occupationnumber_N_subspaces_;
	VEC<VEC<MatReal>> rotationU_N_subspaces_;

public:
	// VEC<MatReal> uormat;					// unitary_orbital_rotation_matrix

private:
    // void initial_uormat();
    Int if_ground_state();
    // bool if_ground_state(Int counter, VecReal sub_energy);

	VEC<VecInt> list_all_posible_nppsos(const VecInt& nppso_i, const VecInt& or_deg) const ;

    Real            calculate_ground_state_energy(const Impdata &impH_i, const VecInt& nppso);
    BlockEnergyResult  scan_orbital_block_energy(const Impdata &impH_i, const VecInt& current_nppso, const VEC<Int>& super_block);
    DeterminacyInfo    calculate_block_determinacy(const BlockEnergyResult& energies);
    AllBlocksScanResult scan_all_blocks_once(const Impdata &impH_i, const VecInt& current_nppso, const VecInt& or_deg, Int determinacy_threshold);
    VecInt             search_uncertain_combinations(const Impdata &impH_i, const VecInt& base_nppso, const VEC<VEC<Int>>& uncertain_super_blocks, const VEC<VEC<Int>>& candidates);
    VecInt             update_nppso_for_block(const VecInt& base_nppso, const VEC<Int>& super_block, Int new_n);
    VEC<VEC<Int>>      build_super_blocks(const VecInt& or_deg) const;
    void                collect_degenerate_N_subspaces(Real tolerance = 1E-6);
    void                prioritize_current_N_subspace();
    // Only coarse candidate-energy solves override the configured NOOC mode.
    Prmtr               make_search_parameters() const;
    // Initialize refinement from this search's matching subspace cache.
    void                initialize_refinement_rotation(NORG& solver) const;
    void                reset_search_state();
    void                remember_search_result(const NORG& solver);
    void                use_cached_search_rotation(const std::string& key);
    void                initialize_rotation(NORG& solver,
                            const VEC<MatReal>& rotation, const Str& context) const;

    Str nppso_str(const VecInt& nppso_i) const{
		Str temp; for_Int(i, 0, nppso_i.size()) { if (i == 0) temp += STR(nppso_i[i]); else /*if (i % 2 == 0)*/ temp += "-" + STR(nppso_i[i]); }	return temp;
	}

	VEC<VEC<Int>> cart_product(const VEC<VEC<Int>> &v) const
	{
		VEC<VEC<Int>> s = {{}};
		for (const auto &u : v)
		{
			VEC<VEC<Int>> r;
			for (const auto &x : s)
			{
				for (const auto y : u)
				{
					r.push_back(x);
					r.back().push_back(y);
				}
			}
			s = move(r);
		}
		return s;
	}



public:
	Occler(const MyMpi& mm_i, Prmtr& prmtr_i);
    NORG find_ground_state_partical(const Impdata &impH_i, const VecInt& or_deg);
    NORG find_ground_state_adaptive(const Impdata &impH_i, const VecInt& or_deg, Int determinacy_threshold = 3);
    const VEC<VecInt>& degenerate_N_subspaces() const { return degenerate_N_subspaces_; }
    const VEC<MatInt>& control_divs_degenerate_N_subspaces() const { return control_divs_degenerate_N_subspaces_; }
    const VEC<VEC<VecReal>>& occupationnumber_N_subspaces() const { return occupationnumber_N_subspaces_; }
    const VEC<VEC<MatReal>>& rotationU_N_subspaces() const { return rotationU_N_subspaces_; }
    // NORG find_ground_state_partical(const Impdata &impH_i);
    // VecInt find_gs_nppso(const Impdata &impH_i, const VecInt& or_deg);


};
