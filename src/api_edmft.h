#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "prmtr.h"
#include "bath.h"
// #include "model.h"
#include "green.h"
#include "impurity.h"
#include "norg.h"
// #include "asnci.h"
#include "occler.h"



class APIedmft{
	const MyMpi& mm;					// parameters
	Prmtr& p;							// parameters
	Int num_orbital;					// The number of orbiatals.
	VecInt or_deg_idx;					// orbitals  degenerate idx.
	VecReal solver_eimp_data;			// Impurity energy level for orbitals, correspondingly mean Impurity energy.
	// ctqmcdata solver_ctqmc_data;
	Real Uc, Jz, mu;
	Int nband;
	Int norbs;
	Int ful_pcl_sch;
	Int iter_count;
	// VecInt artificial_symm;

	
	// NORG coding console
	Str mode;
	Real fit_range;
	VecInt restrain, fit_nbaths;
	VecReal weight_nooc, weight_freze;
	std::vector<double> n_init_vals;		// initial occupancies per spin-orbital from PARAMS.norg
	VecInt or_deg_adaptive_blk;				// super-block degeneracy for adaptive solver, read from PARAMS.norg

	// NORG test part
	//Int nimp;
	VecReal bathose, bathhop;

	// Iteration info
	Real sig_err;
	VecReal n_eles, fit_err;
	bool occupations_available = true;

public:
	VEC<VecReal> t_ose;						// hopping integral for all sites
	VEC<VecReal> t_hyb;						// H_0 onset energy
	VecReal muvec;
private:
	void update(const Str& file);
	// Reading returns whether a previous completed iteration was restored.
	bool edmft_back_up(const Str& status);

	bool if_lock(const Str file) const;

	void read_eDMFT(const Str& file);

	// NORG choose_cauculation_style(Str mode, Impurity &imp);
	// void fitting();

/*
	void test_for_fitting(const Bath& bth, const ImGreen& hby_i, Int num = 666);
	ImGreen fix_se(const ImGreen& se) const;
*/

void read_norg_setting(
	const std::string& filename,
	std::vector<double>& Ed,
	std::vector<int>& Deg,
	double& J,
	std::string& CoulombF,
	double& beta,
	double& U,
	std::vector<double>& noc1,
	std::vector<double>& noc2,
	std::vector<int>& restrain,
	std::vector<int>& fit_nbaths,
	std::vector<double>& n_init,
	std::vector<int>& or_deg_adaptive,
	int& chi_A,
	int& if_n_init,
	Int& if_eigenstate_number,
	Int& eigenstate_number,
	Int& testH0,
	Int& testDelta,
	int& if_fix_N_subspace,
	std::vector<int>& fix_start_search_N_subspace,
	std::vector<std::vector<std::vector<int>>>& fix_N_subspace);

	void process_occupation_sets(
		VEC<VecReal>& occnums,
		VEC<VecReal>& occweights,
		const VecInt& row_sizes,
		const VecReal& occnum_lst);

	void auto_nooc(Str mode, const Impurity &imp);
	void apply_restrain_difference(const VEC<VEC<VecReal>>& occweights_N_subspaces);

	void validate_green_restraint_schemes() const;
	void validate_green_iteration_settings() const;
	void validate_tight_preconvergence_settings() const;
	void validate_green_stopping_settings() const;
	NORG::StopPolicy green_stop_policy(Int scheme_idx) const;
	NORG::StopPolicy whole_subspace_green_stop_policy() const;
	static void validate_green_control(const MatInt& control,
		const Prmtr& solver_p, const Str& context);
	MatInt make_fixed_green_control(Int set) const;
	MatInt make_adaptive_green_control(const Prmtr& subspace_p,
		Int set, const VEC<VecReal>& occupation_weights) const;
	Prmtr make_orbital_parameters(const Prmtr& subspace_p,
		Int set, const VEC<VecReal>& occupation_weights) const;
	// Each subspace uses its own seed; release the tight solver before any wide solve.
	void preconverge_green_subspace(const Impdata& impH, Int subspace_idx,
		const Prmtr& subspace_p, VEC<MatReal>& reference_rotation,
		VEC<VecReal>* occupation_weights = nullptr);
	// Occupations have their own layout: set -> impurity orbital within that set.
	// Counts are 0/1 within a subspace and count subspaces in the ensemble sum.
	struct ImpurityOccupationData {
		VecReal values_by_set;
		VecInt collected_sets;
		// Vec has an explicit default constructor; initialize it directly.
		ImpurityOccupationData() : values_by_set(), collected_sets() {}
	};
	void solve_restrained_green_subspace(const Impdata& impH, Int subspace_idx,
		const Prmtr& subspace_p, ImGreen& assembled_green,
		// Input: this subspace's initial basis. Output: its tight reference basis.
		VEC<MatReal>& reference_rotation, ImpurityOccupationData& occupation_sums);
	void write_unmerged_green(const ImGreen& green, Int subspace_idx, Int set = -1) const;
	ImpurityOccupationData make_impurity_occupation_data() const;
	void validate_impurity_occupations(const ImpurityOccupationData& data,
		Int expected_contributions) const;
	void collect_orbital_occupations(const NORG& norg, Int subspace_idx,
		Int set_idx, ImpurityOccupationData& subspace_occupations) const;
	void finalize_subspace_occupations(Int subspace_idx,
		const ImpurityOccupationData& subspace_occupations,
		ImpurityOccupationData& occupation_sums) const;
	ImpurityOccupationData average_occupations(
		const ImpurityOccupationData& occupation_sums, Int n_subspaces) const;
	void update_occupation_log(const ImpurityOccupationData& occupations);
	void write_occupation_file(const Str& filename,
		const ImpurityOccupationData& occupations) const;

	void print_log(const Str& lbl, std::ostream& os = std::cout) const;

	void log(const Str& lbl) {
		if (mm) {
			print_log(lbl);
			OFS ofs("log.norg", std::ios::app);
			print_log(lbl, ofs);
			ofs.close();
		}
	}

public:
	APIedmft(const MyMpi& mm_i, Prmtr& p, const Str& file = empty_str);

};
