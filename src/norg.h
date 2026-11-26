#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "prmtr.h"
#include "densitymatrix.h"
#include "nocspace.h"
#include "state.h"
#include "crrltfun.h"
#include "crrvec.h"
#include "krylov.h"
#include "impurity.h"


// impurity model
class NORG {
	// typedef VEC<VecInt> Tab;
	//*****************iteration***************
	Int iter_norg_cnt;						// count the NORG optimize iteration times.
	Int norg_stab_cnt;						// count the NORG stability iteration times.
	VecReal occnum_pre;						// occupation number record before one optimize iteration.
	VecReal occnum_lst;						// occupation number record after one optimize iteration.
	Real	groune_pre;						// ground state energy record before one optimize iteration.
	Real energy_err;						// ground state energy Error: error between the two NORG iterations
	Real occupation_err;					// occupation Error			: error between the two NORG iterations
	Real correctionerr;						// correction Error			: error between two correction vector modify
	mutable VecCmplx checkg_pre;			// correction vec check point before one optimize iteration.
	mutable VecCmplx checkg_lst;			// correction vec check point after one optimize iteration.

public:
	const MyMpi &mm;						// parameters
	const Prmtr &p;							// parameters
	VEC<MatReal> uormat;					// unitary_orbital_rotation_matrix

	Impdata impH;
	Real	groune_lst;						// ground state energy record after one optimize iteration.
	MatReal&  final_ground_state;
	VecReal  occnum;
	ImGreen impgreen;

	mutable NocSpace scsp;					// The main space.
	mutable DensityMat oneedm;				// The main space's density matrix.
	
	VecReal nparticle;						// The impurity's particle number.
private:
	/*// ! abandon
	// It assume that we already have the hopint from the Impurity class, but still have not rotated it yet.
	VecReal set_row_primeter_byimpH(const VEC<MatReal>& uormat_i, const MatReal& impH_i);

	// C^+_i C^+_j C_k C_l h_inter from [i][l][j][k] to [alpha][eta][beta][gamma]
	// It assume that we already have the hopint and h_inter from the Impurity class, with four Fermi interaction.
	VecReal set_row_primeter_byimpH(const VEC<MatReal>& uormat_i, const Impdata& impH_i);
	*/

	// Already have the hopint and h_inter from the Impurity class, with four Fermi interaction.
	// Speed up by Operator calss. C^+_i C^+_j C_k C_l h_inter from [i][l][j][k] to [alpha][eta][beta][gamma]
	// void set_row_primeter_byimpH(const VEC<MatReal>& uormat_i, const Impdata& impH_i, std::map<Int, Real>& oper_i);
	void set_row_primeter_byimpH(const VEC<MatReal>& uormat_i, const Impdata& impH_i, VecReal& oper_i);

	// only change for first norg_set of the first div.
	VecInt nppso(const VecInt &a, Int positon)
	{
		VecInt nppso_i(a);
		Int i = abs(positon);
        if (positon > 0) ++nppso_i[i-1];
        if (positon < 0) --nppso_i[i-1];
		return nppso_i;
	}

// ! NOT pass the test.
	VecInt nppso_all(const VecInt &a, Int positon)
	{
		VecInt nppso_i(a);
		Int i = abs(positon);
		if (positon > 0) nppso_i[(i - 1)] += 1;
		if (positon < 0) nppso_i[(i - 1)] -= 1;
		return nppso_i;
	}

	bool converged(Real ge_err = 1.E-10);

	bool green_modify_converged() const;
	bool green_modify_converged(Real correctionerr_i) const;

	void readmatrix(MatReal& m, const Str& file);
/*
	// using the correct vector to modify the shortcut space.
	void upgrade_space(NocSpace& scsp_i, NocSpace& scsp_ipl, VecReal& state_pl, NocSpace& scsp_imi, VecReal& state_mi);
*/

	VEC<MatReal> uormat_initialize();

	void show_the_nozero_number_of_tabel();
/*

	void update_by_correct_vec() const;

	void update_by_correct_vec(const Int orbital, const VecReal& ground_state_temp, const Real& ge,  const VEC<MatReal>& uormat_i) const;

	void rotation_the_space(const VEC<MatReal>& uormat_i) const;
*/
public:
	// One step means one orbital rotation followed by a ground-state solve.
	// The initial solve in the supplied basis is not counted as a rotation step.
	// FixedSteps with zero steps performs only that initial solve, without rotation.
	struct StopPolicy {
		enum class Mode { NormalConvergence, EnergyTolerance, FixedSteps };
		Mode mode;
		Real tolerance;
		Int steps;
		explicit StopPolicy(Mode mode_i = Mode::NormalConvergence,
			Real tolerance_i = 0., Int steps_i = 0)
			: mode(mode_i), tolerance(tolerance_i), steps(steps_i) {}
		bool valid() const;
		bool reached(Int completed_steps, Real relative_energy_error) const;
		// Compare only the setting used by this mode (steps or tolerance).
		bool equivalent_to(const StopPolicy& other) const;
	};

	NORG(const MyMpi& mm_i, const Prmtr& prmtr_i);
	NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, const Tab& table);
	// NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, VecInt nparticals);
	NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, Str tab_name);

	// NORG(const MyMpi& mm_i, const Impurity& imp_i, const Prmtr& prmtr_i);
	
	// for multi-number space.
	void up_date_h0_to_solve(const Impdata& impH, const VecReal sub_energy = Vec<Real>()); 

	void up_date_h0_to_solve(const Impdata& impH, const Int mode);
	void up_date_h0_to_solve(const Impdata& impH_i, const Int mode, const Real error);
	// Controlled whole-subspace or per-block G solve; legacy overloads keep their behavior.
	// Mode 0 can retain its configured degeneracy handling; block solves default to 1.
	void solve_with_policy(const Impdata& impH, const StopPolicy& policy,
		Int ground_state_mode = 1);
	
	void up_date_h0_to_solve(const Impdata& impH, const Real error);

/*
	// In this method we calculate the correction vector using the Krylov-space approach to modify the U.
	void modify_by_krylov_correct_vec();

	// (Deprecated) In this method we calculate the correction vector by the conjugategradient method.
	void modify_by_correct_vector() ;

	// (Deprecated)
	void writematrix(const MatReal& m, const Str U_name, Int iter_norg_cnt) const;
*/

//---------------------------------------calculate the physical operator---------------------------------

	Real sz_imp_sz_bath(const Int imp_postition, const VecReal& vgs_i);

	// To check the NO-interactions check.
	Real return_nointeractions_ground_state_energy(const MatReal& h0_i) const;
//--------------------------------------- for the Green function---------------------------------
	
	// void get_gimp_by_krylov_CV_modify(Green& imp_i) const;
	// // Only use for test the validity.
	// void get_gimp_by_krylov_CV_modify(ReGreen& imp_i) const;

	// Int get_gimp_with_possible_degeneracy(ImGreen& imp_i, Int iter_cont = 999);

	// void get_gimp_by_krylov(const ImGreen& imp_i);

	// void find_g(Green &g) ;
	
	// Int find_g_with_possible_degeneracy(VEC<ReGreen> &g, Int kind = 0) ;

	void get_gimp_diagonal(ImGreen& output);
    void get_gimp_orbital(ImGreen& single, Int orbital);

	void find_g0(Green& imp_i);

	// To calc the excitation spectrum for quasiparticle of holon-doublon bound states.
	void get_gimp_hdQPs(Green& imp_i, Int ex_idx = -1);
//--------------------------------------- for the io---------------------------------
	void write_norg_info(Int iter_cnt) const;
	
	//void write_occupation_info() const;// ! Abandon

	void write_state_info(Int iter_cnt) const;

	MatReal see_MatReal(const VEC<MatReal>& uormat_i) const {
		MatReal transform_uormat(dmat(p.norbit, 1.));
		Int counter(0);
		for (const auto& uormat_ii : uormat_i) {
			for_Int(i, 0, uormat_ii.nrows()) {
				for_Int(j, 0, uormat_ii.ncols()) {
					transform_uormat[i + counter][j + counter] = uormat_ii[i][j];
				}
			}
			counter += uormat_ii.nrows();
		}
		return transform_uormat;
	}

	// Read one set's impurity occupations from the solved density matrix.
	// Bath-only rotation; no file output, susceptibility or extra MPI reduction.
	VecReal get_impurity_occupations(Int set_idx) const;
	VecReal write_impurtiy_occupation(Int iter_cnt = -1, const Str& phy_name = empty_str) const;

	void PIO_occweight(VecReal occnum_lst) const {
		// Calculate sizes for each set
		VecInt row_sizes(p.norg_sets);
		Int start_idx = 0;
		
		// Create vectors to store values for each set
		VEC<VecReal> occnums(p.norg_sets);
		VEC<VecReal> occweights(p.norg_sets);
		
		// Process each set separately
		for_Int(i, 0, p.norg_sets) {
			row_sizes[i] = p.if_norg_imp ? p.nO2sets[i] : p.nI2B[i];
			
			// Create vectors for current set
			occnums[i] = VecReal(row_sizes[i]);
			occweights[i] = VecReal(row_sizes[i]);
			
			// Fill vectors for current set
			for_Int(j, 0, row_sizes[i]) {
				occnums[i][j] = occnum_lst[start_idx + j];
				occweights[i][j] = MIN(occnum_lst[start_idx + j], 
					1 - occnum_lst[start_idx + j]) < 1e-14 ? 0 : 
					MIN(occnum_lst[start_idx + j], 1 - occnum_lst[start_idx + j]);
			}
			start_idx += row_sizes[i];
		}
		
		Str nppso = scsp.nppso_str();
		if (mm) {
			// std::cout << "NORG Set Information:" << std::endl;
			// std::cout << "if_norg_imp: " << p.if_norg_imp << std::endl;
			// std::cout << "nppso: " << nppso << std::endl;
			
			// // Output all occupation numbers first
			// std::cout << "Occupation numbers:" << std::endl;
			// for_Int(i, 0, p.norg_sets) {
			// 	for_Int(j, 0, occnums[i].nrows()) {
			// 		std::cout << occnums[i][j][0] << " ";
			// 	}
			// 	std::cout << std::endl;
			// }
			
			// Then output all occupation weights
			std::cout << "Occupation weights:" << std::endl;
			for_Int(i, 0, p.norg_sets) {
				for_Int(j, 0, occweights[i].size()) {
					std::cout << occweights[i][j] << " ";
				}
				std::cout << std::endl;
			}
		}
	}

	// print the Double Occupancy matrix
	MatReal print_DO(const DensityMat& dm) const {
		MatReal docc = dm.find_imp_double_occupancy();

		// if (mm) PIO(NAV1(docc));
		return docc;
	}

	// For speed up reason, we need to save(read) the nature transform matrix(NTR).
	bool check_NTR() {
		IFS checker("ru" + scsp.nppso_str() + ".bi");
		return checker.good();
	}
	MatReal save_NTR();
	MatReal read_NTR();
};
