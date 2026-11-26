#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "gen/green_krylov_options.h"

class Prmtr {
private:
	const Int np;					// number of mpi processes
public:
    mutable GreenKrylovOptions green_krylov;
	// model related, model parameters
	Str project;					// project name
	Int nband;						// number of band in lattice model.
	Int norbs;						// number of spin-orbitals in lattice model.
	Real bandw;						// The bandwidth of this model.
	
	// square lattice, model parameters
	mutable Real U;					// Hubbard interaction U
	mutable Real jz;				// hund interaction Jz
	mutable Real Uprm;				// U^' as U - 2 * Jz
	mutable Real mu;				// chemical potential
	mutable Real degel;				// Degenerate energy levels
	mutable MatReal	eimp;			// the impurity energy matrix (up-dn-up-dn block structure).
	VecReal	bsr;					// the vector of bath sum rule.
	VecReal bethe_t;				// hopping strength t for bethe lattice
	//ReGreen
	Real freq_upp;					// upper bound of real frequency, 1.5 * (band upper bound) suggested
	Real freq_low;					// lower bound of real frequency, 1.5 * (band lower bound) suggested
	Real dlt_freq;					// delta frequency, freq = n * dlt_freq, freq[i] = freq_low + dlt_freq * i;
	Real eta_freq;					// eta of omega + i * eta
	Int nfreq;						// number of real frequencies = 1 + Int_ROUND((freq_upp - freq_low) / dlt_freq)	// dmft related
    VecCmplx Re_z;		

	//ImGreen		
	mutable Real beta;						// unit_omg = PI/beta;
	mutable Real unit_omg;					// unit imaginary frequency, omg_n = (2 n + 1) unit_omg, 0.01 or 0.02 suggested for zero temperature
	mutable Real max_omg;					// imaginary frequency cutoff, 4 * (half bandwidth) suggested
	mutable Real nmesh;						// imaginary frequency number form Zen.
	mutable Int num_omg;					// number of positive imaginary frequencies
	mutable VecCmplx Im_z;

	//fitting related		
	mutable VEC<Int> fit_points;				// number of positive imaginary frequencies used for bath fitting
	mutable Real fit_max_omg;				// imaginary frequency upper bound used for bath fitting
	mutable Int fit_num_omg;				// number of positive imaginary frequencies used for bath fitting
	mutable Real fit_pow;					// 1 / (omg_n + fit_rsd) ^ fit_pow is the weight used in bath fitting
	mutable Real fit_rsd;					// 1 / (omg_n + fit_rsd) ^ fit_pow is the weight used in bath fitting

	//iter related
	bool imp_backup;				// imp backup
	Int iter_max;					// DMFT max number of iterations
	Int gauss_n_max;				// max gauss_n for Green function integration
	Int gauss_n_min;				// min gauss_n for Green function integration

	// New prmtr for NORG solver.
	Int norg_sets;					// number of norg sets (spinless orbital).
	Int iter_max_norg;				// the max NORG iteration times.
	bool if_norg_imp;				// if norg rotate NO with impurity orbitals, if true, the imp_orb will rotate, treated same as bath orbital.
	Int if_norg_degenerate;			// set 0: the norg running  in the non-degenerate state; set 1: the norg running for finding the degenerate state.
	mutable VecInt nI2B;			// the number of bath sites for each impurity.
	mutable VecInt nO2sets;			// The number of orbital to each sets.
	mutable VecInt npartical;		// number of particals.
	mutable VEC<VecInt> degenerate_N_subspaces; // particle-number subspaces degenerate at the lowest searched ground-state energy.
	mutable VEC<MatInt> control_divs_degenerate_N_subspaces; // control_divs corresponding to degenerate_N_subspaces by index.
	mutable VEC<VEC<VecReal>> occupationnumber_N_subspaces; // occupationnumber corresponding to degenerate_N_subspaces by index.
	mutable VEC<VEC<MatReal>> rotationU_N_subspaces; // orbital rotation matrices corresponding to degenerate_N_subspaces by index.


	mutable Int nbath;				// number of bath sites, must be an integer multiple of 4 
	mutable Int norbit;				// number of NORG orbital(imp + bath).
	mutable Int n_rot_orb;			// rotation orbital number.

	mutable MatInt control_divs;	// to set the number of division and the shortcut restratin.
	mutable MatReal restrain_difference; // optional real matrix read from PARAMS.norg
	mutable Int restrain_difference_dim; // positive: checked row count; 0: disable custom restraints, retaining preset rows
	mutable MatInt restrain_active_orbit_number; // optional integer matrix read from PARAMS.norg
	mutable Int restrain_active_orbit_number_dim; // positive: checked row count; 0: use adaptive active counts, retaining preset rows
	mutable Vec<MatReal> restrain_difference_G; // G real matrices, indexed as [matrix][row][column]
	mutable Vec<MatInt> restrain_active_orbit_number_G; // G integer matrices
	mutable Vec<MatInt> control_divs_G; // G control matrices, indexed as [matrix][row][column]
	mutable Int restrain_G_mode; // G restraint mode from params.dat; default 0
	mutable Int restrain_autonooc; // Integer parameter from params.dat; default 0
	mutable VecInt restrain_tight_divs; // Tight constraint row; all input elements are values (no count prefix).
	mutable VecReal norg_each_orbital_converg_error; // Per-set errors; excludes the input count (0 selects fixed steps).
	mutable VecInt norg_each_orbital_converg_step_number; // Per-set step counts; excludes the input count.
	mutable VecInt templet_restrain;// to set the restrain templet one for all distribute;
	// mutable VecInt templet_control;	// to set the control templet one for all distribute;
	mutable VecInt stage2_restrain;	// to set the stage2 restrain one for all distribute;
	mutable VecInt stage2_control;	// to set the stage2 control one for all distribute;
	mutable Int ndiv;				// the the divsion's nubmer.
	mutable VEC<MatReal> rotationU;	// save the rotation matrix for convience.

	
	mutable Str nooc_mode;
	mutable Int chi_A;				// nematic susceptibility coupling constant from params.dat
	mutable Int chi_nomb;			// number of bosonic Matsubara frequencies for chi_A(i*Omega_n)
	mutable Int if_n_init;						// whether to use n_init (0: no, 1: yes)
	mutable Int if_eigenstate_number;			// whether to force the Lanczos eigenstate number (0: no, 1: yes)
	mutable Int eigenstate_number;				// forced number of Lanczos eigenstates
	mutable Int gs_lanczos_reorthogonalization; // ground-state Lanczos setting from params.dat; default 0
	mutable Int g_lanczos_reorthogonalization;  // Green-function Lanczos setting from params.dat; default 0
	mutable Int testH0;							// whether to test norgH0.txt (0: no, 1: yes)
	mutable Int testDelta;						// whether to test norgDelta.txt (0: no, 1: yes)
	mutable Int if_fix_N_subspace;				// whether to fix the N-subspace (0: no, 1: yes)
	mutable VecInt fix_start_search_N_subspace;	// starting particle numbers for N-subspace search, one per norg_set
	mutable VEC<VEC<VecInt>> fix_N_subspace;	// fixed degenerate subspaces: [subspace][0] is npartical, [subspace][1..] are control_divs rows


	//--------------------------------------------------------special for the hhd function(arXiv:2209.14178v1)-----------------------------------------------------------------
	mutable Real alpha;						// the parameter for the hhd function(arXiv:2209.14178v1) to control the strength of the interaction in two wide band.
	Real delta;								// the parameter for the hhd function(arXiv:2209.14178v1) to control the strength of the interaction in two wide band.


	//---------------------------------------------------------------------------------------------------------------------------------------

	
	// derived
	Str ofx;				// output filename prefix

private:
	void set_inert_values();
	void set_values();
	void print(std::ostream &os, const Str &var, const Str &val, const Str &comment) const {
		using namespace std;
		Str true_var = var == "\"\"" ? "" : var;
		os << rght_justify(true_var, 16) << (true_var == "" ? "   " : " = ") << left_justify(val, w_Real)
			<< "    # " + comment << endl;
	}

	VEC<MatReal> uormat_initialize() const {
		VEC<MatReal> uormat_i;
		for_Int(i, 0, nO2sets.size()) {
			MatReal temp(dmat(nO2sets[i], 1.));
			uormat_i.push_back(std::move(temp));
		}
		return uormat_i;
	}


public:
	Prmtr(const MyMpi& mm);
	// void after_modify_prmtr() const;
	void after_modify_prmtr(const VecInt& nbaths) const;
	void reinit_from_nO2sets() const;

	void according_nppso(const VecInt& nppsos) const;
	// Rebuild bath partitions for this N subspace using the tight constraint row.
	Prmtr make_tight_subspace_parameters() const;
	void according_controler(const Vec<VecInt>& controler, const VecInt& or_deg) const;
	void derive_ImGreen() const;
	void recalc_partical_number() const;
	void print(std::ostream &os = std::cout) const;
	// void change_the_norg_restrain_and_div(VecInt new_restrain, VecInt new_control) const;


	void derive();
	// Imag frequency
	Real Imomg(Int n) const { return imag(Im_z[n]); }
	Cmplx Imz(Int n) const { return Im_z[n]; }
	// Real frequency
	Real Reomg(Int n) const { return real(Re_z[n]); }
	Cmplx Rez(Int n) const { return Re_z[n]; }
};
