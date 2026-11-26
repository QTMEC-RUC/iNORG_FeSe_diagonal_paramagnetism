#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "operator.h"
#include "green.h"

struct ChiIomResult {
	VecReal omega;      // Bosonic Matsubara frequencies, Omega_n = 2*pi*n/beta.
	MatReal chi_iom;    // chi_iom[channel][n]; column n=0 is the static result.
};

// Correlation function The method is easily generalized to calculate correlations 
// $$\left\langle A^{\dagger}(t) B\left(t^{\prime}\right)\right\rangle$$ for any pair of observables $$A^{+}$$and $$B$$ 
// and, once the ground-state properties $$\left(\left|\psi_{0}\right\rangle\right,and\left.E_{0}\right)$$ are known,
// all correlations can be obtained with a small computational effort.
// Here \psi_{0} mean ground state(gs), ground state energy(e_g).
// DOI:: Physical review letters 59.26 (1987): 2999.

class CrrltFun : public Operator
{
	typedef std::tuple<VecReal, VecReal> Vectrdgnl;
	typedef std::tuple<StdVecReal, StdVecReal> VECtrdgnl;
	const NocSpace& old_nosp;				// nature orbital space
	const NocSpace& new_nosp;				// nature orbital space
	//VecReal vgs;							// vector ground state(gs)
	const Int crtorann;
	//const Real& gse;					// ground state energy(gse)	
	//const Int ex_orbital_position;

	//save the continue fraction a and b.
	VecReal a_saved, b_saved;

public:
	const VecReal ex_state;				// scalar-path source; empty for block/static-susceptibility construction

private:
	//  (Deactivate) Using the Lanczos to find the tridiagonal matrix.
	Vectrdgnl find_trdgnl(const VecReal& initial_vector, const Int crtann);

	// Using the Lanczos to mapping H to the tridiagonal matrix(First step).
	//VECtrdgnl find_trdgnl_first(const VecReal& initial_vector);

	// Using the Lanczos to mapping H to the tridiagonal matrix(step by step).
	void find_trdgnl_one_step(const VEC<VecReal>& initial_vector, VecReal& v0, VecReal& v1, Real& a, Real& b, const SparseMatReal& sep_h);


	// mapping the state by using the project operator.
	VecReal project_uplwer_parical_space(const VecReal &initial_vector, const Int crtann, const Int norg_set, const Int orbit_pos_in_div = 0) const;

	// return the ex_state_part_in_rotation
	void add_ex_state_part_in_rotation(const VecReal &initial_vector, VecReal& ex_state_part, const Int& set_n, const VecReal& rotation_coefficients, BasisIndex h_i, BasisIndex subnosp) const;

	bool if_in_this_orbital(const VecOnb &exd_cf, const Int crtann, const Int div_in_one_row_pos, const Int orbit_pos_in_div) const;

public:
	// Static-susceptibility path: old/new particle spaces are identical, so no
	// particle-creation/annihilation projection or ex_state is constructed.
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& nosp_i, const Operator& oper);
	// Block-GF path: prepare only the particle-sector context. Each call to
	// find_gf_orbital projects the supplied state once.
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper);
	// Scalar/HD paths still require an eagerly projected ex_state.
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const VecReal& vgs_i, const Int position = 0);
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper, const VecReal& vgs_i, const Int position = 0);
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper, const VecReal& vgs_i, const Int pos_in_set, const Int pos_in_div);
	CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const Operator& oper, const VecReal ex_state_i, const Int crtorann_i);

	// CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_scsp, const Tab& table, const MatReal& vdegs_i, const Int position = 0);
	// ? CrrltFun(const NocSpace& old_nosp_i, const Operator& main_op, const VecReal& vgs_i, const Int position = 0);

	// // (Deactivate) Caluate the density-density correlation function.
	// $$G_{A}(Z)=\left\langle\psi_{0}\left|A^{\dagger}(Z-H)^{-1}A\right|\psi_{0}\right\rangle,(Z=\omega+i\eta+E_{0})$$
	ImGreen find_density_density_correlation_function(const Real &ge0);

	void find_gf(const Real& ge0, Green &g0, bool is_greater = true);

	// Return chi_A(i*Omega_n) on the bosonic Matsubara mesh:
	//   chi_iom[0][n]     = chi_Sz(i*Omega_n)
	//   chi_iom[1][n]     = chi_Nimp/2(i*Omega_n)
	//   chi_iom[2+p][n]   = chi_Oij(i*Omega_n)
	// The n=0 column is identical to the original static susceptibility.
	//
	/*
	一个重要提醒
	这个函数默认你传入的 f_state 已经满足：
	⟨0∣f⟩=0
	也就是你已经做了：
	∣f⟩=(A−⟨A⟩)∣0⟩
	如果你直接传
	A∣0⟩
	而没有减掉 ⟨A⟩∣0⟩，那么 ∣f⟩ 里面可能含有基态分量。
	此时 H−E0 ,在基态方向有零本征值，(H−E0)^{−1}会发散，代码结果会不稳定。
	|f>没有归一化
	*/
	ChiIomResult find_chi_Sz_N_O_nondeg(const Real& E0, const VecReal& vgs);

	void find_gf_greater(const Real& ge0, Green &g0);

	void find_gf_lesser(const Real& ge0, Green &g0);

    void find_gf_orbital(const Real& ge0, ImGreen& g0, const VecReal& vgs,
        Int set_idx, const SparseMatReal& sep_h, bool is_greater, const Str& monitor_context);

	void find_hd_greater(const Real& ge0, Green &g0, Int position, Int ex_idx = -1);
	VecReal find_hd_exstate(Int position,  Int ex_idx = -1);
	
	// void find_gf_greater(const Real& ge0, Green &g0, Int kind);

	// void find_gf_lesser(const Real& ge0, Green &g0, Int kind);
};
