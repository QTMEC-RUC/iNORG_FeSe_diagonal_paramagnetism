#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "prmtr.h"
#include "nocspace.h"
#include "state.h"


// impurity model
#include "gen/tab.h"

// ! not finished.（may not need）
// class TAB { }; 



class Operator {
public:
	const MyMpi& mm;				// parameters
	const Prmtr& p;					// parameters
	const NocSpace& scsp;			// NocSpace
	const BasisIndex dim;
    // const VecReal& coefficient;      // coefficient for all the H's terms

private:
    mutable double h_build_seconds_ = 0.;
    tab_timing::Report tab_timing_; // Must be initialized before table.
public:
	Tab table;						// The movement by femi operator contain.
	Real groundstate_energy;		// The ground state energy on this shortcut restratin.

	
	MatReal ground_state;			// if degeneracy happened, we only get one of them.
	//MatReal ninj_mat;				// <n_i n_j> impurity correlator; indices run over impurity div-0 orbitals only.
	// std::map<Int, Real> oper_value;	// The Hamiltonian operator in the shortcut space.
	VecReal oper_value;				// The Hamiltonian operator in the shortcut space.
private:

	template<typename T>
	constexpr const T& myclamp(const T& v, const T& lo, const T& hi) {
		return (v < lo) ? lo : ((hi < v) ? hi : v);
	}

public:
	// Expand the Shortcut space under Number of particles(NumberSpa).
	// GS and GF callers can enable snapshots at the exact Tab construction boundary.
	Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i,
		bool monitor_table_memory = false, const char* monitor_scope = "GS",
		const Str& monitor_context = "");
// 	Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const Tab &tab, const VecReal& coefficient_i);
	Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i,const Tab &per_table);
	Operator(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& s_i, Str tab_name);
	
	//! abandon
	// CSR: row_ptr bounds local rows; cols are global; ops are signed operator IDs.
	// Tab find_h_idx();

	// CSR: row_ptr bounds local rows; cols are global; ops are signed operator IDs.
	Tab find_fullH_idx();

	// CSR: row_ptr bounds local rows; cols are global; ops are signed operator IDs.
	Tab find_fullH_offdiagonal_idx(tab_timing::Report* timing = nullptr);

	void clear() { table.reset(); }

	// By using the find_h_idx() to find the idx first then speed up the procedure.
	double h_build_seconds() const { return h_build_seconds_; }
    SparseMatReal find_hmlt(const Tab& h_idx) const;

	// By using the find_fullH_offdiagonal_idx() to find all the offdiagonal elements of table.
	SparseMatReal find_hmlt_V2(const Tab& h_idx) const;


	// Using Lanczos algorithm to find several lowest eigenpairs of a Hermitian matrix
	MatReal lowest_eigpairs(const BasisIndex n, bool if_need_fast = true, Int wish_nev = 1);

	//Compute the Single particle excited state(sn_prtcl_ex_state) 
	//here only support for the div[0] 
	//$$\left|f_{0}\right\rangle=A\left|\psi_{0}\right\rangle$$
	VecReal sn_prtcl_ex_state(const Int ex_orbital_position, const VecReal ground_state, const Int crtann)const;

	void save_the_Tab(const Tab& tab, Str name) const;

	void write_the_multiTab(Str name) const;
	Tab read_the_Tab(Str name) const;

	// 1 Ln: N; 2 Ln: Sz; 3 Ln: P;
	MatReal local_multiplets_state(const MatReal& g_state) const;
};

