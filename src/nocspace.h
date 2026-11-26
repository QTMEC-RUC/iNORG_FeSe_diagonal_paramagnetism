#pragma once

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "gen/basis_index.h"
#include "specs.h"
#include "prmtr.h"
#include "impurity.h"
#include "gen/occupation_vector.h"
#include "gen/nocspace_timing.h"
#include "gen/nocspace_profile.h"
#include "gen/occupation_index.h"
// impurity model
// At the Shortcut space(NocSpace) we set first divison as impurity, and the active orbital(bath site) on the middle.  

#include "gen/tab.h"
class NocSpace {

private:
    // First member: start after synchronization, before member allocations.
    nocspace_profile::Report construction_profile_;
    nocspace_timing::BuildTiming construction_detail_{false};
    std::vector<std::unique_ptr<nocspace_timing::BuildTiming>> construction_workers_;
    bool basis_cache_hit_=false;
    void finish_construction(const char* source);
	const BnmlFast bf{};
public:
    bool basis_cache_hit()const{return basis_cache_hit_;}
    double construction_seconds() const { return construction_profile_.total(); }
	const MyMpi &mm;				// parameters
	const Prmtr& p;					// parameters
	Idx ndivs;						// The amount of divisons's number. 
	MatInt control_divs;			// to set the number of division and the shortcut restratin.
	VecInt orbital_divcnt;			// the SUM in row for each colum.
	MatInt sit_mat;					// spinless - orbits number in each division.
	Idx nspa;						// The amount of partical's number.
	VecInt nppso;					// nppso mean: number of partical per spin orbital.
	
	BasisIndex dim = 0;						// the dimension of the Shortcut space.
	VEC<VecOcc> div;				// row-major occupations; shape is sit_mat.nrows() x ndivs.
	OccupationIndex<BasisIndex> divs_to_idx;	// a set of combined number.
	VEC<BasisIndex> idx_div;				// a set of idx(The idx of the begining with 0) for each subspace.
	VecReal mu;						// -\mu_{k}^{\prime}b_{k,\sigma}^{+}b_{k,\sigma}
	VEC<VecReal> t_ose;				// H_0 onset energy
	VEC<VecReal> t_hyb;				// hopping parameter from bath to imp.
	MatReal hopint;					// transformed hopping integral
	VecReal coefficient;			// coefficient for all the H's terms, consturct by t_ose and t_hyb.
	
	
	Real u_hbd;						// The Hubbard term U.
	Real j_ob;						// The Hunter couping term J.
	
private:
	// Cartesian-product sizes and encoded labels must remain in Idx throughout.
	template<typename T>
	static Idx free_div_combination_count(const VEC<VEC<T>>& v) {
		for (const auto& choices : v) if (choices.empty()) return 0;
		Idx count = 1;
		for (const auto& choices : v) {
			if (count > std::numeric_limits<Idx>::max() / choices.size())
				ERR(STR("free div combination count exceeds Idx range."));
			count *= choices.size();
		}
		return count;
	}

	void set_control();
	void log_storage_after_construction(const char* source) const;
	// Find all the combined number subspaces.
	void find_combined_number_subspaces(const Int mode = 0);
	void find_combined_number_subspaces_no_active_orbital();
	// // Find all the combined number subspaces by nppso.
	// void find_combined_number_subspaces(const VecInt& nppso);

	// Find all the combined number subspaces, with speed up.
	void find_all_noc_subspaces();
	void find_all_noc_subspaces_multi();
	void find_all_noc_subspaces_by_row();
	void find_thought_noc_subspaces();
	void find_thought_noc_subspaces_nestedloops();


	void print(std::ostream& os, const Str& var, const Str& val, const Str& comment) const {
		using namespace std;
		Str true_var = var == "\"\"" ? "" : var;
		os << rght_justify(true_var, 16) << (true_var == "" ? "   " : " = ") << left_justify(val, w_Real)
			<< "    # " + comment << endl;
	}
	bool if_div_in_restraint(const VecInt& restraint, const Int position, const Int max, const Int now) const;
	
	bool if_col_divs_in_restraint(const Int& restraint, const VEC<Int>& divcol_i, Int col_idx) const;
	bool if_row_divs_in_restraint(const Int& restraint, const VEC<Int>& divrow_i, VecInt count_sit_mat) const;

	bool ifin_NocSpace_judge_by_nppso(const MatInt& spilss_div, const VecInt& nppso) const;

	VecInt multi_judger(const VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const;
	VecInt multi_judger_by_row(const VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const;
	Vec<MatInt> multi_judger_with_return(VEC<VEC<int> >& s, const VEC<VEC<int> >& a) const;

	bool check_each_column(const Int& col_pos, const VecInt& div_colsum) const {
		if (p.if_norg_imp) {
			if (col_pos < div_colsum.size() / 2 && div_colsum[col_pos] >= orbital_divcnt[col_pos] + control_divs[0][col_pos] && div_colsum[col_pos] <= orbital_divcnt[col_pos]) return true;
			if (col_pos >= div_colsum.size() / 2 && div_colsum[col_pos] >= 0 && div_colsum[col_pos] <= control_divs[0][col_pos]) return true;
		}
		else {
			if (col_pos == div_colsum.size() / 2) return true;
			if (col_pos < div_colsum.size() / 2 && div_colsum[col_pos] >= orbital_divcnt[col_pos] + control_divs[0][col_pos] && div_colsum[col_pos] <= orbital_divcnt[col_pos]) return true;
			if (col_pos > div_colsum.size() / 2 && div_colsum[col_pos] >= 0 && div_colsum[col_pos] <= control_divs[0][col_pos]) return true;
		}
		return false;
	}

	bool check_correlated_column(const Int& col_pos, const VecInt& div_colsum) const;

	// reference the 10.1103/PhysRevB.96.085139 FIG.1 algorithm
	bool check_if_PHSs(const VecInt& div_colsum) const;	
	bool check_if_PHSs_v2(const VecInt& div_colsum) const;

	VecInt read_from_col_lable(const VEC<Int> x, const VEC<VEC<Int> > a) const;

	void find_all_possible_state(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const;
	void find_all_possible_state_by_col(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const;
	void find_all_possible_state_by_row(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const;
	void find_all_possible_state_by_nooc(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const;
	void find_all_possible_state_suit_for_PHSs(VEC<VEC<Int> >& a, VEC<VEC<Int> >& s) const;

	
	void nestedLoops(int depth, int n, MatInt control_divs, std::vector<int>& current);
	template<typename T>
	VEC<VEC<T>> cart_product(const VEC<VEC<T>>& v) const {
		VEC<VEC<T>> result = {{}};
		for (const auto& u : v) {
			
			VEC<VEC<T>> temp;
			for (const auto& x : result) {
				for (const auto y : u) {
					auto new_combination = x; // Create a new combination
					new_combination.push_back(y); // Add current element
					temp.push_back(std::move(new_combination)); // Move to temp
				}
			}
			
			// Move temp to result
			result = std::move(temp);
		}
		return result;
	};
	VEC<VEC<Int> > cart_product_monitor_col(const VEC<VEC<int> >& v, const VEC<VEC<Int> >& a)const;
	VEC<VEC<Int> > cart_product_monitor_row(const VEC<VEC<int> >& v, const VEC<VEC<int> >& a) const;
	VEC<VEC<Int> > cart_product_monitor_PHS(const VEC<VEC<int> >& v, const VEC<VEC<int> >& a) const;

	
	
public:

	// Expand the Shortcut space under Number of particles(NumberSpa).
	NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i): mm(mm_i), p(prmtr_i) {};
	// NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const Int& NumberSpa);
	NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const Int& NumberSpa);
	NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i);
	NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i, Str tab_name);
	NocSpace(const MyMpi& mm_i, const Prmtr& prmtr_i, const VecInt& nppso_i, const Tab& tab);
	
	// bool ifin_NocSpace(VecInt& ud) const;
	bool ifin_NocSpace(MatInt& ud) const;
	// bool ifin_NocSpace_for_green(MatInt& spilss_div, const VecInt& nppso, const Int& crtorann) const;
	bool ifin_NocSpace(MatInt& spilss_div, const VecInt& nppso) const;
    // Preconditions: source is valid; one same-set annihilation/creation has
    // passed cell capacity checks; counts contains the updated column sums.
    bool two_fermion_columns_allowed(const VecInt& counts, Int ann, Int crt) const;
	bool suit_NOOC(MatInt& spilss_div, const VecInt& nppso) const;
	
	bool ifin_NocSpace(const VecBool new_cfig, const VecInt& nppso) const;

	bool ifin_NocSpace_more_strict(MatInt& spilss_div, const VecInt& nppso) const;

	BasisIndex wherein_NocSpace(BasisIndex h_i) const;

	// One cursor per traversal/thread. The bound space and its group boundaries
	// must remain alive and unchanged for the lifetime of the cursor.
	class GroupCursor {
	public:
		explicit GroupCursor(const NocSpace& space) : space_(space) {}
		BasisIndex locate(BasisIndex h_i);
	private:
		const NocSpace& space_;
		BasisIndex group_ = 0;
		BasisIndex previous_h_ = 0;
		bool initialized_ = false;
	};

	void print(std::ostream& os = std::cout) const;

	Str nppso_str() const	{
		Str temp; for_Int(i, 0, p.npartical.size()) { if (i == 0) temp += STR(p.npartical[i]); else /*if (i % 2 == 0)*/ temp += "-" + STR(p.npartical[i]); }	return temp;
	}

	// Preserve the value type: Int occupations or Idx encoded column labels.
	template<typename T>
	Vec<T> free_div_base_decode(Idx idx, const VEC<VEC<T>>& v) const {
		if (idx >= free_div_combination_count(v))
			ERR(STR("free div base decode error."));
		Vec<T> rep(v.size(), T(0));
		for_Idx(i, 0, v.size()) {
			rep[i] = v[i][idx % v[i].size()];
			idx /= v[i].size();
		}
		return rep;
	}
};



/* // NOT in used code

/// #1 cart_product
	template<typename T>
	VEC<VEC<T>> cart_product(const VEC<VEC<T>> &v) const
	{
		VEC<VEC<T>> s = {{}};
		for (const auto &u : v)
		{
			VEC<VEC<T>> r;
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
	};
///



*/
