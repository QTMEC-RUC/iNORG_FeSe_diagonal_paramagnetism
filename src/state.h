#pragma once

#include <string_view>
#include <cstring>

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "prmtr.h"
#include "nocspace.h"	// cause now This class was only for the NORG imp solver, at the Shortcut space.

/// <summary>
/// In this class will find all the possible state(StateStatistics) for one gived Combined division (ComDiv).
/// impurity model
/// At the Shortcut space(NocSpace) we set first of two divisons as impurity and the active orbital(bath site).
/// For now it's support for find two fermion act on one state.
/// </summary>
class StateStatistics 
{
	typedef VEC<std::tuple<Int, Int, MatInt>> hopdata;
	typedef VEC<std::pair<std::array<int, 4>, MatInt>> furfrm;
private:
	static const BnmlFast bf;
public:
	const NocSpace& space;								// parameters
	const BasisIndex div_idx;										// The divison idx at all Combined division space.
	const BasisIndex idx;											// The idx in one divison.
	const VecOcc& occ_n; // Read-only occupations owned by space.div; do not mutate that container while in use.
	// const MatInt sit_mat;								// spin - orbits number in each division.
	// const VecInt sit_n;									// spin - orbits number in each division.
	const ComDivs cfg;										// The configuration of the state.
	// const hopdata hopup;		// For the upper spin-orbits, a set of occupy number in each division
	// const hopdata hopdw;		// For the down spin-orbits, a set of occupy number in each division
	//ComDivs cfgdw;										// The configuration of the state.

	// const VecOnb cfgup;										// Here only for better reading reason.
	// const VecOnb cfgdw;										// Here only for better reading reason.

	// for the electronic statistics of the state.
	Vec< VEC<Int> > filled_spinless;

	// VEC<VecInt> off_diagonal_term;	// a set of valuable term with the annihilation, creation orbit's positon, and Colum idx(H_i).
private:
	ComDivs cfgdetail();
	// To statistic the occupation situation and filling the filled_spinless matrix.
	void statistics();



	// fermion anticommutativity, for two orbitals changed.
	//Int anticommu(const Int& pi,const Int& pj);

	// find the div's idx according to the "newdiv".
	BasisIndex find_newdiv_idx(const MatInt& newdiv);

	template<typename T>
	constexpr const T& myclamp(const T& v, const T& lo, const T& hi) {
		return (v < lo) ? lo : ((hi < v) ? hi : v);
	}

public:
	//state(Int idxD, VecInt bases_i) :idx(idxD), bases(bases_i) {};
	enum class OccupationStatistics { collect, skip };
	StateStatistics(BasisIndex h_i, BasisIndex comdiv_idx, const NocSpace& s_i,
        OccupationStatistics statistics = OccupationStatistics::collect);

	// The out put<0>[i]: the annihilation divsion position; <1>[i]: the creation dision positon; <2>[i]: The new divs.
	hopdata divocchop_ingroup(BasisIndex ComDiv, Idx sets_n);
	
	// [0]~[3] i-j-k-l orbit's position; (*this).second: The new divs.
	furfrm divs_change_fourFermi(BasisIndex ComDiv, Idx sets_i, Idx sets_j);

	// The out put<0>[i]: the annihilation divsion position; <1>[i]: the creation dision positon; <2>[i]: The new divs.
	VEC<MatInt> interation_soc_hop(BasisIndex ComDiv);

	// Output all off-diagonal H matrix term in one row.(For all sites hoping?)
	// [i][0]:annihilation orbit's position; [i][1]:creation orbit's positon; [i][2]:Colum idx(i); [i][3]:sign(fermion anticommutativity)
	// PS.:The matrix Max size: (N*N-(N-N_{imp})-N_{imp}*N_{imp},2), and it's spinless.
	VEC<VecInt> find_off_diagonal_term(const hopdata &hopup, const hopdata &hopdw);

	// Output all off-diagonal H0 matrix term with one spinless orbit.
	// [i][0]:annihilation orbit's position; [i][1]:creation orbit's positon; [i][2]:Colum idx(i); [i][3]:sign(fermion anticommutativity)
	VEC<std::array<BasisIndex, 4>> find_each_spiless_group_off_diagonal_term(const hopdata &hopspi, const Int sets_n);

	// Output all off-diagonal H0 matrix and all H_{ijkl} term with one spinless orbit.
	//  [0]~[3] i-j-k-l orbit's position; [4]:Colum idx(i);[5]:sign(fermion anticommutativity)
	VEC<std::array<BasisIndex, 6>> find_off_diagonal_term_fourFermi(const furfrm &furfrm, const Int sets_i, const Int sets_j);
	
	// Output all off-diagonal SOC spinless orb term.
	VEC<std::array<BasisIndex, 4>> off_diagonal_soc_term(const VEC<MatInt> &hop_soc);

	Mat<Char> show_cfg();

	VecBool cfg2vecbool();
	
	VecBool cfgex2vecbool(Int ex_pos) ;
};

// Slots borrow the immutable space's compact occupations; one index per Tab build,
// shared read-only by all worker caches. The space must outlive this index.
class TwoFermionGroupIndex {
    const NocSpace& space_;
    // Open-addressed slots hold group numbers, borrowing immutable occupations.
    std::vector<std::vector<Idx>> slots_;
    static constexpr Idx empty_ = std::numeric_limits<Idx>::max();
public:
    explicit TwoFermionGroupIndex(const NocSpace& space);
    const NocSpace& space() const { return space_; }
    BasisIndex at(std::string_view key) const;
};

// Per-builder cache for the if_norg_imp == false, two-fermion Tab path.
// Only the current occupation group is retained; no global/static cache.
class TwoFermionGroupCache {
    struct Rule {
        Int ann_div, crt_div;
        BasisIndex target_begin;
        bool single_to_single;
        std::vector<BasisIndex> target_weights;
    };

    const NocSpace& space_;
    const TwoFermionGroupIndex& index_;
    BasisIndex group_ = -1;
    std::size_t active_rules_ = 0, entries_per_row_ = 0;
    std::vector<Int> orbital_begin_;
    std::vector<Rule> rules_;
    std::vector<BasisIndex> local_rank_; // Reused source-configuration scratch space.
    bool needs_rank_ = false;
    bool single_orbital_ = true;
    MatInt target_;
    VecInt counts_;
    std::string key_;
    std::vector<signed char> allowed_;
    std::vector<Int> occupation_prefix_;
    tab_timing::Worker* timing_ = nullptr;
    bool sample_group_ = false;

    void refresh(BasisIndex group);

public:
    explicit TwoFermionGroupCache(const TwoFermionGroupIndex& index,
        tab_timing::Worker* timing = nullptr);
    // Reserve for a contiguous run of rows from one occupation group.
    void prepare(BasisIndex group, std::size_t rows, TabBuilder& table);
    void append(const StateStatistics& state, BasisIndex local_row, TabBuilder& table);
    bool single_orbital() const { return single_orbital_; }
    // Valid only after prepare(), when every division has at most one orbital.
    void append_single_orbital(BasisIndex local_row, TabBuilder& table) const;
};

// Occupation-only decoding for diagonal observables. One instance per worker;
// the immutable NocSpace must outlive it. No per-basis-row heap allocation.
class OccupationDecoder {
    const NocSpace& space_;
    BasisIndex group_ = -1;
    bool unique_ = false;
    std::vector<Int> capacity_, begin_, set_;
    std::vector<BasisIndex> radix_;
    std::vector<std::vector<Int>> filled_;
    void refresh(BasisIndex group);
public:
    explicit OccupationDecoder(const NocSpace& space);
    // Same ascending global orbital order as StateStatistics::filled_spinless.
    const std::vector<std::vector<Int>>& decode(BasisIndex row, BasisIndex group);
};
