/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "comdivs.h"
const BnmlFast ComDivs::bf = BnmlFast();
using namespace std;

ComDivs::ComDivs(BasisIndex i_i, VecInt ne_i, VecInt ns_i, bool quick) : idx(i_i), ne(ne_i), ns(ns_i), cf(cf_ComDivs(i_i))
{
	if (!quick)
	{
		auto orbitals = find_div_orb();
		div_orb_e = std::move(std::get<0>(orbitals));
		div_orb_c = std::move(std::get<1>(orbitals));
	}
}

ComDivs::ComDivs(BasisIndex i_i, MatInt ne_i, MatInt ns_i, bool quick) : idx(i_i), ne(ne_i.vec()), ns(ns_i.vec()), cf(cf_ComDivs(i_i))
{
	if (!quick)
	{
		auto orbitals = find_div_orb();
		div_orb_e = std::move(std::get<0>(orbitals));
		div_orb_c = std::move(std::get<1>(orbitals));
	}
}

ComDivs::ComDivs(BasisIndex i_i, VecInt ne_i, VecInt ns_i) : idx(i_i), ne(ne_i), ns(ns_i), cf(cf_ComDivs(i_i))
{
	auto orbitals = find_div_orb();
	div_orb_e = std::move(std::get<0>(orbitals));
	div_orb_c = std::move(std::get<1>(orbitals));
}

ComDivs::ComDivs(BasisIndex i_i, MatInt ne_i, MatInt ns_i) : idx(i_i), ne(ne_i.vec()), ns(ns_i.vec()), cf(cf_ComDivs(i_i))
{
	auto orbitals = find_div_orb();
	div_orb_e = std::move(std::get<0>(orbitals));
	div_orb_c = std::move(std::get<1>(orbitals));
}

ComDivs::ComDivs(VecOnb &newcf, VecInt ne_i, VecInt ns_i) : ne(ne_i), ns(ns_i), cf(newcf)
{
	if (ne_i.size() != ns_i.size())
		ERR("some err with input" + NAV2(ne_i, ns_i));
	idx = find_cf_idx(cf);
}

ComDivs::ComDivs(VecOnb &newcf, MatInt ne_i, MatInt ns_i) : ne(ne_i.vec()), ns(ns_i.vec()), cf(newcf)
{
	if (ne_i.size() != ns_i.size())
		ERR("some err with input" + NAV2(ne_i, ns_i));
	idx = find_cf_idx(cf);
}

ComDivs::ComDivs(BasisIndex i_i, const VecOcc& ne_i, const MatInt& ns_i, bool quick)
	: ComDivs(i_i, occupation_counts(ne_i, ns_i.size()), ns_i.vec(), quick)
{}

ComDivs::ComDivs(BasisIndex i_i, const VecOcc& ne_i, const MatInt& ns_i)
	: ComDivs(i_i, occupation_counts(ne_i, ns_i.size()), ns_i.vec())
{}

ComDivs::ComDivs(VecOnb& newcf, const VecOcc& ne_i, const MatInt& ns_i)
	: ComDivs(newcf, occupation_counts(ne_i, ns_i.size()), ns_i.vec())
{}

VecOnb ComDivs::cf_ComDivs(BasisIndex rank)
{
    if (rank < 0) throw std::out_of_range("Negative configuration rank");
    VecOnb cells(ns.size());
    for (Idx i = 0; i < ns.size(); ++i) {
        const BasisIndex radix = bf(ns[i], ne[i]);
        cells[i] = Onb(ne[i], ns[i], rank % radix);
        rank /= radix;
    }
    if (rank != 0) throw std::out_of_range("Configuration rank exceeds group size");
    return cells;
}

ComDivs::OrbTuple ComDivs::find_div_orb()
{
	vector<VecInt> div_orb_es, div_orb_cs;
	div_orb_es.reserve(ns.size());
	div_orb_cs.reserve(ns.size());
	for_Int(i, 0, ns.size()) {
		Int k_e(0), k_c(0);
		div_orb_es.emplace_back(ne[i], -1);
		div_orb_cs.emplace_back(ns[i] - ne[i], -1);
		auto& div_orb_e = div_orb_es.back();
		auto& div_orb_c = div_orb_cs.back();
		for_Int(j, 0, ns[i]) {
			if (cf[i].isocc(j)) div_orb_e[k_e++] = j;
			else if (cf[i].isuno(j)) div_orb_c[k_c++] = j;
			else ERR("Wrong  with count orbit idx: ");
		}
#ifdef _CHECK_BOUNDS_
		for_Int(i, 0, div_orb_e.size()) if (div_orb_e[i] < 0) ERR("Wrong with count orbit idx: " + NAV4(div_orb_e, i, ne, ns));
		for_Int(i, 0, div_orb_c.size()) if (div_orb_c[i] < 0) ERR("Wrong with count orbit idx: " + NAV4(div_orb_c, i, ne, ns));
#endif
	}
#ifdef _ASSERTION_
	Int x(0);
	for_Int(i, 0, ns.size()) {
		for_Int(j, 0, div_orb_es[i].size()) ++x;
		for_Int(j, 0, div_orb_cs[i].size()) ++x;
	}
	if (x != SUM(ns)) ERR("Wrong with count orbit idx: " + NAV2(SUM(ns), x));
#endif
	if (div_orb_es.size() != ns.size()) ERR("Wrong with count div's orbit idx list finding: " + NAV2(div_orb_es.size(), ns.size()));
	if (div_orb_cs.size() != ns.size()) ERR("Wrong with count div's orbit idx list finding: " + NAV2(div_orb_cs.size(), ns.size())); 
	//DBG("find_div_orb() FINISHED  !!!");
	return std::make_tuple(std::move(div_orb_es), std::move(div_orb_cs));
}

BasisIndex ComDivs::find_cf_idx(const VecOnb& cells)
{
    BasisIndex rank = 0;
    for (Idx end = cells.size(); end > 0; --end) {
        const Idx i = end - 1;
        const BasisIndex radix = bf(ns[i], ne[i]);
        const BasisIndex digit = cells[i].idx();
        if (digit < 0 || digit >= radix)
            throw std::out_of_range("Configuration rank exceeds division size");
        rank = basis_count_sum(basis_count_product(rank, radix), digit);
    }
    return rank;
}

// Only suit for fermion sign changed by c_i^+ c_j or c_j^+ c_i
Int ComDivs::count_electrons(const Int& lw, const Int& up) const
{
	Int counter(0), nele(0);

	for_Int(i, 0, ns.size()) {
		for_Int(j, 0, ns[i]) {
			if (counter >= up)	return nele;
			if (counter >= lw && cf[i].isocc(j)) nele++;
			counter++;
		}
	}
#ifdef _ASSERTION_
	if (counter > up - lw + 1) ERR("Wrong with Idxing orbit" + NAV2(lw, up));
#endif
    return nele;
}
