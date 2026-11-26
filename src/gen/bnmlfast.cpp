/*
code developed by
    Rong-Qiang He (rqhe@ruc.edu.cn, RUC, China)
date 2012 - 2017
*/

#include "bnmlfast.h"

#include <array>
#include <numeric>

namespace {
// C(66,33) is the largest central binomial coefficient representable in int64.
constexpr int table_rows = 67;
using Binomials = std::array<std::array<BasisIndex, table_rows>, table_rows>;
constexpr Binomials make_binomials()
{
    Binomials values{};
    for (int n = 0; n < table_rows; ++n) {
        values[n][0] = values[n][n] = 1;
        for (int m = 1; m < n; ++m)
            values[n][m] = values[n - 1][m - 1] + values[n - 1][m];
    }
    return values;
}
constexpr auto binomials = make_binomials();
}

BasisIndex BnmlFast::operator()(Int n, Int m) const
{
    if (m < 0 || m > n) throw std::invalid_argument("Invalid binomial coefficient");
    if (n < table_rows) return binomials[n][m];

    // Cancel the denominator before multiplying, even if the intermediate
    // numerator would overflow. Coefficients increase monotonically up to n/2.
    m = std::min(m, n - m);
    BasisIndex result = 1;
    for (Int k = 1; k <= m; ++k) {
        const BasisIndex numerator = n - m + k;
        const BasisIndex common = std::gcd(numerator, BasisIndex(k));
        result = basis_count_product(result / (k / common), numerator / common);
    }
    return result;
}
