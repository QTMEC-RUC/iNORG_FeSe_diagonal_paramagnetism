#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

// Global basis rows, occupation-group offsets and configuration ranks.
using BasisIndex = std::int64_t;

inline BasisIndex basis_count_product(BasisIndex a, BasisIndex b)
{
    if (a < 0 || b < 0 || (b > 1 && a > std::numeric_limits<BasisIndex>::max() / b))
        throw std::overflow_error("Hilbert space dimension exceeds signed 64-bit range");
    return a * b;
}

inline BasisIndex basis_count_sum(BasisIndex a, BasisIndex b)
{
    if (a < 0 || b < 0 || a > std::numeric_limits<BasisIndex>::max() - b)
        throw std::overflow_error("Hilbert space dimension exceeds signed 64-bit range");
    return a + b;
}
