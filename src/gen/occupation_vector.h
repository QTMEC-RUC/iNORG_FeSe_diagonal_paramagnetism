#pragma once

#include "mat.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

// One owning, row-major byte vector per occupation matrix. Shape belongs to
// NocSpace (sit_mat/ndivs), not to each of its potentially millions of entries.
using VecOcc = Vec<std::uint8_t>;

inline VecOcc compact_occupations(const MatInt& source)
{
    for (Idx k = 0; k < source.size(); ++k) {
        if (source.p()[k] < 0 || source.p()[k] > 255)
            throw std::out_of_range("occupation outside uint8_t range at index "
                + std::to_string(k) + ": " + std::to_string(source.p()[k]));
    }
    VecOcc result(source.size());
    for (Idx k = 0; k < source.size(); ++k)
        result[k] = static_cast<std::uint8_t>(source.p()[k]);
    return result;
}

// Only for editable, signed work matrices: annihilating an empty division
// must produce -1 for the existing constraint checks, not wrap to 255.
inline MatInt expand_occupations(const VecOcc& source, Idx rows, Idx cols)
{
    if ((cols && rows > std::numeric_limits<Idx>::max() / cols)
        || rows * cols != source.size())
        throw std::invalid_argument("occupation vector/shape mismatch");
    MatInt result(rows, cols);
    for (Idx k = 0; k < source.size(); ++k) result.p()[k] = source[k];
    return result;
}

// ComDivs keeps arithmetic wide. Read the bytes directly without making a
// MatInt or allocating any row views.
inline VecInt occupation_counts(const VecOcc& source, Idx expected_size)
{
    if (source.size() != expected_size)
        throw std::invalid_argument("occupation/site vector length mismatch");
    VecInt result(source.size());
    for (Idx k = 0; k < source.size(); ++k) result[k] = source[k];
    return result;
}
