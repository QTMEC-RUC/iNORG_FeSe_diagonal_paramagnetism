#pragma once

#include <cstddef>

// Index only: the actual interaction coefficients remain in oper_value.
// Widen before multiplication; preserve the original i*N^3+j*N^2+k*N+l order.
inline std::size_t four_fermion_index(std::size_t n, std::size_t i,
    std::size_t j, std::size_t k, std::size_t l)
{
    return ((i * n + j) * n + k) * n + l;
}
