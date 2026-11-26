#pragma once
#include <cstdlib>
#include <cmath>
#include <string>
#include <limits>
#include <stdexcept>
namespace runtime_options {
inline unsigned long long nonnegative_integer(const char* name,unsigned long long fallback) {
    const char* raw=std::getenv(name);if(!raw)return fallback;
    std::string value(raw);std::size_t end=0;
    if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument(name);
    const auto result=std::stoull(value,&end);
    if(end!=value.size())throw std::invalid_argument(name);
    return result;
}
inline double nonnegative_real(const char* name,double fallback) {
    const char* raw=std::getenv(name);if(!raw)return fallback;
    std::size_t end=0;const std::string value(raw);const double result=std::stod(value,&end);
    if(end!=value.size() || !std::isfinite(result) || result<0.)throw std::invalid_argument(name);
    return result;
}
}
