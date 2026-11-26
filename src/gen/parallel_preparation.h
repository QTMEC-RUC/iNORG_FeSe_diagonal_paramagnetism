#pragma once
#include "parallel_stage2.h"
#include "vec.h"
#include <cstdint>
#include <limits>
namespace preparation_parallel {
namespace par = stage2_parallel;
// Checked blocked scan: only the small block-total array is serial.
template<class Count, class Write>
std::size_t offsets(std::size_t n, Count count, Write write,
                    std::size_t limit = std::numeric_limits<std::size_t>::max()) {
    const std::size_t parts = par::workers(n);
    std::vector<std::size_t> totals(parts,0), starts(parts,0);
    par::ranges(n,[&](std::size_t first,std::size_t last,std::size_t part) {
        std::size_t sum=0;
        for(auto i=first;i<last;++i) {
            const std::size_t value=count(i);
            if(value>limit-sum) throw std::overflow_error("Parallel prefix sum overflow");
            sum+=value;
        }
        totals[part]=sum;
    });
    std::size_t total=0;
    for(std::size_t part=0;part<parts;++part) {
        starts[part]=total;
        if(totals[part]>limit-total) throw std::overflow_error("Parallel prefix sum overflow");
        total+=totals[part];
    }
    par::ranges(n,[&](std::size_t first,std::size_t last,std::size_t part) {
        std::size_t offset=starts[part];
        for(auto i=first;i<last;++i) {
            const std::size_t value=count(i);
            write(i,offset);offset+=value;
        }
    });
    write(n,total);return total;
}
template<class Predicate> bool all(std::size_t n, Predicate predicate) {
    std::vector<unsigned char> valid(par::workers(n),1);
    par::ranges(n,[&](std::size_t first,std::size_t last,std::size_t part) {
        bool ok=true;
        for(auto i=first;i<last;++i) if(!predicate(i)) ok=false;
        valid[part]=ok;
    });
    return std::all_of(valid.begin(),valid.end(),[](unsigned char x){return x!=0;});
}
inline VecReal slice(const VecReal& source,std::size_t begin,std::size_t end) {
    if(begin>end || end>source.size()) throw std::out_of_range("Parallel vector slice");
    VecReal result(end-begin);
    par::ranges(result.size(),[&](std::size_t first,std::size_t last,std::size_t) {
        for(auto i=first;i<last;++i)result[i]=source[begin+i];
    });
    return result;
}
inline void normalize(VecReal& state) {
    const Real scale=INV(state.norm()); // Preserve MKL norm convention.
    par::ranges(state.size(),[&](std::size_t first,std::size_t last,std::size_t) {
        for(auto i=first;i<last;++i)state[i]*=scale;
    });
}
// Stateless SplitMix64 counter mixing. Global index/stream determine values,
// independent of MPI/thread layout. Only Lanczos starts use this generator.
inline double start_value(std::uint64_t index,std::uint64_t stream) {
    std::uint64_t z=index+UINT64_C(0x9e3779b97f4a7c15)*(stream+1);
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    z^=z>>31;
    return static_cast<double>(z>>11)*0x1.0p-53-0.5;
}
inline void fill_start(VecReal& out,std::uint64_t global_begin,std::uint64_t stream) {
    par::ranges(out.size(),[&](std::size_t first,std::size_t last,std::size_t) {
        for(auto i=first;i<last;++i)out[i]=start_value(global_begin+i,stream);
    });
}
}
