#pragma once
#include "basis_index.h"
#include "runtime_options.h"
#include "parallel_preparation.h"
#include <list>
#include <sstream>
// Main-thread, process-local LRU. Entries own packed data, never references to Prmtr/NocSpace.
// Restore makes independent mutable arrays; no Hamiltonian coefficients are cached.
namespace nocspace_cache {
struct Entry {
    std::string key;
    std::vector<unsigned char> occupations;
    std::vector<BasisIndex> offsets;
    BasisIndex dimension=0;
    Idx width=0;
    std::size_t bytes()const{return occupations.capacity()+offsets.capacity()*sizeof(BasisIndex)+key.capacity();}
};
inline std::list<Entry>& entries(){static std::list<Entry> value;return value;}
struct Seen {std::string key;unsigned long long count;};
inline std::list<Seen>& history(){static std::list<Seen> value;return value;}
inline bool admit(const std::string& key) {
    const auto threshold=runtime_options::nonnegative_integer("NORG_BASIS_CACHE_ADMIT",2);
    const auto capacity=runtime_options::nonnegative_integer("NORG_BASIS_CACHE_KEYS",64);
    auto& seen=history();
    if(!capacity || !threshold){seen.clear();return false;}
    unsigned long long count=1;
    for(auto it=seen.begin();it!=seen.end();++it)if(it->key==key) {
        count=it->count<threshold?it->count+1:threshold;seen.erase(it);break;
    }
    seen.push_front({key,count});
    while(seen.size()>capacity)seen.pop_back();
    return count>=threshold;
}
inline std::size_t budget() {
    const auto mb=runtime_options::nonnegative_integer("NORG_BASIS_CACHE_MB",256);
    if(mb>std::numeric_limits<std::size_t>::max()/(1024*1024))
        throw std::invalid_argument("NORG_BASIS_CACHE_MB is too large");
    return static_cast<std::size_t>(mb)*1024*1024;
}
inline void trim(std::size_t limit) {
    auto& cache=entries();std::size_t used=0;if(!limit)history().clear();
    for(const auto& e:cache)used+=e.bytes();
    while(!cache.empty() && used>limit){used-=cache.back().bytes();cache.pop_back();}
}
template<class Space> std::string key(const Space& s) {
    std::ostringstream out;
    out<<"nppso-v1 "<<s.ndivs<<' '<<s.nspa<<' '<<s.p.norg_sets<<' '
       <<s.p.norbit<<' '<<s.p.if_norg_imp<<' '<<s.p.nooc_mode.size()<<':'<<s.p.nooc_mode<<' ';
    out<<s.control_divs.nrows()<<' '<<s.control_divs.ncols()<<' ';
    for(Idx i=0;i<s.control_divs.size();++i)out<<s.control_divs.p()[i]<<',';
    out<<' '<<s.nppso.size()<<' ';
    for(Idx i=0;i<s.nppso.size();++i)out<<s.nppso[i]<<',';
    return out.str();
}
template<class Space> bool restore(Space& s) {
    std::string wanted;Entry* found=nullptr;
    stage2_parallel::collective(s.mm,[&]{
        trim(budget());wanted=key(s);
        auto& cache=entries();
        for(auto it=cache.begin();it!=cache.end();++it)if(it->key==wanted) {
            cache.splice(cache.begin(),cache,it);found=&cache.front();break;
        }
    });
    // All ranks take the same construction path even if their cache histories differ.
    if(s.mm.Allreduce(found?1:0)!=s.mm.np())return false;
    stage2_parallel::collective(s.mm,[&]{
        const auto groups=found->offsets.size()-1;
        s.div.resize(groups);s.idx_div=found->offsets;s.dim=found->dimension;
        std::vector<std::vector<typename OccupationIndex<BasisIndex>::Pending>> chunks(stage2_parallel::workers(groups));
        stage2_parallel::ranges(groups,[&](std::size_t first,std::size_t last,std::size_t part){
            auto& chunk=chunks[part];chunk.reserve(last-first);
            for(auto g=first;g<last;++g){
                s.div[g].reset(found->width);
                const auto* begin=found->occupations.data()+g*found->width;
                std::copy(begin,begin+found->width,s.div[g].p());
                chunk.emplace_back(s.div[g].string(),s.idx_div[g]);
            }
        });
        s.divs_to_idx.append_chunks(chunks,[]{});
    });
    return true;
}
template<class Space> void remember(const Space& s) {
    stage2_parallel::collective(s.mm,[&]{
        const auto limit=budget();trim(limit);if(!limit)return;
        const std::size_t width=s.sit_mat.size(),groups=s.div.size();
        if(width && groups>limit/width)return;
        const std::size_t size=width*groups;
        if(s.idx_div.size()>limit/sizeof(BasisIndex) || size>limit-s.idx_div.size()*sizeof(BasisIndex))return;
        Entry entry;entry.key=key(s);
        if(!admit(entry.key))return;
        entry.width=width;entry.dimension=s.dim;
        entry.occupations.resize(size);entry.offsets=s.idx_div;
        stage2_parallel::ranges(groups,[&](std::size_t first,std::size_t last,std::size_t){
            for(auto g=first;g<last;++g)
                std::copy(s.div[g].p(),s.div[g].p()+width,entry.occupations.begin()+g*width);
        });
        auto& cache=entries();
        for(auto it=cache.begin();it!=cache.end();)if(it->key==entry.key)it=cache.erase(it);else ++it;
        cache.push_front(std::move(entry));trim(limit);
    });
}
}
