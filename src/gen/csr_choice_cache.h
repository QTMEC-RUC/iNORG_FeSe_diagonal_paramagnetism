#pragma once
#include "runtime_options.h"
#include "parallel_preparation.h"
#include <list>
#include <cstdint>
namespace csr_choice_cache {
using Key=std::vector<unsigned long long>;
struct Entry {Key key;int mode;unsigned long long reuse=0;};
inline std::list<Entry>& entries(){static std::list<Entry> cache;return cache;}
inline unsigned long long mix(unsigned long long hash,unsigned long long value) {
    return (hash^value)*1099511628211ULL;
}
// This fingerprint caches only a performance choice, NEVER matrix values or communication maps.
// A collision can select a slower algorithm, but cannot change the computed operator.
template<class Mpi,class V> Key key(const Mpi& mm,int dim,const V& ptr,const V& col,
    int chunks,int trials) {
    std::vector<unsigned long long> partial(stage2_parallel::workers(ptr.size()-1),1469598103934665603ULL);
    stage2_parallel::ranges(ptr.size()-1,[&](std::size_t first,std::size_t last,std::size_t worker){
        auto& hash=partial[worker];
        for(auto r=first;r<last;++r){
            hash=mix(hash,ptr[r+1]-ptr[r]);
            for(int j=ptr[r];j<ptr[r+1];++j)hash=mix(hash,col[j]);
        }
    });
    unsigned long long hash=1469598103934665603ULL;
    for(auto value:partial)hash=mix(hash,value);
    hash=mix(hash,dim);hash=mix(hash,thread_budget());hash=mix(hash,chunks);
    hash=mix(hash,trials);hash=mix(hash,mm.ms());hash=mix(hash,mkl_get_max_threads());
    char host[MPI_MAX_PROCESSOR_NAME];int length=0;
    MPI_Get_processor_name(host,&length);
    for(int i=0;i<length;++i)hash=mix(hash,static_cast<unsigned char>(host[i]));
    for(const char* name:{"OMP_PLACES","OMP_PROC_BIND","OMP_THREAD_LIMIT","MKL_NUM_THREADS"}) {
        if(const char* value=std::getenv(name))while(*value)hash=mix(hash,static_cast<unsigned char>(*value++));
        hash=mix(hash,0);
    }
    Key result(mm.np());
    if(MPI_Allgather(&hash,1,MPI_UNSIGNED_LONG_LONG,result.data(),1,MPI_UNSIGNED_LONG_LONG,mm.cm())!=MPI_SUCCESS)
        throw std::runtime_error("CSR choice fingerprint exchange failed");
    return result;
}
inline int lookup(const Key& key,unsigned long long limit) {
    auto& cache=entries();if(!limit)return 0;
    for(auto it=cache.begin();it!=cache.end();++it)if(it->key==key) {
        if(it->reuse>=limit){cache.erase(it);return 0;}
        ++it->reuse;const int mode=it->mode;cache.splice(cache.begin(),cache,it);return mode;
    }
    return 0;
}
inline void remember(const Key& key,int mode,unsigned long long capacity) {
    auto& cache=entries();
    for(auto it=cache.begin();it!=cache.end();)if(it->key==key)it=cache.erase(it);else ++it;
    if(capacity)cache.push_front({key,mode,0});
    while(cache.size()>capacity)cache.pop_back();
}
}
