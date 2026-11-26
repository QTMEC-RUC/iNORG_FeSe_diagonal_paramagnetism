#pragma once
#include "csr_choice_cache.h"
#include <array>
#include <cmath>
#include <limits>

// A bounded in-process performance cache: only descriptors and timings.
// In particular, no matrix values, CSR indices, handles or MPI maps survive here.
namespace csr_adaptive_cache {
constexpr int candidates=5;
struct Features {
    csr_choice_cache::Key environment;
    std::array<double,6> shape{};
    double work=1.;
};
struct Profile {
    Features features;
    std::array<double,6> apply{},build{};
    std::array<unsigned,6> measured{};
    unsigned long long reuse=0;
};
struct Suggestion {
    std::array<double,6> apply{},build{};
    std::array<unsigned,6> measured{};
    unsigned long long reuse=0;
    bool matched=false;
};
inline std::list<Profile>& profiles(){static std::list<Profile> value;return value;}
template<class Mpi,class V>
Features describe(const Mpi& mm,int dim,const V& ptr,const V& col,int chunks,int trials) {
    // Bounded samples affect only candidate ordering; exact CSR validation is
    // performed by the caller. Sample counts are tuning costs, not physics sizes.
    constexpr std::size_t row_samples=256,column_samples=1024;
    const std::size_t rows=ptr.size()-1,nnz=col.size();
    const std::size_t nr=std::min(rows,row_samples),nc=std::min(nnz,column_samples);
    double sum=0.,square=0.,maximum=0.,remote=0.;
    for(std::size_t k=0;k<nr;++k) {
        const auto r=k*rows/nr;const double n=ptr[r+1]-ptr[r];
        sum+=n;square+=n*n;maximum=std::max(maximum,n);
    }
    VecPartition part(mm.np(),mm.id(),dim);
    for(std::size_t k=0;k<nc;++k) {
        const int c=col[k*nnz/nc];remote+=c<part.bgn()||c>=part.end();
    }
    const double mean=nr?sum/nr:0.;
    double local[6]={static_cast<double>(nnz),mean,
        mean?square/nr/(mean*mean):0.,mean?maximum/mean:0.,
        nc?remote/nc:0.,static_cast<double>(rows)};
    double global[6]={},total=0.;
    if(MPI_Allreduce(local,global,6,MPI_DOUBLE,MPI_MAX,mm.cm())!=MPI_SUCCESS ||
       MPI_Allreduce(local,&total,1,MPI_DOUBLE,MPI_SUM,mm.cm())!=MPI_SUCCESS)
        throw std::runtime_error("CSR adaptive descriptor exchange failed");
    Features result;
    result.work=std::max(1.,global[0]);
    result.shape={std::log1p(static_cast<double>(dim)),std::log1p(global[1]),
        std::log1p(global[2]),std::log1p(global[3]),global[4],
        total?global[0]*mm.np()/total:1.};
    unsigned long long hash=1469598103934665603ULL;
    for(auto value:{thread_budget(),chunks,trials,mm.ms(),mkl_get_max_threads()})
        hash=csr_choice_cache::mix(hash,value);
    char host[MPI_MAX_PROCESSOR_NAME];int length=0;MPI_Get_processor_name(host,&length);
    for(int i=0;i<length;++i)hash=csr_choice_cache::mix(hash,static_cast<unsigned char>(host[i]));
    for(const char* name:{"OMP_PLACES","OMP_PROC_BIND","OMP_THREAD_LIMIT","MKL_NUM_THREADS"}) {
        if(const char* value=std::getenv(name))
            while(*value)hash=csr_choice_cache::mix(hash,static_cast<unsigned char>(*value++));
        hash=csr_choice_cache::mix(hash,0);
    }
    stage2_parallel::collective(mm,[&]{result.environment.resize(mm.np());});
    if(MPI_Allgather(&hash,1,MPI_UNSIGNED_LONG_LONG,result.environment.data(),1,
                    MPI_UNSIGNED_LONG_LONG,mm.cm())!=MPI_SUCCESS)
        throw std::runtime_error("CSR adaptive environment exchange failed");
    return result;
}
inline double distance(const Features& a,const Features& b) {
    if(a.environment!=b.environment)return std::numeric_limits<double>::infinity();
    double d=0.;
    for(std::size_t i=0;i<a.shape.size();++i)d=std::max(d,std::abs(a.shape[i]-b.shape[i]));
    return d;
}
inline Suggestion lookup(const Features& f,unsigned long long reuse_limit) {
    Suggestion out;
    if(!reuse_limit)return out;
    auto& cache=profiles();auto best=cache.end();double closest=.35;
    for(auto it=cache.begin();it!=cache.end();++it) {
        const double d=distance(f,it->features);
        if(d<closest){closest=d;best=it;}
    }
    if(best==cache.end() || best->reuse>=reuse_limit)return out;
    out.matched=true;out.reuse=++best->reuse;
    const double scale=f.work/best->features.work;
    for(int mode=1;mode<=candidates;++mode) {
        out.apply[mode]=best->apply[mode]*scale;
        out.build[mode]=best->build[mode]*scale;
        out.measured[mode]=best->measured[mode];
    }
    cache.splice(cache.begin(),cache,best);
    return out;
}
inline void remember(const Features& features,const Suggestion& old,
                     const std::array<double,6>& apply,const std::array<double,6>& build,
                     const std::array<unsigned,6>& measured,unsigned long long capacity) {
    if(!capacity)return;
    Profile next;next.features=features;next.reuse=old.matched?old.reuse:0;
    for(int mode=1;mode<=candidates;++mode) {
        next.apply[mode]=measured[mode]?apply[mode]:old.apply[mode];
        next.build[mode]=build[mode]>0.?build[mode]:old.build[mode];
        next.measured[mode]=measured[mode]?measured[mode]:old.measured[mode];
    }
    auto& cache=profiles();
    // Replace only an identical descriptor; nearby entries remain independent.
    for(auto it=cache.begin();it!=cache.end();)
        if(distance(features,it->features)==0.)it=cache.erase(it);else ++it;
    cache.push_front(std::move(next));
    while(cache.size()>capacity)cache.pop_back();
}
}
