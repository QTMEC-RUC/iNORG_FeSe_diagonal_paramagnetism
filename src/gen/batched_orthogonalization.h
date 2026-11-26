#pragma once
#include "parallel_lanczos.h"
#include "mympi.h"
#include "lanczos_timing.h"

namespace lanczos_local {
// Accumulate BOTH passes' coefficients when constructing Q^* H Q.
// method=0: sequential MGS reference; method=1: two array reductions (CGS2).
template<class T>
Vec<T> project_out(const MyMpi& mm,Vec<T>& v,const VEC<Vec<T>>& q,Int count,Int method=1) {
    if(count<0 || static_cast<std::size_t>(count)>q.size() || (method!=0 && method!=1))
        throw std::invalid_argument("Invalid orthogonalization arguments");
    Vec<T> total(count,T(0)),local(count),global(count);
    if(!count) return total;
    for(Int pass=0;pass<2;++pass) {
        lanczos_timing::count(lanczos_timing::OrthPasses);
        lanczos_timing::count(lanczos_timing::OrthProjections,count);
        if(method==1) {
            for(Int j=0;j<count;++j)local[j]=dot(q[j],v);
            mm.Allreduce_into(local,global);
            for(Int j=0;j<count;++j)total[j]+=global[j];
            // One threaded traversal, not one new OpenMP team per basis vector.
            const Idx n=v.size();const int nt=workers(n);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt>1)
            for(Idx i=0;i<n;++i) {
                T correction=0;
                for(Int j=0;j<count;++j)correction+=q[j][i]*global[j];
                v[i]-=correction;
            }
        } else {
            for(Int j=0;j<count;++j) {
                T a=mm.Allreduce(dot(q[j],v));total[j]+=a;axpy(v,-a,q[j]);
            }
        }
    }
    return total;
}
// Non-owning projection basis: no deep copies of multi-million-entry vectors.
// Workspace sizes follow the active rank (including dynamic deflation).
template<class T> struct ProjectionWorkspace {
    Vec<T> total,local,global;
    void prepare(Int count) {
        if(total.size()!=static_cast<Idx>(count)) {
            total.reset(count);local.reset(count);global.reset(count);
        }
        for(Int k=0;k<count;++k)total[k]=T(0);
    }
};
template<class T>
const Vec<T>& project_out_views(const MyMpi& mm,Vec<T>& v,
    const VEC<const Vec<T>*>& q,ProjectionWorkspace<T>& workspace,Int method=1) {
    if(q.size()>static_cast<Idx>(std::numeric_limits<Int>::max()) || (method!=0 && method!=1))
        throw std::invalid_argument("Invalid view projection arguments");
    const Int count=static_cast<Int>(q.size());workspace.prepare(count);
    for(const auto* p:q)if(!p || p->size()!=v.size() || (v.size() && p->p()==v.p()))
        throw std::invalid_argument("Invalid or aliased projection vector");
    auto& total=workspace.total;auto& local=workspace.local;auto& global=workspace.global;
    if(!count)return total;
    for(Int pass=0;pass<2;++pass) {
        lanczos_timing::count(lanczos_timing::OrthPasses);
        lanczos_timing::count(lanczos_timing::OrthProjections,count);
        if(method==1) {
            for(Int j=0;j<count;++j)local[j]=dot(*q[j],v);
            mm.Allreduce_into(local,global);
            for(Int j=0;j<count;++j)total[j]+=global[j];
            const Idx n=v.size();const int nt=workers(n);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt>1)
            for(Idx i=0;i<n;++i) {
                T correction=0;
                for(Int j=0;j<count;++j)correction+=(*q[j])[i]*global[j];
                v[i]-=correction;
            }
        } else {
            for(Int j=0;j<count;++j) {
                T a=mm.Allreduce(dot(*q[j],v));total[j]+=a;axpy(v,-a,*q[j]);
            }
        }
    }
    return total;
}
}
