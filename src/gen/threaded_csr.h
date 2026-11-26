#pragma once
#include "parallel_preparation.h"
#include <algorithm>
#include <cstdint>

// Owning CSR with explicit row ownership per OpenMP slot. No MPI here.
// Local CSR retains empty rows (beta must be applied); remote CSR omits them.
class ThreadedCsrRows {
    VecInt ptr_,col_;
    VecReal values_;
    std::vector<int> output_,cuts_;
public:
    int rows()const{return output_.size();}
    std::size_t nnz()const{return values_.size();}
    int slots()const{return cuts_.empty()?0:static_cast<int>(cuts_.size())-1;}
    void clear(){ptr_.reset();col_.reset();values_.reset();
        std::vector<int>().swap(output_);std::vector<int>().swap(cuts_);}
    void create(int rows,const int* ptr,const int* col,const double* values,
                int begin,int end,const std::vector<int>& lookup,bool remote,int slots,bool progress_master=false) {
        namespace pp=preparation_parallel;
        namespace par=stage2_parallel;
        std::vector<int> counts(rows,0);
        par::ranges(rows,[&](std::size_t first,std::size_t last,std::size_t) {
            for(auto r=first;r<last;++r)for(int j=ptr[r];j<ptr[r+1];++j)
                if((col[j]<begin || col[j]>=end)==remote)++counts[r];
        });
        // Ordered compression preserves each original output row exactly once.
        const auto n=pp::offsets(rows,[&](std::size_t r){return !remote || counts[r]!=0;},
            [](std::size_t,std::size_t){});
        output_.resize(n);
        pp::offsets(rows,[&](std::size_t r){return !remote || counts[r]!=0;},
            [&](std::size_t r,std::size_t pos){if(r<static_cast<std::size_t>(rows) && (!remote || counts[r]))output_[pos]=r;});
        ptr_.reset(n+1);
        const auto nz=pp::offsets(n,[&](std::size_t r){return counts[output_[r]];},
            [&](std::size_t r,std::size_t offset){ptr_[r]=offset;},std::numeric_limits<int>::max());
        col_.reset(nz);values_.reset(nz);
        const int tasks=std::max(1,std::min(slots,std::max(1,static_cast<int>(n))));
        cuts_.resize(tasks+1);cuts_[0]=0;cuts_[tasks]=n;
        // nnz + row count includes the work of empty rows; 64-bit prefix weights.
        const std::uint64_t work=static_cast<std::uint64_t>(nz)+n;
        for(int t=1;t<tasks;++t) {
            const auto target=work*t/tasks;
            int lo=0,hi=n;
            while(lo<hi){const int mid=lo+(hi-lo)/2;
                if(static_cast<std::uint64_t>(ptr_[mid])+mid<target)lo=mid+1;else hi=mid;}
            cuts_[t]=lo;
        }
        // First touch follows exactly the row slots used by the multiply.
const int nt=tasks+(progress_master?1:0);
#pragma omp parallel num_threads(nt)
        {
            const int tid=omp_get_thread_num(),team=omp_get_num_threads();
            const bool reserved=progress_master && team>1;
            const int worker=reserved?tid-1:tid,workers=reserved?team-1:team;
            if(worker>=0)for(int task=worker;task<tasks;task+=workers)for(int i=cuts_[task];i<cuts_[task+1];++i) {
                const int r=output_[i];int dest=ptr_[i];
                for(int j=ptr[r];j<ptr[r+1];++j)if((col[j]<begin || col[j]>=end)==remote) {
                    col_[dest]=remote?lookup[col[j]]-(end-begin):col[j]-begin;
                    values_[dest++]=values[j];
                }
            }
        }
    }
    void slot(int task,const double* x,double* y,double alpha,double beta)const {
        const int* p=ptr_.p();const int* c=col_.p();const double* a=values_.p();
        for(int r=cuts_[task];r<cuts_[task+1];++r) {
            double sum=0.;
#pragma omp simd reduction(+:sum)
            for(int j=p[r];j<p[r+1];++j)sum+=a[j]*x[c[j]];
            double& out=y[output_[r]];
            out=alpha*sum+(beta==0.?0.:beta*out);
        }
    }
    void mv(const double* x,double* y,double alpha,double beta)const {
        const int tasks=slots();
#pragma omp parallel num_threads(tasks)
        {
            const int tid=omp_get_thread_num(),team=omp_get_num_threads();
            for(int task=tid;task<tasks;task+=team)slot(task,x,y,alpha,beta);
        }
    }
};
