#pragma once

#include "mympi.h"
#include "parallel_preparation.h"
#include <unordered_map>

namespace wide_csr_detail {
template<bool distributed>
inline void multiply_rows(std::size_t rows,const BasisIndex* offsets,const BasisIndex* indices,
                          const double* values,const double* local,const double* ghosts,
                          double* result,double alpha,double beta) {
    stage2_parallel::ranges(rows,[&](std::size_t begin,std::size_t end,std::size_t){
        for(std::size_t row=begin;row<end;++row) {
            double sum=0.;
            for(auto j=offsets[row];j<offsets[row+1];++j) {
                const auto c=indices[j];
                sum+=values[j]*(distributed && c<0 ? ghosts[-1-c] : local[c]);
            }
            result[row]=alpha*sum+(beta==0.?0.:beta*result[row]);
        }
    });
}
}

// CSR sizes above the LP64 MKL or MPI-3 limits. A single rank traverses
// the original arrays directly. Distributed ranks exchange only needed columns.
class WideCsrReal {
    const MyMpi& mm_;
    VecPartition part_;
    const Vec<BasisIndex>& ptr_;
    const Vec<BasisIndex>& col_;
    const VecReal& values_;
    std::vector<BasisIndex> send_offsets_, recv_offsets_, send_indices_, compact_col_;
    std::vector<double> send_values_, ghost_;

    static void check(int code) {
        if(code!=MPI_SUCCESS)throw std::runtime_error("Wide CSR MPI communication failed");
    }

    // In phase p, partner(p,r)=(p-r) mod np is symmetric. Thus both ranks
    // perform the same number of Sendrecv calls even when the payloads differ.
    template<class T>
    void exchange(const std::vector<T>& from, const std::vector<BasisIndex>& from_offsets,
                  std::vector<T>& to, const std::vector<BasisIndex>& to_offsets,
                  MPI_Datatype type) const {
        T dummy{};
        for(int phase=0;phase<mm_.np();++phase) {
            int peer=phase-mm_.id();
            if(peer<0)peer+=mm_.np();
            BasisIndex s=from_offsets[peer],r=to_offsets[peer];
            const auto send_end=from_offsets[peer+1],recv_end=to_offsets[peer+1];
            while(s<send_end || r<recv_end) {
                const int ns=static_cast<int>(std::min<BasisIndex>(send_end-s,std::numeric_limits<int>::max()));
                const int nr=static_cast<int>(std::min<BasisIndex>(recv_end-r,std::numeric_limits<int>::max()));
                check(MPI_Sendrecv(ns?from.data()+s:&dummy,ns,type,peer,0,
                                   nr?to.data()+r:&dummy,nr,type,peer,0,mm_.cm(),MPI_STATUS_IGNORE));
                s+=ns;r+=nr;
            }
        }
    }

    void prepare_halo() {
        std::vector<BasisIndex> needed,send_counts,recv_counts;
        stage2_parallel::collective(mm_,[&] {
            std::unordered_map<BasisIndex,BasisIndex> slots;
            for(Idx j=0;j<col_.size();++j) {
                const auto c=col_[j];
                if(c<part_.bgn() || c>=part_.end())slots.emplace(c,0);
            }
            needed.reserve(slots.size());
            for(const auto& item:slots)needed.push_back(item.first);
            std::sort(needed.begin(),needed.end());
            for(Idx i=0;i<needed.size();++i)slots.at(needed[i])=static_cast<BasisIndex>(i);
            const auto& lookup=slots;
            compact_col_.resize(col_.size());
            stage2_parallel::ranges(col_.size(),[&](Idx first,Idx last,Idx){
                for(Idx j=first;j<last;++j) {
                    const auto c=col_[j];
                    compact_col_[j]=(c>=part_.bgn() && c<part_.end())?c-part_.bgn():-1-lookup.at(c);
                }
            });
            recv_offsets_.resize(static_cast<Idx>(mm_.np())+1);
            for(int rank=0;rank<mm_.np();++rank)
                recv_offsets_[rank+1]=std::lower_bound(needed.begin(),needed.end(),part_.end(rank))-needed.begin();
            recv_counts.resize(mm_.np());send_counts.resize(mm_.np());
            for(int i=0;i<mm_.np();++i)recv_counts[i]=recv_offsets_[i+1]-recv_offsets_[i];
        });
        check(MPI_Alltoall(recv_counts.data(),1,MPI_INT64_T,send_counts.data(),1,MPI_INT64_T,mm_.cm()));
        stage2_parallel::collective(mm_,[&] {
            send_offsets_.resize(static_cast<Idx>(mm_.np())+1);
            for(int i=0;i<mm_.np();++i) {
                if(send_counts[i]<0 || send_counts[i]>std::numeric_limits<BasisIndex>::max()-send_offsets_[i])
                    throw std::overflow_error("Wide CSR halo count overflow");
                send_offsets_[i+1]=send_offsets_[i]+send_counts[i];
            }
            send_indices_.resize(send_offsets_.back());
            send_values_.resize(send_indices_.size());ghost_.resize(needed.size());
        });
        exchange(needed,recv_offsets_,send_indices_,send_offsets_,MPI_INT64_T);
        stage2_parallel::collective(mm_,[&] {
            for(auto& c:send_indices_) {
                if(c<part_.bgn() || c>=part_.end())throw std::runtime_error("Wide CSR halo ownership mismatch");
                c-=part_.bgn();
            }
        });
    }


public:
    WideCsrReal(const MyMpi& mm,BasisIndex rows,BasisIndex total,
                const Vec<BasisIndex>& ptr,const Vec<BasisIndex>& col,const VecReal& values)
        :mm_(mm),part_(mm.np(),mm.id(),total),ptr_(ptr),col_(col),values_(values) {
        stage2_parallel::collective(mm_,[&] {
            if(rows!=part_.len() || ptr.size()!=static_cast<Idx>(rows)+1 || ptr[0]!=0 ||
               ptr[rows]<0 || static_cast<Idx>(ptr[rows])!=values.size() || col.size()!=values.size() ||
               !preparation_parallel::all(rows,[&](Idx i){return ptr[i]<=ptr[i+1];}) ||
               !preparation_parallel::all(col.size(),[&](Idx j){return col[j]>=0 && col[j]<total;}))
                throw std::invalid_argument("Invalid 64-bit CSR or MPI row partition");
        });
        if(mm_.np()>1)prepare_halo();
    }
    void apply(const VecReal& x,VecReal& y,double alpha,double beta) {
        if(x.size()!=static_cast<Idx>(part_.len()) || (x.size() && x.p()==y.p()))
            throw std::invalid_argument("Wide CSR input slice or alias mismatch");
        if(y.size()!=x.size()) {
            if(beta!=0.)throw std::invalid_argument("Wide CSR accumulation output size mismatch");
            y.reset(x.size());
        }
        if(mm_.np()==1) {
            wide_csr_detail::multiply_rows<false>(x.size(),ptr_.p(),col_.p(),values_.p(),
                                                 x.p(),nullptr,y.p(),alpha,beta);
            return;
        }
        stage2_parallel::ranges(send_indices_.size(),[&](Idx first,Idx last,Idx){
            for(Idx i=first;i<last;++i)send_values_[i]=x[send_indices_[i]];
        });
        exchange(send_values_,send_offsets_,ghost_,recv_offsets_,MPI_DOUBLE);
        wide_csr_detail::multiply_rows<true>(x.size(),ptr_.p(),compact_col_.data(),values_.p(),
                                            x.p(),ghost_.data(),y.p(),alpha,beta);
    }
};
