#pragma once
#include "milapacke.h"
#include "parallel_stage2.h"
// One workspace per calling thread. LAPACK may overwrite d/e; caller data remain intact.
class LowestTridiagonal {
    std::vector<double> d_,e_,w_,z_,work_;
    std::vector<lapack_int> iwork_,ifail_;
public:
    template<class D,class E> double last_component(const D& d,const E& e,Int k) {
        const std::size_t n=d.size();
        if(!n || e.size()+1!=n || k<0 || static_cast<std::size_t>(k)>=n
           || n>static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())/5)
            throw std::invalid_argument("Invalid residual tridiagonal");
        d_.resize(n);e_.resize(n);w_.resize(n);z_.resize(n);
        work_.resize(5*n);iwork_.resize(5*n);ifail_.resize(n);
        std::copy(d.begin(),d.end(),d_.begin());std::copy(e.begin(),e.end(),e_.begin());
        lapack_int found=0;
        stage2_parallel::LocalMklSerial serial;
        const auto info=LAPACKE_dstevx_work(LAPACK_COL_MAJOR,'V','I',n,d_.data(),e_.data(),
            0.,0.,1,1,0.,&found,w_.data(),z_.data(),n,work_.data(),iwork_.data(),ifail_.data());
        if(info!=0 || found!=1 || !std::isfinite(z_[k]))
            throw std::runtime_error("Lowest tridiagonal eigenpair failed");
        return z_[k];
    }
};
