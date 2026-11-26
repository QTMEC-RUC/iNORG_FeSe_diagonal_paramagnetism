/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/
#pragma once

#include "distributed_csr.h"
#include "wide_csr.h"
#include "matrix_build_profile.h"

template<typename T> class SparseMat {
    const MyMpi& mm;
    VEC<T> pending_values_;
    VEC<BasisIndex> pending_columns_;
    Vec<T> values_;
    mutable VecInt ptr32_,col32_;
    mutable Vec<BasisIndex> ptr64_,col64_;
    mutable bool wide_=false;
    bool fixed_=false;
    BasisIndex last_incremental_row_=-1;
    mutable std::unique_ptr<DistributedCsrReal> narrow_plan_;
    mutable std::unique_ptr<WideCsrReal> wide_plan_;
    mutable unsigned long long expected_calls_=0;

    static constexpr BasisIndex narrow_limit=std::numeric_limits<int>::max();
    void unprepared() const {
        if(narrow_plan_ || wide_plan_)throw std::logic_error("Cannot edit a prepared CSR multiply plan");
    }
    void clear_plan() {
        narrow_plan_.reset();wide_plan_.reset();expected_calls_=0;
    }
    void promote() const {
        if(wide_)return;
        ptr64_.reset(ptr32_.size());
        col64_.reset(col32_.size());
        stage2_parallel::for_each(ptr32_.size(),[&](Idx i){ptr64_[i]=ptr32_[i];});
        stage2_parallel::for_each(col32_.size(),[&](Idx i){col64_[i]=col32_[i];});
        ptr32_.reset();col32_.reset();
        wide_=true;
    }
    BasisIndex row_begin(Idx row) const {
        if(!fixed_ && row>static_cast<Idx>(last_incremental_row_+1))return pending_values_.size();
        return wide_?ptr64_[row]:ptr32_[row];
    }
    void row_offset(Idx row,BasisIndex offset) {
        if(wide_)ptr64_[row]=offset;else ptr32_[row]=static_cast<int>(offset);
    }
    void begin_incremental_row(BasisIndex row) {
        unprepared();
        if(fixed_ || row<0 || row>=length_m || row<last_incremental_row_ ||
           !(wide_?ptr64_.size():ptr32_.size()))
            throw std::invalid_argument("Invalid incremental CSR row or allocation");
        for(BasisIndex i=last_incremental_row_+1;i<row;++i)row_offset(i+1,pending_values_.size());
        last_incremental_row_=row;
    }
    bool pending() const {return !pending_values_.empty();}
    BasisIndex column(Idx j) const {
        if(pending())return pending_columns_[j];
        return wide_?col64_[j]:col32_[j];
    }
    T value(Idx j) const {return pending()?pending_values_[j]:values_[j];}
public:
    BasisIndex length_m,length_k;

    SparseMat(BasisIndex rows,BasisIndex cols,const MyMpi& mpi):mm(mpi),
        wide_(rows>=narrow_limit || cols>=narrow_limit),length_m(rows),length_k(cols) {
        if(rows<0 || cols<0)throw std::invalid_argument("Negative sparse matrix dimensions");
        const Idx count=static_cast<Idx>(rows)+1;
        if(wide_)ptr64_.reset(count,-1);else ptr32_.reset(count,-1);
        row_offset(0,0);
    }
    SparseMat(BasisIndex n,const MyMpi& mpi):SparseMat(n,n,mpi) {}
    template<class I> SparseMat(BasisIndex rows,BasisIndex cols,const Vec<I>& ptr,
            const Vec<I>& col,const Vec<Real>& val,const MyMpi& mpi):SparseMat(rows,cols,mpi) {
        static_assert(std::is_integral<I>::value,"Integer CSR index required");
        if(ptr.size()!=static_cast<Idx>(rows)+1 || val.size()!=col.size() || ptr[0]!=0 ||
           val.size()>static_cast<Idx>(std::numeric_limits<BasisIndex>::max()) ||
           static_cast<std::uintmax_t>(ptr[rows])!=val.size())
            throw std::invalid_argument("CSR constructor size or row endpoints mismatch");
        if(val.size()>static_cast<Idx>(narrow_limit))promote();
        for(Idx i=1;i<ptr.size();++i) {
            if(ptr[i]<ptr[i-1] || static_cast<std::uintmax_t>(ptr[i])>val.size())
                throw std::invalid_argument("Invalid CSR row offsets");
            row_offset(i,static_cast<BasisIndex>(ptr[i]));
        }
        last_incremental_row_=rows-1;
        if(val.size()) {
            pending_values_.assign(val.p(),val.p()+val.size());
            pending_columns_.resize(col.size());
            stage2_parallel::ranges(col.size(),[&](Idx first,Idx last,Idx){
                for(Idx i=first;i<last;++i) {
                    if(static_cast<std::uintmax_t>(col[i])>=static_cast<std::uintmax_t>(cols))
                        throw std::out_of_range("CSR column outside the basis");
                    pending_columns_[i]=static_cast<BasisIndex>(col[i]);
                }
            });
        }
    }
    SparseMat(const SparseMat& r):mm(r.mm),pending_values_(r.pending_values_),pending_columns_(r.pending_columns_),
        values_(r.values_),ptr32_(r.ptr32_),col32_(r.col32_),ptr64_(r.ptr64_),col64_(r.col64_),
        wide_(r.wide_),fixed_(r.fixed_),last_incremental_row_(r.last_incremental_row_),length_m(r.length_m),length_k(r.length_k) {}
    SparseMat& operator=(const SparseMat& rhs) {
        if(this==&rhs)return *this;
        clear_plan();wide_=rhs.wide_;fixed_=rhs.fixed_;last_incremental_row_=rhs.last_incremental_row_;
        length_m=rhs.length_m;length_k=rhs.length_k;
        pending_values_=rhs.pending_values_;pending_columns_=rhs.pending_columns_;
        ptr32_.reset(rhs.ptr32_);col32_.reset(rhs.col32_);ptr64_.reset(rhs.ptr64_);col64_.reset(rhs.col64_);
        values_.reset(rhs.values_);return *this;
    }
    void hint_expected_calls(unsigned long long calls) const {
        if(!narrow_plan_ && !wide_plan_)expected_calls_=calls;
    }
    void addelement(const Real& value,BasisIndex col,BasisIndex row) {
        if(col<0 || col>=length_k)throw std::out_of_range("CSR column outside the basis");
        begin_incremental_row(row);
        if(pending_columns_.size()==static_cast<Idx>(narrow_limit))promote();
        pending_columns_.push_back(col);pending_values_.push_back(value);
        row_offset(row+1,static_cast<BasisIndex>(pending_values_.size()));
    }
    void addelement_one_based(const Real& value,BasisIndex col,BasisIndex row) {
        if(col<0 || col>=length_k-1)throw std::out_of_range("Shifted CSR column outside the basis");
        addelement(value,col+1,row);
    }
    void add_empty_row(BasisIndex row) {
        begin_incremental_row(row);row_offset(row+1,pending_values_.size());
    }
    void shrink_to_fit(){pending_values_.shrink_to_fit();pending_columns_.shrink_to_fit();}
    VEC<T> get_va() const {
        if(pending() || !values_.size())return pending_values_;
        return VEC<T>(values_.p(),values_.p()+values_.size());
    }
    VEC<BasisIndex> get_colnu() const {
        if(pending())return pending_columns_;
        VEC<BasisIndex> result(values_.size());
        for(Idx j=0;j<result.size();++j)result[j]=column(j);
        return result;
    }
    Vec<BasisIndex> get_roin() const {
        if(!(wide_?ptr64_.size():ptr32_.size()))return Vec<BasisIndex>();
        Vec<BasisIndex> ptr(static_cast<Idx>(length_m)+1);
        for(Idx r=0;r<ptr.size();++r)ptr[r]=row_begin(r);
        return ptr;
    }

    void prepare_csr(const std::vector<std::size_t>& counts,matrix_build_profile::Report* profile=nullptr) {
        clear_plan();
        if(counts.size()!=static_cast<Idx>(length_m))throw std::invalid_argument("Sparse CSR row count mismatch");
        if(wide_)ptr64_.reset(counts.size()+1);else ptr32_.reset(counts.size()+1);
        if(profile)profile->mark(matrix_build_profile::CsrRowAllocation);
        auto scan=[&] {
            return preparation_parallel::offsets(counts.size(),[&](Idx r){return counts[r];},
                [&](Idx r,Idx offset){
                    if(wide_ || offset<=static_cast<Idx>(narrow_limit))row_offset(r,static_cast<BasisIndex>(offset));
                },static_cast<Idx>(std::numeric_limits<BasisIndex>::max()));
        };
        const Idx total=scan();
        if(!wide_ && total>static_cast<Idx>(narrow_limit)) {
            wide_=true;ptr32_.reset();col32_.reset();ptr64_.reset(counts.size()+1);scan();
        }
        if(profile)profile->mark(matrix_build_profile::CsrOffsets);
        values_.reset(total);
        if(wide_)col64_.reset(total);else col32_.reset(total);
        if(profile)profile->mark(matrix_build_profile::CsrValuesAllocation);
        VEC<T>().swap(pending_values_);VEC<BasisIndex>().swap(pending_columns_);
        fixed_=true;
        if(profile)profile->mark(matrix_build_profile::CsrLegacyRelease);
    }
    BasisIndex csr_row_begin(Idx row) const {return row_begin(row);}
    void set_csr_entry(BasisIndex pos,BasisIndex col,T v) {
        unprepared();
        if(static_cast<std::uint64_t>(pos)>=values_.size() ||
           static_cast<std::uint64_t>(col)>=static_cast<std::uint64_t>(length_k))
            throw std::out_of_range("CSR entry position or column out of range");
        if(wide_)col64_[pos]=col;else col32_[pos]=static_cast<int>(col);
        values_[pos]=v;
    }
    void fix() {
        if(fixed_)return;
        if(length_m>0)add_empty_row(length_m-1);
        clear_plan();values_.reset(pending_values_);
        if(wide_)col64_.reset(pending_columns_);
        else {
            col32_.reset(pending_columns_.size());
            stage2_parallel::ranges(pending_columns_.size(),[&](Idx first,Idx last,Idx){
                for(Idx i=first;i<last;++i)col32_[i]=static_cast<int>(pending_columns_[i]);
            });
        }
        VEC<T>().swap(pending_values_);VEC<BasisIndex>().swap(pending_columns_);fixed_=true;
    }
    void free() {
        clear_plan();fixed_=false;last_incremental_row_=-1;
        values_.reset();ptr32_.reset();col32_.reset();ptr64_.reset();col64_.reset();
        VEC<T>().swap(pending_values_);VEC<BasisIndex>().swap(pending_columns_);
    }
    void apply(const Vec<T>& x,Vec<T>& y,Real alpha=1.,Real beta=0.) const {
        static_assert(std::is_same<T,Real>::value,"CSR multiply supports real scalars");
        int main_thread=0;MPI_Is_thread_main(&main_thread);
        if(!main_thread)throw std::logic_error("SparseMat::apply requires MPI_THREAD_FUNNELED main thread");
        lanczos_timing::Scope timer(lanczos_timing::Hv);
        if(lanczos_timing::active)lanczos_timing::active->matrix(length_m,values_.size());
        if(!narrow_plan_ && !wide_plan_) {
            lanczos_timing::Scope setup(lanczos_timing::HvSetup);
            const bool need_wide=mm.np()==1 ? wide_ : mm.Allreduce(wide_?1:0)!=0;
            if(need_wide) {
                if(mm.np()>1)stage2_parallel::collective(mm,[&]{promote();});
                wide_plan_.reset(new WideCsrReal(mm,length_m,length_k,ptr64_,col64_,values_));
            } else {
                narrow_plan_.reset(new DistributedCsrReal(mm,static_cast<int>(length_m),static_cast<int>(length_k),
                                                          ptr32_,col32_,values_,expected_calls_));
            }
        }
        if(wide_plan_)wide_plan_->apply(x,y,alpha,beta);else narrow_plan_->apply(x,y,alpha,beta);
    }
    Vec<T> operator*(const Vec<T>& x) const {Vec<T> out(length_m);apply(x,out);return out;}

    void ho(Vec<T>& out,const Vec<T>& ket) {
        VecPartition vp(mm.np(),mm.id(),length_k);
        if(ket.size()!=static_cast<Idx>(length_k) || vp.len()!=length_m)
            throw std::invalid_argument("Full CSR multiplication size mismatch");
        Vec<T> local(length_m);
        stage2_parallel::for_each(local.size(),[&](Idx row){
            T sum{};for(auto j=row_begin(row);j<row_begin(row+1);++j)sum+=value(j)*ket[column(j)];
            local[row]=sum;
        });
        out=mm.Allgatherv(local,vp);
    }
    SparseMat operator+=(SparseMat& rhs) {
        if(length_m!=rhs.length_m || length_k!=rhs.length_k)throw std::invalid_argument("CSR addition size mismatch");
        SparseMat result(length_m,length_k,mm);
        for(BasisIndex r=0;r<length_m;++r) {
            std::map<BasisIndex,T> values;
            for(auto j=row_begin(r);j<row_begin(r+1);++j)values[column(j)]+=value(j);
            for(auto j=rhs.row_begin(r);j<rhs.row_begin(r+1);++j)values[rhs.column(j)]+=rhs.value(j);
            for(const auto& item:values)result.addelement(item.second,item.first,r);
            result.add_empty_row(r);
        }
        return result;
    }
    bool if_hermitian() const {
        if(length_m!=length_k)throw std::invalid_argument("Hermiticity check requires a complete square CSR");
        for(BasisIndex row=0;row<length_m;++row) {
            for(auto j=row_begin(row);j<row_begin(row+1);++j) {
                const auto c=column(j);bool found=false;
                for(auto k=row_begin(c);k<row_begin(c+1);++k)
                    if(column(k)==row && ABS(value(j)-value(k))<1.E-10){found=true;break;}
                if(!found)throw std::runtime_error("CSR is not Hermitian");
            }
        }
        return true;
    }
};
typedef SparseMat<Int> SparseMatInt;
typedef SparseMat<Idx> SparseMatIdx;
typedef SparseMat<Real> SparseMatReal;
typedef SparseMat<Cmplx> SparseMatCmplx;

template<typename T> inline SparseMat<T> operator+(SparseMat<T> lhs,SparseMat<T> rhs){return lhs+=rhs;}



