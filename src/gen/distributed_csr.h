#pragma once
#include "mympi.h"
#include "parallel_lanczos.h"
#include "parallel_preparation.h"
#include "threaded_csr.h"
#include "csr_choice_cache.h"
#include "csr_adaptive_cache.h"
#include <memory>
#include <numeric>
#include <type_traits>
#include <sstream>

// One plan per immutable real CSR matrix. No MPI calls in OpenMP workers.
// CSR arrays outlive this object. Copies of SparseMat create independent plans.
class DistributedCsrReal {
    struct Handle {
        sparse_matrix_t value=nullptr;
        std::vector<MKL_INT> begin,end;
        Handle()=default;
        Handle(const Handle&)=delete;
        ~Handle(){clear();}
        void clear(){if(value){mkl_sparse_destroy(value);value=nullptr;}}
        void create(int cols,const std::vector<int>& rows,const int* ptr,int* index,double* values) {
            begin.reserve(rows.size());end.reserve(rows.size());
            begin.resize(rows.size());end.resize(rows.size());
            stage2_parallel::ranges(rows.size(),[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto i=first;i<last;++i){begin[i]=ptr[rows[i]];end[i]=ptr[rows[i]+1];}
            });
            if(rows.empty() || preparation_parallel::all(rows.size(),[&](std::size_t i){return ptr[rows[i]]==ptr[rows[i]+1];}))return;
            {lanczos_timing::Scope timer(lanczos_timing::MklCreate);
            check(mkl_sparse_d_create_csr(&value,SPARSE_INDEX_BASE_ZERO,rows.size(),cols,
                begin.data(),end.data(),index,values),"create");}
            // Keep the established no-hint optimize path: the installed MKL's
            // hinted kernel fails duplicate-column regression with alpha/beta.
            {lanczos_timing::Scope timer(lanczos_timing::MklOptimize);
            check(mkl_sparse_optimize(value),"optimize");}
        }
        static void check(sparse_status_t status,const char* operation) {
            if(status!=SPARSE_STATUS_SUCCESS)throw std::runtime_error(std::string("CSR MKL ")+operation+" failed: "+std::to_string(status));
        }
        void mv(const double* x,double* y,double alpha=1.,double beta=0.)const {
            if(!value) {
                for(std::size_t i=0;i<begin.size();++i)y[i]=beta==0.?0.:beta*y[i];
                return;
            }
            matrix_descr desc{};desc.type=SPARSE_MATRIX_TYPE_GENERAL;
            check(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,alpha,value,desc,x,beta,y),"mv");
        }
    };
    const MyMpi& mm;
    VecPartition part;
    int rows_,dim_,mode_=0,tune_=3,report_=1,timing_=1,local_chunks_=4;
    bool chosen_=false,choice_cache_hit_=false;
    bool adaptive_=false,automatic_=false;
    unsigned long long expected_calls_=0,tune_calls_=0;
    int exact_hint_=0,tested_candidates_=0;
    double tune_budget_fraction_=.05;
    const VecInt* source_ptr_view_=nullptr;
    const VecInt* source_col_view_=nullptr;
    const int* source_ptr_=nullptr;
    int* source_col_=nullptr;
    double* source_values_=nullptr;
    std::vector<int> halo_lookup_;
    bool halo_ready_=false;
    std::array<bool,6> built_{};
    std::array<double,6> candidate_build_{},candidate_build_max_{},candidate_trial_{},
        candidate_warmup_{},candidate_score_{};
    std::array<unsigned,6> candidate_samples_{};
    std::array<std::vector<double>,6> candidate_observations_;
    csr_adaptive_cache::Features features_;
    csr_adaptive_cache::Suggestion suggestion_;
    double config_validation_=0.,fingerprint_=0.,features_time_=0.,halo_map_time_=0.,
        choose_logic_=0.,release_time_=0.,plan_report_time_=0.;
    double collective_max(double value)const {
        double result=0.;
        mpi_check(MPI_Allreduce(&value,&result,1,MPI_DOUBLE,MPI_MAX,mm.cm()));
        return result;
    }
    void read_policy() {
        int policy=1,invalid=0;
        if(mm) {
            const char* value=std::getenv("NORG_HV_TUNE_POLICY");
            if(value) {
                if(std::string(value)=="full")policy=0;
                else if(std::string(value)!="adaptive")invalid=1;
            }
            value=std::getenv("NORG_HV_TUNE_BUDGET");
            if(value)try {
                std::size_t end=0;tune_budget_fraction_=std::stod(value,&end);
                if(end!=std::string(value).size() || !std::isfinite(tune_budget_fraction_) ||
                   tune_budget_fraction_<0. || tune_budget_fraction_>1.)invalid=1;
            }catch(...){invalid=1;}
        }
        mpi_check(MPI_Bcast(&invalid,1,MPI_INT,mm.ms(),mm.cm()));
        if(invalid)throw std::invalid_argument("NORG_HV_TUNE_POLICY=full|adaptive; NORG_HV_TUNE_BUDGET in [0,1]");
        mpi_check(MPI_Bcast(&policy,1,MPI_INT,mm.ms(),mm.cm()));
        mpi_check(MPI_Bcast(&tune_budget_fraction_,1,MPI_DOUBLE,mm.ms(),mm.cm()));
        unsigned long long hint=expected_calls_;
        mpi_check(MPI_Bcast(&hint,1,MPI_UNSIGNED_LONG_LONG,mm.ms(),mm.cm()));
        collective_require(hint==expected_calls_,"CSR expected call hints differ between ranks");
        automatic_=mode_==0;adaptive_=automatic_ && expected_calls_>0 && policy;
    }
    unsigned long long cache_reuse_=8,cache_entries_=64;
    csr_choice_cache::Key choice_key_;
    void reuse_choice(const VecInt& ptr,const VecInt& col) {
        if(mode_!=0)return;
        int invalid=0;
        if(mm)try {
            cache_reuse_=runtime_options::nonnegative_integer("NORG_HV_CACHE_REUSE",8);
            cache_entries_=runtime_options::nonnegative_integer("NORG_HV_CACHE_ENTRIES",64);
        }catch(...){invalid=1;}
        mpi_check(MPI_Bcast(&invalid,1,MPI_INT,mm.ms(),mm.cm()));
        if(invalid)throw std::invalid_argument("NORG_HV_CACHE_REUSE/ENTRIES must be nonnegative integers");
        mpi_check(MPI_Bcast(&cache_reuse_,1,MPI_UNSIGNED_LONG_LONG,mm.ms(),mm.cm()));
        mpi_check(MPI_Bcast(&cache_entries_,1,MPI_UNSIGNED_LONG_LONG,mm.ms(),mm.cm()));
        if(!cache_reuse_ || !cache_entries_)return;
        const double begin=MPI_Wtime();
        choice_key_=csr_choice_cache::key(mm,dim_,ptr,col,local_chunks_,tune_);
        // Adaptive winners come from a budgeted search. Keep their hints out of
        // the legacy full-search cache, including subsequent ground-state plans.
        if(adaptive_)stage2_parallel::collective(mm,[&] {
            choice_key_.push_back(0x4144415054495645ULL);
        });
        fingerprint_+=MPI_Wtime()-begin;
        int selected=mm?csr_choice_cache::lookup(choice_key_,cache_reuse_):0;
        mpi_check(MPI_Bcast(&selected,1,MPI_INT,mm.ms(),mm.cm()));
        if(selected) {
            choice_cache_hit_=true;
            if(adaptive_)exact_hint_=selected;
            else {mode_=selected;if(mode_==4){mode_=3;local_chunks_=1;}}
        }
    }
    std::vector<int> recv_count_,recv_disp_,send_count_,send_disp_,send_index_,compact_col_;
    std::vector<int> interior_,boundary_;
    std::vector<double> full_x_,compact_x_,send_x_,inner_y_,outer_y_;
    // Split CSR owns its values/indices; handles must die before these buffers.
    std::vector<int> local_ptr_,remote_ptr_,chunk_begin_;
    VecInt local_col_,remote_col_;
    VecReal local_values_,remote_values_;
    std::vector<double> ghost_x_;
    std::vector<std::unique_ptr<Handle>> local_handles_;
    Handle remote_handle_;
    std::unique_ptr<Handle> single_local_;
    ThreadedCsrRows stream_local_,stream_remote_;
    std::vector<double> stream_ghost_;
    std::size_t local_nnz_=0,remote_nnz_=0;
    unsigned long long progress_tests_=0;
    Handle full_,inner_,outer_; // Destroy handles before their borrowed index buffers.
    std::size_t remote_=0,send_total_=0;
    unsigned long long calls_=0;
    inline static unsigned long long next_plan_=0;
    unsigned long long plan_id_=0;
    std::size_t nnz_=0;
    double setup_=0.,tuning_=0.,seconds_=0.,pack_=0.,exchange_=0.,compute_=0.;
    double trial_gather_=0.,trial_halo_=0.,trial_split_=0.,trial_single_=0.,trial_stream_=0.;
    double overlap_=0.,overlap_mpi_=0.;
    // Exclusive rank-wall stages. Only legacy halo's boundary/scatter split
    // adds a clock when detail is enabled. Split reuses its execution clocks.
    // No hot-path formatting, allocation or profiling communication.
    double local_mv_=0.,allgather_=0.,post_=0.,interior_mv_=0.,wait_=0.,boundary_mv_=0.,scatter_=0.,progress_=0.;
    static void mpi_check(int rc){if(rc!=MPI_SUCCESS)throw std::runtime_error("CSR MPI communication failed");}
    void collective_require(bool valid,const char* message)const {
        int bad=valid?0:1,any=0;mpi_check(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,mm.cm()));
        if(any)throw std::runtime_error(message);
    }
    static int displacements(const std::vector<int>& counts,std::vector<int>& offsets) {
        long long total=0;offsets.resize(counts.size());
        for(std::size_t i=0;i<counts.size();++i){
            if(total>std::numeric_limits<int>::max())return -1;
            offsets[i]=static_cast<int>(total);total+=counts[i];
        }
        return total>std::numeric_limits<int>::max()?-1:static_cast<int>(total);
    }
    void read_config() {
        int config[5]={0,3,1,1,4};
        if(mm){
            const char* value=std::getenv("NORG_HV_EXCHANGE");
            const std::string mode=value?value:"auto";
            if(mode=="allgather")config[0]=1;
            else if(mode=="halo")config[0]=2;
            else if(mode=="split")config[0]=3;
            else if(mode=="stream")config[0]=5;
            else if(mode!="auto")config[0]=-1;
            value=std::getenv("NORG_HV_TUNE_STEPS");
            if(value){try{config[1]=parse_thread_budget(value);}catch(...){config[0]=-1;}}
            value=std::getenv("NORG_HV_REPORT");
            if(value){if(std::string(value)=="0")config[2]=0;else if(std::string(value)!="1")config[0]=-1;}
            value=std::getenv("NORG_HV_LOCAL_CHUNKS");
            if(value){try{config[4]=parse_thread_budget(value);}catch(...){config[0]=-1;}}
            value=std::getenv("NORG_HV_TIMING");
            if(value){if(std::string(value)=="0")config[3]=0;else if(std::string(value)!="1")config[0]=-1;}
        }
        mpi_check(MPI_Bcast(config,5,MPI_INT,mm.ms(),mm.cm()));
        if(config[0]<0)throw std::invalid_argument("NORG_HV_EXCHANGE=auto|allgather|halo|split|stream; NORG_HV_LOCAL_CHUNKS positive; NORG_HV_TUNE_STEPS positive; NORG_HV_REPORT=0|1; NORG_HV_TIMING=0|1");
        mode_=mm.np()==1 && config[0]!=0 && config[0]!=5?1:config[0];tune_=config[1];report_=config[2];timing_=config[3];local_chunks_=config[4];
    }
    void make_halo(const int* ptr,int* col,double* values) {
        // O(D + nnz) construction with O(D) temporary lookup, not O(nnz)
        // duplicate request storage and one binary search per matrix entry.
        namespace pp=preparation_parallel;
        namespace par=stage2_parallel;
        std::vector<int> lookup,needed;
        par::collective(mm,[&] {
            lookup.resize(dim_);
            par::ranges(dim_,[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto c=first;c<last;++c)lookup[c]=-1;
            });
            par::ranges(ptr[rows_],[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto j=first;j<last;++j)if(col[j]<part.bgn() || col[j]>=part.end()) {
                    // Duplicate columns can be visited by different workers.
#pragma omp atomic write
                    lookup[col[j]]=0;
                }
            });
            const auto remote=pp::offsets(dim_,[&](std::size_t c){return lookup[c]==0;},
                [&](std::size_t c,std::size_t offset) {
                    if(c<static_cast<std::size_t>(dim_) && lookup[c]==0)
                        lookup[c]=part.len()+offset;
                },std::numeric_limits<int>::max()-part.len());
            needed.resize(remote);
            par::ranges(dim_,[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto c=first;c<last;++c)if(lookup[c]>=part.len())needed[lookup[c]-part.len()]=c;
            });
            remote_=needed.size();recv_count_.assign(mm.np(),0);send_count_.assign(mm.np(),0);
            par::for_each(mm.np(),[&](std::size_t owner) {
                recv_count_[owner]=std::lower_bound(needed.begin(),needed.end(),part.end(owner))
                    -std::lower_bound(needed.begin(),needed.end(),part.bgn(owner));
            });
            displacements(recv_count_,recv_disp_);
        });
        mpi_check(MPI_Alltoall(recv_count_.data(),1,MPI_INT,send_count_.data(),1,MPI_INT,mm.cm()));
        const int total=displacements(send_count_,send_disp_);
        collective_require(total>=0,"CSR halo send index count exceeds MPI int range");
        par::collective(mm,[&] {send_total_=total;send_index_.resize(std::max(1,total));});
        int dummy=0;
        mpi_check(MPI_Alltoallv(needed.empty()?&dummy:needed.data(),recv_count_.data(),recv_disp_.data(),MPI_INT,
            send_index_.data(),send_count_.data(),send_disp_.data(),MPI_INT,mm.cm()));
        par::collective(mm,[&] {
            const bool valid=pp::all(total,[&](std::size_t i) {
                send_index_[i]-=part.bgn();
                return send_index_[i]>=0 && send_index_[i]<part.len();
            });
            if(!valid)throw std::runtime_error("CSR halo received an index outside local ownership");
            send_x_.resize(std::max(1,total));
            if(adaptive_ || mode_==3 || mode_==5)return;
            compact_x_.resize(std::max<std::size_t>(1,part.len()+remote_));
            compact_col_.resize(ptr[rows_]);
            const auto parts=par::workers(rows_);
            std::vector<std::vector<int>> inner(parts),outer(parts);
            par::ranges(rows_,[&](std::size_t first,std::size_t last,std::size_t worker) {
                for(auto r=first;r<last;++r) {
                    bool local=true;
                    for(int j=ptr[r];j<ptr[r+1];++j) {
                        const int c=col[j];
                        if(c>=part.bgn()&&c<part.end())compact_col_[j]=c-part.bgn();
                        else {local=false;compact_col_[j]=lookup[c];}
                    }
                    (local?inner[worker]:outer[worker]).push_back(r);
                }
            });
            const auto concatenate=[&](const std::vector<std::vector<int>>& chunks,std::vector<int>& output) {
                std::vector<std::size_t> start(chunks.size()+1);
                pp::offsets(chunks.size(),[&](std::size_t i){return chunks[i].size();},
                    [&](std::size_t i,std::size_t offset){start[i]=offset;});
                output.resize(start.back());
                par::for_each(chunks.size(),[&](std::size_t i) {
                    std::copy(chunks[i].begin(),chunks[i].end(),output.begin()+start[i]);
                });
            };
            concatenate(inner,interior_);concatenate(outer,boundary_);
            inner_y_.resize(std::max<std::size_t>(1,interior_.size()));outer_y_.resize(std::max<std::size_t>(1,boundary_.size()));
        });
        if(adaptive_){halo_lookup_.swap(lookup);return;}
        if(mode_==0 || mode_==3)make_split(ptr,col,values,lookup);
        if(mode_==0 || mode_==5)make_stream(ptr,col,values,lookup);
        if(mode_==3 || mode_==5)return;
        inner_.create(part.len()+remote_,interior_,ptr,compact_col_.data(),values);
        outer_.create(part.len()+remote_,boundary_,ptr,compact_col_.data(),values);
    }
    void make_split(const int* ptr,const int* col,const double* values,const std::vector<int>& lookup) {
        namespace par=stage2_parallel;
        namespace pp=preparation_parallel;
        par::collective(mm,[&] {
            local_ptr_.resize(static_cast<std::size_t>(rows_)+1);
            remote_ptr_.resize(static_cast<std::size_t>(rows_)+1);
            std::vector<int> counts(rows_,0);
            par::ranges(rows_,[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto r=first;r<last;++r)for(int j=ptr[r];j<ptr[r+1];++j)
                    if(col[j]>=part.bgn() && col[j]<part.end())++counts[r];
            });
            local_nnz_=pp::offsets(rows_,[&](std::size_t r){return counts[r];},
                [&](std::size_t r,std::size_t offset){local_ptr_[r]=offset;},std::numeric_limits<int>::max());
            remote_nnz_=pp::offsets(rows_,[&](std::size_t r){return ptr[r+1]-ptr[r]-counts[r];},
                [&](std::size_t r,std::size_t offset){remote_ptr_[r]=offset;},std::numeric_limits<int>::max());
            // Uninitialized allocation: parallel fill provides first touch for CSR data.
            local_col_.reset(local_nnz_);local_values_.reset(local_nnz_);
            remote_col_.reset(remote_nnz_);remote_values_.reset(remote_nnz_);
            ghost_x_.resize(std::max<std::size_t>(1,remote_));
            // Contiguous row chunks preserve y ownership. Balance by local nnz;
            // clamp boundaries so empty rows and ranks also get beta exactly once.
            const int chunks=std::min(local_chunks_,std::max(1,rows_));
            chunk_begin_.push_back(0);
            for(int c=1;c<chunks;++c) {
                const auto target=local_nnz_*static_cast<std::size_t>(c)/chunks;
                int row=local_nnz_?std::lower_bound(local_ptr_.begin(),local_ptr_.end(),target)-local_ptr_.begin()
                                  :static_cast<long long>(rows_)*c/chunks;
                row=std::max(chunk_begin_.back()+1,std::min(row,rows_-(chunks-c)));
                chunk_begin_.push_back(row);
            }
            chunk_begin_.push_back(rows_);
            // First touch each local chunk with the SAME full worker team that
            // will execute its MKL multiply. Whole-matrix first touch would place
            // individual chunks on only a subset of NUMA domains.
            const int nt=par::workers(rows_);
            for(int c=0;c<chunks;++c) {
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
                for(int r=chunk_begin_[c];r<chunk_begin_[c+1];++r) {
                    int l=local_ptr_[r];
                    for(int j=ptr[r];j<ptr[r+1];++j)if(col[j]>=part.bgn() && col[j]<part.end()) {
                        local_col_[l]=col[j]-part.bgn();local_values_[l++]=values[j];
                    }
                }
            }
            // Remote CSR is one full-row multiply, so use whole-row first touch.
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
            for(int r=0;r<rows_;++r) {
                int g=remote_ptr_[r];
                for(int j=ptr[r];j<ptr[r+1];++j)if(col[j]<part.bgn() || col[j]>=part.end()) {
                    remote_col_[g]=lookup[col[j]]-part.len();remote_values_[g++]=values[j];
                }
            }
            for(int c=0;c<chunks;++c) {
                auto handle=std::make_unique<Handle>();
                std::vector<int> rows(chunk_begin_[c+1]-chunk_begin_[c]);
                std::iota(rows.begin(),rows.end(),chunk_begin_[c]);
                handle->create(part.len(),rows,local_ptr_.data(),local_col_.p(),local_values_.p());
                local_handles_.push_back(std::move(handle));
            }
            std::vector<int> rows(rows_);std::iota(rows.begin(),rows.end(),0);
            remote_handle_.create(remote_,rows,remote_ptr_.data(),remote_col_.p(),remote_values_.p());
            if(mode_==0 && local_chunks_>1 && !adaptive_) {
                single_local_=std::make_unique<Handle>();
                single_local_->create(part.len(),rows,local_ptr_.data(),local_col_.p(),local_values_.p());
            }
        });
    }
    void make_stream(const int* ptr,const int* col,const double* values,const std::vector<int>& lookup) {
        stage2_parallel::collective(mm,[&] {
            const bool reserve=mm.np()>1 && thread_budget()>1;
            stream_local_.create(rows_,ptr,col,values,part.bgn(),part.end(),lookup,false,
                std::max(1,thread_budget()-(reserve?1:0)),reserve);
            stream_remote_.create(rows_,ptr,col,values,part.bgn(),part.end(),lookup,true,thread_budget());
            stream_ghost_.resize(std::max<std::size_t>(1,remote_));
        });
    }
    void ensure_halo() {
        if(halo_ready_ || mm.np()==1)return;
        const double begin=MPI_Wtime();
        make_halo(source_ptr_,source_col_,source_values_);
        halo_map_time_+=MPI_Wtime()-begin;halo_ready_=true;
    }
    void make_halo_candidate() {
        namespace par=stage2_parallel;namespace pp=preparation_parallel;
        par::collective(mm,[&] {
            compact_x_.resize(std::max<std::size_t>(1,part.len()+remote_));
            compact_col_.resize(source_ptr_[rows_]);
            const auto parts=par::workers(rows_);
            std::vector<std::vector<int>> inner(parts),outer(parts);
            par::ranges(rows_,[&](std::size_t first,std::size_t last,std::size_t worker) {
                for(auto r=first;r<last;++r) {
                    bool local=true;
                    for(int j=source_ptr_[r];j<source_ptr_[r+1];++j) {
                        const int c=source_col_[j];
                        if(c>=part.bgn() && c<part.end())compact_col_[j]=c-part.bgn();
                        else {local=false;compact_col_[j]=halo_lookup_[c];}
                    }
                    (local?inner[worker]:outer[worker]).push_back(r);
                }
            });
            const auto concatenate=[&](const std::vector<std::vector<int>>& chunks,
                                       std::vector<int>& output) {
                std::vector<std::size_t> start(chunks.size()+1);
                pp::offsets(chunks.size(),[&](std::size_t i){return chunks[i].size();},
                    [&](std::size_t i,std::size_t offset){start[i]=offset;});
                output.resize(start.back());
                par::for_each(chunks.size(),[&](std::size_t i) {
                    std::copy(chunks[i].begin(),chunks[i].end(),output.begin()+start[i]);
                });
            };
            concatenate(inner,interior_);concatenate(outer,boundary_);
            inner_y_.resize(std::max<std::size_t>(1,interior_.size()));
            outer_y_.resize(std::max<std::size_t>(1,boundary_.size()));
            inner_.create(part.len()+remote_,interior_,source_ptr_,compact_col_.data(),source_values_);
            outer_.create(part.len()+remote_,boundary_,source_ptr_,compact_col_.data(),source_values_);
        });
    }
    void ensure_candidate(int mode) {
        if(built_[mode])return;
        if(mode!=1)ensure_halo();
        if(mode==4)ensure_candidate(3);
        const double begin=MPI_Wtime();
        if(mode==1) {
            stage2_parallel::collective(mm,[&] {
                std::vector<int> all(rows_);std::iota(all.begin(),all.end(),0);
                full_.create(dim_,all,source_ptr_,source_col_,source_values_);
                if(mm.np()>1)full_x_.resize(std::max(1,dim_));
            });
        }else if(mode==2)make_halo_candidate();
        else if(mode==3)make_split(source_ptr_,source_col_,source_values_,halo_lookup_);
        else if(mode==4) {
            stage2_parallel::collective(mm,[&] {
                std::vector<int> all(rows_);std::iota(all.begin(),all.end(),0);
                single_local_=std::make_unique<Handle>();
                single_local_->create(part.len(),all,local_ptr_.data(),local_col_.p(),local_values_.p());
            });
        }else if(mode==5)make_stream(source_ptr_,source_col_,source_values_,halo_lookup_);
        else throw std::logic_error("Invalid adaptive CSR candidate");
        candidate_build_[mode]+=MPI_Wtime()-begin;built_[mode]=true;
        candidate_build_max_[mode]=collective_max(candidate_build_[mode]);
    }
    double measure_candidate(int mode,const VecReal& x,VecReal& scratch) {
        const double begin=MPI_Wtime();
        execute(x,scratch,1.,0.,mode,false);++tune_calls_;
        candidate_warmup_[mode]+=MPI_Wtime()-begin;
        mpi_check(MPI_Barrier(mm.cm()));
        const double sample=MPI_Wtime();
        execute(x,scratch,1.,0.,mode,false);++tune_calls_;
        const double elapsed=MPI_Wtime()-sample;
        candidate_score_[mode]=collective_max(elapsed);
        stage2_parallel::collective(mm,[&] {
            candidate_observations_[mode].push_back(candidate_score_[mode]);
        });
        candidate_trial_[mode]+=MPI_Wtime()-begin;
        ++candidate_samples_[mode];++tested_candidates_;
        return candidate_score_[mode];
    }
    double refine_candidate(int mode,const VecReal& x,VecReal& scratch) {
        const double begin=MPI_Wtime();
        for(unsigned k=candidate_samples_[mode];k<static_cast<unsigned>(tune_);++k) {
            mpi_check(MPI_Barrier(mm.cm()));
            const double sample=MPI_Wtime();execute(x,scratch,1.,0.,mode,false);++tune_calls_;
            const double value=collective_max(MPI_Wtime()-sample);
            stage2_parallel::collective(mm,[&]{candidate_observations_[mode].push_back(value);});
            ++candidate_samples_[mode];
        }
        auto& samples=candidate_observations_[mode];std::sort(samples.begin(),samples.end());
        const auto mid=samples.size()/2;
        candidate_score_[mode]=samples.size()%2?samples[mid]:.5*(samples[mid-1]+samples[mid]);
        candidate_trial_[mode]+=MPI_Wtime()-begin;
        return candidate_score_[mode];
    }
    void choose_adaptive(const VecReal& x) {
        const double begin=MPI_Wtime();
        features_=csr_adaptive_cache::describe(mm,dim_,
            // Use borrowed arrays through the existing vector-independent views.
            *source_ptr_view_,*source_col_view_,local_chunks_,tune_);
        stage2_parallel::collective(mm,[&] {
            if(mm && cache_entries_)suggestion_=csr_adaptive_cache::lookup(features_,cache_reuse_);
        });
        int matched=suggestion_.matched;
        mpi_check(MPI_Bcast(&matched,1,MPI_INT,mm.ms(),mm.cm()));suggestion_.matched=matched;
        mpi_check(MPI_Bcast(suggestion_.apply.data(),6,MPI_DOUBLE,mm.ms(),mm.cm()));
        mpi_check(MPI_Bcast(suggestion_.build.data(),6,MPI_DOUBLE,mm.ms(),mm.cm()));
        mpi_check(MPI_Bcast(suggestion_.measured.data(),6,MPI_UNSIGNED,mm.ms(),mm.cm()));
        mpi_check(MPI_Bcast(&suggestion_.reuse,1,MPI_UNSIGNED_LONG_LONG,mm.ms(),mm.cm()));
        features_time_+=MPI_Wtime()-begin;
        std::vector<int> candidates;
        stage2_parallel::collective(mm,[&] {
            candidates=mm.np()==1?std::vector<int>{1,5}:std::vector<int>{1,2,3,5};
            if(mm.np()>1 && local_chunks_>1)candidates.push_back(4);
        });
        int order[5]={},count=static_cast<int>(candidates.size());
        stage2_parallel::collective(mm,[&] {if(mm) {
            // Structural heuristic affects order only. It never skips validation
            // or installs a previous matrix's communication/data structures.
            const int preferred=exact_hint_?exact_hint_:
                (mm.np()>1 && features_.shape[4]<1.-1./mm.np()?5:1);
            const int challenger=preferred==1?5:1;
            std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b) {
                const auto cost=[&](int mode) {
                    if(suggestion_.measured[mode])
                        return suggestion_.build[mode]+expected_calls_*suggestion_.apply[mode];
                    return std::numeric_limits<double>::infinity();
                };
                if(cost(a)!=cost(b))return cost(a)<cost(b);
                if(a==preferred || b==preferred)return a==preferred && b!=preferred;
                if(a==challenger || b==challenger)return a==challenger && b!=challenger;
                return a<b;
            });
            for(int i=0;i<count;++i)order[i]=candidates[i];
        }});
        mpi_check(MPI_Bcast(order,count,MPI_INT,mm.ms(),mm.cm()));
        VecReal scratch;stage2_parallel::collective(mm,[&]{scratch.reset(rows_);});
        const int first=order[0];ensure_candidate(first);
        double best=measure_candidate(first,x,scratch);mode_=first;
        const double exploration_start=MPI_Wtime();
        double exploration_spent=0.;
        // The budget applies to additional exploration. The first valid backend
        // and its short current-matrix confirmation are unavoidable baseline work.
        const double budget=tune_budget_fraction_*static_cast<double>(expected_calls_)*best;
        for(int i=1;i<count;++i) {
            const int candidate=order[i];
            int explore=0;
            if(mm) {
                const bool known=suggestion_.measured[candidate]>0;
                const double predicted_apply=known?suggestion_.apply[candidate]:best*.5;
                double predicted_build=known?suggestion_.build[candidate]:candidate_build_max_[first];
                if(candidate==4) {
                    if(known && built_[3])predicted_build=std::max(0.,predicted_build-suggestion_.build[3]);
                    if(!known && !built_[3])predicted_build+=candidate_build_max_[first];
                }
                if(built_[candidate])predicted_build=0.;
                // Half the current time is an optimistic search prior, not an
                // assumed performance result; every accepted candidate is measured.
                const double possible_gain=static_cast<double>(expected_calls_)*
                    std::max(0.,best-predicted_apply);
                const double search_cost=predicted_build+2.*predicted_apply;
                const bool new_case=!suggestion_.matched || !suggestion_.measured[mode_];
                const bool first_alternative=i==1 && new_case;
                // A new case tests a competitive alternative when its call budget
                // can amortize even an optimistic improvement. History is advisory.
                explore=(possible_gain>search_cost) &&
                    (first_alternative || exploration_spent+search_cost<=budget);
            }
            mpi_check(MPI_Bcast(&explore,1,MPI_INT,mm.ms(),mm.cm()));
            if(!explore)continue;
            ensure_candidate(candidate);
            double value=measure_candidate(candidate,x,scratch);
            // A short sample is noisy. Close contenders get the configured
            // median sample count when its extra measurements can be amortized.
            const double close_scale=std::max(best,value);
            int refine=0;
            if(mm) {
                const double extra=(tune_-1)*(best+value);
                const double potential=static_cast<double>(expected_calls_)*std::abs(best-value);
                refine=tune_>1 && std::abs(best-value)<=.1*close_scale &&
                    extra<std::max(potential,budget-exploration_spent);
            }
            mpi_check(MPI_Bcast(&refine,1,MPI_INT,mm.ms(),mm.cm()));
            if(refine) {
                best=refine_candidate(mode_,x,scratch);
                value=refine_candidate(candidate,x,scratch);
            }
            // Search cost is now sunk: choose the measured faster valid backend.
            // Whether to spend it was decided from the remaining-call objective.
            if(value<best){best=value;mode_=candidate;}
            exploration_spent=collective_max(MPI_Wtime()-exploration_start);
        }
        // The master supplies one decision, even in the presence of noisy clocks.
        mpi_check(MPI_Bcast(&mode_,1,MPI_INT,mm.ms(),mm.cm()));
        trial_gather_=candidate_score_[1];trial_halo_=candidate_score_[2];
        trial_split_=candidate_score_[3];trial_single_=candidate_score_[4];
        trial_stream_=candidate_score_[5];
        auto cache_build=candidate_build_max_;
        if(built_[4])cache_build[4]+=candidate_build_max_[3];
        stage2_parallel::collective(mm,[&] {
            if(mm)csr_adaptive_cache::remember(features_,suggestion_,candidate_score_,
                cache_build,candidate_samples_,cache_entries_);
        });
        double child=features_time_+halo_map_time_;
        for(int i=1;i<=5;++i)child+=candidate_build_[i]+candidate_trial_[i];
        choose_logic_=std::max(0.,MPI_Wtime()-begin-child);
    }
    void release_split() {
        local_handles_.clear();single_local_.reset();remote_handle_.clear();
        std::vector<int>().swap(local_ptr_);std::vector<int>().swap(remote_ptr_);
        local_col_.reset();remote_col_.reset();
        local_values_.reset();remote_values_.reset();
        std::vector<double>().swap(ghost_x_);
    }
    void execute(const VecReal& x,VecReal& y,double alpha,double beta,int mode,bool record) {
        double pack=0.,comm=0.,compute=0.;
        const bool detail=record && timing_;
        double post=0.,interior=0.,wait=0.,boundary=0.,scatter=0.,progress=0.;
        unsigned long long tests=0;
        double overlap=0.,overlap_mpi=0.;
        if(mode==1){
            const double start=MPI_Wtime();
            if(mm.np()==1){
                // No collective or copy for a purely local multiply.
            }else{
                lanczos_timing::Scope gather(lanczos_timing::VectorAllgatherv);
                mpi_check(MPI_Allgatherv(x.p(),part.len(),MPI_DOUBLE,full_x_.data(),part.len_p(),part.dsp_p(),MPI_DOUBLE,mm.cm()));
            }
            comm=MPI_Wtime()-start;
            const double mv=MPI_Wtime();
            {lanczos_timing::Scope timer(lanczos_timing::MklMultiply);full_.mv(mm.np()==1?x.p():full_x_.data(),y.p(),alpha,beta);}
            compute=MPI_Wtime()-mv;
        }else if(mode==5){
            if(mm.np()==1) {
                const double start=MPI_Wtime();
                stream_local_.mv(x.p(),y.p(),alpha,beta);
                compute=MPI_Wtime()-start;
            }else{
                double start=MPI_Wtime();
                const int pack_threads=lanczos_local::workers(send_total_);
#pragma omp parallel for schedule(static) num_threads(pack_threads) if(pack_threads > 1)
                for(std::size_t i=0;i<send_total_;++i)send_x_[i]=x[send_index_[i]];
                pack=MPI_Wtime()-start;
                MPI_Request request=MPI_REQUEST_NULL;
                start=MPI_Wtime();
                mpi_check(MPI_Ialltoallv(send_x_.data(),send_count_.data(),send_disp_.data(),MPI_DOUBLE,
                    stream_ghost_.data(),recv_count_.data(),recv_disp_.data(),MPI_DOUBLE,mm.cm(),&request));
                post=MPI_Wtime()-start;
                start=MPI_Wtime();
                const int tasks=stream_local_.slots();
                const int nt=std::min(thread_budget(),tasks+1);
                int rc=MPI_SUCCESS;
#pragma omp parallel num_threads(nt) shared(request,rc,overlap_mpi)
                {
                    const int tid=omp_get_thread_num(),team=omp_get_num_threads();
                    if(team==1) {
                        for(int task=0;task<tasks;++task)stream_local_.slot(task,x.p(),y.p(),alpha,beta);
                    }else if(tid!=0) {
                        for(int task=tid-1;task<tasks;task+=team-1)
                            stream_local_.slot(task,x.p(),y.p(),alpha,beta);
                    }
                    // OpenMP primary thread is the MPI-initializing thread.
                    // No MPI calls or timers in workers; thread budget includes it.
                    if(tid==0) {
                        const double begin=MPI_Wtime();
                        rc=MPI_Wait(&request,MPI_STATUS_IGNORE);
                        overlap_mpi=MPI_Wtime()-begin;
                    }
                }
                mpi_check(rc);
                overlap=MPI_Wtime()-start; // includes both compute and MPI: count ONCE
                start=MPI_Wtime();
                if(stream_remote_.nnz())stream_remote_.mv(stream_ghost_.data(),y.p(),alpha,1.);
                boundary=MPI_Wtime()-start;
                comm=post;compute=overlap+boundary;
            }
        }else if(mode>=3){
            double start=MPI_Wtime();
            {lanczos_timing::Scope timer(lanczos_timing::HaloPack);
            const int nt=lanczos_local::workers(send_total_);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
            for(std::size_t i=0;i<send_total_;++i)send_x_[i]=x[send_index_[i]];}
            pack=MPI_Wtime()-start;
            MPI_Request request=MPI_REQUEST_NULL;
            start=MPI_Wtime();
            {lanczos_timing::Scope timer(lanczos_timing::HaloPost);
            mpi_check(MPI_Ialltoallv(send_x_.data(),send_count_.data(),send_disp_.data(),MPI_DOUBLE,
                ghost_x_.data(),recv_count_.data(),recv_disp_.data(),MPI_DOUBLE,mm.cm(),&request));}
            post=MPI_Wtime()-start;
            int complete=0;
            // MPI calls stay on the initializing thread (MPI_THREAD_FUNNELED).
            // Test between MKL calls drives progress without a new MPI thread.
            const std::size_t chunks=mode==4?1:local_handles_.size();
            for(std::size_t c=0;c<chunks;++c) {
                start=MPI_Wtime();
                {lanczos_timing::Scope timer(lanczos_timing::MklMultiply);
                const Handle& handle=mode==4?*single_local_:*local_handles_[c];
                const int row=mode==4?0:chunk_begin_[c];
                handle.mv(x.p(),rows_?y.p()+row:y.p(),alpha,beta);}
                interior+=MPI_Wtime()-start;
                if(c+1<chunks && !complete) {
                    start=MPI_Wtime();
                    mpi_check(MPI_Test(&request,&complete,MPI_STATUS_IGNORE));
                    progress+=MPI_Wtime()-start;++tests;
                }
            }
            start=MPI_Wtime();
            {lanczos_timing::Scope timer(lanczos_timing::HaloWait);
            mpi_check(MPI_Wait(&request,MPI_STATUS_IGNORE));}
            wait=MPI_Wtime()-start;
            start=MPI_Wtime();
            {lanczos_timing::Scope timer(lanczos_timing::MklMultiply);
            if(remote_nnz_)remote_handle_.mv(ghost_x_.data(),y.p(),alpha,1.);}
            boundary=MPI_Wtime()-start;
            comm=post+progress+wait;compute=interior+boundary;
        }else{
            double start=MPI_Wtime();
            {lanczos_timing::Scope pack_timer(lanczos_timing::HaloPack);
            const int nt=lanczos_local::workers(std::max<std::size_t>(part.len(),send_total_));
#pragma omp parallel num_threads(nt) if(nt > 1)
            {
#pragma omp for schedule(static) nowait
                for(int i=0;i<part.len();++i)compact_x_[i]=x[i];
#pragma omp for schedule(static)
                for(std::size_t i=0;i<send_total_;++i)send_x_[i]=x[send_index_[i]];
            }
            }
            pack=MPI_Wtime()-start;
            MPI_Request request=MPI_REQUEST_NULL;
            start=MPI_Wtime();
            {lanczos_timing::Scope post_timer(lanczos_timing::HaloPost);
            mpi_check(MPI_Ialltoallv(send_x_.data(),send_count_.data(),send_disp_.data(),MPI_DOUBLE,
                compact_x_.data()+part.len(),recv_count_.data(),recv_disp_.data(),MPI_DOUBLE,mm.cm(),&request));}
            post=MPI_Wtime()-start;comm+=post;
            start=MPI_Wtime();
            {lanczos_timing::Scope interior_timer(lanczos_timing::HaloInterior);
            lanczos_timing::Scope timer(lanczos_timing::MklMultiply);inner_.mv(compact_x_.data(),inner_y_.data());}
            interior=MPI_Wtime()-start;compute+=interior;
            start=MPI_Wtime();
            {lanczos_timing::Scope wait_timer(lanczos_timing::HaloWait);
            mpi_check(MPI_Wait(&request,MPI_STATUS_IGNORE));}
            wait=MPI_Wtime()-start;comm+=wait;
            start=MPI_Wtime();
            {lanczos_timing::Scope boundary_timer(lanczos_timing::HaloBoundary);
            lanczos_timing::Scope timer(lanczos_timing::MklMultiply);outer_.mv(compact_x_.data(),outer_y_.data());}
            if(detail)boundary=MPI_Wtime()-start;
            {lanczos_timing::Scope scatter_timer(lanczos_timing::HaloScatter);
            const int nrows=rows_,ni=interior_.size(),workers=lanczos_local::workers(rows_);
#pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1)
            for(int j=0;j<nrows;++j){
                const int row=j<ni?interior_[j]:boundary_[j-ni];
                const double value=j<ni?inner_y_[j]:outer_y_[j-ni];
                y[row]=alpha*value+(beta==0.?0.:beta*y[row]);
            }
            }
            const double tail=MPI_Wtime()-start;
            compute+=tail;
            if(detail)scatter=tail-boundary;
        }
        if(record){
            pack_+=pack;exchange_+=comm;compute_+=compute;progress_tests_+=tests;
            if(detail){
                if(mode==1 || (mode==5 && mm.np()==1)){if(mode==1 && mm.np()>1)allgather_+=comm;local_mv_+=compute;}
                else {post_+=post;interior_mv_+=interior;wait_+=wait;boundary_mv_+=boundary;scatter_+=scatter;progress_+=progress;overlap_+=overlap;overlap_mpi_+=overlap_mpi;}
            }
            if(lanczos_timing::active){
                auto* timing=lanczos_timing::active;
                timing->add(lanczos_timing::HvGather,timing->state,comm);
                timing->add(lanczos_timing::HvBuffers,timing->state,pack);
                timing->add(lanczos_timing::HvSparse,timing->state,compute);
            }
        }
    }
    void choose(const VecReal& x) {
        if(chosen_)return;
        lanczos_timing::Scope tune_timer(lanczos_timing::HvTune);
        const double start=MPI_Wtime();
        lanczos_timing::Session exclude_calibration(false);
        if(adaptive_)choose_adaptive(x);
        else if(mode_==0){
            VecReal scratch(rows_,0.);
            // Candidate IDs describe algorithms, never rank counts.
            std::vector<int> candidates;
            if(mm.np()==1)candidates={1,5};
            else {candidates={1,2,3};if(single_local_)candidates.push_back(4);candidates.push_back(5);}
            std::vector<double> samples[5];
            stage2_parallel::collective(mm,[&] {
                for(int mode:candidates)samples[mode-1].reserve(tune_);
            });
            tested_candidates_=static_cast<int>(candidates.size());
            for(int mode:candidates){execute(x,scratch,1.,0.,mode,false);++tune_calls_;}
            for(int k=0;k<tune_;++k) {
                double timings[5]={},maximum[5]={};
                for(std::size_t order=0;order<candidates.size();++order) {
                    const int mode=candidates[(k+order)%candidates.size()];
                    mpi_check(MPI_Barrier(mm.cm()));
                    const double begin=MPI_Wtime();execute(x,scratch,1.,0.,mode,false);
                    timings[mode-1]=MPI_Wtime()-begin;++tune_calls_;
                }
                // Calibration only: include the slowest rank of EACH sample.
                mpi_check(MPI_Allreduce(timings,maximum,5,MPI_DOUBLE,MPI_MAX,mm.cm()));
                for(int mode:candidates)samples[mode-1].push_back(maximum[mode-1]);
            }
            double score[5]={};
            for(int mode:candidates) {
                auto& sample=samples[mode-1];std::sort(sample.begin(),sample.end());
                const auto mid=sample.size()/2;
                score[mode-1]=sample.size()%2?sample[mid]:.5*(sample[mid-1]+sample[mid]);
            }
            trial_gather_=score[0];trial_halo_=score[1];
            trial_split_=score[2];trial_single_=score[3];trial_stream_=score[4];
            mode_=*std::min_element(candidates.begin(),candidates.end(),
                [&](int a,int b){return score[a-1]<score[b-1];});
        }
        if(adaptive_)stage2_parallel::collective(mm,[&] {
            if(!choice_key_.empty() && mm)csr_choice_cache::remember(choice_key_,mode_,cache_entries_);
        });
        else if(!choice_cache_hit_ && !choice_key_.empty() && mm)
            csr_choice_cache::remember(choice_key_,mode_,cache_entries_);
        if(mode_==4) {
            local_handles_.clear();local_handles_.push_back(std::move(single_local_));
            chunk_begin_.assign({0,rows_});mode_=3;
        }
        single_local_.reset();
        chosen_=true;tuning_=MPI_Wtime()-start;
        const double release_begin=MPI_Wtime();
        // Release the unused candidate's buffers/handles after calibration.
        if(mode_!=1){full_.clear();std::vector<double>().swap(full_x_);}
        if(mode_!=2){
            inner_.clear();outer_.clear();
            std::vector<int>().swap(compact_col_);
            std::vector<double>().swap(compact_x_);
            std::vector<double>().swap(inner_y_);std::vector<double>().swap(outer_y_);
        }
        if(mode_!=3)release_split();
        if(mode_!=5){stream_local_.clear();stream_remote_.clear();std::vector<double>().swap(stream_ghost_);}
        if(mode_==1){std::vector<int>().swap(send_index_);std::vector<double>().swap(send_x_);}
        std::vector<int>().swap(halo_lookup_);
        release_time_=MPI_Wtime()-release_begin;
        const double report_begin=MPI_Wtime();
        if(report_&&mm)std::cout<<"HV_PLAN: dim="<<dim_<<" ranks="<<mm.np()<<" threads="<<thread_budget()
            <<" mode="<<mode_name()<<" cache_hit="<<choice_cache_hit_
            <<" policy="<<(adaptive_?"adaptive":automatic_?"full":"fixed")
            <<" expected_calls_hint="<<expected_calls_<<" tested_candidates="<<tested_candidates_
            <<" tune_calls="<<tune_calls_<<" feature_hit="<<suggestion_.matched
            <<" tune_stat="<<(adaptive_?"short_or_refined_median_rank_max":"median_rank_max")<<" tune_allgather_s="<<trial_gather_<<" tune_halo_s="<<trial_halo_<<" tune_split_s="<<trial_split_
            <<" tune_stream_s="<<trial_stream_<<" tune_split_single_s="<<trial_single_<<" local_chunks="<<local_handles_.size()<<std::endl;
        plan_report_time_=MPI_Wtime()-report_begin;
    }
public:
    DistributedCsrReal(const MyMpi& mpi,int rows,int dim,const VecInt& ptr,const VecInt& col,const VecReal& values,
                       unsigned long long expected_calls=0)
      :mm(mpi),part(mpi.np(),mpi.id(),dim),rows_(rows),dim_(dim),expected_calls_(expected_calls),
       source_ptr_view_(&ptr),source_col_view_(&col),source_ptr_(ptr.p()),source_col_(col.p()),
       source_values_(values.p()){
        static_assert(std::is_same<Int,MKL_INT>::value,"CSR requires matching MPI/MKL LP64 integer indices");
        const double start=MPI_Wtime();
        namespace pp=preparation_parallel;
        bool valid=rows==part.len()&&ptr.size()==static_cast<Idx>(rows+1)&&col.size()==values.size();
        if(valid)valid=ptr[0]==0 && ptr[rows]>=0 && static_cast<Idx>(ptr[rows])==values.size();
        collective_require(valid,"Invalid fixed CSR or MPI row partition");
        plan_id_=++next_plan_;nnz_=values.size();
        stage2_parallel::collective(mm,[&] {
            if(!pp::all(rows,[&](std::size_t r){return ptr[r]>=0&&ptr[r]<=ptr[r+1];}) ||
               !pp::all(col.size(),[&](std::size_t j){return col[j]>=0&&col[j]<dim;}))
                throw std::runtime_error("Invalid fixed CSR indices");
        });
        read_config();read_policy();
        config_validation_=MPI_Wtime()-start;
        reuse_choice(ptr,col);
        if(!adaptive_ && (mode_==0 || mode_==1)){
            std::vector<int> all(rows);
            stage2_parallel::ranges(rows,[&](std::size_t first,std::size_t last,std::size_t) {
                for(auto i=first;i<last;++i)all[i]=i;
            });
            full_.create(dim,all,ptr.p(),col.p(),values.p());
            if(mm.np()>1)full_x_.resize(std::max(1,dim));
        }
        if(!adaptive_ && mode_!=1) {
            if(mm.np()>1)make_halo(ptr.p(),col.p(),values.p());
            else make_stream(ptr.p(),col.p(),values.p(),{});
        }
        setup_=MPI_Wtime()-start;
    }
    DistributedCsrReal(const DistributedCsrReal&)=delete;
    const char* mode_name()const{return mode_==5?"stream":mm.np()==1?"local":mode_==2?"halo":mode_==3?"split":"allgather";}
    std::size_t remote_count()const{return remote_;}
    void apply(const VecReal& x,VecReal& y,double alpha,double beta){
        if(x.size()!=static_cast<Idx>(part.len()))throw std::invalid_argument("CSR multiply requires the rank-local wavefunction slice");
        if(x.p()==y.p() && x.size())throw std::invalid_argument("CSR input/output buffers must not alias");
        if(y.size()!=x.size()){if(beta!=0.)throw std::invalid_argument("CSR accumulation output size mismatch");y.reset(x.size());}
        choose(x);const double start=MPI_Wtime();execute(x,y,alpha,beta,mode_,true);
        seconds_+=MPI_Wtime()-start;++calls_;
    }
    ~DistributedCsrReal(){
        if(report_&&calls_){
            const double format_begin=MPI_Wtime();
            std::ostringstream out;out<<"HV_PROFILE: rank="<<mm.id()<<" dim="<<dim_<<" rows="<<rows_
              <<" mode="<<mode_name()<<" calls="<<calls_<<" needed_remote="<<remote_
              <<" received_doubles_per_call="<<(mm.np()==1?0:mode_!=1?remote_:dim_-part.len())
              <<" halo_sent_doubles="<<send_total_<<" interior_rows="<<interior_.size()
              <<" setup_s="<<setup_<<" tuning_s="<<tuning_<<" apply_s="<<seconds_
              <<" pack_s="<<pack_<<" exposed_comm_s="<<exchange_<<" compute_s="<<compute_<<" plan="<<plan_id_<<'\n';
            if(timing_){
                const double stages=pack_+allgather_+local_mv_+post_+interior_mv_+wait_+boundary_mv_+scatter_+progress_+overlap_;
                // Per-rank only: destruction is not guaranteed collective. In
                // particular, do not add a gather/barrier to order these lines.
                out.precision(12);
                out<<"HV_DETAIL: schema=3 plan="<<plan_id_<<" rank="<<mm.id()<<" ranks="<<mm.np()
                   <<" dim="<<dim_<<" rows="<<rows_<<" nnz="<<nnz_<<" threads="<<thread_budget()
                   <<" mode="<<mode_name()<<" calls="<<calls_<<" clock=MPI_Wtime measurement=rank_wall"
                   <<" exclusive=1 reporting_excluded=1 calibration_excluded=1 instrumentation_included=1"
                   <<" extra_clock_reads="<<(mode_==2?calls_:0)
                   <<" received_doubles_per_call="<<(mm.np()==1?0:mode_!=1?remote_:dim_-part.len())
                   <<" interior_rows="<<(mm.np()==1?rows_:mode_==2?interior_.size():0)
                   <<" local_chunks="<<local_handles_.size()<<" local_nnz="<<(mode_==5?stream_local_.nnz():local_nnz_)<<" remote_nnz="<<(mode_==5?stream_remote_.nnz():remote_nnz_)
                   <<" split_local_mv_s="<<(mode_==3?interior_mv_:0.)<<" split_remote_mv_s="<<(mode_==3?boundary_mv_:0.)
                   <<" progress_tests="<<progress_tests_<<" progress_s="<<progress_
                   <<" overlap_s="<<overlap_<<" overlap_mpi_s="<<overlap_mpi_
                   <<" stream_slots="<<stream_local_.slots()<<" stream_remote_rows="<<stream_remote_.rows()
                   <<" pack_s="<<pack_<<" allgather_s="<<allgather_<<" local_mv_s="<<local_mv_
                   <<" post_s="<<post_<<" interior_mv_s="<<interior_mv_<<" wait_s="<<wait_
                   <<" boundary_mv_s="<<boundary_mv_<<" scatter_s="<<scatter_
                   <<" other_s="<<seconds_-stages<<" apply_s="<<seconds_
                   <<" mean_us="<<seconds_*1.e6/calls_<<" setup_s="<<setup_<<" tuning_s="<<tuning_<<'\n';
            }
            if(adaptive_) {
                out<<"HV_SETUP_DETAIL: schema=1 plan="<<plan_id_<<" rank="<<mm.id()
                   <<" policy=adaptive expected_calls_hint="<<expected_calls_
                   <<" config_validation_s="<<config_validation_<<" fingerprint_s="<<fingerprint_
                   <<" features_s="<<features_time_<<" halo_map_s="<<halo_map_time_
                   <<" selection_other_s="<<choose_logic_<<" release_s="<<release_time_
                   <<" plan_report_s="<<plan_report_time_
                   <<" constructor_setup_s="<<setup_<<" selection_total_s="<<tuning_
                   <<" lazy_build_nested_in_tuning=1 trial_warmup_nested_in_trial=1"
                   <<" setup_tuning_exclusive=1 stages_rank_wall=1 hot_path_extra_collectives=0"
                   <<" tested_candidates="<<tested_candidates_<<" tune_calls="<<tune_calls_<<'\n';
                if(mm)for(int candidate=1;candidate<=5;++candidate)if(built_[candidate])
                    out<<"HV_CANDIDATE: plan="<<plan_id_<<" candidate="<<candidate
                       <<" build_rank_s="<<candidate_build_[candidate]
                       <<" build_max_s="<<candidate_build_max_[candidate]
                       <<" trial_rank_s="<<candidate_trial_[candidate]
                       <<" warmup_nested_s="<<candidate_warmup_[candidate]
                       <<" measured_rank_max_s="<<candidate_score_[candidate]
                       <<" samples="<<candidate_samples_[candidate]<<'\n';
            }
            const auto report=out.str();
            const double format_elapsed=MPI_Wtime()-format_begin;
            const double report_begin=MPI_Wtime();
            std::cout.write(report.data(),report.size());
            if(adaptive_) {
                std::cout.flush();
                const double report_elapsed=MPI_Wtime()-report_begin;
                std::cout<<"HV_REPORT_COST: plan="<<plan_id_<<" rank="<<mm.id()
                    <<" summary_format_s="<<format_elapsed
                    <<" summary_write_s="<<report_elapsed<<" self_report_excluded=1"
                    <<" per_apply_output=0"<<std::endl;
            }
        }
    }
};
