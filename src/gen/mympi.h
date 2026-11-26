/*
code developed by
    Rong-Qiang He (rqhe@ruc.edu.cn, RUC, China) date 2013 - 2017
	Jia-Ming Wang (jmw@ruc.edu.cn, RUC, China) date 2020 - 2024
*/

#ifndef _MYMPI_H_
#define _MYMPI_H_

#include "stdpfx.h"
#include "mpi.h"
#include "vec.h"
#include "myomp.h"
#include "safebool.h"
#include "lanczos_timing.h"
#include "full_vector_gather.h"

#include "basis_index.h"

class VecPartition {
    int np_, id_;
    BasisIndex size_;
    std::vector<BasisIndex> starts_;
    std::vector<int> small_starts_, small_lengths_;

    void finish() {
        if (starts_.back() != size_) throw std::invalid_argument("Partition size mismatch");
        if (size_ <= std::numeric_limits<int>::max()) {
            small_starts_.assign(starts_.begin(), starts_.end());
            small_lengths_.resize(np_);
            for (int i=0; i<np_; ++i) small_lengths_[i] = small_starts_[i+1] - small_starts_[i];
        }
    }
    void validate() const {
        if (np_ <= 0 || id_ < 0 || id_ >= np_ || size_ < 0)
            throw std::invalid_argument("Invalid vector partition");
    }
    void require_small() const {
        if (small_starts_.empty()) throw std::overflow_error("MPI-3 partition requires chunked counts");
    }
public:
    VecPartition(int np, int id, BasisIndex n): np_(np), id_(id), size_(n) {
        validate();
        starts_.resize(static_cast<std::size_t>(np)+1);
        for (int i=0; i<np; ++i)
            starts_[i+1] = starts_[i] + n/np + (i < n%np);
        finish();
    }
    template<class I>
    VecPartition(int np, int id, BasisIndex n, const Vec<I>& sizes): np_(np), id_(id), size_(n) {
        validate();
        if (sizes.size()!=static_cast<Idx>(np))
            throw std::invalid_argument("Wrong number of partition sizes");
        starts_.resize(static_cast<std::size_t>(np)+1);
        for (int i=0; i<np; ++i) {
            if (sizes[i]<0 || static_cast<std::uint64_t>(sizes[i]) > static_cast<std::uint64_t>(n-starts_[i]))
                throw std::invalid_argument("Invalid partition length");
            starts_[i+1] = starts_[i] + sizes[i];
        }
        finish();
    }
    BasisIndex total_size() const { return size_; }
    BasisIndex bgn(int rank) const { return starts_[rank]; }
    BasisIndex end(int rank) const { return starts_[rank+1]; }
    BasisIndex len(int rank) const { return end(rank)-bgn(rank); }
    BasisIndex dsp(int rank) const { return bgn(rank); }
    BasisIndex bgn() const { return bgn(id_); }
    BasisIndex end() const { return end(id_); }
    BasisIndex len() const { return len(id_); }
    BasisIndex dsp() const { return bgn(); }
    bool fits_mpi_int() const { return !small_starts_.empty(); }
    const int* bgn_p() const { require_small(); return small_starts_.data(); }
    const int* end_p() const { return bgn_p()+1; }
    const int* len_p() const { require_small(); return small_lengths_.data(); }
    const int* dsp_p() const { return bgn_p(); }

    // Each window is contiguous in global order and fits MPI-3 int displacements.
    template<class F> void windows(F fn) const {
        if (fits_mpi_int()) {
            fn(BasisIndex(0),BasisIndex(0),small_lengths_.data(),small_starts_.data());
            return;
        }
        std::vector<int> counts(np_), displs(np_);
        for (BasisIndex first=0; first<size_;) {
            const BasisIndex last=first+std::min<BasisIndex>(size_-first,std::numeric_limits<int>::max());
            for (int rank=0; rank<np_; ++rank) {
                const auto lo=std::max(first,bgn(rank)), hi=std::min(last,end(rank));
                counts[rank]=hi>lo?static_cast<int>(hi-lo):0;
                displs[rank]=hi>lo?static_cast<int>(lo-first):0;
            }
            fn(first,counts[id_]?std::max(first,bgn())-bgn():0,counts.data(),displs.data());
            first=last;
        }
    }
};

class MyMpi: public safe_bool<MyMpi> {
private:
	typedef std::complex<double> _Cmplx;
	MPI_Comm comm;
	int nprc;
	int myid;
	int mstr;
private:
    static MPI_Datatype mpi_type(const int*) { return MPI_INT; }
    static MPI_Datatype mpi_type(const BasisIndex*) { return MPI_INT64_T; }
    static MPI_Datatype mpi_type(const double*) { return MPI_DOUBLE; }
    static MPI_Datatype mpi_type(const _Cmplx*) { return MPI_DOUBLE_COMPLEX; }
    static MPI_Datatype mpi_type(const char*) { return MPI_CHAR; }
    int Bcast(BasisIndex* p, int n) const { return MPI_Bcast(p,n,MPI_INT64_T,mstr,comm); }
    int Allreduce(const BasisIndex* a, BasisIndex* b, int n) const { return MPI_Allreduce(a,b,n,MPI_INT64_T,MPI_SUM,comm); }
    template<class F> static int chunked(Idx n,F op) {
        Idx pos=0;
        do {
            const int count=static_cast<int>(std::min<Idx>(n-pos,std::numeric_limits<int>::max()));
            const int rc=op(pos,count);
            if(rc!=MPI_SUCCESS)return rc;
            pos+=static_cast<Idx>(count);
        } while(pos<n);
        return MPI_SUCCESS;
    }
    template<class T> Vec<T> gather_partition(const Vec<T>& send,const VecPartition& part,bool all) const {
        if(send.size()!=static_cast<Idx>(part.len()))throw std::invalid_argument("MPI partition buffer mismatch");
        if(nprc==1)return send;
        Vec<T> result(all || myid==mstr ? part.total_size() : 0);
        T dummy{};
        auto dtype=mpi_type(send.p());
        part.windows([&](BasisIndex start,BasisIndex local,const int* counts,const int* offsets){
            const T* src=counts[myid]?send.p()+local:&dummy;
            T* dst=result.size()?result.p()+start:&dummy;
            const int rc=all ? MPI_Allgatherv(src,counts[myid],dtype,dst,counts,offsets,dtype,comm)
                             : MPI_Gatherv(src,counts[myid],dtype,dst,counts,offsets,dtype,mstr,comm);
            if(rc!=MPI_SUCCESS)throw std::runtime_error("Partition gather failed");
        });
        return result;
    }
    template<class T> int send_range(const T* ptr,Idx size,int dest,int tag) const {
        return chunked(size,[&](Idx off,int n){return MPI_Send(n?ptr+off:ptr,n,mpi_type(ptr),dest,tag,comm);});
    }
    template<class T> int receive_range(T* ptr,Idx size,int source,int tag,MPI_Status& status) const {
        return chunked(size,[&](Idx off,int n){
            const int rc=MPI_Recv(n?ptr+off:ptr,n,mpi_type(ptr),source,tag,comm,&status);
            if(rc==MPI_SUCCESS){source=status.MPI_SOURCE;tag=status.MPI_TAG;}
            return rc;
        });
    }
    template<class T> void reduce_range(const T* src,T* dst,Idx size) const {
        T dummy{};
        if(chunked(size,[&](Idx off,int n){
            return MPI_Reduce(n?src+off:src,n && myid==mstr?dst+off:&dummy,n,
                              mpi_type(src),MPI_SUM,mstr,comm);
        })!=MPI_SUCCESS)throw std::runtime_error("MPI reduce failed");
    }
	int Bcast(char *buff, int count) const {
		return MPI_Bcast(buff, count, MPI_CHAR, mstr, comm);
	}
	int Bcast(int *buff, int count) const {
		return MPI_Bcast(buff, count, MPI_INT, mstr, comm);
	}
	int Bcast(double *buff, int count) const {
		return MPI_Bcast(buff, count, MPI_DOUBLE, mstr, comm);
	}
	int Bcast(_Cmplx *buff, int count) const {
		return MPI_Bcast(buff, count, MPI_DOUBLE_COMPLEX, mstr, comm);
	}

	int Allreduce(const char *sendbuf, char *recvbuf, int sendcount) const {
		return MPI_Allreduce(sendbuf, recvbuf, sendcount, MPI_CHAR, MPI_SUM, comm);
	}
	int Allreduce(const int *sendbuf, int *recvbuf, int sendcount) const {
		return MPI_Allreduce(sendbuf, recvbuf, sendcount, MPI_INT, MPI_SUM, comm);
	}
	int Allreduce(const double *sendbuf, double *recvbuf, int sendcount) const {
		lanczos_timing::Scope timer(lanczos_timing::ScalarAllreduce);
		return MPI_Allreduce(sendbuf, recvbuf, sendcount, MPI_DOUBLE, MPI_SUM, comm);
	}
	int Allreduce(const _Cmplx *sendbuf, _Cmplx *recvbuf, int sendcount) const {
		return MPI_Allreduce(sendbuf, recvbuf, sendcount, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
	}

public:
	MyMpi(MPI_Comm comm_i = MPI_COMM_WORLD, int mstr_i = 0) : comm(comm_i), mstr(mstr_i) {
		MPI_Comm_size(comm, &nprc);
		MPI_Comm_rank(comm, &myid);
		if (myid < 0 || myid >= nprc || mstr < 0 || mstr >= nprc) ERR(NAV4(comm, nprc, myid, mstr));
		display();
	}
    // Reuse the receive buffer; callers supply independent local and global arrays.
    template<class T>
    void Allreduce_into(const Vec<T>& local,Vec<T>& global) const {
        if(global.size()!=local.size())global.reset(local.size());
        if(chunked(local.size(),[&](Idx offset,int count) {
            return Allreduce(count?local.p()+offset:local.p(),count?global.p()+offset:global.p(),count);
        })!=MPI_SUCCESS)throw std::runtime_error("Allreduce into reusable buffer failed");
    }
    // Return only the equal-row segment owned by this rank; no full receive vector.
    VecReal Reduce_scatter(const VecReal& local, const VecPartition& vp) const {
        if(nprc==1) {
            if(local.size()!=static_cast<Idx>(vp.total_size()) || vp.len()!=vp.total_size())
                throw std::invalid_argument("Reduce_scatter partition/buffer mismatch");
            return local;
        }
        VecReal result;
        std::exception_ptr error;
        try {
            if(vp.total_size() < 0 || local.size() != static_cast<Idx>(vp.total_size()))
                throw std::invalid_argument("Reduce_scatter partition/buffer mismatch");
            result.reset(vp.len(), 0.);
        } catch(...) { error = std::current_exception(); }
        if(Allreduce(error ? 1 : 0)) {
            if(error) std::rethrow_exception(error);
            throw std::runtime_error("Reduce_scatter allocation failed on another rank");
        }
        Real dummy=0.;
        vp.windows([&](BasisIndex start,BasisIndex offset,const int* counts,const int*) {
            if(MPI_Reduce_scatter(local.size()?local.p()+start:&dummy,
                    counts[myid]?result.p()+offset:&dummy,counts,MPI_DOUBLE,MPI_SUM,comm)!=MPI_SUCCESS)
                throw std::runtime_error("MPI_Reduce_scatter failed");
        });
        return result;
    }
    // Lanczos output only: every rank receives the complete, ordered vector.
    VecReal Allgatherv_full(const VecReal& local, const VecPartition& vp) const {
        if(local.size()!=static_cast<Idx>(vp.len()))throw std::invalid_argument("Full-vector gather buffer mismatch");
        if(!vp.fits_mpi_int())return gather_partition(local,vp,true);
        VecReal result(vp.total_size());
        namespace lt = lanczos_timing;
        lt::Scope timer(lt::VectorAllgatherv);
        full_vector_gather::Timing measured;
        const int rc = full_vector_gather::allgather(local.p(),
            static_cast<int>(local.size()), result.p(), vp.len_p(), vp.dsp_p(),
            comm, mstr, lt::active ? &measured : nullptr);
        if(rc != MPI_SUCCESS)
            throw std::runtime_error("Full-vector gather failed: check partition, "
                "NORG_FULL_VECTOR_GATHER=ring|native and "
                "NORG_FULL_VECTOR_CHUNK_BYTES (positive MPI-count-compatible bytes)");
        if(lt::active) {
            const int state = lt::active->state;
            lt::active->add(lt::FullVectorSetup, state, measured.setup);
            lt::active->add(lt::FullVectorCopy, state, measured.copy);
            lt::active->add(measured.native ? lt::FullVectorNative : lt::FullVectorRing,
                            state, measured.exchange);
        }
        return result;
    }
	MPI_Comm cm() const { return comm; }
	int np() const { return nprc; }
	int id() const { return myid; }
	int ms() const { return mstr; }
	bool boolean_test() const { return myid == mstr; }
	int barrier() const { return MPI_Barrier(comm); }
	void print_process_names(std::ostream &os = std::cout) const;
	void display() const {
		if (*this) std::cout << "MPI: number of processes = " << np() << std::endl;
		if (*this) std::cout << "MPI: master process id = " << ms() << std::endl;
		print_process_names();
	}
public:
	template <typename T>
	int Send(const T& v, int dest, int tag) const {
		return send_range(&v,1,dest,tag);
	}
	template <typename T>
	int Send(const Vec<T>& v, int dest, int tag) const {
		return send_range(v.p(),v.size(),dest,tag);
	}
	template <typename T>
	int Send(const Mat<T>& a, int dest, int tag) const {
		return send_range(a.p(),a.size(),dest,tag);
	}
	template <typename T>
	int Recv(T& v, MPI_Status& status, int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const {
		return receive_range(&v,1,source,tag,status);
	}
	template <typename T>
	int Recv(Vec<T>& v, MPI_Status& status, int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const {
		return receive_range(v.p(),v.size(),source,tag,status);
	}
	template <typename T>
	int Recv(Mat<T>& a, MPI_Status& status, int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const {
		return receive_range(a.p(),a.size(),source,tag,status);
	}
	template <typename T>
	int Bcast(T &v) const {
		return Bcast(&v, 1);
	}
	template <typename T>
	int Bcast(Vec<T> &v) const {
		return chunked(v.size(),[&](Idx offset,int count){return Bcast(count?v.p()+offset:v.p(),count);});
	}
	template <typename T>
	int Bcast(Mat<T>& a) const {
		return chunked(a.size(),[&](Idx offset,int count){return Bcast(count?a.p()+offset:a.p(),count);});
	}
	template <typename T>
	T Reduce(const T &sendbuf) const {
		T recvbuf{};
		reduce_range(&sendbuf,&recvbuf,1);
		return recvbuf;
	}
	template <typename T>
	Vec<T> Reduce(const Vec<T> &sendbuf) const {
		Vec<T> recvbuf(*this ? sendbuf.size() : 0);
		reduce_range(sendbuf.p(),recvbuf.p(),sendbuf.size());
		return *this ? recvbuf : sendbuf;
	}
	template <typename T>
	Mat<T> Reduce(const Mat<T> &sendbuf) const {
		Mat<T> recvbuf(*this ? sendbuf.nrows() : 0, *this ? sendbuf.ncols() : 0);
		reduce_range(sendbuf.p(),recvbuf.p(),sendbuf.size());
		return *this ? recvbuf : sendbuf;
	}
	template <typename T>
	T Allreduce(const T &sendbuf) const {
		T recvbuf;
		Allreduce(&sendbuf, &recvbuf, 1);
		return recvbuf;
	}
	template <typename T>
	Vec<T> Allreduce(const Vec<T> &sendbuf) const {
		Vec<T> recvbuf(sendbuf.size());
		if(chunked(sendbuf.size(),[&](Idx offset,int count){
            return Allreduce(count?sendbuf.p()+offset:sendbuf.p(),count?recvbuf.p()+offset:recvbuf.p(),count);
        })!=MPI_SUCCESS)throw std::runtime_error("MPI allreduce failed");
		return recvbuf;
	}
	template <typename T>
	Mat<T> Allreduce(const Mat<T>& sendbuf) const {
		Mat<T> recvbuf(sendbuf.nrows(), sendbuf.ncols());
		if(chunked(sendbuf.size(),[&](Idx offset,int count){
            return Allreduce(count?sendbuf.p()+offset:sendbuf.p(),count?recvbuf.p()+offset:recvbuf.p(),count);
        })!=MPI_SUCCESS)throw std::runtime_error("MPI allreduce failed");
		return recvbuf;
	}
	template <typename T>
	Vec<T> Gatherv(const Vec<T> &sendbuf, const VecPartition &vp) const {
		return gather_partition(sendbuf,vp,false);
	}
	template <typename T>
	Vec<Mat<T>> Gatherv(const Vec<Mat<T>>& vm, Idx mat_nrows, Idx mat_ncols, const VecPartition& vp) const;

	template <typename T>
	Vec<T> Allgatherv(const Vec<T>& sendbuf, const VecPartition& vp) const {
		return gather_partition(sendbuf,vp,true);
	}
	template <typename T>
	Vec<T> Gather(const T &sendbuf) const {
		Vec<T> recvbuf(*this ? np() : 0);
		MPI_Gather(&sendbuf,1,mpi_type(&sendbuf),recvbuf.p(),1,mpi_type(&sendbuf),mstr,comm);
		return recvbuf;
	}
	template <typename T>
	Vec<T> Gather(const Vec<T> &sendbuf) const {
        if (sendbuf.size()>static_cast<Idx>(std::numeric_limits<BasisIndex>::max()/nprc))
            throw std::overflow_error("MPI gathered vector size exceeds 64-bit range");
        if(sendbuf.size()>static_cast<Idx>(std::numeric_limits<int>::max()))
            return gather_partition(sendbuf,VecPartition(nprc,myid,static_cast<BasisIndex>(sendbuf.size())*nprc),false);

		Vec<T> recvbuf((*this ? np() : 0) * sendbuf.size());
		MPI_Gather(sendbuf.p(),static_cast<int>(sendbuf.size()),mpi_type(sendbuf.p()),recvbuf.p(),static_cast<int>(sendbuf.size()),mpi_type(sendbuf.p()),mstr,comm);
		return recvbuf;
	}
	template <typename T>
	Vec<T> Allgather(const T &sendbuf) const {
		Vec<T> recvbuf(np());
		MPI_Allgather(&sendbuf,1,mpi_type(&sendbuf),recvbuf.p(),1,mpi_type(&sendbuf),comm);
		return recvbuf;
	}
	template <typename T>
	Vec<T> Allgather(const Vec<T> &sendbuf) const {
        if (sendbuf.size()>static_cast<Idx>(std::numeric_limits<BasisIndex>::max()/nprc))
            throw std::overflow_error("MPI gathered vector size exceeds 64-bit range");
        if(sendbuf.size()>static_cast<Idx>(std::numeric_limits<int>::max()))
            return gather_partition(sendbuf,VecPartition(nprc,myid,static_cast<BasisIndex>(sendbuf.size())*nprc),true);

		Vec<T> recvbuf(np() * sendbuf.size());
		MPI_Allgather(sendbuf.p(),static_cast<int>(sendbuf.size()),mpi_type(sendbuf.p()),recvbuf.p(),static_cast<int>(sendbuf.size()),mpi_type(sendbuf.p()),comm);
		return recvbuf;
	}
	int Allreduce_MIN(const double* sendbuf, double* recvbuf, int sendcount) const {
		return MPI_Allreduce(sendbuf, recvbuf, sendcount, MPI_DOUBLE, MPI_MIN, comm);
	}
};

template <typename T>
Vec<Mat<T>> MyMpi::Gatherv(const Vec<Mat<T>>& vm, Idx mat_nrows, Idx mat_ncols, const VecPartition& vp) const
{
	Mat<Vec<T>> sendbuf(mat_nrows, mat_ncols, Vec<T>(vm.size()));
	for_Idx(i, 0, mat_nrows) {
		for_Idx(j, 0, mat_ncols) {
			for_Idx(k, 0, vm.size()) {
				sendbuf[i][j][k] = vm[k][i][j];
			}
		}
	}

	Mat<Vec<T>> recvbuf(mat_nrows, mat_ncols);
	for_Idx(i, 0, mat_nrows) {
		for_Idx(j, 0, mat_ncols) {
			recvbuf[i][j] = Gatherv(sendbuf[i][j], vp);
		}
	}

	Idx len = (*this ? vp.total_size() : 0);
	Vec<Mat<T>> res(len, Mat<T>(mat_nrows, mat_ncols));
	for_Idx(i, 0, mat_nrows) {
		for_Idx(j, 0, mat_ncols) {
			for_Idx(k, 0, len) {
				res[k][i][j] = recvbuf[i][j][k];
			}
		}
	}
	return res;
}

inline void use_omp(const MyMpi &mm, Int nthreads = 1)
{
	use_omp(nthreads);
	const Int actual = probe_omp_team_size();
	if (mm) std::cout << "OpenMP: actual_team = " << actual << std::endl;
}

inline void use_mkl(const MyMpi &mm, Int nthreads = 1)
{
	use_mkl(nthreads);
	if (mm) std::cout << "MKL: mkl_num_threads = " << mkl_get_max_threads() << std::endl;
}

// Collective, called only by MPI's initializing thread after the OpenMP probe.
// A single rank prints all records to avoid interleaved multi-process output.
inline void print_parallel_config(const MyMpi& mm, Int provided, Int budget, Int actual)
{
	int omp_max = 1, openmp_version = 0;
#ifdef _OPENMP
	omp_max = omp_get_max_threads();
	openmp_version = _OPENMP;
#endif
	const int fields = 5;
	const int local[fields] = {provided, budget, omp_max, actual, mkl_get_max_threads()};
	std::vector<int> all(mm ? static_cast<std::size_t>(mm.np()) * fields : 0);
	if (MPI_Gather(local, fields, MPI_INT, all.data(), fields, MPI_INT, mm.ms(), mm.cm()) != MPI_SUCCESS)
		throw std::runtime_error("Cannot gather parallel startup configuration");
	if (mm) {
		std::cout << "PARALLEL: ranks=" << mm.np()
			<< " required=MPI_THREAD_FUNNELED(" << MPI_THREAD_FUNNELED << ")"
			<< " openmp=" << openmp_version
			<< " omp_dynamic=0 max_active_levels=1 mkl_dynamic=0" << std::endl;
		for (int rank = 0; rank < mm.np(); ++rank) {
			const int* record = all.data() + static_cast<std::size_t>(rank) * fields;
			std::cout << "PARALLEL: rank=" << rank << " provided=" << record[0]
				<< " budget=" << record[1] << " omp_max=" << record[2]
				<< " omp_actual=" << record[3] << " mkl_max=" << record[4] << std::endl;
		}
	}
}

#endif /* _MYMPI_H_ */
