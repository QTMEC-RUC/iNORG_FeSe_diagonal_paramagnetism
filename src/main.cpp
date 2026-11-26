/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "specs.h"
#include "prmtr.h"
// #include "model.h"
// #include "api_zen.h"
#include "api_edmft.h"
// #include "dmft_bethe.h"

int main(int argc, char* argv[])
{
	using namespace std;
	int provided = MPI_THREAD_SINGLE;
	const int init_status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (init_status != MPI_SUCCESS) {
		cerr << "MPI_Init_thread failed: " << init_status << endl;
		return EXIT_FAILURE;
	}
	MyMpi mm;
	if (provided < MPI_THREAD_FUNNELED) {
		cerr << "rank " << mm.id() << ": MPI_THREAD_FUNNELED required, provided=" << provided << endl;
		MPI_Abort(mm.cm(), EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	try {
		const Int budget = thread_budget();
		configure_threads(budget, budget);
		const Int actual = probe_omp_team_size();
		print_parallel_config(mm, provided, budget, actual);
		const int local_ok = actual == budget && mkl_get_max_threads() == budget;
		int all_ok = 0;
		if (MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, mm.cm()) != MPI_SUCCESS)
			throw std::runtime_error("Cannot confirm parallel startup across ranks");
		if (!all_ok)
			throw std::runtime_error("Actual OpenMP team or MKL limit differs from thread budget");
	} catch (const std::exception& error) {
		cerr << "rank " << mm.id() << ": parallel startup failed: " << error.what() << endl;
		MPI_Abort(mm.cm(), EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	// Exercise startup without loading solver inputs or writing calculation output.
	if (argc == 2 && std::strcmp(argv[1], "--parallel-check") == 0) {
		if (mm) cout << "PARALLEL: startup check passed" << endl;
		return MPI_Finalize() == MPI_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	if (mm) cout << "\n\nVersion: v1.9.14.p3@2024.12.03\
(running "<< present() <<")\n\n" << endl;
	if (mm) cout << NAV(pwd()) << endl; 
	// if (mm) io_init();
	ElapsedTimer t_program_bgn;
	if (mm) TIME_BGN("program", t_program_bgn);
	try {
	Prmtr prmtr(mm);

	// APIzen norg(mm, prmtr, "solver");
	APIedmft norg(mm, prmtr, "solver");
	if(mm) WRN(NAV(prmtr.control_divs));
	// DMFT dmft(mm, prmtr, 1);
	
    if (mm)	TIME_END("program", t_program_bgn);
    } catch(const std::exception& error) {
        cerr << "rank " << mm.id() << ": solver failed: " << error.what() << endl;
        MPI_Abort(mm.cm(),EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    MPI_Finalize();
	return 0;
}
