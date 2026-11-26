/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "norg.h"
#include "gen/memory_monitor.h"
#include "gen/green_profile.h"
#include <chrono>
#include <thread>
#include <cmath>
#include "gen/runtime_options.h"
#define condition iter_norg_cnt < p.iter_max_norg  && !converged()

namespace {
std::uint64_t global_table_entries(const MyMpi& mm, const Tab& table)
{
    const std::uint64_t local = table.nnz();
    std::uint64_t total = 0;
    if (MPI_Allreduce(&local, &total, 1, MPI_UINT64_T, MPI_SUM, mm.cm()) != MPI_SUCCESS)
        throw std::runtime_error("Failed to sum operator table entry counts");
    return total;
}

// Optional observation delay, disabled unless explicitly requested.
void pause_before_ground_state_table(const MyMpi& mm)
{
    double seconds=0.;
    if(mm) {
        try { seconds=runtime_options::nonnegative_real("NORG_PRE_TAB_PAUSE_SECONDS",0.); }
        catch(...) { seconds=-1.; }
    }
    mm.Bcast(seconds);
    if(seconds<0.)throw std::invalid_argument("NORG_PRE_TAB_PAUSE_SECONDS must be finite and nonnegative");
    if(seconds==0.)return;
    log_ground_state_memory(mm,"PRE_TAB_SLEEP_BEGIN",memory_monitor::unknown_dimension);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    log_ground_state_memory(mm,"PRE_TAB_SLEEP_END",memory_monitor::unknown_dimension);
}
} // namespace

NORG::NORG(const MyMpi& mm_i, const Prmtr& prmtr_i) :
	mm(mm_i), p(prmtr_i), impgreen(prmtr_i.norbit - prmtr_i.nbath, prmtr_i), uormat(uormat_initialize()), final_ground_state(oneedm.ground_state),
	occupation_err(1.), energy_err(1.), correctionerr(1.), occnum_pre(prmtr_i.n_rot_orb, 0.),
	iter_norg_cnt(0), groune_pre(1.e99), groune_lst(0.), occnum_lst(prmtr_i.n_rot_orb, 0.),
	// Measure before scsp construction and immediately after it, before oneedm builds tab.
	scsp((log_ground_state_memory(mm_i, "NOCSPACE_BEFORE", memory_monitor::unknown_dimension), mm_i),
		prmtr_i, prmtr_i.npartical),
	oneedm((log_ground_state_memory(mm, "NOCSPACE_AFTER", scsp.dim, true, scsp.construction_seconds()),
		pause_before_ground_state_table(mm), mm), prmtr_i, scsp),norg_stab_cnt(0)
{
	show_the_nozero_number_of_tabel();
	if(mm) scsp.print();
}

// NORG::NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, VecInt nparticals) :
// 	mm(mm_i), p(prmtr_i), impgreen(prmtr_i.norbs, prmtr_i), uormat(uormat_initialize()),
// 	occupation_err(1.), energy_err(1.), correctionerr(1.), occnum_pre(prmtr_i.n_rot_orb, 0.),
// 	iter_norg_cnt(0), groune_pre(1.e99), groune_lst(0.), occnum_lst(prmtr_i.n_rot_orb, 0.),
// 	scsp(mm_i, prmtr_i, nparticals), oneedm(mm, prmtr_i, scsp),norg_stab_cnt(0)
// {
// 	show_the_nozero_number_of_tabel();
// }

NORG::NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, const Tab& table) :	
	mm(mm_i), p(prmtr_i), impgreen(prmtr_i.norbs, prmtr_i), uormat(uormat_initialize()), final_ground_state(oneedm.ground_state),
	occupation_err(1.), energy_err(1.), correctionerr(1.), occnum_pre(prmtr_i.n_rot_orb, 0.),
	iter_norg_cnt(0), groune_pre(1.e99), groune_lst(0.), occnum_lst(prmtr_i.n_rot_orb, 0.),
	scsp((log_ground_state_memory(mm_i, "NOCSPACE_BEFORE", memory_monitor::unknown_dimension), mm_i),
		prmtr_i, prmtr_i.npartical, table),
	oneedm((log_ground_state_memory(mm, "NOCSPACE_AFTER", scsp.dim, true, scsp.construction_seconds()),
		pause_before_ground_state_table(mm), mm), prmtr_i, scsp, table),norg_stab_cnt(0)
{

}

NORG::NORG(const MyMpi& mm_i, const Prmtr& prmtr_i, Str tab_name) :
	mm(mm_i), p(prmtr_i), impgreen(prmtr_i.norbs, prmtr_i), uormat(uormat_initialize()), final_ground_state(oneedm.ground_state),
	occupation_err(1.), energy_err(1.), correctionerr(1.), occnum_pre(prmtr_i.n_rot_orb, 0.),
	iter_norg_cnt(0), groune_pre(1.e99), groune_lst(0.), occnum_lst(prmtr_i.n_rot_orb, 0.),
	scsp((log_ground_state_memory(mm_i, "NOCSPACE_BEFORE", memory_monitor::unknown_dimension), mm_i),
		prmtr_i, prmtr_i.npartical, tab_name),
	oneedm((log_ground_state_memory(mm, "NOCSPACE_AFTER", scsp.dim, true, scsp.construction_seconds()),
		pause_before_ground_state_table(mm), mm), prmtr_i, scsp, tab_name),norg_stab_cnt(0)
{
	show_the_nozero_number_of_tabel();
}

bool NORG::StopPolicy::valid() const
{
	switch(mode) {
	case Mode::NormalConvergence: return true;
	case Mode::EnergyTolerance: return std::isfinite(tolerance) && tolerance > 0.;
	case Mode::FixedSteps: return steps >= 0;
	}
	return false;
}

bool NORG::StopPolicy::reached(Int completed_steps, Real relative_energy_error) const
{
	switch(mode) {
	case Mode::FixedSteps: return completed_steps >= 0 && completed_steps >= steps;
	case Mode::EnergyTolerance:
		return completed_steps > 0 && std::isfinite(relative_energy_error)
			&& std::abs(relative_energy_error) < tolerance;
	case Mode::NormalConvergence: return false; // Evaluated by NORG::converged.
	}
	return false;
}

bool NORG::StopPolicy::equivalent_to(const StopPolicy& other) const
{
	if(!valid() || !other.valid() || mode != other.mode) return false;
	switch(mode) {
	case Mode::NormalConvergence: return true;
	case Mode::EnergyTolerance: return tolerance == other.tolerance;
	case Mode::FixedSteps: return steps == other.steps;
	}
	return false;
}

void NORG::solve_with_policy(const Impdata& impH_i, const StopPolicy& policy,
	Int ground_state_mode)
{
	if(!policy.valid()) ERR("Invalid NORG stopping policy")
	if(policy.mode != StopPolicy::Mode::FixedSteps && p.iter_max_norg <= 0) {
		ERR("NORG convergence requires a positive iter_max_norg")
	}
	impH = impH_i;
	scsp.divs_to_idx.clear();
	iter_norg_cnt = 0;
	norg_stab_cnt = 0;
	energy_err = occupation_err = 1.;

	// Use the caller's degeneracy handling (block solves default to recomputing it).
	// Occupations below are diagonals in the actual current basis, so adaptive
	// G constraints refer to the same orbital ordering as the saved uormat.
	auto solve_current_basis = [&]() {
		set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);
		oneedm.update(ground_state_mode);
		groune_lst = oneedm.groundstate_energy;
		if(!std::isfinite(groune_lst)) ERR("Non-finite NORG ground-state energy")
		VEC<VecReal> occupations;
		for_Int(i, 0, p.norg_sets) {
			const Int first = p.if_norg_imp ? 0 : p.nO2sets[i] - p.nI2B[i];
			VecReal values(p.nO2sets[i] - first, 0.);
			for_Int(j, 0, values.size()) {
				values[j] = oneedm.dm[i][first + j][first + j];
				if(!std::isfinite(values[j])) ERR("Non-finite NORG occupation")
			}
			occupations.push_back(values);
		}
		oneedm.occupationnumber = occupations;
		occnum_lst.reset(VECVectoVec(occupations));
		// Report weights in the basis of this solve, including zero-rotation runs.
		if(mm) PIO_occweight(occnum_lst);
	};

	// Establish a ground state in the supplied basis before the first rotation.
	solve_current_basis();
	const Int limit = policy.mode == StopPolicy::Mode::FixedSteps
		? policy.steps : p.iter_max_norg;
	// Zero rotation steps still require the initial solve above. Keep the
	// supplied basis and finalize its newly solved state through the common path.
	bool stopped = policy.reached(iter_norg_cnt, energy_err);
	while(!stopped && iter_norg_cnt < limit) {
		groune_pre = groune_lst;
		occnum_pre = occnum_lst;
		const VEC<MatReal> rotation = oneedm.find_unitary_orbital_rotation_matrix();
		for_Int(i, 0, uormat.size()) uormat[i] = rotation[i] * uormat[i];
		solve_current_basis();
		++iter_norg_cnt;
		energy_err = 2 * (groune_pre - groune_lst)
			/ (ABS(groune_pre) + ABS(groune_lst) + 1.);
		occupation_err = SQRT(SUM(SQR(occnum_pre - occnum_lst)) / p.norbit);
		if(mm) WRN("Controlled NORG: " + NAV3(iter_norg_cnt, energy_err, occupation_err));
		stopped = policy.mode == StopPolicy::Mode::NormalConvergence
			? converged(1.E-12) : policy.reached(iter_norg_cnt, energy_err);
		if(stopped) break;
	}
	if(!stopped) {
		ERR("NORG did not converge within iter_max_norg=" + STR(p.iter_max_norg)
			+ "; relative energy error=" + STR(energy_err))
	}
	occnum = occnum_lst;
	p.rotationU = uormat;
	if(mm) {
		WRN("NORG " + Str(policy.mode == StopPolicy::Mode::FixedSteps
			? "completed requested steps: " : "converged: ")
			+ NAV2(iter_norg_cnt, groune_lst));
	}
}

// mode 0: for not rotation; mode 1: for rotation the orbitals; 
void NORG::up_date_h0_to_solve(const Impdata& impH_i, const Int mode) {
	// oneedm.write_the_multiTab("0.25-");
	if(mm) std::cout << std::endl;						// blank line
	impH = impH_i;
	// //! testing-20240503 begin
	// scsp.div = std::vector<MatInt>();
	scsp.divs_to_idx.clear(); // Release shard nodes/buckets in parallel.
	// scsp.idx_div = std::vector<Int>();
	// //! testing-20240503 end
	if (mm) WRN(NAV2(impH.first,scsp.dim));
	set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);

	oneedm.update(p.if_norg_degenerate); 
	// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
	groune_lst = oneedm.groundstate_energy;
	if (mm)	{
		// scsp.print();
		// write_norg_info(iter_norg_cnt);
		// write_state_info(iter_norg_cnt);
	}
	// if (mm) WRN(NAV2(oneedm.dm[0], uormat[0]));
	
	while (iter_norg_cnt < p.iter_max_norg && !converged(1.E-12)) {
		iter_norg_cnt++;
		if (mm) PIO("The iteration counting: " + NAV2(iter_norg_cnt, norg_stab_cnt));
		if (mode) {
			VEC<MatReal> uormat_new(oneedm.find_unitary_orbital_rotation_matrix());
			// if(mm) WRN(NAV(see_MatReal(uormat_new)));
			for_Int(i, 0, uormat.size()) uormat[i] = uormat_new[i] * uormat[i];
		}
		// if (mm) WRN(NAV2(oneedm.dm[0], uormat[0]));
		set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);
		// if (mm)PIO("ground_state size" + NAV(oneedm.ground_state.size()));
		groune_pre = groune_lst;	occnum_pre = occnum_lst;
		oneedm.update(p.if_norg_degenerate);
		// if (mm) PIO(NAV(see_MatReal(oneedm.dm)));
		// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
		occnum_lst = VECVectoVec(oneedm.occupationnumber);				PIO_occweight(occnum_lst);
		groune_lst = oneedm.groundstate_energy;
		// if(mm) WRN(NAV(oneedm.dm[0]));
		if (mm) {
			// WRN(NAV(iter_norg_cnt));
			// for_Int(i, 0, uormat.size()) WRN(NAV(uormat_new[i]));
			// write_norg_info(iter_norg_cnt);
			// write_state_info(iter_norg_cnt);
		}
		occupation_err = SQRT(SUM(SQR(occnum_pre - occnum_lst)) / p.norbit);
		energy_err = 2 * (groune_pre - groune_lst) / (ABS(groune_pre) + ABS(groune_lst) + 1.);
		if (mm) {
			PIO(NAV2(energy_err, occupation_err));
			std::cout << "groundE_pre " << iofmt("sci") << groune_pre << std::setw(4) << "   groundE_lst " << groune_lst  << "  " << present() << std::endl;
		}
		if(mm) std::cout << std::endl;						// blank line
	}
	iter_norg_cnt = 0;		occupation_err = 1.;		energy_err = 1.;
	norg_stab_cnt = 0.; occnum = occnum_lst;
	//----------------------print put--------------------------------
	if(mm) std::cout << "---------- THE NORG CONVERGED ----------" << std::endl;;
	PIO_occweight(occnum);
	if(mm) {
        MatReal density(p.nband, p.nband, 0.);
        for_Int(m, 0, p.nband) density[m][m] = oneedm.dm[2 * m][0][0];
        std::cout << "---------- IMPURITY UP-SPIN DENSITY MATRIX ----------" << std::endl
                  << iofmt("sci") << density << std::endl;
	}
	p.rotationU = uormat;
	// write_impurtiy_occupation();

	// if (mm) WRN(NAV4(oneedm.dm[0], uormat[0],oneedm.dm[2], uormat[2]));

	if(mm) {
		std::cout << "\nnorg ground state energy: " << groune_lst << "  " << present() << std::endl;
		std::cout << std::endl;							// blank line
	}
}

// mode 0: for not rotation; mode 1: for rotation the orbitals; 
void NORG::up_date_h0_to_solve(const Impdata& impH_i, const Int mode, const Real error) {
	// oneedm.write_the_multiTab("0.25-");
	if(mm) std::cout << std::endl;						// blank line
	impH = impH_i;
	// //! testing-20240503 begin
	// scsp.div = std::vector<MatInt>();
	scsp.divs_to_idx.clear(); // Release shard nodes/buckets in parallel.
	// scsp.idx_div = std::vector<Int>();
	// //! testing-20240503 end
	if (mm) WRN(NAV2(impH.first,scsp.dim));
	set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);

	oneedm.update(p.if_norg_degenerate); 
	// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
	groune_lst = oneedm.groundstate_energy;
	if (mm)	{
		// scsp.print();
		// write_norg_info(iter_norg_cnt);
		// write_state_info(iter_norg_cnt);
	}
	// if (mm) WRN(NAV2(oneedm.dm[0], uormat[0]));
	
	while (iter_norg_cnt < p.iter_max_norg && !converged(error)) {
		iter_norg_cnt++;
		if (mm) PIO("The iteration counting: " + NAV2(iter_norg_cnt, norg_stab_cnt));
		if (mode) {
			VEC<MatReal> uormat_new(oneedm.find_unitary_orbital_rotation_matrix());
			// if(mm) WRN(NAV(see_MatReal(uormat_new)));
			for_Int(i, 0, uormat.size()) uormat[i] = uormat_new[i] * uormat[i];
		}
		// if (mm) WRN(NAV2(oneedm.dm[0], uormat[0]));
		set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);
		// if (mm)PIO("ground_state size" + NAV(oneedm.ground_state.size()));
		groune_pre = groune_lst;	occnum_pre = occnum_lst;
		oneedm.update(p.if_norg_degenerate);
		// if (mm) PIO(NAV(see_MatReal(oneedm.dm)));
		// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
		occnum_lst = VECVectoVec(oneedm.occupationnumber);				PIO_occweight(occnum_lst);
		groune_lst = oneedm.groundstate_energy;
		// if(mm) WRN(NAV(oneedm.dm[0]));
		if (mm) {
			// WRN(NAV(iter_norg_cnt));
			// for_Int(i, 0, uormat.size()) WRN(NAV(uormat_new[i]));
			// write_norg_info(iter_norg_cnt);
			// write_state_info(iter_norg_cnt);
		}
		occupation_err = SQRT(SUM(SQR(occnum_pre - occnum_lst)) / p.norbit);
		energy_err = 2 * (groune_pre - groune_lst) / (ABS(groune_pre) + ABS(groune_lst) + 1.);
		if (mm) {
			PIO(NAV2(energy_err, occupation_err));
			std::cout << "groundE_pre " << iofmt("sci") << groune_pre << std::setw(4) << "   groundE_lst " << groune_lst  << "  " << present() << std::endl;
		}
		if(mm) std::cout << std::endl;						// blank line
	}
	iter_norg_cnt = 0;		occupation_err = 1.;		energy_err = 1.;
	norg_stab_cnt = 0.; occnum = occnum_lst;
	//----------------------print put--------------------------------
	if(mm) std::cout << "---------- THE NORG CONVERGED ----------" << std::endl;;
	PIO_occweight(occnum);
	if(mm) {
        MatReal density(p.nband, p.nband, 0.);
        for_Int(m, 0, p.nband) density[m][m] = oneedm.dm[2 * m][0][0];
        std::cout << "---------- IMPURITY UP-SPIN DENSITY MATRIX ----------" << std::endl
                  << iofmt("sci") << density << std::endl;
	}
	p.rotationU = uormat;
	// write_impurtiy_occupation();

	// if (mm) WRN(NAV4(oneedm.dm[0], uormat[0],oneedm.dm[2], uormat[2]));

	if(mm) {
		std::cout << "\nnorg ground state energy: " << groune_lst << "  " << present() << std::endl;
		std::cout << std::endl;							// blank line
	}
}

// Here we only need the energy of the impurity, so we don't need iter too much.
void NORG::up_date_h0_to_solve(const Impdata& impH_i, const VecReal sub_energy) {
	std::streambuf* old = std::cout.rdbuf();
	std::cout.rdbuf(NULL); // redirect std::cout to nowhere
	// oneedm.write_the_multiTab("0.25-");
	if(mm) std::cout << std::endl;						// blank line
	impH = impH_i;
	// if (mm) PIO(NAV2(h0,scsp.dim));
	// scsp.coefficient = set_row_primeter_byimpH(uormat, impH_i);
	set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);
	oneedm.update(0);
	// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
	groune_lst = oneedm.groundstate_energy;
	if (mm)	{
		// scsp.print();
		// write_norg_info(iter_norg_cnt);
		// write_state_info(iter_norg_cnt);
	}

	// if(!p.if_norg_imp && sub_energy.size() != 0 && MIN(sub_energy) < groune_lst * (1/* + 2e-5*/) )  {
	// 	if(mm) std::cout <<iofmt("def")<< "The energy level is so hight, "<< NAV(groune_lst) << ".\nWhich reach to " \
	// 	<<100*MIN(sub_energy)/groune_lst<<"%, so, the Ground state mustn't in this sub-space, and calc. will be stoped." <<"\n"<< std::endl;
	// 	return ;}
	
	//! while (iter_norg_cnt < p.iter_max_norg && !converged()) {
	while (iter_norg_cnt < 10 && !converged(1.E-5)) {
		iter_norg_cnt++;
		if (mm) PIO("The iteration counting: " + NAV2(iter_norg_cnt, norg_stab_cnt));
		VEC<MatReal> uormat_new(oneedm.find_unitary_orbital_rotation_matrix());
		// if(mm) WRN(NAV(see_MatReal(uormat_new)));
		for_Int(i, 0, uormat.size()) uormat[i] = uormat_new[i] * uormat[i];
		// if(mm) WRN(NAV(uormat[0]));
		// scsp.coefficient = set_row_primeter_byimpH(uormat, impH_i);
		set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);	//if (mm) scsp.print();
		// if (mm)PIO("ground_state size" + NAV(oneedm.ground_state.size()));
		groune_pre = groune_lst;	occnum_pre = occnum_lst;
		oneedm.update(0);
		// if(mm) PIO(NAV(see_MatReal(oneedm.dm)));
		// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
		occnum_lst = VECVectoVec(oneedm.occupationnumber);
		groune_lst = oneedm.groundstate_energy;
		// if(mm) WRN(NAV(oneedm.dm[0]));
		if (mm) {
			// WRN(NAV(iter_norg_cnt));
			// for_Int(i, 0, uormat.size()) WRN(NAV(uormat_new[i]));
			// write_norg_info(iter_norg_cnt);
			// write_state_info(iter_norg_cnt);
		}
		occupation_err = SQRT(SUM(SQR(occnum_pre - occnum_lst)) / p.norbit);
		energy_err = 2 * (groune_pre - groune_lst) / (ABS(groune_pre) + ABS(groune_lst) + 1.);
		if (mm) {
			PIO(NAV2(energy_err, occupation_err));
			std::cout << "groundE_pre " << iofmt("sci") << groune_pre << std::setw(4) << "   groundE_lst " << groune_lst  << "  " << present() << std::endl;
		}
		if(mm) std::cout << std::endl;						// blank line
	}
	iter_norg_cnt = 0;		occupation_err = 1.;		energy_err = 1.;
	norg_stab_cnt = 0.; occnum = occnum_lst;
	/*-------------------------------- print put --------------------------------*/
	// MatReal occnum, occweight;
	// occnum = occnum_lst.mat(p.norg_sets, p.n_rot_orb/p.norg_sets);
	// occweight = occnum;
	// for_Int(i, 0, p.norg_sets) for_Int(j, 0, occnum.ncols()) occweight[i][j] = MIN(occweight[i][j],1 - occnum[i][j]) < 1e-14 ? 0 : MIN(occweight[i][j],1 - occnum[i][j]);
	// Str nppso = scsp.nppso_str();
	// if (mm) WRN(NAV4(p.if_norg_imp, nppso, occnum, occweight));
	PIO_occweight(occnum);
	p.rotationU = uormat;
	// write_impurtiy_occupation();

	std::cout.rdbuf(old);
	// Finish the NORGiteration.
	// oneedm.save_the_Tab(oneedm.table, "mainspace");
}

// Here we only need the energy of the impurity, so we don't need iter too much.
void NORG::up_date_h0_to_solve(const Impdata& impH_i, const Real error) {
	std::streambuf* old = std::cout.rdbuf();
	std::cout.rdbuf(NULL); // redirect std::cout to nowhere
	// oneedm.write_the_multiTab("0.25-");
	if(mm) std::cout << std::endl;						// blank line
	impH = impH_i;
	// if (mm) PIO(NAV2(h0,scsp.dim));
	// scsp.coefficient = set_row_primeter_byimpH(uormat, impH_i);
	set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);
	oneedm.update(0);
	// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
	groune_lst = oneedm.groundstate_energy;
	if (mm)	{
		// scsp.print();
		// write_norg_info(iter_norg_cnt);
		// write_state_info(iter_norg_cnt);
	}

	
	//! while (iter_norg_cnt < p.iter_max_norg && !converged()) {
	while (iter_norg_cnt < 10 && !converged(error)) {
		iter_norg_cnt++;
		if (mm) PIO("The iteration counting: " + NAV2(iter_norg_cnt, norg_stab_cnt));
		VEC<MatReal> uormat_new(oneedm.find_unitary_orbital_rotation_matrix());
		// if(mm) WRN(NAV(see_MatReal(uormat_new)));
		for_Int(i, 0, uormat.size()) uormat[i] = uormat_new[i] * uormat[i];
		// if(mm) WRN(NAV(uormat[0]));
		// scsp.coefficient = set_row_primeter_byimpH(uormat, impH_i);
		set_row_primeter_byimpH(uormat, impH_i, oneedm.oper_value);	//if (mm) scsp.print();
		// if (mm)PIO("ground_state size" + NAV(oneedm.ground_state.size()));
		groune_pre = groune_lst;	occnum_pre = occnum_lst;
		oneedm.update(0);
		// if(mm) PIO(NAV(see_MatReal(oneedm.dm)));
		// if(mm) PIO(NAV(oneedm.sum_off_diagonal()));
		occnum_lst = VECVectoVec(oneedm.occupationnumber);
		groune_lst = oneedm.groundstate_energy;
		// if(mm) WRN(NAV(oneedm.dm[0]));
		if (mm) {
			// WRN(NAV(iter_norg_cnt));
			// for_Int(i, 0, uormat.size()) WRN(NAV(uormat_new[i]));
			// write_norg_info(iter_norg_cnt);
			// write_state_info(iter_norg_cnt);
		}
		occupation_err = SQRT(SUM(SQR(occnum_pre - occnum_lst)) / p.norbit);
		energy_err = 2 * (groune_pre - groune_lst) / (ABS(groune_pre) + ABS(groune_lst) + 1.);
		if (mm) {
			PIO(NAV2(energy_err, occupation_err));
			std::cout << "groundE_pre " << iofmt("sci") << groune_pre << std::setw(4) << "   groundE_lst " << groune_lst  << "  " << present() << std::endl;
		}
		if(mm) std::cout << std::endl;						// blank line
	}
	iter_norg_cnt = 0;		occupation_err = 1.;		energy_err = 1.;
	norg_stab_cnt = 0.; occnum = occnum_lst;
	/*-------------------------------- print put --------------------------------*/
	// MatReal occnum, occweight;
	// occnum = occnum_lst.mat(p.norg_sets, p.n_rot_orb/p.norg_sets);
	// occweight = occnum;
	// for_Int(i, 0, p.norg_sets) for_Int(j, 0, occnum.ncols()) occweight[i][j] = MIN(occweight[i][j],1 - occnum[i][j]) < 1e-14 ? 0 : MIN(occweight[i][j],1 - occnum[i][j]);
	// Str nppso = scsp.nppso_str();
	// if (mm) WRN(NAV4(p.if_norg_imp, nppso, occnum, occweight));
	PIO_occweight(occnum);
	p.rotationU = uormat;
	// write_impurtiy_occupation();

	std::cout.rdbuf(old);
	// Finish the NORGiteration.
	// oneedm.save_the_Tab(oneedm.table, "mainspace");
}

void NORG::get_gimp_diagonal(ImGreen& output)
{
    if (output.norbs != p.nband || output.nomgs != p.num_omg) ERR("Expected one Green channel per spatial orbital.")
    for_Int(m, 0, p.nband) {
        ImGreen single(1, p);
        get_gimp_orbital(single, m);
        for_Int(n, 0, output.nomgs) output[n][m][m] = single[n][0][0];
    }
}

void NORG::get_gimp_orbital(ImGreen& single, Int orbital)
{
    if (orbital < 0 || orbital >= p.nband || single.norbs != 1 || single.nomgs != p.num_omg)
        ERR("Invalid paramagnetic orbital Green function.")
    const Int set_idx = 2 * orbital;
	if(p.degel <= 0 || static_cast<Int>(final_ground_state.nrows()) < p.degel) {
		ERR("No ground-state ensemble is available for the orbital Green function.")
	}

	const StdVecInt difference = {set_idx + 1, -(set_idx + 1)};
	for(const auto ii : difference) {
		const Str context = "set=" + STR(set_idx) + " sector=" + (ii > 0 ? "N+1" : "N-1");
		green_profile::Profile profile(mm,context,"sector",p.green_krylov.solver);
		const auto log_sector = [&](const char* stage, Idx dimension, double measured_s = -1.) {
			green_profile::Scope timer(&profile,green_profile::Monitor);
			log_computation_memory(mm, stage, dimension, "GF", context, true, measured_s);
		};
		// A completely full/empty set has no addition/removal contribution.
		if((ii > 0 && p.npartical[set_idx] == p.nO2sets[set_idx])
			|| (ii < 0 && p.npartical[set_idx] == 0)) {
			log_sector("SECTOR_SKIPPED_FULL_OR_EMPTY", 0);
			profile.finish(mm,"GF_CHECK "+context+" stage=sector status=SKIPPED reason=full_or_empty\n");
			continue;
		}
		log_sector("SECTOR_BEGIN", memory_monitor::unknown_dimension);
		log_sector("NOCSPACE_BEFORE", memory_monitor::unknown_dimension);
		green_profile::Scope space_timer(&profile,green_profile::Nocspace);
		NocSpace scsp_sub(mm, p, nppso(p.npartical, ii));
		space_timer.stop();
		log_sector("NOCSPACE_AFTER", scsp_sub.dim, scsp_sub.construction_seconds());
		if(scsp_sub.dim == 0) {
			log_sector("SECTOR_SKIPPED_ZERO_DIM", 0);
			log_sector("SECTOR_END", 0);
			profile.finish(mm,"GF_CHECK "+context+" stage=sector status=SKIPPED reason=zero_dimension\n");
			continue;
		}
		green_profile::Scope table_timer(&profile,green_profile::Table);
		Operator opr_sub(mm, p, scsp_sub, true, "GF", context);
		table_timer.stop();
		const auto table_entries = global_table_entries(mm, opr_sub.table);
		if(mm) std::cout << "GF: " << context << " dim=" << scsp_sub.dim
			<< " table_entries_sum=" << table_entries << std::endl;
		log_sector("H_COEFFICIENTS_BEFORE", scsp_sub.dim);
		{green_profile::Scope timer(&profile,green_profile::Coefficients);
		set_row_primeter_byimpH(uormat, impH, opr_sub.oper_value);}
		log_sector("H_COEFFICIENTS_AFTER", scsp_sub.dim);
		// One matrix per particle sector, shared by all degenerate ground states.
		log_sector("H_BEFORE", scsp_sub.dim);
		green_profile::Scope h_timer(&profile,green_profile::Hamiltonian);
		const SparseMatReal sep_h = opr_sub.find_hmlt_V2(opr_sub.table);
		h_timer.stop();
		log_sector("H_AFTER", scsp_sub.dim, opr_sub.h_build_seconds());
		log_sector("TAB_RELEASE_BEFORE", scsp_sub.dim);
		{green_profile::Scope timer(&profile,green_profile::Release);
		opr_sub.table.reset();} // CrrltFun below now copies an empty table.
		log_sector("TAB_RELEASE_AFTER", scsp_sub.dim);
		for_Int(egs_idx, 0, p.degel) {
			const Str state_context = context + " gs=" + STR(egs_idx);
			log_computation_memory(mm, "STATE_PREP_BEFORE", scsp_sub.dim, "GF", state_context);
			// The orbital source uses the same set as the particle-number shift.
			green_profile::Scope prep_timer(&profile,green_profile::StatePrep);
			CrrltFun temp_green(mm, p, scsp, scsp_sub, opr_sub);
			ImGreen gf(1, p);
			prep_timer.stop();
			log_computation_memory(mm, "STATE_PREP_AFTER", scsp_sub.dim, "GF", state_context);
			temp_green.find_gf_orbital(groune_lst, gf, final_ground_state[egs_idx],
				set_idx, sep_h, ii > 0, state_context);
			green_profile::Scope ensemble_timer(&profile,green_profile::Ensemble);
			for_Int(n, 0, gf.nomgs) 
				single[n][0][0] += gf[n][0][0] / p.degel;
		}
		// Includes all ground states; sector H/space are still alive at this snapshot.
		log_sector("SECTOR_END", scsp_sub.dim);
		profile.finish(mm,"GF_CHECK "+context+" stage=sector status=COMPLETE h_hermiticity=NOT_CHECKED truncation_convergence=NOT_CHECKED\n");
	}
}

// special modify for the HHD case.(arXiv:2209.14178v1)
void NORG::get_gimp_hdQPs(Green& imp_i, Int ex_idx)
{
	for_Int(i, 0, p.nband)
	{
		// StdVecInt difference = {(i+1), -(i+1)};
		// for(const auto ii: difference)
		{
			NocSpace scsp_sub(mm, p, nppso(p.npartical, 2*i+1));
			Operator opr_sub(mm, p, scsp_sub);
			set_row_primeter_byimpH(uormat, impH, opr_sub.oper_value);
			for_Int(egs_idx, 0, p.degel){
				CrrltFun temp_green(mm, p, scsp, scsp_sub, opr_sub, final_ground_state[egs_idx], i * 2);
				if(imp_i.type_info() == STR("ImGreen")) {
					ImGreen green_function(1, p);
					temp_green.find_hd_greater(groune_lst, green_function, i * 2, ex_idx);
					for_Int(n, 0, green_function.nomgs) imp_i[n][i][i] += green_function[n][0][0] / p.degel;
				}
				if(imp_i.type_info() == STR("ReGreen")) {
					ReGreen green_function(1, p);
					temp_green.find_hd_greater(groune_lst, green_function, i * 2, ex_idx);
					for_Int(n, 0, green_function.nomgs) imp_i[n][i][i] += green_function[n][0][0] / p.degel;
				}
			}
		}
		// if (mm) PIO("finished the " + STR(i) + " find_g_norg   " + present());
	}
}




void NORG::readmatrix(MatReal& m, const Str& file)
{
	IFS ifs(file);
	if (!ifs) {
		if (mm) WRN(STR("file opening failed with ") + NAV(file))
	}
	else {
	for_Int(i, 0, m.nrows()) {
		for_Int(j, 0, m.ncols()) ifs >> m[i][j];
	}
	}
	ifs.close();
}

VEC<MatReal> NORG::uormat_initialize()
{
	VEC<MatReal> uormat_i;

	for_Int(i, 0, p.nO2sets.size()) {
		MatReal temp(dmat(p.nO2sets[i], 1.));
		uormat_i.push_back(std::move(temp));
	}

	return uormat_i;
}

bool NORG::converged(Real ge_err) 
{
		const Real gseerr = ABS(energy_err);
		if (gseerr > 1.E-2) { return false; }
		else if (gseerr < ge_err) { return true; }
		else if (ABS(energy_err) < 1.E-7) { 
			norg_stab_cnt++;
			if (norg_stab_cnt > 20) return true;
			if (norg_stab_cnt > 10 && ABS(energy_err) < 1.E-9 && energy_err < 0) return true;
			}
		else if (iter_norg_cnt > 40) {
			const Real occerr = ABS(occupation_err);
			if (occerr < 1.E-8) {
				WRN("The iteration stoped since the occerr is less than 1.E-8.");
				return true;
			}
		}
		return false;
}

bool NORG::green_modify_converged() const
{
	const Real cverr = ABS(correctionerr);
	if (cverr < 1.E-5) { return true; }
	return false;
}

bool NORG::green_modify_converged(Real correctionerr_i) const
{
	const Real cverr = ABS(correctionerr_i);
	if (cverr < 1.E-5) { return true; }
	return false;
}

//------------------------------------------------------------------ spin-spin ------------------------------------------------------------------

Real NORG::sz_imp_sz_bath(const Int imp_postition, const VecReal& vgs_i)
{
	VecPartition row_H(mm.np(), mm.id(), scsp.dim);
	NocSpace::GroupCursor group_cursor(scsp);
	Real sz_imp_sz_bath_i(0.);
	// VecReal gstate_part = vgs_i.truncate(row_H.bgn(), row_H.end());
	for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i)
	{
		StateStatistics a(h_i, group_cursor.locate(h_i), scsp);
		Int conuter = 0;
		// for imp up spin:
#define ndiv scsp.ndivs
		// Preserve the original bath divisions [1, 5), accumulating in Int.
		Int bath_up = 0, bath_down = 0;
		if (a.occ_n[imp_postition * scsp.ndivs] == 1
			|| a.occ_n[(imp_postition + 1) * scsp.ndivs] == 1) {
			for_Int(j, 1, 5) {
				bath_up += a.occ_n[imp_postition * scsp.ndivs + j];
				bath_down += a.occ_n[(imp_postition + 1) * scsp.ndivs + j];
			}
		}
		if(a.occ_n[imp_postition * scsp.ndivs] == 1) {
			conuter += bath_up;
			conuter -= bath_down;

		}
		// for imp down spin:
		if(a.occ_n[(imp_postition + 1) * scsp.ndivs] == 1){
			conuter -= bath_up;
			conuter += bath_down;
		}
		sz_imp_sz_bath_i += conuter * vgs_i[h_i] * vgs_i[h_i];
	}
	Real sz_imp_sz_bath = mm.Allreduce(sz_imp_sz_bath_i) / 4.;
	return sz_imp_sz_bath;
}

//------------------------------------------------------------------- private -------------------------------------------------------------------

void NORG::show_the_nozero_number_of_tabel() 
{
	// Real size_one(oneedm.table.nnz()), size_two(n1mone.table.nnz()), size_tree(n1pone.table.nnz());
	// LLInt size_of_main_t(mm.Allreduce(size_one)), size_of_main_n1mt(mm.Allreduce(size_two)), size_of_main_n1pt(mm.Allreduce(size_tree));
	// if(mm) PIO(NAV3(size_of_main_t, size_of_main_n1mt, size_of_main_n1pt));
    const auto size_of_main_t = global_table_entries(mm, oneedm.table);
	if(mm) PIO(NAV2(size_of_main_t, scsp.dim)+"   "+present());
}

MatReal NORG::save_NTR() {
	MatReal transform_uormat(see_MatReal(uormat));
	if(mm) {
		OFS ofs_a;
		ofs_a.open("ru" + scsp.nppso_str() + ".bi");
		for_Int(i, 0, uormat.size()) biwrite(ofs_a, CharP(uormat[i].p()), uormat[i].szof());
	}
	return transform_uormat;
}

MatReal NORG::read_NTR() {
	IFS ifs_a("ru" + scsp.nppso_str() + ".bi"); 
	if (ifs_a) for_Int(i, 0, uormat.size()) biread(ifs_a, CharP(uormat[i].p()), uormat[i].szof());

	MatReal transform_uormat(see_MatReal(uormat));
	return transform_uormat;
}

void NORG::set_row_primeter_byimpH(const VEC<MatReal>& uormat_i, const Impdata& impH_i, VecReal& oper_i)
{
	using namespace std;
	MatReal hopint(impH_i.first), full_uormat(dmat(p.norbit, 1.));
	Int counter(0), n(p.n_rot_orb);
	if (uormat_i.empty() || oper_i.size() == 0) ERR("the uormat_i or oper_value is empty!!");
	
	if (!uormat_i.empty()) {
		for (const auto& uormat_ii : uormat_i) {
			for_Int(i, 0, uormat_ii.nrows()) {
				for_Int(j, 0, uormat_ii.ncols()) {
					full_uormat[i + counter][j + counter] = uormat_ii[i][j];
				}
			}
			counter += uormat_ii.nrows();
		}
	}

	if(mm) WRN(NAV(hopint));

	hopint = full_uormat * hopint * full_uormat.ct();

	if(mm) WRN(NAV(hopint));

	//! share memory mode.  TESTED :)
	if (p.if_norg_imp) {
		Int n3(n * n * n);
		MatReal t_m; VecReal ijkl(impH_i.second), t_v;
		VecReal& ijkL(ijkl),IjkL(ijkl),kLIj(ijkl),KLIj(ijkl),KLIJ(ijkl),IJKL(ijkl);
		// The tensor rotation as follows:
		ijkL = t_v.sm(t_m.sm(n3, n, ijkl) * full_uormat.ct());
		IjkL = t_v.sm(full_uormat * t_m.sm(n, n3, ijkL));
		kLIj = t_v.sm(t_m.sm(n * n, n * n, IjkL).tr());
		KLIj = t_v.sm(full_uormat * t_m.sm(n, n3, kLIj));
		KLIJ = t_v.sm(t_m.sm(n3, n, KLIj) * full_uormat.ct());
		IJKL = t_v.sm(t_m.sm(n * n, n * n, KLIJ).tr());
		
		oper_i = concat(concat(VecReal{ 0. }, t_v.sm(hopint)), IJKL);
	} else { 
		VecReal t_v;
		oper_i = concat(concat(VecReal{ 0. }, t_v.sm(hopint)), impH_i.second);
	}
	// if(mm) WRN(NAV(full_uormat));
}

//------------------------------------------------------------------ io ------------------------------------------------------------------

void NORG::write_norg_info(Int iter_cnt) const {
	using namespace std;
	Real sc = 0;
	for_Int(i, 0, final_ground_state[0].size()) sc -= SQR(final_ground_state[0][i]) * log(SQR(final_ground_state[0][i]));
	OFS ofs_app_scorrelation(tox + "slater_correlation.txt", std::ios::app);
	ofs_app_scorrelation << iofmt("sci");
	ofs_app_scorrelation << setw(4) << iter_cnt;
	ofs_app_scorrelation << "\t" << setw(w_Real) << sc;
	ofs_app_scorrelation << endl; ofs_app_scorrelation.close();

	OFS ofs_app_energy(tox + "ground_energy.txt", std::ios::app);
	ofs_app_energy << iofmt("sci");
	ofs_app_energy << setw(4) << iter_cnt;
	ofs_app_energy << "\t" << setw(w_Real) << groune_lst;
	ofs_app_energy << endl; ofs_app_energy.close();

	// Only show for the upper spin.
	if(iter_cnt != 0)for_Int(i, 0, p.nband) {
		OFS ofs_app_occupation(tox +"imp"+ STR(i*2) + "_bath_occupation_number.txt", std::ios::app);
		ofs_app_occupation << iofmt("sci");
		ofs_app_occupation << setw(4) << iter_cnt;
		for_Int(j, 0, p.nI2B[i*2]) { ofs_app_occupation << "  " << setw(w_Real) << occnum_lst[SUM_0toX(p.nI2B, i*2) + j]; }
		ofs_app_occupation << endl;
		ofs_app_occupation.close();
	}
	
	// Only show for the upper spin.
	if(iter_cnt != 0)for_Int(i, 0, p.nband) {
		VEC<VecReal> dmeigen = oneedm.check_dm_get_occupation_number();
		OFS ofs_app_occupation_all(tox +"imp"+ STR(i*2) + "_occupation_number.txt", std::ios::app);
		ofs_app_occupation_all << iofmt("sci");
		ofs_app_occupation_all << setw(4) << iter_cnt;
		for_Int(j, 0, dmeigen[i].size()) { ofs_app_occupation_all << "  " << setw(w_Real) << dmeigen[i][j]; }
		ofs_app_occupation_all << endl;
		ofs_app_occupation_all.close();
	}

}
/*// ! Abandon
void NORG::write_occupation_info() const {
	using namespace std;
	OFS ofs; ofs.open("nmat.txt");
	VEC<MatReal> dmtemp(oneedm.dm);
	VecReal counter(3);
	ofs << "#   < n_i >   data:"<< endl;
	for_Int(orb_i, 0, p.norbs)	{
		ofs << iofmt();
		std::string temp = (orb_i % 2) == 0 ? STR(Int(orb_i / 2) + 1) + "up" : STR(Int(orb_i / 2) + 1) + "dn";
		ofs << setw(6) << temp << setw(p_Real) << dmtemp[orb_i][0][0] << endl;
		(orb_i % 2) == 0 ? counter[0] += dmtemp[orb_i][0][0] : counter[1] += dmtemp[orb_i][0][0];
	}
	counter[2] = counter[0] + counter[1];
	ofs << setw(6) << "sup" << setw(p_Real) << counter[0] << endl;
	ofs << setw(6) << "sdn" << setw(p_Real) << counter[1] << endl;
	ofs << setw(6) << "sum" << setw(p_Real) << counter[2];
	ofs.close();
}
*/
void NORG::write_state_info(Int iter_cnt) const {
	using namespace std;
	OFS ofs_app_state; ofs_app_state.open(STR(iter_cnt) + "write_state_info.txt");
	ofs_app_state << setw(4) << "iter_cnt";
	ofs_app_state << "\t" << setw(w_Real) << "state";
	ofs_app_state << "\t" << setw(w_Real) << "state_norm";
	ofs_app_state << "\t" << setw(w_Real) << "norm__sort";
	ofs_app_state << endl; 
	VecReal temp_norm_state = SQR(final_ground_state[0]);
	VecReal temp_norm__sort = temp_norm_state; sort(temp_norm__sort);
	if(mm) WRN(NAV3(final_ground_state.norm(),temp_norm_state.norm(),temp_norm__sort.norm()));
	for_Int(i, 0, final_ground_state.size()) 
	{
		ofs_app_state << iofmt("sci");
		ofs_app_state << setw(4) << iter_cnt;
		ofs_app_state << "\t" << setw(w_Real) << final_ground_state[i];
		ofs_app_state << "\t" << setw(w_Real) << temp_norm_state[i];
		ofs_app_state << "\t" << setw(w_Real) << temp_norm__sort[final_ground_state.size()-1-i];
		ofs_app_state << endl; 
	}
	ofs_app_state.close();
}

VecReal NORG::get_impurity_occupations(Int set_idx) const
{
	if(p.if_norg_imp || set_idx < 0 || set_idx >= p.norg_sets
		|| set_idx >= static_cast<Int>(p.nO2sets.size())
		|| set_idx >= static_cast<Int>(p.nI2B.size())
		|| set_idx >= static_cast<Int>(oneedm.dm.size())) {
		ERR("Invalid impurity occupation request; a valid set and bath-only rotation are required.")
	}
	const Int n_imp = p.nO2sets[set_idx] - p.nI2B[set_idx];
	const MatReal& dm = oneedm.dm[set_idx];
	if(n_imp <= 0 || dm.nrows() != static_cast<Idx>(p.nO2sets[set_idx])
		|| dm.ncols() != dm.nrows()) {
		ERR("Impurity occupations do not match the solved density-matrix dimensions.")
	}
	VecReal occupations(n_imp, 0.);
	// dm is already assembled on every rank. Preserve its ground-state
	// averaging; do not divide by p.degel or reduce over MPI ranks again.
	for_Int(i, 0, n_imp) occupations[i] = dm[i][i];
	return occupations;
}

VecReal NORG::write_impurtiy_occupation(Int iter_cnt, const Str& phy_name) const {
    using namespace std;
    VecReal prtil(2 * p.nband, 0.);
    // Every rank holds the density matrices; callers can use these occupations on any rank.
    for_Int(s, 0, 2 * p.nband) prtil[s] = oneedm.dm[s][0][0];

	// chi_A uses MPI collective ops → must run on all ranks before if (mm)
	ChiIomResult chi_result;
	const bool calculate_chi = p.restrain_G_mode == 0 && p.degel == 1 && p.chi_A == 1;
	if (calculate_chi) {
		// Susceptibility stays in scsp and does not create/annihilate a particle.
		CrrltFun chi_calc(mm, p, scsp, oneedm);
        chi_result = chi_calc.find_chi_Sz_N_O_nondeg(groune_lst, final_ground_state[0]);
	}

	if (mm) {
		OFS ofs;
		if (iter_cnt < 0) {
			if (phy_name.empty()) ofs.open("nmat.txt");
			else ofs.open(phy_name + "nmat.out");
		}
		if (iter_cnt > 0) ofs.open(iox + "zic" + prefill0(iter_cnt, 3) + ".nmat.txt");

        Real counter_up = 0., counter_dn = 0.;
        ofs << "#   < n_i >   data:" << endl;
        for_Int(m, 0, p.nband) {
            const Real up = prtil[2*m], down = prtil[2*m+1];
            const Str prefix = "b" + STR(m) + "o0";
            ofs << iofmt() << setw(10) << prefix+"up" << setw(p_Real) << up << endl;
            ofs << iofmt() << setw(10) << prefix+"dn" << setw(p_Real) << down << endl;
            counter_up += up;
            counter_dn += down;
        }
		ofs << setw(10) << "sup" << setw(p_Real) << counter_up << endl;
		ofs << setw(10) << "sdn" << setw(p_Real) << counter_dn << endl;
		ofs << setw(10) << "sum" << setw(p_Real) << counter_up + counter_dn << endl;


		if (calculate_chi) {
			ofs << setw(10) << "chi_Sz"   << setw(p_Real) << chi_result.chi_iom[0][0] << endl;
			ofs << setw(10) << "chi_Nimp" << setw(p_Real) << chi_result.chi_iom[1][0] << endl;

            const Int n_imp_orb = p.nband;
			Int pair_k = 0;
			for_Int(i, 0, n_imp_orb) {
				for_Int(j, i + 1, n_imp_orb) {
					Str lbl = "chi_O_" + STR(i) + STR(j);
					ofs << setw(10) << lbl << setw(p_Real) << chi_result.chi_iom[2 + pair_k][0] << endl;
					pair_k++;
				}
			}

			OFS chi_iom_ofs;
			if (iter_cnt < 0) {
				if (phy_name.empty()) chi_iom_ofs.open("chi_iom.dat");
				else chi_iom_ofs.open(phy_name + "chi_iom.out");
			}
			else {
				chi_iom_ofs.open(iox + "zic" + prefill0(iter_cnt, 3) + ".chi_iom.dat");
			}

			chi_iom_ofs << "# Omega_n chi_Sz chi_Nimp";
			for_Int(i, 0, n_imp_orb) {
				for_Int(j, i + 1, n_imp_orb) {
					chi_iom_ofs << " chi_O_" << i << j;
				}
			}
			chi_iom_ofs << endl;
			chi_iom_ofs << iofmt("sci");

			for_Int(im, 0, chi_result.omega.size()) {
				chi_iom_ofs << setw(p_Real) << chi_result.omega[im];
				for_Int(channel, 0, chi_result.chi_iom.nrows()) {
					chi_iom_ofs << " " << setw(p_Real) << chi_result.chi_iom[channel][im];
				}
				chi_iom_ofs << endl;
			}
			chi_iom_ofs.close();
		}

		ofs.close();
	}
	return prtil;
}



Real NORG::return_nointeractions_ground_state_energy(const MatReal& h0_i) const {
	MatReal htemp = h0_i;
	VecReal vtemp(htemp.ncols(),0.);
	Real energy = 0.;
	heevr(htemp, vtemp);
	for_Int(i, 0, vtemp.size()) {
		if(vtemp[i]<0) energy += vtemp[i];
	}
	return energy;
}
