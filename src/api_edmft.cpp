/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cctype>
#include <cmath>
#include <limits>
#include <cerrno>

//for the stand C++ package ------------------------------------------------------------------------------------------------------------------------
#include "api_edmft.h"
#include "search_restart.h"


namespace {

// Count each particle-number sector together with its complete truncation.
// The first control row contains global division limits; all other rows
// follow the spin sets (m up, m down).
void validate_pm_ensemble(const VEC<VecInt>& particles, const VEC<MatInt>& controls,
                          Int nsets, const Str& context)
{
    if (particles.size() != controls.size() || nsets % 2 != 0)
        ERR("Invalid PM ensemble dimensions: " + context)
    std::map<std::vector<Int>, Int> balance;
    for_Idx(d, 0, particles.size()) {
        const MatInt& control = controls[d];
        if (particles[d].size() != static_cast<Idx>(nsets)
            || control.nrows() != static_cast<Idx>(nsets + 1))
            ERR("Invalid spin-set dimensions: " + context)
        for_Int(flip, 0, 2) {
            std::vector<Int> key;
            key.reserve(nsets + control.size());
            for_Int(set, 0, nsets) key.push_back(particles[d][set ^ flip]);
            key.insert(key.end(), control[0].begin(), control[0].end());
            for_Int(set, 0, nsets) {
                const auto& row = control[1 + (set ^ flip)];
                key.insert(key.end(), row.begin(), row.end());
            }
            balance[key] += flip ? -1 : 1;
        }
    }
    for (const auto& item : balance)
        if (item.second != 0)
            ERR("Spin reversal must preserve the weighted (N, control_divs) ensemble: " + context)
}

template<class T>
void require_pm_rows(const Mat<T>& input, Int first, const Str& context)
{
    for (Idx row = first; row + 1 < input.nrows(); row += 2)
        if (!(input[row] == input[row + 1]))
            ERR("G restraint rows must match for opposite spin sets: " + context
                + ", sets " + STR(row - first) + " and " + STR(row - first + 1))
}

} // namespace

APIedmft::APIedmft(const MyMpi& mm_i, Prmtr& prmtr_i, const Str& file)
    : mm(mm_i), p(prmtr_i), Uc(0.), Jz(0.), mu(0.), nband(0), norbs(0),
      ful_pcl_sch(1), iter_count(0), fit_range(5.), sig_err(0.)
{
	if(mm) WRN("enter APIedmft");

	update(file);
	if(p.restrain_G_mode != 0 && p.restrain_G_mode != 1) {
		ERR("restrain_G_mode must be 0 or 1.")
	}
	const bool orbital_restraints = p.restrain_G_mode == 1;
	if(p.restrain_G_mode == 0) validate_tight_preconvergence_settings();
	// Mode 0 solves all orbitals together: validate a single shared policy
	// before bath fitting/search. Omitted stopping settings retain the normal wide solve.
	const NORG::StopPolicy whole_subspace_policy = p.restrain_G_mode == 0
		? whole_subspace_green_stop_policy() : NORG::StopPolicy();

    if (orbital_restraints) {
        validate_green_restraint_schemes();
        validate_green_iteration_settings();
    }
    Bath bth(mm, p);
    Impurity imp(mm, p, bth, or_deg_idx);
    ImGreen hb(nband, p);
    const Str delta_file = p.testDelta ? "norgDelta.inp" : "Delta.inp";
    hb.read_edmft(delta_file);
    if (mm) hb.write_edmft("hb_read.txt");
    bth.bth_read_fvb("ose_hop");
    bth.number_bath_fit(hb, or_deg_idx);
    if (mm) bth.bth_write_fvb();
    mm.barrier();

    imp.update("eDMFT");
    if (mm) imp.write_H0info(bth, nband);
    ImGreen hb_imp(nband, p);
    imp.find_hb(hb_imp);
    if (mm) hb_imp.write_edmft("hb_fit.txt");
	const bool restored_search_start = edmft_back_up("read");


    if (!restored_search_start && iter_count == 0 && p.if_n_init && !n_init_vals.empty()) {
        const MatReal& h0 = imp.impH.first;
        Int offset = 0;
        for_Int(s, 0, norbs) {
            Int occupied_bath = 0;
            for_Int(j, 0, p.nI2B[s]) if (h0[offset+1+j][offset+1+j] < 0) ++occupied_bath;
            p.npartical[s] = static_cast<Int>(occupied_bath + n_init_vals[s] + 0.5);
            offset += p.nO2sets[s];
        }
    }

	if(restored_search_start) {
		if(mm) WRN("Search starts from the previous EDMFT ground-state subspace: " + NAV(p.npartical));
	} else if(p.if_n_init){
		if(mm) WRN(NAV(p.if_n_init));
		if(mm) WRN(NAV(p.npartical));
	}else{
		if(mm) WRN(NAV(p.if_n_init));
		if(mm) WRN(NAV(p.npartical));
		if(mm) WRN(NAV(p.fix_start_search_N_subspace));
		p.npartical = p.fix_start_search_N_subspace;
		if(mm) WRN(NAV(p.npartical));
	}



    ImGreen g0imp(nband, p);
    imp.find_g0(g0imp);
    if (mm) g0imp.write_edmft("g0imp.txt");

	if(mm) WRN(NAV(g0imp[0]));
	if(mm) WRN(NAV(g0imp.particle_number()));

	//auto_nooc("ful_pcl_sch", imp);
	
	if(p.if_fix_N_subspace){
		if(mm) WRN(NAV(p.if_fix_N_subspace));
		if (p.fix_N_subspace.empty()) {
			ERR("if_fix_N_subspace is enabled, but fix_N_subspace is empty.")
		}

		p.degenerate_N_subspaces.clear();
		p.control_divs_degenerate_N_subspaces.clear();
		p.occupationnumber_N_subspaces.clear();
		p.rotationU_N_subspaces.clear();

		const Int expected_rows = p.control_divs.nrows() + 1;
		const Int expected_npartical_size = p.npartical.size();
		for_Int(d, 0, (Int)p.fix_N_subspace.size()) {
			const VEC<VecInt>& block = p.fix_N_subspace[d];
			if((Int)block.size() != expected_rows) {
				ERR("Fixed N-subspace [" + STR(d) + "] has " + STR(block.size())
					+ " rows; expected " + STR(expected_rows)
					+ " (one npartical row plus all control_divs rows).")
			}
			if((Int)block[0].size() != expected_npartical_size) {
				ERR("Fixed N-subspace [" + STR(d) + "] npartical size is "
					+ STR(block[0].size()) + "; expected " + STR(expected_npartical_size) + ".")
			}

			MatInt control_divs_d(p.control_divs.nrows(), p.control_divs.ncols(), 0);
			for_Int(i, 0, control_divs_d.nrows()) {
				if((Int)block[i + 1].size() != control_divs_d.ncols()) {
					ERR("Fixed N-subspace [" + STR(d) + "] control_divs row [" + STR(i)
						+ "] has size " + STR(block[i + 1].size())
						+ "; expected " + STR(control_divs_d.ncols()) + ".")
				}
				control_divs_d[i] = block[i + 1];
			}
            validate_green_control(control_divs_d, p, "fixed N-subspace " + STR(d));
            for_Int(s, 0, norbs) {
                if(block[0][s] < 0 || block[0][s] > p.nO2sets[s]) ERR("Invalid fixed particle count")
            }
			p.degenerate_N_subspaces.push_back(block[0]);
			p.control_divs_degenerate_N_subspaces.push_back(control_divs_d);
			p.rotationU_N_subspaces.push_back(p.rotationU);
		}

		// Keep the legacy single-subspace fields synchronized with the first block.
		p.npartical = p.degenerate_N_subspaces[0];
		p.control_divs = p.control_divs_degenerate_N_subspaces[0];
		p.rotationU = p.rotationU_N_subspaces[0];
		if(mm) {
			PIO("Fixed degenerate N-subspace information:");
			for_Int(d, 0, (Int)p.degenerate_N_subspaces.size()) {
				WRN("Fixed N-subspace [" + STR(d) + "]: "
					+ NAV2(p.degenerate_N_subspaces[d],
						   p.control_divs_degenerate_N_subspaces[d]));
			}
		}
	}else{
		if(mm) WRN(NAV(p.if_fix_N_subspace));
		if(ful_pcl_sch){
			if(mm) WRN(NAV(ful_pcl_sch));
			auto_nooc("ful_pcl_sch", imp);
		}else{
			if(mm) WRN(NAV(ful_pcl_sch));
			auto_nooc("blk_pcl_sch", imp);
		}
		if(mm) WRN(NAV(p.control_divs));
		if(mm) {
			PIO("Degenerate N-subspace information:");
			for_Int(d, 0, (Int)p.degenerate_N_subspaces.size()) {
				WRN("Degenerate N-subspace [" + STR(d) + "]: "
					+ NAV2(p.degenerate_N_subspaces[d],
						   p.control_divs_degenerate_N_subspaces[d]));
				for_Int(i, 0, (Int)p.occupationnumber_N_subspaces[d].size()) {
					WRN("occupationnumber_N_subspaces[" + STR(d) + "][" + STR(i) + "] = "
						+ STR(p.occupationnumber_N_subspaces[d][i]));
				}
			}
		}
	}
	
	

	VEC<VecInt> green_N_subspaces;
	VEC<MatInt> green_control_divs;
	VEC<VEC<MatReal>> green_rotationU;
	const bool use_degenerate_N_subspaces = !p.degenerate_N_subspaces.empty();

	if(use_degenerate_N_subspaces) {
		const Int n_degenerate = static_cast<Int>(p.degenerate_N_subspaces.size());
		if(n_degenerate != static_cast<Int>(p.control_divs_degenerate_N_subspaces.size())
			|| n_degenerate != static_cast<Int>(p.rotationU_N_subspaces.size())) {
			ERR("The sizes of the degenerate N-subspace data do not match.")
		}
		if(!p.if_fix_N_subspace
			&& n_degenerate != static_cast<Int>(p.occupationnumber_N_subspaces.size())) {
			ERR("The sizes of the automatically searched N-subspace occupation data do not match.")
		}
		green_N_subspaces = p.degenerate_N_subspaces;
		green_control_divs = p.control_divs_degenerate_N_subspaces;
		green_rotationU = p.rotationU_N_subspaces;
	} else {
		green_N_subspaces.push_back(p.npartical);
		green_control_divs.push_back(p.control_divs);
		green_rotationU.push_back(p.rotationU);
	}

	const Int n_ground_subspaces = static_cast<Int>(green_N_subspaces.size());
	if(n_ground_subspaces == 0) ERR("No particle-number subspace is available for the Green-function calculation.")

    validate_pm_ensemble(green_N_subspaces, green_control_divs, norbs, "ground subspaces");

    // In mode 1 the tight solve determines the occupation-weight bins for each
    // orbital. Keep the actual wide controls and validate the ensemble before
    // using any up-spin propagators. Each tight solve still runs exactly once.
    VEC<VEC<MatInt>> orbital_controls;
    if (orbital_restraints) {
        orbital_controls.resize(nband);
        for_Int(d, 0, n_ground_subspaces) {
            Prmtr sector(p);
            sector.npartical = green_N_subspaces[d];
            sector.control_divs = green_control_divs[d];
            sector.rotationU = green_rotationU[d];
            VEC<VecReal> weights;
            preconverge_green_subspace(imp.impH, d, sector, green_rotationU[d],
                                       sector.if_fix_N_subspace ? nullptr : &weights);
            for_Int(m, 0, nband) {
                const Prmtr orbital = make_orbital_parameters(sector, 2 * m, weights);
                orbital_controls[m].push_back(orbital.control_divs);
            }
        }
        for_Int(m, 0, nband)
            validate_pm_ensemble(green_N_subspaces, orbital_controls[m], norbs,
                                 "G orbital " + STR(m));
    }
    ImGreen gfimp(nband, p);
	VEC<Str> occupation_files;
	ImpurityOccupationData occupation_sums = orbital_restraints
		? make_impurity_occupation_data() : ImpurityOccupationData();
	occupations_available = !orbital_restraints;
	n_eles = occupations_available ? 0. : std::numeric_limits<Real>::quiet_NaN();

	for_Int(d, 0, n_ground_subspaces) {
		Prmtr subspace_p(p);
		subspace_p.npartical = green_N_subspaces[d];
		subspace_p.control_divs = green_control_divs[d];
		subspace_p.rotationU = green_rotationU[d];

		if(orbital_restraints) {
            // Pass the validated controls in their existing spin-set layout.
            subspace_p.control_divs_G.reset(norbs);
            for_Int(m, 0, nband)
                subspace_p.control_divs_G[2 * m].reset(orbital_controls[m][d]);
			ImGreen gfimp_subspace(nband, subspace_p);
			solve_restrained_green_subspace(imp.impH, d, subspace_p,
                gfimp_subspace, green_rotationU[d], occupation_sums);
			if(use_degenerate_N_subspaces) {
				p.rotationU_N_subspaces[d] = green_rotationU[d];
			}
			write_unmerged_green(gfimp_subspace, d);
			gfimp += gfimp_subspace;
			if(mm) WRN("N-subspace [" + STR(d) + "] assembled diagonal Green: "
				+ NAV(gfimp_subspace[0]));
			continue;
		}

		preconverge_green_subspace(imp.impH, d, subspace_p, green_rotationU[d]);
		subspace_p.rotationU = green_rotationU[d];
		NORG subspace_norg(mm, subspace_p);

		if(mm) {
			std::ostringstream oss;
			oss << std::fixed << std::setprecision(16)
				<< subspace_norg.return_nointeractions_ground_state_energy(imp.impH.first);
			WRN("N-subspace [" + STR(d) + "] non-interacting ground state energy = " + oss.str());
		}

		// Mode 0 must use the freshly converged tight basis, never an NTR override.
		subspace_norg.uormat = green_rotationU[d];

		// Shared by searched and fixed N-subspaces, including degenerate ensembles.
		if(whole_subspace_policy.mode == NORG::StopPolicy::Mode::NormalConvergence) {
			subspace_norg.up_date_h0_to_solve(imp.impH, 1);
		} else {
			if(mm) WRN("Controlled whole-subspace G solve: N-subspace [" + STR(d)
				+ "], " + (whole_subspace_policy.mode == NORG::StopPolicy::Mode::FixedSteps
					? "rotation_steps=" + STR(whole_subspace_policy.steps)
					: "energy_tolerance=" + STR(whole_subspace_policy.tolerance)));
			subspace_norg.solve_with_policy(imp.impH, whole_subspace_policy,
				subspace_p.if_norg_degenerate);
		}
		green_rotationU[d] = subspace_norg.uormat;
		if(use_degenerate_N_subspaces) {
			p.rotationU_N_subspaces[d] = subspace_norg.uormat;
		}

		{	// Accumulate impurity occupations in the same equal-weight ensemble as gfimp.
			const Str occupation_file_prefix = "degenerate_N_subspace_" + STR(d) + "_";
			occupation_files.push_back(occupation_file_prefix + "nmat.out");
			if(mm) WRN(NAV(occupation_files.back()));
			VecReal prtil = subspace_norg.write_impurtiy_occupation(
                -1, occupation_file_prefix);
            for_Int(k, 0, nband) n_eles[k] += prtil[2*k] + prtil[2*k+1];
		}

		// Release the ground-state operator table after occupation/optional chi_A
		// evaluation. Green functions build their own excited-sector tables.
		subspace_norg.oneedm.table.reset();

		ImGreen gfimp_subspace(nband, subspace_p);
        subspace_norg.get_gimp_diagonal(gfimp_subspace);
		write_unmerged_green(gfimp_subspace, d);
		gfimp += gfimp_subspace;

		if(mm) {
			WRN("N-subspace [" + STR(d) + "]: "
				+ NAV2(gfimp_subspace[0], gfimp_subspace.particle_number()));
		}
	}

	const Real inverse_degeneracy = 1. / static_cast<Real>(n_ground_subspaces);
	gfimp *= Cmplx(inverse_degeneracy, 0.);
	if(orbital_restraints) {
		const ImpurityOccupationData averaged = average_occupations(
			occupation_sums, n_ground_subspaces);
		update_occupation_log(averaged);
		write_occupation_file("nmat.txt", averaged);
		occupations_available = true;
	} else if(occupations_available) {
		n_eles *= inverse_degeneracy;
	}

	if(use_degenerate_N_subspaces) {
		p.npartical = p.degenerate_N_subspaces[0];
		p.control_divs = p.control_divs_degenerate_N_subspaces[0];
		p.rotationU = p.rotationU_N_subspaces[0];
	} else {
		p.rotationU = green_rotationU[0];
	}

    if (mm) gfimp.write_edmft("Gf.out");

	if(mm) WRN(NAV(gfimp[0]));
	if(mm) WRN(NAV(gfimp.particle_number()));

	mm.barrier();
	SLEEP(5);

	// The restrained path writes its in-memory ensemble once above. Only the
	// legacy path averages per-subspace occupation files here.
	if(mm && occupations_available && !orbital_restraints) {
		VEC<Str> occupation_labels;
		VEC<Real> occupation_sums;

		for_Int(d, 0, (Int)occupation_files.size()) {
			IFS ifs(occupation_files[d]);
			if(!ifs) {
				ERR("Cannot open the N-subspace occupation file: " + occupation_files[d])
			}

			bool found_occupation_section = false;
			VEC<Str> file_labels;
			VEC<Real> file_values;
			Str line;
			while(std::getline(ifs, line)) {
				if(!found_occupation_section) {
					if(line.find("#   < n_i >   data:") != Str::npos)
						found_occupation_section = true;
					continue;
				}

				const std::size_t first_non_space = line.find_first_not_of(" \t\r\n");
				if(first_non_space == Str::npos) continue;
				if(line[first_non_space] == '#') break;

				std::istringstream iss(line);
				Str label;
				Real value;
				if(!(iss >> label >> value)) {
					ERR("Invalid occupation line in " + occupation_files[d] + ": " + line)
				}
				file_labels.push_back(label);
				file_values.push_back(value);
			}

			if(!found_occupation_section) {
				ERR("Cannot find '#   < n_i >   data:' in " + occupation_files[d])
			}
			if(file_labels.empty()) {
				ERR("The occupation section is empty in " + occupation_files[d])
			}

			if(d == 0) {
				occupation_labels = file_labels;
				occupation_sums = file_values;
			} else {
				if(file_labels.size() != occupation_labels.size()) {
					ERR("The occupation files contain different numbers of entries.")
				}
				for_Int(i, 0, (Int)occupation_labels.size()) {
					if(file_labels[i] != occupation_labels[i]) {
						ERR("The occupation labels do not match between N-subspace files.")
					}
					occupation_sums[i] += file_values[i];
				}
			}
		}

		if(occupation_files.empty()) {
			ERR("No N-subspace occupation file is available for averaging.")
		}

		OFS ofs("nmat.txt");
		if(!ofs) ERR("Cannot open nmat.txt for writing.")
		ofs << "#   < n_i >   data:" << std::endl;
		for_Int(i, 0, (Int)occupation_labels.size()) {
			ofs << iofmt() << std::setw(10) << occupation_labels[i]
				<< std::setw(p_Real)
				<< occupation_sums[i] / static_cast<Real>(occupation_files.size())
				<< std::endl;
		}
		ofs.close();
	}
	mm.barrier();


    ImGreen seimp(nband, p);
    for_Int(w, 0, seimp.nomgs) for_Int(m, 0, nband)
        seimp[w][m][m] = 1. / g0imp[w][m][m] - 1. / gfimp[w][m][m];
	fit_err = bth.info.tr()[1];
	{ mm.barrier(); SLEEP(1); }
	if (mm) {
        ImGreen last_sig(nband, p);
        if (std::ifstream("Sig.out")) {
            last_sig.read_edmft("Sig.out");
            sig_err = seimp.error(last_sig);
        }
        log("sigerr_update");
        seimp.write_edmft("Sig.out");
	}
	// p holds the first selected subspace and its matching converged rotation.
	// Commit only after the Green function, occupations and self-energy are done.
	edmft_back_up("save");
}

void APIedmft::read_eDMFT(const Str& file) {
	{
		std::vector<double> Ed;
		std::vector<int> Deg;
        double J = 0., beta = 0., U = 0.;
        std::string CoulombF;
		std::vector<double> weight_noc1;
		std::vector<double> weight_noc2;
		std::vector<int> restrain_t;
		std::vector<int> fit_nbaths_t;
		std::vector<double> n_init_t;
		std::vector<int> or_deg_adaptive_t;
		int chi_A_t = 0;
		int if_n_init_t = 0;
		int if_fix_N_subspace_t = 0;
		Int if_eigenstate_number_t = 0;
		Int eigenstate_number_t = 1;
		Int testH0_t = 0;
		Int testDelta_t = 0;
		std::vector<int> fix_start_search_N_subspace_t;
		std::vector<std::vector<std::vector<int>>> fix_N_subspace_t;
		// std::vector<int> distribute_t; //! delete in v1.9.13.p3@2024.11.28
		read_norg_setting("PARAMS.norg", Ed, Deg, J, CoulombF, beta, U, weight_noc1, weight_noc2, restrain_t, fit_nbaths_t, n_init_t, or_deg_adaptive_t, chi_A_t, if_n_init_t, if_eigenstate_number_t, eigenstate_number_t, testH0_t, testDelta_t, if_fix_N_subspace_t, fix_start_search_N_subspace_t, fix_N_subspace_t);

		if(mm) WRN(NAV2(if_eigenstate_number_t,eigenstate_number_t));

        if (CoulombF != "'Ising'" && CoulombF != "Ising")
            ERR("The paramagnetic solver requires density-density CoulombF='Ising'")
        if (!(beta > 0.) || !std::isfinite(beta)) ERR("Expected a positive finite beta")
        Uc = U; Jz = J; p.beta = beta;
        nband = static_cast<Int>(Deg.size());
        norbs = 2 * nband;
        if ((nband != 2 && nband != 3 && nband != 5) || Ed.size() != Deg.size())
            ERR("Expected 2, 3 or 5 physical diagonal impurity levels and matching Deg")
        if (fit_nbaths_t.size() != static_cast<size_t>(norbs))
            ERR("fit_nbaths must contain one bath count per spin set")
        if (!n_init_t.empty() && n_init_t.size() != static_cast<size_t>(norbs))
            ERR("n_init must contain 2M spin occupations")
        if (!if_n_init_t && fix_start_search_N_subspace_t.size() != static_cast<size_t>(norbs))
            ERR("fix_start_search_N_subspace must contain 2M particle counts when if_n_init=0")
        if (!if_fix_N_subspace_t && (weight_noc1.size() != Deg.size() || weight_noc2.size() != Deg.size()
            || or_deg_adaptive_t.size() != static_cast<size_t>(norbs)))
            ERR("Automatic search requires M orbital weights and 2M adaptive labels")
        p.eimp.reset(norbs, norbs, 0.);
        or_deg_idx.reset(norbs, 0);
        if (!if_fix_N_subspace_t) {
            // Search symmetry labels group particle counts, independently of
            // the physical spin-degenerate one-particle inputs.
            std::set<Int> labels(or_deg_adaptive_t.begin(), or_deg_adaptive_t.end());
            if (labels.empty() || *labels.begin() != 1
                || *labels.rbegin() != static_cast<Int>(labels.size()))
                ERR("Adaptive search labels must contain consecutive integers starting at 1")
        }
        for_Int(m, 0, nband) {
            if (Deg[m] != m+1 || !std::isfinite(Ed[m])) ERR("Expected ordered physical Deg=1..M and finite Ed")
            if (fit_nbaths_t[2*m] < 1 || fit_nbaths_t[2*m] != fit_nbaths_t[2*m+1])
                ERR("Each orbital requires identical positive up/down bath counts")
            or_deg_idx[2*m] = or_deg_idx[2*m+1] = m+1;
            p.eimp[2*m][2*m] = p.eimp[2*m+1][2*m+1] = Ed[m];
        }
		weight_nooc.reset(Vec(weight_noc1));
		weight_freze.reset(Vec(weight_noc2));
		restrain = VecInt(restrain_t);
		fit_nbaths.reset(fit_nbaths_t);
		n_init_vals = n_init_t;
		or_deg_adaptive_blk.reset(or_deg_adaptive_t);
		p.chi_A = chi_A_t;
		p.if_n_init = if_n_init_t;
		p.if_fix_N_subspace = if_fix_N_subspace_t;
		p.if_eigenstate_number = if_eigenstate_number_t;
		p.eigenstate_number = eigenstate_number_t;
		p.testH0 = testH0_t;
		p.testDelta = testDelta_t;
		p.fix_start_search_N_subspace.reset(fix_start_search_N_subspace_t);
		p.fix_N_subspace.clear();
		for (const auto& subspace : fix_N_subspace_t) {
			VEC<VecInt> converted_subspace;
			for (const auto& row : subspace)
				converted_subspace.push_back(VecInt(row));
			p.fix_N_subspace.push_back(converted_subspace);
		}
		// distribute = VecInt(distribute_t); //! delete in v1.9.13.p3@2024.11.28
		// if (mm) WRN(NAV2(VecReal(Ed), VecInt(Deg)));
		//--------------------------------------------------

	}

	if (mm) WRN(NAV2(p.eimp, or_deg_idx));
	if (mm) WRN(NAV7(restrain, fit_nbaths, Uc, Jz, p.beta, weight_nooc, weight_freze));

}

void APIedmft::update(const Str& file) {
	{// modify the parameters from edmft.in
		read_eDMFT(file);
		p.U = Uc; p.mu = mu; p.jz = Jz; p.nband = nband; p.norg_sets = p.norbs = norbs;
		p.templet_restrain.reset(restrain);
		p.ndiv = static_cast<Int>(p.templet_restrain.size());
		if (p.ndiv < 4 || p.ndiv % 2 != 0) {
			ERR("restrain must contain an even number of elements >= 4.")
		}
		p.project = NAV(nband) + "band";
		// if(p.if_norg_degenerate == 1) p.bandw = 4 * Uc;
		p.bandw = 4 * fit_range;

		p.after_modify_prmtr(fit_nbaths);
		p.recalc_partical_number(); p.derive();
		if(mm) WRN(NAV(p.control_divs));
		p.Uprm = p.U - 2 * p.jz;
		p.degel = 0;
        n_eles.reset(nband, 0.);
        fit_err.reset(nband, 0.);
		// if (mm) p.print();
	}

	// if (mm) std::cout << std::endl;						// blank line
}

bool APIedmft::if_lock(const Str file) const {
	Str lock_file = file + ".lock";

	IFS ifs(lock_file);
	{ mm.barrier(); SLEEP(1); }
	if (ifs) {
		return true;
	}
	else {
		{ mm.barrier(); SLEEP(5000); }
		return false;
	}
}


void APIedmft::print_log(const Str& lbl, std::ostream& os) const {
    using namespace std;
    Str temp;
    for_Int(m, 0, nband) {
        if (m) temp += "-";
        temp += STR(p.npartical[2*m]);
    }

    os << iofmt();
    os << setw(4) << iter_count;
    // os << setw(8) << p.U;
    os << iofmt("sci");
    os << "  " << setw(w_Real) << sig_err;

    os << setw(3 + 2 * p.nband) << temp;

    os << iofmt() << fixed << setprecision(8);
    for_Int(m, 0, nband) {
        os << "  " << setw(15) << fit_err[m] << "~" << setw(10);
        if (occupations_available) os << n_eles[m];
        else os << "N/A";
    }
    
    os << "  " << present();
    os << "  " << lbl << endl;
}


namespace {

// The constraint row defines the two halves. Each has one special column;
// the bath columns on either side are mirror images of one another.
void classify_search_baths(Vec<VecInt>& controler, const VecInt& groups,
    const VecInt& particles, const VEC<VecReal>& weights,
    const VecReal& freeze_threshold, const VecReal& nooc_threshold)
{
    const Int width = static_cast<Int>(controler[0].size());
    const Int middle = width / 2;
    const Int outer = 1;
    const Int inner = middle - 1;
    const Int restricted = std::min(outer + 1, inner);

    for_Int(group, 1, controler.size()) {
        Int rep = 0;
        while (rep < groups.size() && groups[rep] != group) ++rep;
        if (rep == groups.size()) ERR("Missing orbital representative for group " + STR(group))
        VecInt& row = controler[group];
        if (row.size() != controler[0].size()) ERR("Search row width differs from restrain")
        const VecReal& bath = weights[rep];
        const Int active = particles[rep] - 1;
        if (active < 0 || static_cast<Idx>(active) >= bath.size())
            ERR("No active bath orbital for search set " + STR(rep))
        row = 0;
        row[0] = 1;       // impurity
        row[middle] = 1;  // active bath orbital
        for_Int(j, 0, bath.size()) {
            if (j == active) continue;
            Int level = inner;
            if (bath[j] < freeze_threshold[rep / 2]) level = outer;
            else if (bath[j] < nooc_threshold[rep / 2]) level = restricted;
            ++row[j < active ? level : width - level];
        }
    }
}

} // namespace

void APIedmft::auto_nooc(Str mode, const Impurity& imp) {
	//if(mm) WRN("start auto_nooc");
	p.degenerate_N_subspaces.clear();
	p.control_divs_degenerate_N_subspaces.clear();
	p.occupationnumber_N_subspaces.clear();
	p.rotationU_N_subspaces.clear();
	VecInt or_deg_idx_tmp(p.norg_sets,0.);
	if (mode == "ful_pcl_sch") {
        or_deg_idx_tmp = or_deg_adaptive_blk;

		if (mm) WRN(NAV2(or_deg_idx,or_deg_idx_tmp));

		Occler opcler(mm, p);
		VEC<MatReal> uormat;
		VEC<VecInt> nppsos_N_subspaces;
		VEC<Vec<VecInt>> controlers_N_subspaces;
		// MatReal occnum, occweight;
		VEC<VEC<VecReal>> occnums_N_subspaces;
		VEC<VEC<VecReal>> occweights_N_subspaces;
		VecInt row_sizes(p.norg_sets);
		Int start_idx = 0;
		if(ful_pcl_sch) {
			NORG norg(opcler.find_ground_state_partical(imp.impH, or_deg_idx_tmp));
			p.degenerate_N_subspaces = opcler.degenerate_N_subspaces();
			p.control_divs_degenerate_N_subspaces = opcler.control_divs_degenerate_N_subspaces();
			p.occupationnumber_N_subspaces = opcler.occupationnumber_N_subspaces();
			p.rotationU_N_subspaces = opcler.rotationU_N_subspaces();

			const Int n_degenerate = static_cast<Int>(p.degenerate_N_subspaces.size());
			if(n_degenerate == 0) ERR("No degenerate N-subspace was found.")
			if(n_degenerate != static_cast<Int>(p.control_divs_degenerate_N_subspaces.size())
				|| n_degenerate != static_cast<Int>(p.occupationnumber_N_subspaces.size())
				|| n_degenerate != static_cast<Int>(p.rotationU_N_subspaces.size())) {
				ERR("The sizes of the degenerate N-subspace data do not match.")
			}
			if(norg.scsp.nppso != p.degenerate_N_subspaces[0]) {
				ERR("The returned NORG subspace is not the first degenerate N-subspace.")
			}

			nppsos_N_subspaces = p.degenerate_N_subspaces;
			controlers_N_subspaces.resize(
				n_degenerate,
				Vec<VecInt>(MAX(or_deg_idx_tmp) + 1, VecInt(p.ndiv, 0)));
			occnums_N_subspaces.resize(n_degenerate, VEC<VecReal>(p.norg_sets));
			occweights_N_subspaces.resize(n_degenerate, VEC<VecReal>(p.norg_sets));

			for_Int(d, 0, n_degenerate) {
				if((Int)p.control_divs_degenerate_N_subspaces[d][0].size() != p.ndiv) {
					ERR("control_divs row size mismatch.");
				}
				
				controlers_N_subspaces[d][0] = p.control_divs_degenerate_N_subspaces[d][0];

				VecReal occnum_lst = VECVectoVec(p.occupationnumber_N_subspaces[d]);
				//if(mm) WRN(NAV2(occnum_lst,p.nI2B));
				if((Int)occnum_lst.size() != SUM(p.nI2B)) {
					ERR("Occupation size mismatch on rank "
						+ NAV4(mm.id(), d, occnum_lst.size(), SUM(p.nI2B)));
				}

				if((Int)occnums_N_subspaces[d].size() != p.norg_sets
					|| (Int)occweights_N_subspaces[d].size() != p.norg_sets
					|| (Int)p.nI2B.size() != p.norg_sets) {
					ERR("Occupation container size mismatch.");
				}

				process_occupation_sets(
					occnums_N_subspaces[d],
					occweights_N_subspaces[d],
					p.nI2B,
					occnum_lst);
			}

			uormat = norg.uormat;
			p.npartical = p.degenerate_N_subspaces[0];
			p.rotationU = p.rotationU_N_subspaces[0];
			norg.PIO_occweight(norg.occnum);
		} else {
			// No candidate search: retain the NOOC mode read from the configuration.
			Prmtr p_temp(p);
			// p_temp.templet_restrain[1] = -1; p_temp.templet_restrain[p_temp.ndiv - 1] = 1;
			p_temp.according_nppso(p_temp.npartical);

			NORG norg(mm, p_temp);	norg.read_NTR(); 	norg.up_date_h0_to_solve(imp.impH, 1);
			uormat = norg.uormat;
			// occnum = norg.occnum.mat(p_temp.norg_sets, p_temp.n_rot_orb / p_temp.norg_sets);occweight = occnum;

			nppsos_N_subspaces.push_back(norg.scsp.nppso);
			controlers_N_subspaces.push_back(
				Vec<VecInt>(MAX(or_deg_idx_tmp) + 1, VecInt(p.ndiv, 0)));
			occnums_N_subspaces.push_back(VEC<VecReal>(p.norg_sets));
			occweights_N_subspaces.push_back(VEC<VecReal>(p.norg_sets));
			controlers_N_subspaces[0][0] = p.control_divs[0];

			// Process the single non-degenerate N-subspace.
			VecReal occnum_lst = VECVectoVec(norg.oneedm.occupationnumber);
			process_occupation_sets(
				occnums_N_subspaces[0],
				occweights_N_subspaces[0],
				p.nI2B,
				occnum_lst);

			p.rotationU = uormat;
			norg.PIO_occweight(norg.occnum);
		}

		const Int n_subspaces = static_cast<Int>(nppsos_N_subspaces.size());
		for_Int(d, 0, n_subspaces) {
			const VecInt& nppso = nppsos_N_subspaces[d];
			const VEC<VecReal>& occweights = occweights_N_subspaces[d];
			Vec<VecInt>& controler = controlers_N_subspaces[d];
			classify_search_baths(controler, or_deg_idx_tmp, nppso, occweights, weight_freze, weight_nooc);
			if (mm) WRN(NAV3(d, p.nooc_mode, controler[0]));
			if (ful_pcl_sch) {
				Prmtr subspace_p(p);
				subspace_p.npartical = nppso;
				subspace_p.control_divs = p.control_divs_degenerate_N_subspaces[d];
				subspace_p.according_controler(controler, or_deg_idx_tmp);
				p.control_divs_degenerate_N_subspaces[d] = subspace_p.control_divs;
			} else {
				p.according_controler(controler, or_deg_idx_tmp);
			}
		}

		if(ful_pcl_sch) {
			p.npartical = p.degenerate_N_subspaces[0];
			p.control_divs = p.control_divs_degenerate_N_subspaces[0];
		}

		
		//if(mm) WRN("end ful_pcl_sch");
		apply_restrain_difference(occweights_N_subspaces);

	} else if (mode == "blk_pcl_sch") {
		//if(mm) WRN("enter blk_pcl_sch");

		VecInt or_deg_adaptive = or_deg_adaptive_blk;
        or_deg_idx_tmp = or_deg_idx;
		if (mm) WRN(NAV3(or_deg_idx, or_deg_adaptive,or_deg_idx_tmp));

		Occler opcler(mm, p);
		VEC<MatReal> uormat;
		VEC<VecInt> nppsos_N_subspaces;
		VEC<Vec<VecInt>> controlers_N_subspaces;
		VEC<VEC<VecReal>> occnums_N_subspaces;
		VEC<VEC<VecReal>> occweights_N_subspaces;
		VecInt row_sizes(p.norg_sets);
		Int start_idx = 0;
		NORG norg(opcler.find_ground_state_adaptive(imp.impH, or_deg_adaptive, MAX(or_deg_adaptive)/2.0));
		p.degenerate_N_subspaces = opcler.degenerate_N_subspaces();
		p.control_divs_degenerate_N_subspaces = opcler.control_divs_degenerate_N_subspaces();
		p.occupationnumber_N_subspaces = opcler.occupationnumber_N_subspaces();
		p.rotationU_N_subspaces = opcler.rotationU_N_subspaces();

		const Int n_degenerate = static_cast<Int>(p.degenerate_N_subspaces.size());
		if(n_degenerate == 0) ERR("No degenerate N-subspace was found.")
		if(n_degenerate != static_cast<Int>(p.control_divs_degenerate_N_subspaces.size())
			|| n_degenerate != static_cast<Int>(p.occupationnumber_N_subspaces.size())
			|| n_degenerate != static_cast<Int>(p.rotationU_N_subspaces.size())) {
			ERR("The sizes of the degenerate N-subspace data do not match.")
		}
		if(norg.scsp.nppso != p.degenerate_N_subspaces[0]) {
			ERR("The returned NORG subspace is not the first degenerate N-subspace.")
		}

		nppsos_N_subspaces = p.degenerate_N_subspaces;
		controlers_N_subspaces.resize(
			n_degenerate,
			Vec<VecInt>(MAX(or_deg_idx_tmp) + 1, VecInt(p.ndiv, 0)));
		occnums_N_subspaces.resize(n_degenerate, VEC<VecReal>(p.norg_sets));
		occweights_N_subspaces.resize(n_degenerate, VEC<VecReal>(p.norg_sets));

		for_Int(d, 0, n_degenerate) {
			controlers_N_subspaces[d][0] = p.control_divs_degenerate_N_subspaces[d][0];
			VecReal occnum_lst = VECVectoVec(p.occupationnumber_N_subspaces[d]);
			process_occupation_sets(
				occnums_N_subspaces[d],
				occweights_N_subspaces[d],
				p.nI2B,
				occnum_lst);
		}

		uormat = norg.uormat;
		p.npartical = p.degenerate_N_subspaces[0];
		p.rotationU = p.rotationU_N_subspaces[0];
		norg.PIO_occweight(norg.occnum);

		for_Int(d, 0, n_degenerate) {
			const VecInt& nppso = nppsos_N_subspaces[d];
			const VEC<VecReal>& occweights = occweights_N_subspaces[d];
			Vec<VecInt>& controler = controlers_N_subspaces[d];
			classify_search_baths(controler, or_deg_idx_tmp, nppso, occweights, weight_freze, weight_nooc);
			if (mm) WRN(NAV3(d, p.nooc_mode, controler[0]));
			Prmtr subspace_p(p);
			subspace_p.npartical = nppso;
			subspace_p.control_divs = p.control_divs_degenerate_N_subspaces[d];
			subspace_p.according_controler(controler, or_deg_idx_tmp);
			p.control_divs_degenerate_N_subspaces[d] = subspace_p.control_divs;
		}

		p.npartical = p.degenerate_N_subspaces[0];
		p.control_divs = p.control_divs_degenerate_N_subspaces[0];

		//if(mm) WRN("end blk_pcl_sch");
		apply_restrain_difference(occweights_N_subspaces);
	}

	//if(mm) WRN("end auto_nooc");
}


namespace {

// Distribute one side from the innermost division towards the outermost one.
// A zero threshold consumes the remaining interval and terminates that side.
void count_nooc_weight_bands(VecInt& row, const VecReal& thresholds,
	const VecReal& weights, Idx begin, Idx end, Int first, Int stop, Int step,
	const Str& context)
{
	Real upper = std::numeric_limits<Real>::infinity();
	Idx assigned = 0;
	bool terminated = false;
	for (Int column = first; column != stop; column += step) {
		row[column] = 0;
		if (terminated) continue;
		const Real lower = thresholds[column];
		if (!std::isfinite(lower) || lower < 0. || lower > upper) {
			ERR("Side thresholds must be finite, nonnegative, and nonincreasing outwards: " + context)
		}
		for (Idx m = begin; m < end; ++m) {
			if (weights[m] < upper && (lower == 0. || weights[m] >= lower)) {
				++row[column];
				++assigned;
			}
		}
		upper = lower;
		terminated = lower == 0.;
	}
	if (assigned != end - begin) {
		ERR("Side thresholds do not cover every occupation weight; add an outer zero threshold: " + context)
	}
}

// Shared by the legacy restraint path and the per-Green-block path.
// Fixed active counts preserve the base control's left/right intervals.
MatInt build_restrained_control_divs(const MatInt& base_control,
	const MatReal& differences, const MatInt* active_counts,
	const VEC<VecReal>& occweights, const Str& source)
{
	const Int nsets = static_cast<Int>(occweights.size());
	const bool fixed_active = active_counts != nullptr;
	if(differences.nrows() != static_cast<Idx>(nsets)
		|| (fixed_active && (active_counts->nrows() != static_cast<Idx>(nsets)
			|| active_counts->ncols() != 2))) {
		ERR("Restraint matrix row/column mismatch: " + source)
	}
	MatInt control(base_control);
	if (control.nrows() != static_cast<Idx>(nsets) + 1
		|| control.ncols() < 2 || control.ncols() % 2 != 0
		|| control.ncols() > static_cast<Idx>(std::numeric_limits<Int>::max())
		|| control.ncols() != differences.ncols()) {
		ERR("Control/threshold matrix or occupation-weight shape mismatch: " + source)
	}
	const Int num = static_cast<Int>(control.ncols());
	const Int middle = num / 2;
	for (Int i = 0; i < nsets; ++i) {
		VecInt& row = control[i + 1];
		const VecReal& thresholds = differences[i];
		const VecReal& weights = occweights[i];
		const Str context = source + ", set " + STR(i);
		if (weights.size() > static_cast<Idx>(std::numeric_limits<Int>::max())) {
			ERR("Too many occupation weights for Int division counts: " + context)
		}
		for (Int column = 0; column < num; ++column) {
			if (row[column] < 0) ERR("Negative control-division count: " + context)
		}
		for (Idx m = 0; m < weights.size(); ++m) {
			if (!std::isfinite(weights[m]) || weights[m] < 0.) {
				ERR("Occupation weights must be finite and nonnegative: " + context)
			}
		}
		if (fixed_active) {
			const VecInt& active = (*active_counts)[i];
			if (active[0] < 0 || active[1] < 0) {
				ERR("Active-orbital counts must be nonnegative: " + context)
			}
			row[0] = active[0];
			row[middle] = active[1];
		}

		LLInt left_count = 0, right_count = 0;
		for (Int column = 1; column < middle; ++column) left_count += row[column];
		for (Int column = middle + 1; column < num; ++column) right_count += row[column];
		// occweights contains bath orbitals only: row[0] is not part of it.
		const LLInt bath_count = left_count + row[middle] + right_count;
		if (bath_count != static_cast<LLInt>(weights.size())) {
			ERR("N_left + active_bath + N_right must equal the occupation-weight count: "
				+ context + "; got " + STR(bath_count) + ", expected " + STR(weights.size())
				+ ". Fixed active counts keep the original left/right intervals.")
		}
		Idx left = static_cast<Idx>(left_count);
		Idx active_count = static_cast<Idx>(row[middle]);
		Idx right = static_cast<Idx>(right_count);
		if (!fixed_active) {
			const Real active_threshold = thresholds[middle];
			if (!std::isfinite(active_threshold) || active_threshold < 0.) {
				ERR("Active threshold must be finite and nonnegative: " + context)
			}
			Idx first_active = weights.size(), last_active = 0;
			active_count = 0;
			for (Idx m = 0; m < weights.size(); ++m) {
				if (weights[m] >= active_threshold) {
					if (active_count == 0) first_active = m;
					last_active = m;
					++active_count;
				}
			}
			if (active_count != 0) {
				// Equivalent to the two threshold crossings, including endpoints.
				if (last_active - first_active + 1 != active_count) {
					ERR("Above-threshold weights must form one contiguous active interval: " + context)
				}
				left = first_active;
			} // With no active weights, retain the old left boundary.
			right = weights.size() - left - active_count;
			row[middle] = static_cast<Int>(active_count);
		}

		// Keep row[0]; replace every side bin, including bins beyond a zero.
		count_nooc_weight_bands(row, thresholds, weights,
			0, left, middle - 1, 0, -1, context);
		const Idx right_begin = left + active_count;
		count_nooc_weight_bands(row, thresholds, weights,
			right_begin, right_begin + right, middle + 1, num, 1, context);
	}
	return control;
}

} // namespace

void APIedmft::apply_restrain_difference(
	const VEC<VEC<VecReal>>& occweights_N_subspaces)
{
	if (p.if_norg_imp || p.restrain_difference_dim == 0) return;
	if (p.norg_sets < 0 || p.restrain_difference_dim != p.norg_sets
		|| p.restrain_difference.nrows() != static_cast<Idx>(p.norg_sets)) {
		ERR("restrain_difference must have one row per norg_set.")
	}
	const bool fixed_active = p.restrain_active_orbit_number_dim != 0;
	if (fixed_active && (p.restrain_active_orbit_number_dim != p.norg_sets
		|| p.restrain_active_orbit_number.nrows() != static_cast<Idx>(p.norg_sets)
		|| p.restrain_active_orbit_number.ncols() != 2)) {
		ERR("restrain_active_orbit_number must have one [impurity, active_bath] row per norg_set.")
	}

	// Work on copies so invalid input cannot leave partially updated subspaces.
	const bool has_subspaces = !p.control_divs_degenerate_N_subspaces.empty();
	VEC<MatInt> updated(p.control_divs_degenerate_N_subspaces);
	if (!has_subspaces) updated.push_back(p.control_divs);
	if (updated.size() != occweights_N_subspaces.size()) {
		ERR("Control-matrix and occupation-weight subspace counts do not match.")
	}
	for (Idx j = 0; j < updated.size(); ++j) {
		updated[j].reset(build_restrained_control_divs(updated[j],
			p.restrain_difference,
			fixed_active ? &p.restrain_active_orbit_number : nullptr,
			occweights_N_subspaces[j], "subspace " + STR(j)));
	}

	// Synchronize the entire first matrix, not its constraint row [0].
	p.control_divs.reset(updated[0]);
	if (has_subspaces) p.control_divs_degenerate_N_subspaces.swap(updated);
}



NORG::StopPolicy APIedmft::green_stop_policy(Int scheme_idx) const
{
	if(scheme_idx < 0
		|| scheme_idx >= static_cast<Int>(p.norg_each_orbital_converg_error.size())
		|| scheme_idx >= static_cast<Int>(p.norg_each_orbital_converg_step_number.size())) {
		ERR("Missing NORG stopping settings for G scheme " + STR(scheme_idx))
	}
	const Real error = p.norg_each_orbital_converg_error[scheme_idx];
	if(!std::isfinite(error) || error < 0.) {
		ERR("NORG convergence error must be finite and nonnegative: scheme " + STR(scheme_idx))
	}
	const NORG::StopPolicy policy(error == 0.
		? NORG::StopPolicy::Mode::FixedSteps : NORG::StopPolicy::Mode::EnergyTolerance,
		error, p.norg_each_orbital_converg_step_number[scheme_idx]);
	if(!policy.valid()) {
		ERR("Zero convergence error requires a nonnegative NORG step count: scheme " + STR(scheme_idx))
	}
	return policy;
}

void APIedmft::validate_green_iteration_settings() const
{
	validate_tight_preconvergence_settings();
	validate_green_stopping_settings();
}

void APIedmft::validate_tight_preconvergence_settings() const
{
	if(p.if_norg_imp) {
		ERR("Tight G preconvergence requires bath-only rotation")
	}
	if(p.ndiv <= 0 || p.restrain_tight_divs.size() != static_cast<Idx>(p.ndiv)) {
		ERR("restrain_tight_divs must contain one constraint per NORG division (no count prefix)")
	}
	if(p.iter_max_norg <= 0) {
		ERR("Tight G preconvergence requires a positive iter_max_norg")
	}
}

void APIedmft::validate_green_stopping_settings() const
{
	if(p.norg_sets <= 0
		|| p.norg_each_orbital_converg_error.size() != static_cast<Idx>(p.norg_sets)
		|| p.norg_each_orbital_converg_step_number.size() != static_cast<Idx>(p.norg_sets)) {
		ERR("NORG orbital stopping arrays must contain one entry per spin-resolved G scheme")
	}
	for_Int(i, 0, p.norg_sets) green_stop_policy(i);
		if(p.norg_sets % 2 != 0) ERR("Paramagnetic NORG stopping settings require up/down pairs")
		for(Int i = 0; i < p.norg_sets; i += 2) {
			const NORG::StopPolicy up = green_stop_policy(i);
			const NORG::StopPolicy down = green_stop_policy(i + 1);
			if(!up.equivalent_to(down)) {
				ERR("Paramagnetic up/down NORG stopping policies must match: pair " + STR(i / 2))
			}
		}
}

NORG::StopPolicy APIedmft::whole_subspace_green_stop_policy() const
{
	if(p.norg_each_orbital_converg_error.size() == 0
		&& p.norg_each_orbital_converg_step_number.size() == 0) {
		return NORG::StopPolicy();
	}
	// Mode 0 has one whole-subspace solve: both configured values must match
	// across all sets, even when a positive error makes the step count inactive.
	// Tight settings are checked separately; the common policy selects how to stop.
	validate_green_stopping_settings();
	const NORG::StopPolicy policy = green_stop_policy(0);
	for_Int(i, 1, p.norg_sets) {
		const NORG::StopPolicy candidate = green_stop_policy(i);
		if(policy.tolerance != candidate.tolerance || policy.steps != candidate.steps) {
			ERR("restrain_G_mode=0 solves all orbitals together and requires identical "
				"norg_each_orbital_converg_error and norg_each_orbital_converg_step_number "
				"values across all sets; mismatch at set " + STR(i)
				+ "; set 0: error=" + STR(policy.tolerance) + ", steps=" + STR(policy.steps)
				+ "; set " + STR(i) + ": error=" + STR(candidate.tolerance)
				+ ", steps=" + STR(candidate.steps))
		}
	}
	return policy;
}

void APIedmft::validate_green_restraint_schemes() const
{
	const Int nsets = p.norg_sets;
	const bool fixed = p.if_fix_N_subspace != 0;
	if(fixed) {
		if(static_cast<Int>(p.control_divs_G.size()) != nsets) {
			ERR("Fixed N-subspaces require one complete control_divs_G matrix per NORG set.")
		}
	} else if(static_cast<Int>(p.restrain_difference_G.size()) != nsets
		|| static_cast<Int>(p.restrain_active_orbit_number_G.size()) != nsets
		|| (p.control_divs_G.size() != 0
			&& static_cast<Int>(p.control_divs_G.size()) != nsets)) {
		ERR("Adaptive G restraints must contain one entry per NORG set.")
	}

	// Check every scheme, including down-spin entries skipped in PM mode.
	for_Int(b, 0, nsets) {
		if(!fixed) {
			const MatReal& thresholds = p.restrain_difference_G[b];
			const MatInt& active = p.restrain_active_orbit_number_G[b];
			if(thresholds.nrows() != static_cast<Idx>(nsets)
				|| thresholds.ncols() != static_cast<Idx>(p.ndiv)
				|| active.nrows() != static_cast<Idx>(nsets) || active.ncols() != 2) {
				ERR("Invalid G restraint shape for scheme " + STR(b))
			}
            require_pm_rows(thresholds, 0, "thresholds for G scheme " + STR(b));
            require_pm_rows(active, 0, "active orbitals for G scheme " + STR(b));
			for_Int(i, 0, nsets) {
				if(active[i][0] != 1
					|| active[i][1] < 0 || active[i][1] > p.nI2B[i]) {
					ERR("G restraints must preserve impurity/bath dimensions: scheme "
						+ STR(b) + ", set " + STR(i))
				}
			}
		}
		if(p.control_divs_G.size() != 0) {
			const MatInt& control = p.control_divs_G[b];
			if(fixed) {
				validate_green_control(control, p, "Fixed G scheme " + STR(b));
			} else if(control.nrows() != static_cast<Idx>(nsets + 1)
				|| control.ncols() != static_cast<Idx>(p.ndiv)) {
				ERR("Invalid control_divs_G shape for scheme " + STR(b))
			}
            require_pm_rows(control, 1, "control_divs_G scheme " + STR(b));
		}
	}
		for(Int b = 0; b < nsets; b += 2) {
			if((!fixed
					&& (!(p.restrain_difference_G[b] == p.restrain_difference_G[b + 1])
						|| !(p.restrain_active_orbit_number_G[b] == p.restrain_active_orbit_number_G[b + 1])))
				|| (p.control_divs_G.size() != 0
					&& !(p.control_divs_G[b] == p.control_divs_G[b + 1]))) {
				ERR("Paramagnetic up/down G schemes must be identical: pair " + STR(b / 2))
			}
		}
}

void APIedmft::validate_green_control(const MatInt& control,
	const Prmtr& solver_p, const Str& context)
{
	if(control.nrows() != static_cast<Idx>(solver_p.norg_sets + 1)
		|| control.ncols() != static_cast<Idx>(solver_p.ndiv)
		|| solver_p.nO2sets.size() != static_cast<Idx>(solver_p.norg_sets)
		|| solver_p.nI2B.size() != static_cast<Idx>(solver_p.norg_sets)) {
		ERR("G control does not match the solver dimensions: " + context)
	}
	for_Int(i, 0, solver_p.norg_sets) {
		LLInt count = 0;
		for_Int(c, 0, solver_p.ndiv) {
			if(control[i + 1][c] < 0) {
				ERR("Negative G control-division count: " + context + ", set " + STR(i))
			}
			count += control[i + 1][c];
		}
		if(count != solver_p.nO2sets[i]
			|| control[i + 1][0] != solver_p.nO2sets[i] - solver_p.nI2B[i]) {
			ERR("G control must preserve each set's impurity/bath counts: " + context + ", set " + STR(i))
		}
	}
}

MatInt APIedmft::make_fixed_green_control(Int set) const
{
	if(set < 0 || set >= static_cast<Int>(p.control_divs_G.size())) {
		ERR("Missing control_divs_G matrix for fixed G scheme " + STR(set))
	}
	// Copy the entire supplied matrix, including its constraint row.
	return p.control_divs_G[set];
}

MatInt APIedmft::make_adaptive_green_control(const Prmtr& subspace_p,
	Int set, const VEC<VecReal>& occupation_weights) const
{
	const Int b = set;
	// An explicit G control supplies the constraint row and left/right intervals;
	// otherwise use this N-subspace's control. Never modify the search results.
	const MatInt& base = p.control_divs_G.size() != 0
		? p.control_divs_G[b] : subspace_p.control_divs;
	const Str context = "G scheme " + STR(b);
	if(occupation_weights.size() != static_cast<Idx>(subspace_p.norg_sets)) {
		ERR("Occupation weights must contain one entry per NORG set: " + context)
	}
	for_Int(i, 0, subspace_p.norg_sets) {
		if(occupation_weights[i].size() != static_cast<Idx>(subspace_p.nI2B[i])) {
			ERR("Occupation weights do not match the bath count: " + context + ", set " + STR(i))
		}
	}
	return build_restrained_control_divs(base,
		p.restrain_difference_G[b], &p.restrain_active_orbit_number_G[b],
		occupation_weights, context);
}

Prmtr APIedmft::make_orbital_parameters(const Prmtr& subspace_p,
	Int set, const VEC<VecReal>& occupation_weights) const
{
	Prmtr block_p(subspace_p);
	block_p.control_divs.reset(subspace_p.if_fix_N_subspace
		? make_fixed_green_control(set)
		: make_adaptive_green_control(subspace_p, set, occupation_weights));
	const Str context = "G scheme " + STR(set);
	validate_green_control(block_p.control_divs, block_p, context);
	if(block_p.npartical.size() != static_cast<Idx>(block_p.norg_sets)) {
		ERR("G particle counts do not match the solver dimensions: " + context)
	}
	for_Int(i, 0, block_p.norg_sets) {
		if(block_p.npartical[i] < 0 || block_p.npartical[i] > block_p.nO2sets[i]) {
			ERR("G particle count is outside the orbital range: " + context + ", set " + STR(i))
		}
	}
	return block_p;
}

void APIedmft::write_unmerged_green(const ImGreen& green, Int subspace_idx, Int set) const
{
    if (!mm) return;
    Str name = "Gf_N_subspace_" + STR(subspace_idx);
    if (set >= 0) name += "_scheme_" + STR(set) + "_set_" + STR(set);
    green.write_edmft(name + ".txt");
}

void APIedmft::preconverge_green_subspace(const Impdata& impH, Int subspace_idx,
	const Prmtr& subspace_p, VEC<MatReal>& reference_rotation,
	VEC<VecReal>* occupation_weights)
{
	if(subspace_idx < 0 || subspace_p.norg_sets <= 0
		|| subspace_p.nO2sets.size() != static_cast<Idx>(subspace_p.norg_sets)
		|| reference_rotation.size() != static_cast<Idx>(subspace_p.norg_sets)) {
		ERR("Missing rotation data for G subspace " + STR(subspace_idx))
	}
	for_Int(i, 0, subspace_p.norg_sets) {
		if(reference_rotation[i].nrows() != static_cast<Idx>(subspace_p.nO2sets[i])
			|| reference_rotation[i].ncols() != static_cast<Idx>(subspace_p.nO2sets[i])) {
			ERR("Initial G rotation matrix has incompatible dimensions.")
		}
	}
	Prmtr tight_p = subspace_p.make_tight_subspace_parameters();
	validate_green_control(tight_p.control_divs, tight_p, "Tight N-subspace " + STR(subspace_idx));
	tight_p.rotationU = reference_rotation;
	if(mm) WRN("Tight preconvergence for N-subspace [" + STR(subspace_idx) + "]");
	NORG tight_norg(mm, tight_p);
	// Preserve the matching search result (or fixed-subspace initial basis).
	// Constructor identity matrices and NTR files must not replace it.
	tight_norg.uormat = reference_rotation;
	tight_norg.solve_with_policy(impH, NORG::StopPolicy());
	reference_rotation = tight_norg.uormat;
	if(occupation_weights) {
		VEC<VecReal> occnums(subspace_p.norg_sets);
		occupation_weights->resize(subspace_p.norg_sets);
		process_occupation_sets(occnums, *occupation_weights, tight_p.nI2B, tight_norg.occnum);
	}
} // Destroy the tight solver and its space/table before constructing a wide solver.

void APIedmft::solve_restrained_green_subspace(const Impdata& impH, Int subspace_idx,
    const Prmtr& subspace_p, ImGreen& assembled_green,
	VEC<MatReal>& reference_rotation, ImpurityOccupationData& occupation_sums)
{
	ImpurityOccupationData subspace_occupations = make_impurity_occupation_data();

    for_Int(m, 0, nband) {
        const Int set = 2*m;
        Prmtr block_p(subspace_p);
        block_p.control_divs.reset(subspace_p.control_divs_G[set]);
		block_p.rotationU = reference_rotation;
		NORG block_norg(mm, block_p);
		// Every scheme starts from the same tight reference, independently.
		block_norg.uormat = reference_rotation;
        if(mm) WRN("N-subspace [" + STR(subspace_idx) + "], orbital [" + STR(m) + "]")
        block_norg.solve_with_policy(impH, green_stop_policy(set));
		// Subsequent G/occupation calculations need the solved state and density
		// matrix, but not the ground-state operator table.
		block_norg.oneedm.table.reset();
        ImGreen orbital_green(1, block_p);
        block_norg.get_gimp_orbital(orbital_green, m);
        write_unmerged_green(orbital_green, subspace_idx, set);
        for_Int(w, 0, assembled_green.nomgs) assembled_green[w][m][m] = orbital_green[w][0][0];
        collect_orbital_occupations(block_norg, subspace_idx, set,
			subspace_occupations);
	}
	finalize_subspace_occupations(subspace_idx, subspace_occupations, occupation_sums);
}

APIedmft::ImpurityOccupationData APIedmft::make_impurity_occupation_data() const
{
	if(p.norg_sets <= 0 || p.norg_sets % 2 != 0
		|| p.nO2sets.size() != static_cast<Idx>(p.norg_sets)
		|| p.nI2B.size() != static_cast<Idx>(p.norg_sets)) {
		ERR("Impurity occupations require matching up/down set pairs and orbital counts.")
	}
	ImpurityOccupationData data;
	data.collected_sets.reset(p.norg_sets, 0);
    data.values_by_set.reset(p.norg_sets, 0.);
	return data;
}

void APIedmft::validate_impurity_occupations(const ImpurityOccupationData& data,
	Int expected_contributions) const
{
	if(expected_contributions < 0
		|| data.values_by_set.size() != static_cast<Idx>(p.norg_sets)
		|| data.collected_sets.size() != data.values_by_set.size()) {
		ERR("Invalid impurity occupation container.")
	}
	for_Int(s, 0, p.norg_sets) {
        if (data.collected_sets[s] != expected_contributions || !std::isfinite(data.values_by_set[s]))
            ERR("Invalid or incomplete orbital occupation in set " + STR(s))
	}
}

void APIedmft::collect_orbital_occupations(const NORG& norg, Int subspace_idx,
	Int set_idx, ImpurityOccupationData& subspace_occupations) const
{
    if(set_idx < 0 || set_idx > p.norg_sets - 2 || set_idx % 2 != 0
		|| subspace_occupations.values_by_set.size() != static_cast<Idx>(p.norg_sets)
		|| subspace_occupations.collected_sets.size() != static_cast<Idx>(p.norg_sets)) {
		ERR("Invalid occupation target in N-subspace " + STR(subspace_idx))
	}
	// PM collects actual up/down occupations from the same solved ground state;
	// it does not copy up to down or evaluate a down-spin Green function.
	for_Int(s, set_idx, set_idx + 2) {
		if(subspace_occupations.collected_sets[s] != 0) {
			ERR("Occupation set collected twice: N-subspace " + STR(subspace_idx) + ", set " + STR(s))
		}
        subspace_occupations.values_by_set[s] = norg.get_impurity_occupations(s)[0];
		subspace_occupations.collected_sets[s] = 1;
	}
}

void APIedmft::finalize_subspace_occupations(Int subspace_idx,
	const ImpurityOccupationData& subspace_occupations,
	ImpurityOccupationData& occupation_sums) const
{
	validate_impurity_occupations(subspace_occupations, 1);
	// N-subspaces arrive in order; this also catches repeated/missing subspaces.
	validate_impurity_occupations(occupation_sums, subspace_idx);
	for_Int(s, 0, p.norg_sets) {
		occupation_sums.values_by_set[s] += subspace_occupations.values_by_set[s];
		++occupation_sums.collected_sets[s];
	}
}

APIedmft::ImpurityOccupationData APIedmft::average_occupations(
	const ImpurityOccupationData& occupation_sums, Int n_subspaces) const
{
	if(n_subspaces <= 0) ERR("Cannot average an empty occupation ensemble.")
	validate_impurity_occupations(occupation_sums, n_subspaces);
	ImpurityOccupationData averaged(occupation_sums);
	for_Int(s, 0, p.norg_sets) {
		averaged.values_by_set[s] *= 1. / static_cast<Real>(n_subspaces);
		averaged.collected_sets[s] = 1;
	}
	return averaged;
}

void APIedmft::update_occupation_log(const ImpurityOccupationData& occupations)
{
	validate_impurity_occupations(occupations, 1);
    for_Int(m, 0, nband)
        n_eles[m] = occupations.values_by_set[2*m] + occupations.values_by_set[2*m+1];
}

void APIedmft::write_occupation_file(const Str& filename,
	const ImpurityOccupationData& occupations) const
{
	validate_impurity_occupations(occupations, 1);
	if(mm) {
		OFS ofs(filename);
		if(!ofs) ERR("Cannot open impurity occupation file: " + filename)
		ofs << "#   < n_i >   data:" << std::endl;
		Real sum_up = 0., sum_down = 0.;
		for_Int(b, 0, p.norg_sets / 2) {
            const Real up = occupations.values_by_set[2*b], down = occupations.values_by_set[2*b+1];
            const Str label = "b" + STR(b) + "o0";
            ofs << iofmt() << std::setw(10) << label+"up" << std::setw(p_Real) << up << std::endl;
            ofs << iofmt() << std::setw(10) << label+"dn" << std::setw(p_Real) << down << std::endl;
            sum_up += up;
            sum_down += down;
		}
		ofs << std::setw(10) << "sup" << std::setw(p_Real) << sum_up << std::endl;
		ofs << std::setw(10) << "sdn" << std::setw(p_Real) << sum_down << std::endl;
		ofs << std::setw(10) << "sum" << std::setw(p_Real) << sum_up + sum_down << std::endl;
		ofs.close();
		if(!ofs) ERR("Cannot finish writing impurity occupation file: " + filename)
	}
}

//change for the stand C++----------------------------------------------------------------------------------------------------------------------------

namespace {

// Read a numeric vector with an optional count prefix into Vec<T>.
// Accept the same separators and trailing comments as read_counted_matrix.
template<typename T>
void read_numeric_vector(std::istream& input, const std::string& key,
	const std::string& filename, Vec<T>& vector, bool has_count)
{
	const std::string context = key + " in " + filename;
	std::string value_text;
	std::getline(input, value_text);
	for (char& ch : value_text) if (ch == ',') ch = ' ';
	std::istringstream values(value_text);
	char bracket;
	Int declared_count = -1;
	if (!(values >> bracket) || bracket != '[') {
		ERR(context + " must start with [")
	}
	if (has_count) {
		if (!(values >> declared_count) || declared_count < 0) {
			ERR(context + " must start with [nonnegative_integer_count")
		}
		const int after_count = values.peek();
		if (after_count != ']' && after_count != std::char_traits<char>::eof()
			&& !std::isspace(static_cast<unsigned char>(after_count))) {
			ERR("Invalid integer count or missing separator in " + context)
		}
	}
	std::vector<T> parsed;
	values >> std::ws;
	while (values.peek() != ']') {
		T value;
		if (!(values >> value) || !std::isfinite(value)) {
			ERR("Invalid, out-of-range, or incomplete numeric value in " + context)
		}
		const int next = values.peek();
		if (next != ']' && next != std::char_traits<char>::eof()
			&& !std::isspace(static_cast<unsigned char>(next))) {
			ERR("Invalid numeric value or missing separator in " + context)
		}
		parsed.push_back(value);
		values >> std::ws;
	}
	values.get(); // closing bracket
	values >> std::ws;
	if (!values.eof() && values.peek() != '#') {
		ERR("Unexpected text after " + context)
	}
	if (has_count && parsed.size() != static_cast<std::size_t>(declared_count)) {
		ERR(context + " declares " + STR(declared_count) + " values, but "
			+ STR(parsed.size()) + " were parsed")
	}
	vector.reset(parsed.size());
	for (Idx i = 0; i < parsed.size(); ++i) vector[i] = parsed[i];
}

// All elements in [value_0 ...] are data, including a leading zero.
template<typename T>
void read_uncounted_vector(std::istream& input, const std::string& key,
	const std::string& filename, Vec<T>& vector)
{
	read_numeric_vector(input, key, filename, vector, false);
}

// The first element in [count value_0 ...] is metadata.
template<typename T>
void read_counted_vector(std::istream& input, const std::string& key,
	const std::string& filename, Vec<T>& vector)
{
	read_numeric_vector(input, key, filename, vector, true);
}

// Read [declared_rows [row_0] [row_1] ...] into any numeric Mat<T>.
// Commas are optional; both dimensions are inferred from the parsed rows.
// A declared count of zero disables use of the matrix but may retain preset rows.
template<typename T>
void read_counted_matrix(std::istream& input, const std::string& key,
	const std::string& filename, Mat<T>& matrix, Int& declared_dim)
{
	const std::string context = key + " in " + filename;
	std::string value_text;
	std::getline(input, value_text);
	for (char& ch : value_text) if (ch == ',') ch = ' ';
	std::istringstream values(value_text);
	char bracket;
	Int declared_rows = -1;
	if (!(values >> bracket) || bracket != '['
		|| !(values >> declared_rows) || declared_rows < 0) {
		ERR(context + " must start with [nonnegative_integer_row_count")
	}

	std::vector<std::vector<T>> rows;
	values >> std::ws;
	while (values.peek() == '[') {
		values.get();
		std::vector<T> row;
		values >> std::ws;
		while (values.peek() != ']') {
			T value;
			if (!(values >> value) || !std::isfinite(value)) {
				ERR("Invalid, out-of-range, or incomplete numeric value in " + context)
			}
			// Reject partially read numbers, e.g. 1.5 or 1e2 for MatInt.
			const int next = values.peek();
			if (next != ']' && next != std::char_traits<char>::eof()
				&& !std::isspace(static_cast<unsigned char>(next))) {
				ERR("Invalid numeric value or missing separator in " + context)
			}
			row.push_back(value);
			values >> std::ws;
		}
		values.get(); // closing bracket of this row
		if (row.empty() || (!rows.empty() && row.size() != rows.front().size())) {
			ERR(context + " rows must be nonempty and have equal column counts")
		}
		rows.push_back(row);
		values >> std::ws;
	}
	if (!(values >> bracket) || bracket != ']') {
		ERR("Missing closing bracket or invalid row in " + context)
	}
	values >> std::ws;
	if (!values.eof() && values.peek() != '#') {
		ERR("Unexpected text after " + context)
	}
	if (declared_rows > 0 && rows.size() != static_cast<std::size_t>(declared_rows)) {
		ERR(context + " declares " + STR(declared_rows) + " rows, but "
			+ STR(rows.size()) + " were parsed")
	}

	const Idx nrows = rows.size();
	const Idx ncols = rows.empty() ? 0 : rows.front().size();
	matrix.reset(nrows, ncols, T{});
	for (Idx i = 0; i < nrows; ++i) {
		for (Idx j = 0; j < ncols; ++j) {
			matrix[i][j] = rows[i][j];
		}
	}
	declared_dim = declared_rows;
}

// Read [matrix_count [row_count [row_0] ...] ...] into Vec<Mat<T>>.
// G counts are actual sizes (zero means empty); mode is stored separately.
template<typename T>
void read_counted_matrix_vector(std::istream& input, const std::string& key,
	const std::string& filename, Vec<Mat<T>>& matrices)
{
	const std::string context = key + " in " + filename;
	std::string value_text;
	std::getline(input, value_text);
	for (char& ch : value_text) if (ch == ',') ch = ' ';
	std::istringstream values(value_text);
	char bracket;
	Int declared_count = -1;
	if (!(values >> bracket) || bracket != '['
		|| !(values >> declared_count) || declared_count < 0) {
		ERR(context + " must start with [nonnegative_integer_matrix_count")
	}
	VEC<Mat<T>> parsed;
	values >> std::ws;
	while (values.peek() == '[') {
		// Isolate a complete block: read_counted_matrix consumes its input line.
		std::string block;
		Idx depth = 0;
		do {
			char ch;
			if (!values.get(ch)) ERR("Incomplete matrix block in " + context)
			block += ch;
			if (ch == '[') ++depth;
			else if (ch == ']') --depth;
		} while (depth != 0);
		std::istringstream matrix_input(block);
		Mat<T> matrix;
		Int declared_rows = 0;
		read_counted_matrix(matrix_input, key + "[" + STR(parsed.size()) + "]",
			filename, matrix, declared_rows);
		if (matrix.nrows() != static_cast<Idx>(declared_rows)) {
			ERR(context + " matrix " + STR(parsed.size()) + " declares "
				+ STR(declared_rows) + " rows, but " + STR(matrix.nrows()) + " were parsed")
		}
		parsed.push_back(matrix);
		values >> std::ws;
	}
	if (!(values >> bracket) || bracket != ']') {
		ERR("Missing closing bracket or invalid matrix block in " + context)
	}
	values >> std::ws;
	if (!values.eof() && values.peek() != '#') {
		ERR("Unexpected text after " + context)
	}
	if (parsed.size() != static_cast<Idx>(declared_count)) {
		ERR(context + " declares " + STR(declared_count) + " matrices, but "
			+ STR(parsed.size()) + " were parsed")
	}
	// Reset both container and elements: Mat assignment requires matching shapes.
	matrices.reset(parsed.size());
	for (Idx g = 0; g < parsed.size(); ++g) matrices[g].reset(parsed[g]);
}

} // namespace

void APIedmft::read_norg_setting(
	const std::string& filename,
	std::vector<double>& Ed,
	std::vector<int>& Deg,
	double& J,
	std::string& CoulombF,
	double& beta,
	double& U,
	std::vector<double>& noc1,
	std::vector<double>& noc2,
	std::vector<int>& restrain,
	std::vector<int>& fit_nbaths,
	std::vector<double>& n_init,
	std::vector<int>& or_deg_adaptive,
	int& chi_A,
	int& if_n_init,
	Int& if_eigenstate_number,
	Int& eigenstate_number,
	Int& testH0,
	Int& testDelta,
	int& if_fix_N_subspace,
	std::vector<int>& fix_start_search_N_subspace,
	std::vector<std::vector<std::vector<int>>>& fix_N_subspace
) {
	p.restrain_difference.reset();
	p.restrain_difference_dim = 0;
	bool found_restrain_difference = false;
	p.restrain_active_orbit_number.reset();
	p.restrain_active_orbit_number_dim = 0;
	bool found_restrain_active_orbit_number = false;
	p.restrain_difference_G.reset();
	p.restrain_active_orbit_number_G.reset();
	p.control_divs_G.reset();
	p.restrain_G_mode = 0;
	p.restrain_autonooc = 0;
	p.restrain_tight_divs.reset();
	p.norg_each_orbital_converg_error.reset();
	p.norg_each_orbital_converg_step_number.reset();
	p.gs_lanczos_reorthogonalization = 0;
	p.g_lanczos_reorthogonalization = 0;
    p.green_krylov = GreenKrylovOptions();
    std::set<Str> found_green_options;
	bool found_gs_lanczos_reorthogonalization = false;
	bool found_g_lanczos_reorthogonalization = false;
	bool found_restrain_difference_G = false;
	bool found_restrain_active_orbit_number_G = false;
	bool found_control_divs_G = false;
	bool found_restrain_G_mode = false;
	bool found_restrain_autonooc = false;
	bool found_restrain_tight_divs = false;
	bool found_norg_each_orbital_converg_error = false;
	bool found_norg_each_orbital_converg_step_number = false;
	std::ifstream file(filename);
    if (!file) ERR("Cannot read " + filename)
    std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string key;
		iss >> key;
        if (key == "Ed") {
            VecReal levels;
            read_uncounted_vector(iss, key, filename, levels);
            Ed.assign(levels.begin(), levels.end());
        } else if (key == "Deg") {
            VecInt labels;
            read_uncounted_vector(iss, key, filename, labels);
            Deg.assign(labels.begin(), labels.end());
        }
		else if (key == "J") {
			iss >> J;
		}
		else if (key == "CoulombF") {
			iss >> CoulombF;
		}
		else if (key == "fit_range") {
			iss >> fit_range;
		}
		else if (key == "NOOC") {
			iss >> p.nooc_mode;
		}
		else if (key == "beta") {
			iss >> beta;
		}
		else if (key == "U") {
			iss >> U;
		}
		else if (key == "Minsulator") {
			iss >> p.if_norg_degenerate;
		}
        else if (key == "magnetic") {
            int legacy = -1;
            if (!(iss >> legacy) || legacy != 0) ERR("Only paramagnetic input is supported")
            iss >> std::ws;
            if (!iss.eof() && iss.peek() != '#') ERR("Expected integer zero for magnetic")
        }
		else if (key == "nmesh") {
			iss >> p.nmesh;
		}
		else if (key == "ful_pcl_sch") {
			iss >> ful_pcl_sch;
		}
		else if (key == "weight_nooc") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					double value;
					iss.unget();
					iss >> value;
					noc1.push_back(value);
				}
			}
		}
		else if (key == "fit_points") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					double value;
					iss.unget();
					iss >> value;
					p.fit_points.push_back(value);
				}
			}
		}
		else if (key == "weight_freze") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					double value;
					iss.unget();
					iss >> value;
					noc2.push_back(value);
				}
			}
		}
		else if (key == "fit_nbaths") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					iss.unget();
					int value;
					iss >> value;
					fit_nbaths.push_back(value);
				}
			}
		}
		else if (key == "n_init") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					iss.unget();
					double value;
					iss >> value;
					n_init.push_back(value);
				}
			}
		}
		else if (key == "or_deg_adaptive") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch != ',' && ch != '[') {
					iss.unget();
					int value;
					iss >> value;
					or_deg_adaptive.push_back(value);
				}
			}
		}
        else if (key == "nondiag" || key == "nondiag_blk" || key == "OffDiagonal")
            ERR("Block layout options are not part of the diagonal solver input")
		else if (key == "chi_A") {
			iss >> chi_A;
		}
		else if (key == "chi_nomb") {
			iss >> p.chi_nomb;
		}
		else if (key == "if_n_init") {
			iss >> if_n_init;
		}
		else if (key == "if_eigenstate_number") {
			iss >> if_eigenstate_number;
		}
		else if (key == "eigenstate_number") {
			iss >> eigenstate_number;
		}
        else if (p.green_krylov.parse(key, iss, found_green_options)) {}
		else if (key == "gs_lanczos_reorthogonalization"
			|| key == "g_lanczos_reorthogonalization") {
			const bool is_ground_state = key == "gs_lanczos_reorthogonalization";
			bool& found = is_ground_state ? found_gs_lanczos_reorthogonalization
				: found_g_lanczos_reorthogonalization;
			if (found) ERR("Duplicate " + key + " field in " + filename)
			Int value;
			if (!(iss >> value)) ERR(key + " must be an Int in " + filename)
			iss >> std::ws;
			if (!iss.eof() && iss.peek() != '#') {
				ERR("Unexpected text after integer " + key + " in " + filename)
			}
			if (value != 0 && value != 1) {
				ERR(key + " must be 0 or 1 in " + filename)
			}
			if (is_ground_state) p.gs_lanczos_reorthogonalization = value;
			else p.g_lanczos_reorthogonalization = value;
			found = true;
		}
		else if (key == "testH0") {
			iss >> testH0;
		}
		else if (key == "testDelta") {
			iss >> testDelta;
			if (testDelta) {
				const std::string eimp_filename = "norgEimp.inp";
				std::ifstream eimp_file(eimp_filename);
				if (!eimp_file) {
					ERR("Cannot open " + eimp_filename)
				}

				std::vector<double> norg_eimp;
				std::string eimp_line;
				bool found_norg_eimp = false;
				while (std::getline(eimp_file, eimp_line)) {
					std::istringstream eimp_iss(eimp_line);
					std::string eimp_key;
					eimp_iss >> eimp_key;
					if (eimp_key != "norgEimp") continue;
					if (found_norg_eimp) {
						ERR("Duplicate norgEimp field in " + eimp_filename)
					}

					char ch = '\0';
					eimp_iss >> std::ws;
					if (!eimp_iss.get(ch) || ch != '[') {
						ERR("Expected '[' after norgEimp in " + eimp_filename)
					}

					while (true) {
						eimp_iss >> std::ws;
						if (eimp_iss.peek() == ']') {
							eimp_iss.get();
							break;
						}

						double value = 0.;
						if (!(eimp_iss >> value)) {
							ERR("Invalid norgEimp value in " + eimp_filename
								+ ": " + eimp_line)
						}
						norg_eimp.push_back(value);

						eimp_iss >> std::ws;
						if (eimp_iss.peek() == ',') eimp_iss.get();
					}

					eimp_iss >> std::ws;
					if (!eimp_iss.eof() && eimp_iss.peek() != '#') {
						ERR("Unexpected text after norgEimp in " + eimp_filename
							+ ": " + eimp_line)
					}
					if (norg_eimp.empty()) {
						ERR("norgEimp is empty in " + eimp_filename)
					}
					found_norg_eimp = true;
				}

				if (!eimp_file.eof()) {
					ERR("Error while reading " + eimp_filename)
				}
				if (!found_norg_eimp) {
					ERR("Cannot find norgEimp field in " + eimp_filename)
				}
				Ed = norg_eimp;
			}
		}
		else if (key == "if_fix_N_subspace") {
			iss >> if_fix_N_subspace;
		}
		else if (key == "fix_start_search_N_subspace") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch == '-' || isdigit(ch)) {
					iss.unget();
					int value;
					iss >> value;
					fix_start_search_N_subspace.push_back(value);
				}
			}
		}
		else if (key == "fix_N_subspace") {
			// Format: [count [[[npartical] [control row] ...] ...]].
			// Bracket depths 1, 3, and 4 denote count, subspace, and row.
			Int declared_count = -1;
			Int depth = 0;
			char ch;
			fix_N_subspace.clear();
			while (iss.get(ch)) {
				if (ch == '[') {
					++depth;
					if(depth == 3) fix_N_subspace.emplace_back();
					else if(depth == 4) fix_N_subspace.back().emplace_back();
				}
				else if (ch == ']') {
					--depth;
					if(depth == 0) break;
				}
				else if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
					iss.unget();
					int value;
					iss >> value;
					if(depth == 1) declared_count = value;
					else if(depth == 4) fix_N_subspace.back().back().push_back(value);
				}
			}
			if(declared_count <= 0) {
				ERR("fix_N_subspace must declare a positive number of subspaces.")
			}
			if(declared_count != static_cast<Int>(fix_N_subspace.size())) {
				ERR("fix_N_subspace declares " + STR(declared_count) + " subspaces, but "
					+ STR(fix_N_subspace.size()) + " were parsed.")
			}
		}
		else if (key == "restrain_difference") {
			if (found_restrain_difference) {
				ERR("Duplicate restrain_difference field in " + filename)
			}
			read_counted_matrix(iss, key, filename,
				p.restrain_difference, p.restrain_difference_dim);
			found_restrain_difference = true;
		}
		else if (key == "restrain_active_orbit_number") {
			if (found_restrain_active_orbit_number) {
				ERR("Duplicate restrain_active_orbit_number field in " + filename)
			}
			read_counted_matrix(iss, key, filename,
				p.restrain_active_orbit_number, p.restrain_active_orbit_number_dim);
			found_restrain_active_orbit_number = true;
		}
		else if (key == "restrain_difference_G") {
			if (found_restrain_difference_G) ERR("Duplicate restrain_difference_G field in " + filename)
			read_counted_matrix_vector(iss, key, filename, p.restrain_difference_G);
			found_restrain_difference_G = true;
		}
		else if (key == "restrain_active_orbit_number_G") {
			if (found_restrain_active_orbit_number_G) ERR("Duplicate restrain_active_orbit_number_G field in " + filename)
			read_counted_matrix_vector(iss, key, filename, p.restrain_active_orbit_number_G);
			found_restrain_active_orbit_number_G = true;
		}
		else if (key == "control_divs_G") {
			if (found_control_divs_G) ERR("Duplicate control_divs_G field in " + filename)
			read_counted_matrix_vector(iss, key, filename, p.control_divs_G);
			found_control_divs_G = true;
		}
		else if (key == "restrain_tight_divs") {
			if (found_restrain_tight_divs) ERR("Duplicate restrain_tight_divs field in " + filename)
			read_uncounted_vector(iss, key, filename, p.restrain_tight_divs);
			found_restrain_tight_divs = true;
		}
		else if (key == "norg_each_orbital_converg_error") {
			if (found_norg_each_orbital_converg_error) ERR("Duplicate norg_each_orbital_converg_error field in " + filename)
			read_counted_vector(iss, key, filename, p.norg_each_orbital_converg_error);
			found_norg_each_orbital_converg_error = true;
		}
		else if (key == "norg_each_orbital_converg_step_number") {
			if (found_norg_each_orbital_converg_step_number) ERR("Duplicate norg_each_orbital_converg_step_number field in " + filename)
			read_counted_vector(iss, key, filename, p.norg_each_orbital_converg_step_number);
			found_norg_each_orbital_converg_step_number = true;
		}
		else if (key == "restrain_G_mode") {
			if (found_restrain_G_mode) ERR("Duplicate restrain_G_mode field in " + filename)
			Int value;
			if (!(iss >> value)) ERR("restrain_G_mode must be an Int in " + filename)
			iss >> std::ws;
			if (!iss.eof() && iss.peek() != '#') {
				ERR("Unexpected text after integer restrain_G_mode in " + filename)
			}
			p.restrain_G_mode = value;
			found_restrain_G_mode = true;
		}
		else if (key == "restrain_autonooc") {
			if (found_restrain_autonooc) ERR("Duplicate restrain_autonooc field in " + filename)
			Int value;
			if (!(iss >> value)) ERR("restrain_autonooc must be an Int in " + filename)
			iss >> std::ws;
			if (!iss.eof() && iss.peek() != '#') {
				ERR("Unexpected text after integer restrain_autonooc in " + filename)
			}
			p.restrain_autonooc = value;
			found_restrain_autonooc = true;
		}
		else if (key == "restrain") {
			char ch;
			while (iss >> ch && ch != ']') {
				if (ch == '0') {
					restrain.push_back(0);
				}
				else if (ch == '-' || isdigit(ch)) {
					iss.unget();
					int value;
					iss >> value;
					restrain.push_back(value);
				}
			}
		}
		// else if (key == "distribute") {
		// 	char ch;
		// 	while (iss >> ch && ch != ']') {
		// 		if (ch == '0') {
		// 			distribute.push_back(0);
		// 		}
		// 		else if (ch == '-' || isdigit(ch)) {
		// 			iss.unget();
		// 			int value;
		// 			iss >> value;
		// 			distribute.push_back(value);
		// 		}
		// 	}
		// }
	}
}

namespace {
norg_restart::Layout search_restart_layout(const Prmtr& p)
{
	norg_restart::Layout layout;
	layout.rotate_impurity = p.if_norg_imp;
	for_Int(i, 0, p.nO2sets.size()) layout.orbitals.push_back(p.nO2sets[i]);
	for_Int(i, 0, p.nI2B.size()) layout.baths.push_back(p.nI2B[i]);
	norg_restart::require(layout.orbitals.size() == static_cast<Idx>(p.norg_sets),
		"set count does not match solver");
	return layout;
}

void apply_search_restart_rotation(const norg_restart::State& state, const Prmtr& p)
{
	VEC<MatReal> rotations;
	for_Int(i, 0, p.norg_sets) {
		const Int size = p.nO2sets[i];
		MatReal matrix(size, size, 0.);
		for_Int(r, 0, size) for_Int(c, 0, size)
			matrix[r][c] = state.rotation[i][static_cast<Idx>(r) * size + c];
		rotations.push_back(matrix);
	}
	p.rotationU = rotations;
}
} // namespace

bool APIedmft::edmft_back_up(const Str& status)
{
	const Str filename = "edmft_back_up";
	if(status == "read") {
		// Read once, then give every MPI rank exactly the same snapshot.
		std::string contents;
		Int present = 0;
		if(mm) {
			errno = 0;
			std::ifstream input(filename);
			if(input) {
				std::ostringstream buffer;
				buffer << input.rdbuf();
				contents = buffer.str();
				present = input.bad() || contents.size() > static_cast<Idx>(std::numeric_limits<Int>::max())
					? -1 : 1;
			} else if(errno != ENOENT) {
				present = -1;
			}
		}
		mm.Bcast(present);
		if(present < 0) ERR("Cannot read EDMFT search checkpoint")
		Int length = static_cast<Int>(contents.size());
		mm.Bcast(length);
		contents.resize(length);
		if(length > 0) {
			// Use MyMpi's public vector interface; the pointer overload is private.
			Vec<char> buffer(length);
			if(mm) contents.copy(buffer.p(), length);
			mm.Bcast(buffer);
			contents.assign(buffer.p(), length);
		}
		try {
			const auto layout = search_restart_layout(p);
			norg_restart::State state = norg_restart::identity(layout);
			if(present) {
				std::istringstream input(contents);
				state = norg_restart::read(input, layout);
				p.npartical.reset(state.particles.size(), 0);
				for_Int(i, 0, p.npartical.size()) p.npartical[i] = state.particles[i];
				if(!state.has_rotation && mm) {
					WRN("Legacy EDMFT backup has no rotation: restoring its particle counts with identity matrices.");
				}
			}
			iter_count = state.iteration;
			apply_search_restart_rotation(state, p);
		} catch(const std::exception& error) {
			ERR(Str(error.what()))
		}
		return present != 0;
	}
	if(status == "save") {
		Int success = 1;
		if(mm) {
			try {
				const auto layout = search_restart_layout(p);
				norg_restart::State state = norg_restart::identity(layout);
				norg_restart::require(iter_count < std::numeric_limits<Int>::max(), "iteration count overflow");
				state.iteration = iter_count + 1;
				state.has_rotation = true;
				for_Int(i, 0, p.npartical.size()) state.particles.push_back(p.npartical[i]);
				norg_restart::require(p.rotationU.size() == layout.orbitals.size(), "missing final rotation");
				for_Int(i, 0, p.norg_sets) {
					const Int size = p.nO2sets[i];
					norg_restart::require(p.rotationU[i].nrows() == static_cast<Idx>(size)
						&& p.rotationU[i].ncols() == static_cast<Idx>(size), "invalid final rotation dimensions");
					for_Int(r, 0, size) for_Int(c, 0, size)
						state.rotation[i][static_cast<Idx>(r) * size + c] = p.rotationU[i][r][c];
				}
				const Str temporary = filename + ".tmp";
				std::ofstream output(temporary, std::ios::trunc);
				norg_restart::write(output, state, layout);
				output.close();
				norg_restart::require(bool(output), "cannot finish writing checkpoint");
#ifdef _MSC_VER
				const bool replaced = MoveFileExA(temporary.c_str(), filename.c_str(),
					MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
				const bool replaced = std::rename(temporary.c_str(), filename.c_str()) == 0;
#endif
				norg_restart::require(replaced, "cannot replace checkpoint");
			} catch(const std::exception& error) {
				WRN(Str(error.what()));
				success = 0;
			}
		}
		mm.Bcast(success);
		if(!success) ERR("EDMFT search checkpoint was not saved")
		return true;
	}
	ERR("Unknown EDMFT backup operation: " + status)
	return false;
}


void APIedmft::process_occupation_sets(
    VEC<VecReal>& occnums,
    VEC<VecReal>& occweights,
    const VecInt& row_sizes,
    const VecReal& occnum_lst)
{
    if((Int)row_sizes.size() != p.norg_sets
        || (Int)occnums.size() != p.norg_sets
        || (Int)occweights.size() != p.norg_sets) {
        ERR("Occupation-set container size mismatch.");
    }

    if((Int)occnum_lst.size() != SUM(row_sizes)) {
        ERR("Occupation-number total size mismatch.");
    }

    Int start_idx = 0;

    for_Int(i, 0, p.norg_sets) {
        if(row_sizes[i] < 0) {
            ERR("Negative occupation row size.");
        }

        occnums[i].reset(row_sizes[i], 0.);
        occweights[i].reset(row_sizes[i], 0.);

        for_Int(j, 0, row_sizes[i]) {
            const Real occupation = occnum_lst[start_idx + j];

            occnums[i][j] = occupation;
            occweights[i][j] =
                MIN(occupation, 1. - occupation) < 1E-14
                ? 0.
                : MIN(occupation, 1. - occupation);
        }

        start_idx += row_sizes[i];
    }
}
