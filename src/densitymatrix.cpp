/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "densitymatrix.h"
#include "gen/density_timing.h"
#include "gen/parallel_stage2.h"
using namespace std;

DensityMat::DensityMat(const MyMpi& mm_i, const Prmtr& prmtr_i, NocSpace& scsp_i, bool imp_rotation) :Operator(mm_i, prmtr_i, scsp_i, true)
	,dm(dm_initialize())
{
}

DensityMat::DensityMat(const MyMpi& mm_i, const Prmtr& prmtr_i, NocSpace& scsp_i, Str tab_name, bool imp_rotation) :Operator(mm_i, prmtr_i, scsp_i, tab_name)
	,dm(dm_initialize())
{
}

DensityMat::DensityMat(const MyMpi& mm_i, const Prmtr& prmtr_i, NocSpace& scsp_i, const Tab& tab, bool imp_rotation) :Operator(mm_i, prmtr_i, scsp_i, tab)
	,dm(dm_initialize())
{
}

//----------------------------- basic function -----------------------------

VEC<MatReal> DensityMat::find_unitary_orbital_rotation_matrix()
{
	VEC<MatReal> rotaionU;
	if(p.if_norg_imp){
		rotaionU = dm;
		VEC<VecReal> evalue;
		for_Int(i, 0, p.norg_sets) {
			VecReal evalu_i(rotaionU[i].nrows(), 0.);
			heevr(rotaionU[i], evalu_i); rotaionU[i] = rotaionU[i].tr(); evalue.push_back(std::move(evalu_i));
			for_Int(j, 0, (p.nO2sets[i] / 2.)) {
				SWAP(rotaionU[i][j], rotaionU[i][p.nO2sets[i] - j - 1]);
				SWAP(evalue[i][j], evalue[i][p.nO2sets[i] - j - 1]);
			}
			/*
			VEC<MatReal> rotaionU_temp = rotaionU; VEC<VecReal> evalue_temp = evalue;
			for_Int(j, 0, p.nO2sets[i]) {
				Int nimp = p.nO2sets[i] - p.nI2B[i];
				if (j < nimp) {
					rotaionU[i][j] = rotaionU_temp[i][j + SUM_0toX(scsp.sit_mat[i], p.ndiv / 2) - scsp.sit_mat[i][0]];
					evalue[i][j] = evalue_temp[i][j + SUM_0toX(scsp.sit_mat[i], p.ndiv / 2) - scsp.sit_mat[i][0]];
				} else {
					if (p.nI2B[i] % 2 != 0 && j - nimp < p.nI2B[i] / 2. - 1) {
						rotaionU[i][j] = rotaionU_temp[i][j - nimp];
						evalue[i][j] = evalue_temp[i][j - nimp];
					}
					if (p.nI2B[i] % 2 == 0 && j - nimp < p.nI2B[i] / 2.) {
						rotaionU[i][j] = rotaionU_temp[i][j - nimp];
						evalue[i][j] = evalue_temp[i][j - nimp];
					}
				}
			}
			// if(mm && i == 0) WRN(NAV4(rotaionU_temp[0], rotaionU[0], evalue_temp[0], evalue[0]));
			*/
			// if(mm) WRN(NAV(evalu_i));
		}
		// for_Int(i, 0, rotaionU.size()) rotaionU[i] = rotaionU[i - (i%2)]; //! using the spin inversion symmetry
		occupationnumber = evalue;
	} else {
		VEC<MatReal> rotaionU_bath;
		for_Int(i, 0, p.norg_sets) {
			if(mm) WRN(NAV2(i,dm[i]));
			Int nimp = p.nO2sets[i] - p.nI2B[i];
			rotaionU_bath.push_back(dm[i].truncate(nimp, nimp, p.nO2sets[i], p.nO2sets[i]));
			// if(mm) WRN(NAV(dm[i][0][0]))
			Real imp_occ = 0.; for_Int(r, 0, nimp) imp_occ += dm[i][r][r];
			if(mm) std::cout << "The "<<i<<"-th impurity occupation number: "<<iofmt()<<imp_occ<< std::endl;
		}
		// if (mm) WRN(NAV3(dm[0], dm[1], dm[2]));
		// for_Int(spin, 0, 2) rotaionU_bath[0 + spin] = rotaionU_bath[2 + spin] = 0.5 * (rotaionU_bath[0 + spin] + rotaionU_bath[2 + spin]); //! set band 0 same as band 1.
		//for_Int(i, 0, p.nband) rotaionU_bath[i*2] = rotaionU_bath[i*2 + 1] = 0.5 * (rotaionU_bath[i*2] + rotaionU_bath[i*2 + 1]); //! using the spin inversion symmetry(suit for SC).

		VEC<VecReal> evalue;
		for_Int(i, 0, p.norg_sets) {
			VecReal evalu_i(rotaionU_bath[i].nrows(), 0.);
			// if(mm) WRN("New dm for bath" + NAV3(i, rotaionU_bath[i], rotaionU_bath.size()));
			heevr(rotaionU_bath[i], evalu_i); rotaionU_bath[i] = rotaionU_bath[i].tr(); evalue.push_back(std::move(evalu_i));
			// if(mm) WRN("New U for bath" + NAV2(rotaionU_bath[i], evalue[i]));
			for_Int(j, 0, (p.nI2B[i] / 2.)) {
				SWAP(rotaionU_bath[i][j], rotaionU_bath[i][p.nI2B[i] - j - 1]);
				SWAP(evalue[i][j], evalue[i][p.nI2B[i] - j - 1]);
			}
			// if(mm) WRN("New uorm" + NAV3(i, rotaionU_bath[i], evalue[i]));
		}
		// if(mm) WRN(NAV3(evalue[0].mat(1,p.nI2B[0]), evalue[1].mat(1,p.nI2B[1]), evalue[2].mat(1,p.nI2B[2])));
		// for_Int(i, 0, rotaionU_bath.size()) rotaionU_bath[i] = rotaionU_bath[i - (i%2)]; //! using the spin inversion symmetry(suit for SC).

		occupationnumber = evalue;
		for_Int(i, 0, p.nO2sets.size()) {
			MatReal temp(dmat(p.nO2sets[i], 1.));
			Int gap(p.nO2sets[i] - p.nI2B[i]);
			for_Int(j, 0, p.nI2B[i]) for_Int(k, 0, p.nI2B[i]) temp[j + gap][k + gap] = rotaionU_bath[i][j][k];
			rotaionU.push_back(temp);
		}
	}
	return rotaionU;
}

void DensityMat::update(Int mode) {
	dm = dm_initialize();
	if (mode == 1) {
		p.degel = 1;
		MatReal egses(lowest_eigpairs(scsp.dim, false, p.degel + 1));
		for_Int(egs_idx, 0, p.degel) {
			VEC<MatReal> temp_dm;
			temp_dm = find_one_electron_density_matrix(egses[egs_idx].mat(1, scsp.dim), table);
			// if(mm) WRN(NAV(temp_dm[0]));
			for_Int(dm_i, 0, dm.size()) dm[dm_i] += temp_dm[dm_i] * Real(1 / p.degel);
		}
	}
	else dm = find_one_electron_density_matrix(lowest_eigpairs(scsp.dim), table);
}

VEC<MatReal> DensityMat::find_one_electron_density_matrix(const MatReal& state, const Tab& table_i)
{
    namespace dt = density_timing;
    namespace mp = matrix_build_profile;
    namespace par = stage2_parallel;
    VecInt timing_options(2, 0);
    if(mm) { timing_options[0] = dt::requested(); timing_options[1] = dt::sample_interval(); }
    mm.Bcast(timing_options);
    static std::uint64_t build_id = 0;
    const auto this_build = ++build_id;
    mp::Report profile(mm,timing_options[0]!=0);
    VEC<MatReal> D;
    std::vector<dt::Timing> worker_timing;
    dt::Timing timing(timing_options[0] != 0, timing_options[1], mm.id(), this_build);
    {
    dt::Timing::Scope state_timer(timing, dt::StatePrepare);
    VecReal state_eff;
    par::collective(mm, [&] {
        if(state.ncols() != scsp.dim) throw std::invalid_argument("Density state dimension mismatch");
        state_eff.reset(state.ncols());
        par::ranges(state.ncols(),[&](std::size_t first,std::size_t last,std::size_t) {
            for(auto col=first;col<last;++col) {
                Real sum=0.;
                for_Int(row,0,state.nrows())sum+=state[row][col];
                state_eff[col]=sum;
            }
        });
        preparation_parallel::normalize(state_eff); // Same effective-state convention.
        profile.mark(mp::StatePrepare);
    });
    profile.mark(mp::CollectiveSync);
    state_timer.stop();
    dt::Timing::Scope setup_timer(timing, dt::Setup);
    VecPartition row_H(mm.np(), mm.id(), scsp.dim);
    const Tab& hop_op(table_i);
    const std::size_t parts = std::max(par::workers(row_H.len()), par::workers(hop_op.nnz()));
    std::vector<VEC<MatReal>> partial;
    std::vector<std::pair<Int, Int>> orbital_location;
    VecReal packed, global;
    par::collective(mm, [&] {
        table_i.require_layout(scsp.dim, row_H.bgn(), row_H.len());
        for_Int(block, 0, p.nI2B.size())
            for_Int(local, 0, p.nO2sets[block])
                orbital_location.emplace_back(block, local);
        Idx size = 0;
        for_Int(i, 0, p.nI2B.size()) {
            D.push_back(MatReal(p.nO2sets[i], p.nO2sets[i], 0.));
            if(D.back().size() > static_cast<Idx>(std::numeric_limits<int>::max()) - size)
                throw std::overflow_error("Packed density matrix exceeds MPI count range");
            size += D.back().size();
        }
        packed.reset(size, 0.); global.reset(size, 0.);
        partial.resize(parts, D);
        worker_timing.reserve(parts);
        for(std::size_t part = 0; part < parts; ++part)
            worker_timing.emplace_back(timing_options[0] != 0, timing_options[1],
                mm.id(), this_build + part);
        profile.mark(mp::DensitySetup);
    });
    profile.mark(mp::CollectiveSync);
    timing.rows = row_H.len(); timing.entries = table_i.nnz();
    setup_timer.stop();

    dt::Timing::Scope diagonal_timer(timing, dt::Diagonal);
    par::collective(mm, [&] {
        par::ranges(row_H.len(), [&](std::size_t first, std::size_t last, std::size_t part) {
            NocSpace::GroupCursor group_cursor(scsp);
            OccupationDecoder occupation_decoder(scsp);
            auto& D_splited = partial[part];
            auto& timing = worker_timing[part];
	auto accumulate_row = [&](std::size_t offset, auto selected) {
		constexpr bool sampled = decltype(selected)::value;
		const BasisIndex h_i = row_H.bgn() + static_cast<BasisIndex>(first + offset);
		dt::SampleScope<sampled> row_timer(timing, dt::Row);
		dt::SampleScope<sampled> lookup_timer(timing, dt::GroupLookup);
		const BasisIndex group = group_cursor.locate(h_i);
		lookup_timer.stop();
		dt::SampleScope<sampled> decode_timer(timing, dt::StateDecode);
		const auto& filled = occupation_decoder.decode(h_i, group);
		decode_timer.stop();
		dt::SampleScope<sampled> add_timer(timing, dt::DiagonalAdd);

		// Occupations retain the original ascending orbital order.
            const Real amplitude = state_eff[h_i] * state_eff[h_i];
            for (const auto& orbitals : filled) {
                for (const Int orbital : orbitals) {
                    if (static_cast<std::size_t>(orbital) >= orbital_location.size()) continue;
                    const auto location = orbital_location[orbital];
                    D_splited[location.first][location.second][location.second] += amplitude;
                }
            }
    };

            timing.rows=last-first;
            timing.traverse(last - first, 0, accumulate_row);
        });
        profile.mark(mp::DensityDiagonal);
    });
    profile.mark(mp::CollectiveSync);
    diagonal_timer.stop();

    dt::Timing::Scope offdiagonal_timer(timing, dt::OffDiagonal);
    par::collective(mm, [&] {
        // Contiguous entry ranges balance local Tab work without changing MPI row ownership.
        par::ranges(hop_op.nnz(), [&](std::size_t first, std::size_t last, std::size_t part) {
            if(first == last) return;
            std::size_t tab_row = std::upper_bound(hop_op.row_ptr().begin(),
                hop_op.row_ptr().end(), first) - hop_op.row_ptr().begin() - 1;
            auto& D_splited = partial[part];
            auto& timing = worker_timing[part];
	auto accumulate_entry = [&](std::size_t index, auto selected) {
		const std::size_t pos = first + index;
		constexpr bool sampled = decltype(selected)::value;
		dt::SampleScope<sampled> entry_timer(timing, dt::Entry);
		dt::SampleScope<sampled> filter_timer(timing, dt::Filter);
		while (pos >= hop_op.row_ptr()[tab_row + 1]) ++tab_row;
		const bool one_body = abs(hop_op.ops()[pos]) <= scsp.hopint.size();
		filter_timer.stop();
		if (one_body)
		{
			dt::SampleScope<sampled> orbital_timer(timing, dt::OrbitalDecode);
			Int row_n((abs(hop_op.ops()[pos]) - 1) / p.norbit), col_n((abs(hop_op.ops()[pos]) - 1) % p.norbit); // "1" mean the real_length = abs(hop_op.ops()[pos]) - 1, since to distinguish the minus sign.
			Int check_point(0);
			orbital_timer.stop();
			dt::SampleScope<sampled> match_add_timer(timing, dt::OffDiagonalAdd);
			for_Int(i, 0, p.nI2B.size()) {
				check_point = 0;
				if (row_n >= 0 && row_n < (p.nO2sets[i]) && col_n >= 0 && col_n < (p.nO2sets[i]))
				{
					Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
					Real amplitude = coefficient_comm * state_eff[tab_row + hop_op.row_begin()] * state_eff[hop_op.cols()[pos]];
					D_splited[i][row_n][col_n] += amplitude;
					check_point++;
					break;
				}
				else {
					row_n -= p.nO2sets[i];
					col_n -= p.nO2sets[i];
				}
			}
#ifdef _ASSERTION_
			if (check_point == 0)ERR("Not in the hopint but you save this bases(Many body base), why? For D mapping");
#endif
		}
	};

            timing.entries=last-first;
            timing.traverse(last - first, 1, accumulate_entry);
        });
        profile.mark(mp::DensityOffDiagonal);
    });
    profile.mark(mp::CollectiveSync);
    offdiagonal_timer.stop();

    dt::Timing::Scope reduce_timer(timing, dt::Reduce);
    par::collective(mm, [&] {
        // Stable part order, no atomics in density hot loops.
        for(std::size_t part = 0; part < parts; ++part) {
            Idx offset = 0;
            for(const auto& block : partial[part]) {
                for(Idx i = 0; i < block.size(); ++i) packed[offset + i] += block.p()[i];
                offset += block.size();
            }
        }
        profile.mark(mp::LocalReduce);
    });
    profile.mark(mp::CollectiveSync);
    {
        dt::Timing::Scope collective_timer(timing, dt::ReduceCall);
        mm.Allreduce_into(packed, global);
        profile.mark(mp::Allreduce);
    }
    Idx offset = 0;
    for(auto& block : D) {
        std::copy(global.p() + offset, global.p() + offset + block.size(), block.p());
        offset += block.size();
    }
    profile.mark(mp::Unpack);
    reduce_timer.stop();
    dt::Timing::Scope check_timer(timing, dt::Check);
#ifdef _ASSERTION_
    for(const auto& d : D)
        if(!d.if_symmetric()) ERR("The one_electron_density_matrix is not hermitian!!");
#endif
    check_timer.stop();
    profile.mark(mp::Check);
    } // Computational temporaries are destroyed before the build clock stops.
    profile.mark(mp::Cleanup);profile.finish();
    density_build_seconds_=profile.total();
    timing.finish();
    VecPartition measured_rows(mm.np(),mm.id(),scsp.dim);
    mp::emit(mm,profile,"DENSITY_PROFILE","one_particle_from_tab",scsp.dim,
        measured_rows.len(),table_i.nnz(),this_build);
    if(timing_options[0]) {
        for(const auto& worker : worker_timing) timing.merge_samples(worker);
        const VecReal packed_timing(timing.pack());
        const VecReal gathered = mm.Gather(packed_timing);
        if(mm) cout << timing.report(gathered.p(), mm.np(), scsp.dim,
            scsp.div.size(), state.nrows(), p.norg_sets, this_build);
    }
    return D;
}


MatReal DensityMat::find_imp_double_occupancy() const
{
    namespace par = stage2_parallel;
    VecPartition row_H(mm.np(), mm.id(), scsp.dim);
    VecInt global_off, imp_off, imp_size;
    Int n_imp = 0;
    MatReal ninj_loc;
    // All ranks leave setup together, including allocation/validation failures.
    par::collective(mm, [&] {
        if(p.degel < 1 || ground_state.nrows() < p.degel ||
           ground_state.ncols() != scsp.dim)
            throw std::invalid_argument("Impurity double occupancy: invalid ground states");
        global_off.reset(p.norg_sets, 0);
        imp_off.reset(p.norg_sets, 0);
        imp_size.reset(p.norg_sets, 0);
        for_Int(k, 0, p.norg_sets) {
            imp_size[k] = p.control_divs[k + 1][0];
            imp_off[k] = n_imp;
            n_imp += imp_size[k];
            if(k) global_off[k] = global_off[k - 1] + p.nO2sets[k - 1];
        }
        ninj_loc.reset(n_imp, n_imp, 0.);
    });
    if(mm) WRN("Impurity double occupancy BEGIN: dim=" + STR(scsp.dim)
        + ", ranks=" + STR(mm.np()) + ", thread_budget=" + STR(thread_budget()));

    // Preserve normalization and the equal-weight average over ground states.
    for_Int(egs_idx, 0, p.degel) {
        par::collective(mm, [&] {
            VecReal gs=preparation_parallel::slice(ground_state[egs_idx],0,ground_state.ncols());
            preparation_parallel::normalize(gs);
            VecReal gs_part=preparation_parallel::slice(gs,row_H.bgn(),row_H.end());
            MatReal ngs(n_imp, row_H.len(), 0.);

            // Each range owns its cursor and state temporaries. Different ranges
            // write disjoint columns of ngs; the shared space is read-only.
            par::ranges(row_H.len(), [&](std::size_t begin, std::size_t end,
                                         std::size_t) {
                NocSpace::GroupCursor group_cursor(scsp);
                for(std::size_t local = begin; local < end; ++local) {
                    const auto li = local;
                    const BasisIndex h_i = row_H.bgn() + static_cast<BasisIndex>(li);
                    StateStatistics a(h_i, group_cursor.locate(h_i), scsp);
                    for_Int(k, 0, p.norg_sets) {
                        const Int imp_begin = global_off[k];
                        const Int imp_end = imp_begin + imp_size[k];
                        for(const auto& N : a.filled_spinless[k]) {
                            if(N >= imp_begin && N < imp_end)
                                ngs[imp_off[k] + N - imp_begin][li] = gs_part[li];
                        }
                    }
                }
            });

            // One writer per matrix element. The helper limits MKL to one
            // thread per worker and restores it afterwards (no nested teams).
            // Skip zero-length DOT on MPI ranks with no local basis rows.
            if(row_H.len()) {
                const std::size_t pairs = static_cast<std::size_t>(n_imp) * n_imp;
                par::for_each(pairs, [&](std::size_t pair) {
                    const Int r = static_cast<Int>(pair / n_imp);
                    const Int c = static_cast<Int>(pair % n_imp);
                    ninj_loc[r][c] += DOT(ngs[r], ngs[c]) / Real(p.degel);
                });
            }
        });
        if(mm) WRN("Impurity double occupancy state complete: "
            + STR(egs_idx + 1) + "/" + STR(p.degel));
    }

    // MPI stays on the initializing thread, after all OpenMP workers join.
    MatReal DOcc = mm.Allreduce(ninj_loc);
    if(mm) {
        WRN("impurity <n_i n_j> matrix:");
        WRN(NAV(DOcc));
        WRN("Impurity double occupancy END");
    }
    return DOcc;
}

MatReal DensityMat::find_full_double_occupancy() const {
	MatReal docc_splited(p.norbit, p.norbit, 0.);
	for_Int(i, 0, ground_state.nrows()) {
		VecReal state_eff = ground_state[i];
		state_eff.normalize();
		VecPartition row_H(mm.np(), mm.id(), scsp.dim);

		const Tab& hop_op(table);
		hop_op.require_layout(scsp.dim, row_H.bgn(), row_H.len());
		for_Idx(tab_row, 0, hop_op.nrows())
		for (Tab::Offset pos = hop_op.row_ptr()[tab_row]; pos < hop_op.row_ptr()[tab_row + 1]; ++pos) {
			if (abs(hop_op.ops()[pos]) > scsp.hopint.size()) {
				// four-fermion operator terms for C^+_i C^+_j C_k C_l; 
				Int tensor_idx = abs(hop_op.ops()[pos]) - scsp.hopint.size() - 1;
				Int row_n = Int(tensor_idx / scsp.hopint.size()) / p.norbit, col_n = Int(tensor_idx / scsp.hopint.size()) % p.norbit; // N_i:row_n;N_j:col_n;
				if (row_n == (tensor_idx % scsp.hopint.size()) / p.norbit && col_n == (tensor_idx % scsp.hopint.size()) % p.norbit) {
						row_n = row_n / p.nO2sets[0]; col_n = col_n / p.nO2sets[0];
						if(row_n>p.norbs||col_n>p.norbs)WRN(NAV5(tensor_idx, row_n, col_n, (tensor_idx % scsp.hopint.size()) / p.norbit, (tensor_idx % scsp.hopint.size()) % p.norbit));
						Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
						Real amplitude = coefficient_comm * state_eff[tab_row + hop_op.row_begin()] * state_eff[hop_op.cols()[pos]];
						// WRN(NAV3(row_n, col_n,amplitude));
						docc_splited[row_n][col_n] += amplitude / p.degel ;
				}
			}
		}
	}
	// MatReal DOcc(mm.Allreduce(docc_splited));
	
	MatReal DOcc(mm.Allreduce(docc_splited));
	return DOcc;
}


//----------------------------- append function -----------------------------

Real DensityMat::sum_off_diagonal() const{
	Real sum = 0 ;
	auto test_dm(dm);
	for_Int(i, 0, p.norg_sets) if(i%2 == 0) {
		MatReal test_dm_bath = test_dm[i].truncate(scsp.sit_mat[i][0], scsp.sit_mat[i][0], p.nO2sets[i], p.nO2sets[i]);
		for_Int(r, 0, test_dm_bath.nrows()) test_dm_bath[r][r] -= test_dm_bath[r][r];
		VecReal vectemp(test_dm_bath.vec());
		// if(mm) WRN(NAV(test_dm_bath));
		sum += SQRT(vectemp.norm_sqr() / (vectemp.size() - test_dm_bath.nrows()));
	}
	return sum/(p.norg_sets / 2);
}

VEC<VecReal> DensityMat::check_dm_get_occupation_number() const{
	auto test_dm(dm);
	VEC<VecReal> occupation_nubmer;
	for_Int(i, 0, p.norg_sets) {
		VecReal evalu_i(test_dm[i].nrows(), 0.);
		heevr(test_dm[i], evalu_i); test_dm[i] = test_dm[i].tr();
		if(mm) WRN(NAV3(i,evalu_i,SUM(evalu_i)));
		if (i % 2 == 0) occupation_nubmer.push_back(evalu_i);
	}
	return occupation_nubmer;
}

//----------------------------- correction function -----------------------------

void DensityMat::find_density_matrix_by_Crrvec(VEC < MatReal>& D_splited, const Crrvec& corstate_i)
{
	VecPartition row_H(mm.np(), mm.id(), corstate_i.opr.scsp.dim);
	MatReal corrvec_real = real(corstate_i.correct_vecs); for_Int(i, 0, corrvec_real.nrows()) corrvec_real[i].normalize();
	MatReal corrvec_imag = imag(corstate_i.correct_vecs); for_Int(i, 0, corrvec_imag.nrows()) corrvec_imag[i].normalize();
	VecReal oneparticalex = corstate_i.ex_state; oneparticalex.normalize();
	// Real d_weight_y    = 0.8; 	//4times
	Real d_weight_y = 1.; 		//2times
	// Real d_weight_y = (8./7.); 	//1times

	const Tab& hop_op(corstate_i.opr.table);
	hop_op.require_layout(corstate_i.opr.scsp.dim, row_H.bgn(), row_H.len());
	for_Idx(tab_row, 0, hop_op.nrows())
	for (Tab::Offset pos = hop_op.row_ptr()[tab_row]; pos < hop_op.row_ptr()[tab_row + 1]; ++pos)
	{
		if (abs(hop_op.ops()[pos]) <= corstate_i.opr.scsp.hopint.size())
		{
			Int row_n((abs(hop_op.ops()[pos]) - 1) / p.norbit), col_n((abs(hop_op.ops()[pos]) - 1) % p.norbit); // "1" mean the real_length = abs(hop_op.ops()[pos]) - 1, since to distinguish the minus sign.
			Int check_point(0);
			for_Int(i, 0, p.nI2B.size()) {
				check_point = 0;
				if (row_n >= 0 && row_n < (p.nO2sets[i]) && col_n >= 0 && col_n < (p.nO2sets[i]))
				{
					Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
					for_Int(j, 0, corstate_i.correct_vecs.nrows()){
						Real amplitude = coefficient_comm * corrvec_real[j][tab_row + hop_op.row_begin()] * corrvec_real[j][hop_op.cols()[pos]];
							amplitude += coefficient_comm * corrvec_imag[j][tab_row + hop_op.row_begin()] * corrvec_imag[j][hop_op.cols()[pos]];
							amplitude += coefficient_comm * oneparticalex[tab_row + hop_op.row_begin()] * oneparticalex[hop_op.cols()[pos]];
						D_splited[i][row_n][col_n] += d_weight_y * 0.25 * amplitude * INV(corstate_i.correct_vecs.nrows() * 2.);// fist 2: G^gatter, G^lesser; second 2: Rell and imag part.
					}
					check_point++;
					break;
				}
				else {
					row_n -= p.nO2sets[i];
					col_n -= p.nO2sets[i];
				}
			}
#ifdef _ASSERTION_
			if (check_point == 0)ERR("Not in the hopint but you save this bases(Many body base), why? For D mapping");
#endif
		}
	}
}

VEC<MatReal> DensityMat::correct_one_electron_density_matrix(const VecReal& state, Crrvec& corstate_p, Crrvec& corstate_m, const VecReal& omega_point)
{
	VecReal state_eff = state;
	state_eff.normalize();
	VecPartition row_H(mm.np(), mm.id(), scsp.dim);

	VEC<MatReal> D_splited;
	for_Int(i, 0, p.nI2B.size()) {
		MatReal temp(p.nO2sets[i], p.nO2sets[i], 0.);
		D_splited.push_back(std::move(temp));
	}
	// for the ground state's D.
	const Tab& hop_op(table);
	hop_op.require_layout(scsp.dim, row_H.bgn(), row_H.len());
	for_Idx(tab_row, 0, hop_op.nrows())
	for (Tab::Offset pos = hop_op.row_ptr()[tab_row]; pos < hop_op.row_ptr()[tab_row + 1]; ++pos)
	{
		if (abs(hop_op.ops()[pos]) <= scsp.hopint.size())
		{
			Int row_n((abs(hop_op.ops()[pos]) - 1) / p.norbit), col_n((abs(hop_op.ops()[pos]) - 1) % p.norbit); // "1" mean the real_length = abs(hop_op.ops()[pos]) - 1, since to distinguish the minus sign.
			Int check_point(0);
			for_Int(i, 0, p.nI2B.size()) {
				check_point = 0;
				if (row_n >= 0 && row_n < (p.nO2sets[i]) && col_n >= 0 && col_n < (p.nO2sets[i]))
				{
					Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
					Real amplitude = coefficient_comm * state_eff[tab_row + hop_op.row_begin()] * state_eff[hop_op.cols()[pos]];
					D_splited[i][row_n][col_n] += amplitude;
					check_point++;
					break;
				}
				else {
					row_n -= p.nO2sets[i];
					col_n -= p.nO2sets[i];
				}
			}
#ifdef _ASSERTION_
			if (check_point == 0)ERR("Not in the hopint but you save this bases(Many body base), why? For D mapping");
#endif
		}
	}

	// for_Int(j, 0, omega_point.size()){
		corstate_p.krylov_update_state( I * cmplx(omega_point));
		// for the corstate_p's D.
		find_density_matrix_by_Crrvec(D_splited, corstate_p);
		corstate_m.krylov_update_state( I * cmplx(omega_point));
		// for the corstate_m's D.
		find_density_matrix_by_Crrvec(D_splited, corstate_m);
	// }

	VEC<MatReal> D;
	for_Int(i, 0, p.norg_sets) {
		MatReal D_i(mm.Allreduce(D_splited[i]));
		D.push_back(std::move(D_i));
	}
#ifdef _ASSERTION_
	for (const auto& d : D)
		if (d.if_symmetric() == false)ERR("The one_electron_density_matrix is not hermitian!!");
#endif
	return std::move(D);
}


VEC<MatReal> DensityMat::correct_one_electron_density_matrix(const VecReal& state, Crrvec& corstate1_p, Crrvec& corstate1_m, Crrvec& corstate2_p, Crrvec& corstate2_m)
{
	VecReal state_eff = state;
	state_eff.normalize();
	VecPartition row_H(mm.np(), mm.id(), scsp.dim);

	VEC<MatReal> D_splited;
	for_Int(i, 0, p.nI2B.size()) {
		MatReal temp(p.nO2sets[i], p.nO2sets[i], 0.);
		D_splited.push_back(std::move(temp));
	}
	// for the ground state's D.
	const Tab& hop_op(table);
	hop_op.require_layout(scsp.dim, row_H.bgn(), row_H.len());
	for_Idx(tab_row, 0, hop_op.nrows())
	for (Tab::Offset pos = hop_op.row_ptr()[tab_row]; pos < hop_op.row_ptr()[tab_row + 1]; ++pos)
	{
		if (abs(hop_op.ops()[pos]) <= scsp.hopint.size())
		{
			Int row_n((abs(hop_op.ops()[pos]) - 1) / p.norbit), col_n((abs(hop_op.ops()[pos]) - 1) % p.norbit); // "1" mean the real_length = abs(hop_op.ops()[pos]) - 1, since to distinguish the minus sign.
			Int check_point(0);
			for_Int(i, 0, p.nI2B.size()) {
				check_point = 0;
				if (row_n >= 0 && row_n < (p.nO2sets[i]) && col_n >= 0 && col_n < (p.nO2sets[i]))
				{
					Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
					Real amplitude = coefficient_comm * state_eff[tab_row + hop_op.row_begin()] * state_eff[hop_op.cols()[pos]];
					D_splited[i][row_n][col_n] += amplitude * 0.5;
					check_point++;
					break;
				}
				else {
					row_n -= p.nO2sets[i];
					col_n -= p.nO2sets[i];
				}
			}
#ifdef _ASSERTION_
			if (check_point == 0)ERR("Not in the hopint but you save this bases(Many body base), why? For D mapping");
#endif
		}
	}
	
	// for the corstate_p's D.
	find_density_matrix_by_Crrvec(D_splited, corstate1_p);
	find_density_matrix_by_Crrvec(D_splited, corstate2_p);

	// for the corstate_m's D.	
	find_density_matrix_by_Crrvec(D_splited, corstate1_m);
	find_density_matrix_by_Crrvec(D_splited, corstate2_m);

	VEC<MatReal> D;
	for_Int(i, 0, p.norg_sets) {
		MatReal D_i(mm.Allreduce(D_splited[i]));
		D.push_back(std::move(D_i));
	}
#ifdef _ASSERTION_
	for (const auto& d : D)
		if (d.if_symmetric() == false)ERR("The one_electron_density_matrix is not hermitian!!");
#endif
	return std::move(D);
}

VEC<MatReal> DensityMat::correct_one_electron_density_matrix(const VecReal& state, const Crrvec& corstate_p, const Crrvec& corstate_m)
{
	VecReal state_eff = state;
	state_eff.normalize();
	VecPartition row_H(mm.np(), mm.id(), scsp.dim);
	// Real d_weight_x    = 1.6; 	//4times
	Real d_weight_x = 1.; 		//2times
	// Real d_weight_x = (4./7.); 	//1times

	VEC<MatReal> D_splited;
	for_Int(i, 0, p.nI2B.size()) {
		MatReal temp(p.nO2sets[i], p.nO2sets[i], 0.);
		D_splited.push_back(std::move(temp));
	}
	// for the ground state's D.
	const Tab& hop_op(table);
	hop_op.require_layout(scsp.dim, row_H.bgn(), row_H.len());
	for_Idx(tab_row, 0, hop_op.nrows())
	for (Tab::Offset pos = hop_op.row_ptr()[tab_row]; pos < hop_op.row_ptr()[tab_row + 1]; ++pos)
	{
		if (abs(hop_op.ops()[pos]) <= scsp.hopint.size())
		{
			Int row_n((abs(hop_op.ops()[pos]) - 1) / p.norbit), col_n((abs(hop_op.ops()[pos]) - 1) % p.norbit); // "1" mean the real_length = abs(hop_op.ops()[pos]) - 1, since to distinguish the minus sign.
			Int check_point(0);
			for_Int(i, 0, p.nI2B.size()) {
				check_point = 0;
				if (row_n >= 0 && row_n < (p.nO2sets[i]) && col_n >= 0 && col_n < (p.nO2sets[i]))
				{
					Int coefficient_comm = hop_op.ops()[pos] >= 0 ? 1 : -1;
					Real amplitude = coefficient_comm * state_eff[tab_row + hop_op.row_begin()] * state_eff[hop_op.cols()[pos]];
					D_splited[i][row_n][col_n] += amplitude * 0.25 *d_weight_x;
					check_point++;
					break;
				}
				else {
					row_n -= p.nO2sets[i];
					col_n -= p.nO2sets[i];
				}
			}
#ifdef _ASSERTION_
			if (check_point == 0)ERR("Not in the hopint but you save this bases(Many body base), why? For D mapping");
#endif
		}
	}

	// for the corstate_p's D.
	find_density_matrix_by_Crrvec(D_splited, corstate_p);

	// for the corstate_m's D.
	find_density_matrix_by_Crrvec(D_splited, corstate_m);

	VEC<MatReal> D;
	for_Int(i, 0, p.norg_sets) {
		MatReal D_i(mm.Allreduce(D_splited[i]));
		D.push_back(std::move(D_i));
	}
#ifdef _ASSERTION_
	for (const auto& d : D)
		if (d.if_symmetric() == false)ERR("The one_electron_density_matrix is not hermitian!!");
#endif
	return std::move(D);
}


//----------------------------- deprecated function -----------------------------

/*
//! (Deactivate) In Here the state a row is a state, and the number of row means degeneracy
MatReal DensityMat::find_one_electron_density_matrix(const MatReal& state)
{
//#ifdef _CHECK_DIMENSION_MATCH_
//	ASSERT_EQ(state.ncols(), degeneracy);
//#endif
	// if (mm)WRN("find_one_electron_density_matrix BEGIN: ");
	VecReal state_eff(state.ncols(), 0.);
	for_Int(i, 0, state.nrows()) state_eff += state[i];
	state_eff.normalize();
	// if (mm)WRN("find_one_electron_density_matrix BEGIN: " + NAV2(state_eff.isnormalized(), SUM(state_eff)) );
	VecPartition row_H(mm.np(), mm.id(), scsp.dim);
	MatReal D_splited(p.norbit, p.norbit, 0.);
	for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i)
	{
		StateStatistics a(h_i, scsp.wherein_NocSpace(h_i), scsp);
		// For diagonally terms.
		for (const auto &x : a.filled_spinless)
		{
			for (const auto &i : x)
			{
				D_splited[i][i] += state_eff[h_i] * state_eff[h_i];
			}
		}
		vector<VecInt> off_dt;
		for_Int(idx_sets, 0, p.norg_sets){
			vector<VecInt> off_dt_next(a.find_each_spiless_group_off_diagonal_term(a.divocchop_ingroup(a.div_idx, idx_sets),idx_sets));
			off_dt. insert(off_dt. end(), off_dt_next. begin(), off_dt_next. end());
		}
		for_Int(i, 0, off_dt.size()) {
			D_splited[off_dt[i][1]][off_dt[i][0]] += off_dt[i][3] * state_eff[h_i] * state_eff[off_dt[i][2]];
		}
	}
	MatReal D(mm.Allreduce(D_splited));
	if (mm)WRN("find_one_electron_density_matrix FINISHED!!! " + NAV(D));
#ifdef _ASSERTION_
	if (D.if_symmetric() == false)ERR("The one_electron_density_matrix is not hermitian!!");
#endif
// #ifdef _ASSERTIONFORVSCODE_
// 	if (D.if_symmetric() == false)ERR("The one_electron_density_matrix is not hermitian!!");
// #endif
	return D;
}

// (Deactivate) for the OrthonormalityRecover.
MatReal DensityMat::OrthonormalityRecover(const MatReal& mat_i)
{
	MatReal mat(mat_i.tr());
	for_Int(i, 0, mat.nrows()) {
		for_Int(j, 0, i) {
			mat[i] -= mat[j] * DOT(mat[i], mat[j]);
		}
		mat[i].normalize();
	}
	return mat.tr();
}
*/
