/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "bath.h"
#include "gen/parallel_stage2.h"
#include <numeric>

Bath::Bath(const MyMpi& mm_i, const Prmtr& prmtr_i) :
    mm(mm_i), p(prmtr_i), npart(p.nband), nb(npart), uur(mm.id()), info(npart, 6, 0.)
{
    { SLEEP(1); mm.barrier(); }
    for_Int(m, 0, npart) nb[m] = p.nI2B[2 * m];
    fvb_init();
}

void Bath::number_bath_fit(const ImGreen& hb, const VecInt or_deg)
{
    if (hb.norbs != npart || or_deg.size() != 2 * npart)
        ERR("PM bath layout does not match the spatial orbitals");
    VecInt first(MAX(or_deg), -1);
    for_Int(m, 0, npart) {
        const Int label = or_deg[2 * m] - 1;
        if (label < 0 || or_deg[2 * m + 1] != label + 1)
            ERR("Invalid PM degeneracy label");
        if (first[label] >= 0) {
            const Int rep = first[label];
            if (nb[rep] != nb[m]) ERR("Equivalent orbitals require equal bath sizes");
            fvb[m] = fvb[rep];
            for_Int(j, 0, info.ncols()) info[m][j] = info[rep][j];
            continue;
        }
        first[label] = m;
        ImGreen scalar(1, p);
        for_Int(n, 0, hb.nomgs) scalar[n][0][0] = hb[n][m][m];
        const VecReal fit = number_bath_fit_part(scalar, m);
        for_Int(k, 0, nb[m]) {
            fvb[m][0][k] = fit[k];
            fvb[m][1][k] = fit[nb[m] + k];
        }
    }
}

VecReal Bath::number_bath_fit_part(const ImGreen& vhb_i, Int idx)
{
	Real err;
	VecReal a;
	Int nmin;
	VecReal a_mrq = fvb2arr(idx);
	std::tie(err, a, nmin) = bath_fit_number_contest(a_mrq, vhb_i, idx);
	// if (mm) WRN("HERE is fine ")
	if (mm) {
		HybErr hyberr(p, vhb_i, nb[idx]);
		Real err = hyberr(a);
		Real err_crv = hyberr.err_curve(a);
		Real err_bsr = hyberr.err_bsr(a);
		Real err_reg = hyberr.err_reg(a);
		Real a_norm = a.norm();
		using namespace std;
		Int i = idx;
		cout << setw(4) << "  " << NAV7(nb[i],nmin, err, err_crv, err_bsr, err_reg, a_norm) << "  " << present() << endl;
		NAV6(Int(info[idx][0]=Real(nmin)), info[idx][1]=err, info[idx][2]=err_crv, info[idx][3]=err_bsr, info[idx][4]=err_reg, info[idx][5]=a_norm);
	}

	return a;
}

std::tuple<Real, VecReal, Int> Bath::bath_fit_number_contest(const VecReal& a0, const ImGreen& vhb_i, Int idx)
{
    // Preserve the historical trial counts and tolerances; only scheduling changes.
    const Int np = a0.size();
    const Int ntry_fine = MAX(16, mm.np() - 1);
    const Real tol = SQR(MIN(1.e-8, std::pow(10, -5 - ((nb[idx] / 4 + 1) / 2))));
    HybErr initial_error(p, vhb_i, nb[idx]);
    VecReal a_optm(a0);
    Real err_optm = initial_error(a_optm);
    Int nmin = 0;
    MatReal candidates;
    try {
        for (Int round = 0; round < 2; ++round) {
            const Int ntry = round == 0 ? ntry_fine : MAX(1, mm.np() - 1);
            const Int itmax = round == 0 ? 500 : 1500;
            const Int ndone = 16;
            const Int workers = MAX(1, mm.np() - 1);
            const Int batch_size = MIN(thread_budget(), 1 + (ntry - 1) / workers);
            MatReal starts, results;
            VecReal errors;
            VecInt fit_status;
            if (mm) {
                starts.reset(ntry, np); results.reset(ntry, np);
                errors.reset(ntry); fit_status.reset(ntry);
                Int trial = 0;
                for (Int job = 0; job < ntry; ++job) {
                    if (round == 0) starts[job] = next_initial_fitting_parameters(a0, ntry_fine, trial);
                    else starts[job] = candidates[job];
                }
            }
            const auto fit_batch = [&](MatReal& parameters, VecReal& values, VecInt& states) {
                const std::size_t count = parameters.nrows();
                values.reset(count); states.reset(count);
                const auto fit_range = [&](std::size_t first, std::size_t last, std::size_t) {
                    HybErr local_error(p, vhb_i, nb[idx]);
                    for (std::size_t job = first; job < last; ++job) {
                        FitMrq<HybErr> mrq(local_error.x, local_error.y, local_error.sig,
                            parameters[job], local_error, tol, itmax, ndone);
                        states[job] = mrq.fit();
                        parameters[job] = mrq.a;
                        values[job] = local_error(mrq.a);
                        if (!std::isfinite(values[job]))
                            throw std::runtime_error("Nonfinite bath fitting error");
                    }
                };
                // A single task uses frequency parallelism. A batch uses task parallelism;
                // the worker-aware helper suppresses nested frequency teams automatically.
                if (count == 1) fit_range(0, 1, 0);
                else stage2_parallel::ranges(count, fit_range);
            };
            if (mm.np() == 1) {
                results = starts;
                fit_batch(results, errors, fit_status);
            } else if (mm) {
                Int next = 0, completed = 0;
                const auto dispatch = [&](Int rank) {
                    const Int count = MIN(batch_size, ntry - next);
                    VecInt header{next, count};
                    mm.Send(header, rank, 1);
                    if (count) {
                        MatReal payload(count, np);
                        for (Int j = 0; j < count; ++j) payload[j] = starts[next + j];
                        mm.Send(payload, rank, 2);
                        next += count;
                    }
                };
                for (Int rank = 0; rank < mm.np(); ++rank) if (rank != mm.ms()) dispatch(rank);
                while (completed < ntry) {
                    VecInt header(2); MPI_Status status;
                    mm.Recv(header, status, MPI_ANY_SOURCE, 3);
                    const Int rank = status.MPI_SOURCE, first = header[0], count = header[1];
                    if (count < 1 || first < 0 || first > ntry - count)
                        throw std::runtime_error("Invalid bath result batch");
                    MatReal payload(count, np); VecReal values(count); VecInt states(count);
                    mm.Recv(payload, status, rank, 4);
                    mm.Recv(values, status, rank, 5);
                    mm.Recv(states, status, rank, 6);
                    for (Int j = 0; j < count; ++j) {
                        results[first + j] = payload[j]; errors[first + j] = values[j];
                        fit_status[first + j] = states[j];
                    }
                    completed += count;
                    dispatch(rank);
                }
            } else {
                while (true) {
                    VecInt header(2); MPI_Status status;
                    mm.Recv(header, status, mm.ms(), 1);
                    if (header[1] == 0) break;
                    MatReal payload(header[1], np); VecReal values; VecInt states;
                    mm.Recv(payload, status, mm.ms(), 2);
                    fit_batch(payload, values, states);
                    mm.Send(header, mm.ms(), 3); mm.Send(payload, mm.ms(), 4);
                    mm.Send(values, mm.ms(), 5); mm.Send(states, mm.ms(), 6);
                }
            }
            if (mm) {
                // Deterministic task order; completion order never changes ties or round-two starts.
                Int unfinished = 0, singular = 0;
                for (Int job = 0; job < ntry; ++job) {
                    const Real err = errors[job];
                    if (err_optm - sqrt(tol) > err) nmin = 0;
                    if (err_optm > err) { a_optm = results[job]; err_optm = err; }
                    nmin += err - err_optm < sqrt(tol);
                    unfinished += fit_status[job] == -1;
                    singular += fit_status[job] == -2;
                }
                if (round == 0) {
                    std::vector<Int> order(ntry); std::iota(order.begin(), order.end(), 0);
                    std::stable_sort(order.begin(), order.end(), [&](Int x, Int y) { return errors[x] < errors[y]; });
                    candidates.reset(ntry, np);
                    for (Int job = 0; job < ntry; ++job) candidates[job] = results[order[job]];
                }
                std::cout << "BATH_FIT_BATCH: block=" << idx << " round=" << round
                    << " tasks=" << ntry << " batch=" << batch_size
                    << " threads=" << thread_budget() << " iteration_limit=" << unfinished
                    << " singular=" << singular << std::endl;
            }
        }
        mm.Bcast(err_optm); mm.Bcast(a_optm); mm.Bcast(nmin);
    } catch (const std::exception& error) {
        // Main-thread only: a failed worker must not leave the dispatcher waiting forever.
        std::cerr << "Bath fit rank " << mm.id() << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        throw;
    }
    return std::make_tuple(err_optm, a_optm, nmin);
}

VecReal Bath::next_initial_fitting_parameters(const VecReal& a0, const Int& ntry_fine, Int& itry)
{
	VecReal a = a0;
	++itry;
	if (itry == 1) {
	}
	else if (itry <= ntry_fine * 1) {
		a = perturb(a, 0.1);
	}
	else if (itry <= ntry_fine * 2) {
		a = perturb(a, 0.2);
	}
	else if (itry <= ntry_fine * 3) {
		a = perturb(a, 0.4);
	}
	else if (itry <= ntry_fine * 4) {
		a = perturb(a, 0.8);
	}
	else {
		a = perturb(a, 1.6);
	}
	return a;
}


void Bath::fvb_init()
{
    fvb.reset(npart);
    for_Int(m, 0, npart) {
        fvb[m].reset(2, nb[m]);
        for_Int(row, 0, 2) {
            uur(fvb[m][row]);
            fvb[m][row] -= 0.5;
            fvb[m][row] *= 4;
        }
    }
}

void Bath::bth_write_fvb(Int iter_cnt, const Str bath_name) const
{
    using namespace std;
    const Str filename = iter_cnt > 0 ? iox + "zic" + prefill0(iter_cnt, 3) + ".ose_hop.txt"
        : (bath_name.empty() ? Str("ose_hop") : bath_name + "ose_hop.out");
    OFS ofs(filename);
    ofs << iofmt("sci");
    for_Int(m, 0, npart) {
        ofs << "set[" << m << "]" << endl;
        for_Int(row, 0, 2) {
            ofs << setw(w_Real) << (row == 0 ? "the" : "epi") << endl;
            for_Int(k, 0, nb[m]) ofs << setw(w_Real) << fvb[m][row][k];
            ofs << endl;
        }
    }
}

void Bath::bth_read_fvb(const Str& file)
{
    IFS ifs(file);
    if (!ifs) {
        if (mm) WRN(STR("file opening failed with ") + NAV(file));
        return;
    }
    for_Int(m, 0, npart) {
        Str tag;
        if (!(ifs >> tag) || tag != "set[" + STR(m) + "]")
            ERR("Invalid bath set in " + file);
        for_Int(row, 0, 2) {
            if (!(ifs >> tag) || tag != (row == 0 ? "the" : "epi"))
                ERR("Invalid bath row in " + file);
            for_Int(k, 0, nb[m])
                if (!(ifs >> fvb[m][row][k])) ERR("Invalid bath data in " + file);
        }
    }
    Str extra;
    if (ifs >> extra) ERR("Unexpected extra bath data in " + file);
}

VecReal Bath::fvb2arr(const Int idx)
{
    return fvb[idx].vec();
}

MatReal Bath::find_hop() const
{
    MatReal h0(0, 0, 0.);
    for_Int(m, 0, npart) {
        VecReal hopping(fvb[m][0]);
        VecReal energy(fvb[m][1]);
        slctsort(energy, hopping);
        MatReal star(1 + nb[m], 1 + nb[m], 0.);
        for_Int(k, 0, nb[m]) {
            star[0][1 + k] = star[1 + k][0] = hopping[k];
            star[1 + k][1 + k] = energy[k];
        }
        h0.reset(direct_sum(h0, direct_sum(star, star)));
    }
    return h0;
}
