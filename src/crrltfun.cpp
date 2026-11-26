/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "crrltfun.h"
#include "gen/memory_monitor.h"
#include "gen/parallel_lanczos.h"
#include "gen/green_krylov.h"
#include "gen/parallel_stage2.h"
using namespace std;

CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& nosp_i, const Operator& oper)
    :Operator(oper),
    old_nosp(nosp_i), new_nosp(nosp_i), crtorann(0),
    ex_state()
{}

CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper)
    :Operator(oper),
    old_nosp(old_nosp_i), new_nosp(main_nosp), crtorann(main_nosp.nspa - old_nosp.nspa),
    ex_state()
{}

CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const VecReal& vgs_i, const Int position)
    :Operator(mm_i, prmtr_i, main_nosp),
    old_nosp(old_nosp_i), new_nosp(main_nosp), crtorann(main_nosp.nspa - old_nosp.nspa),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, position))
{}

CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper, const VecReal& vgs_i, const Int position)
    :Operator(oper),
    old_nosp(old_nosp_i), new_nosp(main_nosp), crtorann(main_nosp.nspa - old_nosp.nspa),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, position))
{}

CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const NocSpace& old_nosp_i, const NocSpace& main_nosp, const Operator& oper, const VecReal& vgs_i, const Int pos_in_set, const Int pos_in_div)
    :Operator(oper),
    old_nosp(old_nosp_i), new_nosp(main_nosp), crtorann(main_nosp.nspa - old_nosp.nspa),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, pos_in_set, pos_in_div))
{}

// CrrltFun::CrrltFun(const MyMpi& mm_i, const Prmtr& prmtr_i, const Operator& oper, const VecReal ex_state_i, const Int crtorann_i)
//     :Operator(oper),
//     old_nosp(NocSpace(mm_i, prmtr_i)), new_nosp(NocSpace(mm_i, prmtr_i)), crtorann(crtorann_i),
//     ex_state(ex_state_i)
// {}


// ?CrrltFun::CrrltFun(const NocSpace& old_nosp_i, const Operator& main_op, const VecReal& vgs_i, const Int position = 0)
// ?    :Operator(main_op),
// ?    old_nosp(old_nosp_i), new_nosp(main_op.scsp), crtorann(main_op.scsp.nspa - old_nosp.nspa),
// ?    ex_state(project_uplwer_parical_space(vgs_i, crtorann, position))
// ?{}

// (Deactivate) Only for test
ImGreen CrrltFun::find_density_density_correlation_function(const Real &ge0) {

    Real upper_fraction(mm.Allreduce(DOT(ex_state, ex_state)));
    Vectrdgnl trdgnl(find_trdgnl(ex_state, -1));
    // if (mm) WRN("Finish the find_trdgnl");
    VecReal zerod(get<0>(trdgnl).size(), 0.);
    VecReal zeros(get<1>(trdgnl).size(), 0.);
    VecCmplx a(cmplx(get<0>(trdgnl), zerod));
    VecCmplx b(cmplx(get<1>(trdgnl), zeros));
    Int n(a.size());
    ImGreen green(1, p);
    for_Int(w, 0, green.nomgs) {
        //Cmplx z(cmplx(ge0, green.omg(w)));
        Cmplx z = green.z(w) + ge0;
        Cmplx c(z + a[n - 2] - b[n - 2] * b[n - 2] / (z + a[n - 1]));
        for (Int i = n; i >= 3; i--) c = z + a[i - 3] - b[i - 3] * b[i - 3] / c;
        Cmplx under_fraction(c);
        Cmplx gaz(upper_fraction / under_fraction);
        green[w][0][0] = gaz;
    }
    return green;
}


void CrrltFun::find_gf(const Real& ge0, Green &g0, bool is_greater)
{
    if(p.g_lanczos_reorthogonalization != 0 && p.g_lanczos_reorthogonalization != 1)
        ERR("g_lanczos_reorthogonalization must be 0 or 1.");
    if(new_nosp.dim == 0) return;
    const SparseMatReal sep_h = find_hmlt_V2(table);
    const Str context = Str("entry=single_source sector=") + (is_greater ? "N+1" : "N-1");
    green_profile::Profile profile(mm,context,"contribution",p.green_krylov.solver);
    std::ostringstream checks;
    green_krylov::FrequencyGrid frequencies;
    Int adaptive_grid=p.green_krylov.lanczos_step==0;
    mm.Bcast(adaptive_grid);
    if(adaptive_grid) {
        frequencies.z.reset(g0.nomgs);
        for_Int(w,0,g0.nomgs) frequencies.z[w]=g0.z(w);
        frequencies.ge0=ge0;
        frequencies.is_greater=is_greater;
    }
    // Historical grid size is only an initial planning hint in adaptive mode.
    const Int initial_hint = g0.type_info() == STR("ImGreen") ? 501 : 3001;
    const auto spectra=green_krylov::solve(mm,ex_state,sep_h,new_nosp.dim,
        initial_hint,p.g_lanczos_reorthogonalization==1,p.green_krylov,
        context,green_krylov::Monitor(),frequencies);
    green_krylov::accumulate_frequency(mm,spectra,g0,ge0,is_greater,&profile,checks);
    profile.finish(mm,checks.str());
}

ChiIomResult CrrltFun::find_chi_Sz_N_O_nondeg(const Real& E0, const VecReal& vgs)
{
    /*
        Compute static susceptibilities:

            chi_A = 2 <f_A| (H - E0)^(-1) |f_A>

        by Lanczos + continued fraction.

        Here:

            |f_A> = (A - <A>) |0>

        for diagonal impurity operators:

            chi[0]      : S_z^imp
            chi[1]      : N_imp / 2 (CTQMC charge convention)
            chi[2 + p]  : O_ij, i < j

        Important convention in this version:

            vgs is a full global ground-state vector on every MPI process.

        However, all parallel operations inside this function are still
        performed on the local Hilbert-space segment owned by each process:

            h in [vp.bgn(), vp.bgn() + vp.len())

        Therefore we first build:

            vgs_loc[h_i] = vgs[vp.bgn() + h_i]

        and all DOT, norm, projection, f_state construction use vgs_loc.
    */

    const Real eps_norm = 1E-30;
    const Real eps_beta = 1E-14;
    const Real eps_conv = 1E-12;
    const Int  max_iter = 3000;

    // ------------------------------------------------------------
    // Construct Hamiltonian only once.
    // Every chi_A below reuses this sep_h.
    // ------------------------------------------------------------
    const SparseMatReal sep_h = find_hmlt_V2(table);

    // ------------------------------------------------------------
    // Parallel partition of the Hilbert space.
    // ------------------------------------------------------------
    VecPartition vp(mm.np(), mm.id(), new_nosp.dim);

    // ------------------------------------------------------------
    // In this version, vgs must be a full global vector on every rank.
    // But this function still computes in parallel using only local slices.
    // ------------------------------------------------------------
    if (vgs.size() != new_nosp.dim) {
        ERR("find_chi_Sz_N_O_nondeg: vgs.size() != full Hilbert-space size: "
            + NAV2(vgs.size(), new_nosp.dim));
    }

    VecReal vgs_loc(vp.len(), 0.0);

    for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) {
        const BasisIndex global_h = vp.bgn() + h_i;
        vgs_loc[h_i] = vgs[global_h];
    }

    // ------------------------------------------------------------
    // Build physical f_state:
    //
    //     |f_A> = (A - <A>) |0>
    //
    // for diagonal impurity operators:
    //
    //     chi[0]      : S_z^imp
    //     chi[1]      : N_imp / 2 (CTQMC charge convention)
    //     chi[2 + p]  : O_ij, i < j
    //
    // Orbital order:
    //
    //     0 : z^2
    //     1 : x^2-y^2
    //     2 : xz
    //     3 : yz
    //     4 : xy
    //
    const Int n_imp_orb = p.nband;

    // ------------------------------------------------------------
    // Build O_ij list, i < j.
    // For 5 orbitals, there are 10 O_ij.
    // ------------------------------------------------------------

    StdVecInt O_i;
    StdVecInt O_j;

    for_Int(i, 0, n_imp_orb) {
        for_Int(j, i + 1, n_imp_orb) {
            O_i.push_back(i);
            O_j.push_back(j);
        }
    }

    const Int n_O = O_i.size();
    const Int n_chi = 2 + n_O;

    if (p.beta <= 0.0) {
        ERR("find_chi_Sz_N_O_nondeg: beta must be positive: " + NAV(p.beta));
    }
    if (p.chi_nomb <= 0) {
        ERR("find_chi_Sz_N_O_nondeg: chi_nomb must be positive: " + NAV(p.chi_nomb));
    }

    ChiIomResult result;
    result.omega.reset(p.chi_nomb, 0.0);
    result.chi_iom.reset(n_chi, p.chi_nomb, 0.0);
    for_Int(im, 0, p.chi_nomb) {
        result.omega[im] = 2.0 * pi_Real * Real(im) / p.beta;
    }

    // chi index:
    //   0          -> Sz
    //   1          -> Nimp / 2
    //   2 + pair_k -> O_{O_i[pair_k], O_j[pair_k]}

    // ------------------------------------------------------------
    // Given a global basis index h, compute all diagonal A(h).
    //
    // A_values[0]     = Sz_imp(h)
    // A_values[1]     = N_imp(h) / 2
    // A_values[2 + p] = O_ij(h)
    // ------------------------------------------------------------

    auto get_A_values_on_state = [&](const BasisIndex global_h, VecReal& A_values,
        NocSpace::GroupCursor& group_cursor) {

        A_values.reset(n_chi, 0.0);

        VecInt n_up(n_imp_orb, 0);
        VecInt n_dn(n_imp_orb, 0);

        const BasisIndex subnosp = group_cursor.locate(global_h);

        StateStatistics st(global_h, subnosp, new_nosp);

        // The first division of each set contains its single physical impurity orbital.
        for_Int(m, 0, n_imp_orb) {
            n_up[m] = st.cfg.cf[(2*m) * new_nosp.ndivs].isocc(0);
            n_dn[m] = st.cfg.cf[(2*m+1) * new_nosp.ndivs].isocc(0);
        }

        // Total N_imp and S_z^imp.  CTQMC defines its charge operator as
        // N_imp / 2, so A_values[1] uses the same normalization below.
        Real Nimp_total = 0.0;
        Real Sz         = 0.0;

        for_Int(m, 0, n_imp_orb) {
            Nimp_total += Real(n_up[m] + n_dn[m]);
            Sz         += 0.5 * Real(n_up[m] - n_dn[m]);
        }

        A_values[0] = Sz;
        A_values[1] = 0.5 * Nimp_total;

        // O_ij = sum_sigma (n_i_sigma - n_j_sigma), i < j
        for_Int(pij, 0, n_O) {

            const Int i = O_i[pij];
            const Int j = O_j[pij];

            const Real ni = Real(n_up[i] + n_dn[i]);
            const Real nj = Real(n_up[j] + n_dn[j]);

            A_values[2 + pij] = ni - nj;
        }
    };

    // ------------------------------------------------------------
    // Compute <A> = <0|A|0> for all A.
    //
    // Since every process has full vgs but only computes its own
    // local segment, use vgs_loc here.
    //
    // If vgs is not exactly normalized, use
    //
    //     <A> = sum_h |v_h|^2 A(h) / sum_h |v_h|^2
    //
    // This also guarantees <0|f_A> = 0 numerically.
    // ------------------------------------------------------------

    VecReal A_sum_local(n_chi, 0.0);
    Real norm_local = 0.0;

    VecReal A_values(n_chi, 0.0);

    NocSpace::GroupCursor average_cursor(new_nosp);

    for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) {

        const BasisIndex global_h = vp.bgn() + h_i;

        get_A_values_on_state(global_h, A_values, average_cursor);

        const Real weight = vgs_loc[h_i] * vgs_loc[h_i];

        norm_local += weight;

        for_Int(a, 0, n_chi) {
            A_sum_local[a] += weight * A_values[a];
        }
    }

    const Real norm_gs = mm.Allreduce(norm_local);

    if (norm_gs < eps_norm) {
        ERR("find_chi_Sz_N_O_nondeg: ground-state norm is zero.");
    }

    VecReal A_sum_global(mm.Allreduce(A_sum_local));
    VecReal A_avg(n_chi, 0.0);

    for_Int(a, 0, n_chi) {
        A_avg[a] = A_sum_global[a] / norm_gs;
    }

    if (mm) {
        PIO("==================================================");
        PIO("find_chi_Sz_N_O_nondeg: diagonal operator averages");
        PIO("<Sz_imp> = " + STR(A_avg[0]));
        PIO("<N_imp/2> = " + STR(A_avg[1]));

        for_Int(pij, 0, n_O) {
            PIO("<O_" + STR(O_i[pij]) + STR(O_j[pij]) + "> = "
                + STR(A_avg[2 + pij]));
        }

        PIO("Ground-state norm <0|0> = " + STR(norm_gs));
        PIO("==================================================");
    }

    // ------------------------------------------------------------
    // Build local f_state for one operator:
    //
    //     f_A(h) = [A(h) - <A>] * vgs(h)
    //
    // Since f_state is local, use vgs_loc[h_i].
    //
    // Then
    //
    //     <0|f_A> = 0
    //
    // up to numerical error.
    // ------------------------------------------------------------

    auto make_f_state_from_diag_A = [&](const Int A_idx) -> VecReal {

        VecReal f_state(vp.len(), 0.0);

        VecReal A_values_tmp(n_chi, 0.0);

        NocSpace::GroupCursor source_cursor(new_nosp);

        for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) {

            const BasisIndex global_h = vp.bgn() + h_i;

            get_A_values_on_state(global_h, A_values_tmp, source_cursor);

            f_state[h_i] = (A_values_tmp[A_idx] - A_avg[A_idx]) * vgs_loc[h_i];
        }

        // Optional check: <0|f_A> should be zero.
        const Real overlap_local = DOT(vgs_loc, f_state);
        const Real overlap = mm.Allreduce(overlap_local);

        if (mm) {
            PIO("make_f_state_from_diag_A: A_idx = " + STR(A_idx)
                + ", <0|f_A> = " + STR(overlap));
        }

        return f_state;
    };

    // ------------------------------------------------------------
    // Project out the ground-state component:
    //
    //     x <- x - |0> <0|x> / <0|0>
    //
    // Since x is local, use vgs_loc here:
    //
    //     x_loc <- x_loc - proj * vgs_loc
    //
    // This avoids numerical singular contamination from
    //
    //     (H - E0)^(-1)
    //
    // because the ground-state component has zero denominator.
    // ------------------------------------------------------------

    auto project_out_ground_state = [&](VecReal& x) -> Real {

        const Real proj_local = DOT(vgs_loc, x);
        const Real proj = mm.Allreduce(proj_local) / norm_gs;

        x -= proj * vgs_loc;

        return proj;
    };

    // ------------------------------------------------------------
    // Core Lanczos + continued fraction for one f_state.
    // ------------------------------------------------------------

    auto calc_chi_from_f_state = [&](VecReal f_state, const Int label) -> VecReal {

        VecReal chi_iom(result.omega.size(), 0.0);

        // ------------------------------------------------------------
        // Name of the susceptibility.
        //
        // label = 0      -> Sz
        // label = 1      -> Nimp / 2 (printed as chi_Nimp)
        // label = 2+pij  -> O_ij
        // ------------------------------------------------------------

        auto chi_name = [&](const Int label_i) -> Str {

            if (label_i == 0) return STR("Sz");
            if (label_i == 1) return STR("Nimp");

            const Int pij = label_i - 2;

            if (pij >= 0 && pij < n_O) {
                // 0-based orbital label:
                // O_01, O_02, ...
                return STR("O_") + STR(O_i[pij]) + STR(O_j[pij]);

                // 如果你想打印成 1-based 轨道编号，用下面这一行替换上面 return：
                // return STR("O_") + STR(O_i[pij] + 1) + STR(O_j[pij] + 1);
            }

            return STR("unknown");
        };

        const Str name = chi_name(label);

        // ------------------------------------------------------------
        // First project out possible numerical ground-state component:
        //
        //     |f> <- |f> - |0><0|f>/<0|0>
        //
        // Ideally <0|f> = 0 already, but this removes residual
        // roundoff error before applying (H - E0)^(-1).
        // ------------------------------------------------------------

        const Real proj_f = project_out_ground_state(f_state);

        if (mm) {
            PIO("chi_" + name
                + ": initial ground-state projection <0|f>/<0|0> = "
                + STR(proj_f));
        }

        // ------------------------------------------------------------
        // b0^2 = <f|f>, after ground-state projection
        // ------------------------------------------------------------

        Real b0_square = mm.Allreduce(DOT(f_state, f_state));

        if (b0_square < eps_norm) {
            if (mm) {
                PIO("find_chi_Sz_N_O_nondeg: |f_" + name
                    + "> norm is zero after ground-state projection, chi_"
                    + name + " = 0.");
            }
            return chi_iom;
        }

        // ------------------------------------------------------------
        // q0 = |f> / sqrt(<f|f>)
        // ------------------------------------------------------------

        VecReal q_curr(std::move(f_state));
        q_curr *= INV(SQRT(b0_square));

        VecReal q_prev(q_curr.size(), 0.0);

        VEC<Real> ltd;      // alpha_j
        VEC<Real> lt_sd;    // beta_{j+1}; lt_sd[0] = beta_1

        Real beta_prev = 0.0;
        Real chi_prev  = 0.0;
        Real chi_now   = 0.0;
        bool lanczos_finished = false;

        for_Int(k, 0, max_iter) {

            // --------------------------------------------------------
            // w = H |q_k>
            // --------------------------------------------------------

            VecReal w(sep_h * q_curr);

            // --------------------------------------------------------
            // First remove possible ground-state component from H|q_k>.
            //
            // This makes the Lanczos iteration closer to working in the
            // projected subspace:
            //
            //     P_perp H P_perp
            //
            // where
            //
            //     P_perp = 1 - |0><0|/<0|0>
            //
            // In exact arithmetic this projection should be zero, because
            // q_k is orthogonal to |0>.  But numerically it can remove a
            // dangerous tiny component along |0>.
            // --------------------------------------------------------

            const Real proj_Hq = project_out_ground_state(w);

            if (mm && k % 50 == 0) {
                PIO("chi_" + name
                    + " iter = " + STR(k)
                    + ", projected ground-state component of Hq = "
                    + STR(proj_Hq));
            }

            // --------------------------------------------------------
            // w = P_perp H|q_k> - beta_k |q_{k-1}>
            // --------------------------------------------------------

            if (k > 0) {
                w -= beta_prev * q_prev;
            }

            // --------------------------------------------------------
            // alpha_k = <q_k| P_perp H P_perp |q_k>
            //
            // Since q_k is already normalized and orthogonal to |0>,
            // this is the projected-space Lanczos diagonal element.
            // --------------------------------------------------------

            Real alpha_k = mm.Allreduce(DOT(q_curr, w));

            // --------------------------------------------------------
            // w = P_perp H|q_k> - beta_k |q_{k-1}> - alpha_k |q_k>
            // --------------------------------------------------------

            w -= alpha_k * q_curr;

            // --------------------------------------------------------
            // Project again after subtracting beta and alpha terms.
            //
            // This second projection removes roundoff errors accumulated
            // in the Lanczos residual vector before computing beta_{k+1}.
            // --------------------------------------------------------

            const Real proj_w = project_out_ground_state(w);

            if (mm && k % 50 == 0) {
                PIO("chi_" + name
                    + " iter = " + STR(k)
                    + ", projected ground-state component of residual w = "
                    + STR(proj_w));
            }

            ltd.push_back(alpha_k);

            // --------------------------------------------------------
            // Continued fraction:
            //
            // chi_A = 2 <f|f> [T_m - E0 I]^(-1)_{00}
            //
            // [T_m - E0 I]^(-1)_{00}
            //
            // = 1 /
            //   (alpha_0 - E0
            //    - beta_1^2 / (alpha_1 - E0
            //    - beta_2^2 / (...)))
            // --------------------------------------------------------

            const Int n = ltd.size();

            Real c = ltd[n - 1] - E0;

            for (Int i = n - 2; i >= 0; --i) {
                c = (ltd[i] - E0) - lt_sd[i] * lt_sd[i] / c;
            }

            Real inv00 = INV(c);

            // chi_A = 2 * <f|f> * [T_m - E0 I]^(-1)_{00}
            chi_now = 2.0 * b0_square * inv00;

            // --------------------------------------------------------
            // Convergence check
            // --------------------------------------------------------

            if (k > 5) {

                Real err = ABS(chi_now - chi_prev);

                if (k % 50 == 0) {
                    // Check orthogonality between q_curr and q_prev.
                    // Without full re-orthogonalization, loss of orthogonality
                    // (|<q_k|q_{k-1}>| >> 0) indicates ghost eigenvalues may appear.
                    // These are MPI collectives, so every rank must call them.
                    Real orth_curr_prev = ABS(mm.Allreduce(DOT(q_curr, q_prev)));
                    Real norm_curr      = SQRT(mm.Allreduce(DOT(q_curr, q_curr)));
                    Real norm_prev      = SQRT(mm.Allreduce(DOT(q_prev, q_prev)));
                    Real orth_rel       = (norm_curr * norm_prev > 0.0)
                                         ? orth_curr_prev / (norm_curr * norm_prev)
                                         : 0.0;

                    // Restrict only output to the master rank.
                    if (mm) {
                        PIO("chi_" + name
                            + " iter = " + STR(k)
                            + ", chi_" + name + " = " + STR(chi_now)
                            + ", err = " + STR(err));

                        PIO("chi_" + name
                            + " orthogonality check at iter " + STR(k)
                            + ": |<q_k|q_{k-1}>|/(|q_k||q_{k-1}|) = "
                            + STR(orth_rel));

                        if (orth_rel > 1e-6) {
                            PIO("WARNING: chi_" + name
                                + " orthogonality loss at iter " + STR(k)
                                + ", orth_rel = " + STR(orth_rel)
                                + ". Lanczos may produce ghost eigenvalues.");
                        }
                    }
                }

                if (err < eps_conv * (1.0 + ABS(chi_now))) {
                    if (mm) {
                        PIO("chi_" + name
                            + " converged, iter = " + STR(k)
                            + ", chi_" + name + " = " + STR(chi_now));

                        PIO("The size of a and b for chi_" + name
                            + ": " + NAV2(ltd.size(), lt_sd.size()));
                    }

                    lanczos_finished = true;
                    break;
                }
            }

            chi_prev = chi_now;

            // --------------------------------------------------------
            // beta_{k+1} = ||w||
            // --------------------------------------------------------

            Real beta_next = SQRT(mm.Allreduce(DOT(w, w)));

            // --------------------------------------------------------
            // Krylov space closed.
            // --------------------------------------------------------

            if (beta_next < eps_beta) {
                if (mm) {
                    PIO("chi_" + name
                        + " Krylov space closed, iter = " + STR(k)
                        + ", chi_" + name + " = " + STR(chi_now));

                    PIO("The size of a and b for chi_" + name
                            + ": " + NAV2(ltd.size(), lt_sd.size()));
                }

                lanczos_finished = true;
                break;
            }

            // --------------------------------------------------------
            // q_{k+1} = w / beta_{k+1}
            // --------------------------------------------------------

            w *= INV(beta_next);

            // --------------------------------------------------------
            // Update Lanczos vectors.
            // --------------------------------------------------------

            q_prev = std::move(q_curr);
            q_curr = std::move(w);

            lt_sd.push_back(beta_next);

            beta_prev = beta_next;
        }

        // ------------------------------------------------------------
        // Reached max_iter.
        // ------------------------------------------------------------

        if (mm && !lanczos_finished) {
            PIO("chi_" + name
                + " reached max_iter = " + STR(max_iter)
                + ", chi_" + name + " = " + STR(chi_now));

            PIO("The size of a and b for chi_" + name
                + ": " + NAV2(ltd.size(), lt_sd.size()));
        }

        // Reuse the converged Lanczos tridiagonal matrix for every bosonic
        // Matsubara frequency.  For z=i*Omega,
        //
        //   chi_A(i*Omega) = 2 <f|f> Re[(T-E0+i*Omega)^(-1)]_00
        //                  = 2 <f| (H-E0)/((H-E0)^2+Omega^2) |f>.
        //
        // No additional Hamiltonian-vector multiplication or MPI reduction
        // is required here.
        const Int n_lanczos = ltd.size();
        for_Int(im, 0, result.omega.size()) {
            const Real omega = result.omega[im];
            Cmplx c(ltd[n_lanczos - 1] - E0, omega);

            for (Int i = n_lanczos - 2; i >= 0; --i) {
                c = Cmplx(ltd[i] - E0, omega) - lt_sd[i] * lt_sd[i] / c;
            }

            chi_iom[im] = 2.0 * b0_square * std::real(Cmplx(1.0, 0.0) / c);
        }

        return chi_iom;
    };

    // ------------------------------------------------------------
    // Compute chi for:
    //
    //   chi[0]      = chi_Sz
    //   chi[1]      = chi_{Nimp/2}, printed as chi_Nimp for CTQMC compatibility
    //   chi[2+pij]  = chi_Oij
    //
    // For five orbitals, chi.size() = 12.
    // ------------------------------------------------------------

    {
        VecReal f_Sz = make_f_state_from_diag_A(0);
        result.chi_iom[0] = calc_chi_from_f_state(std::move(f_Sz), 0);
    }

    {
        VecReal f_N = make_f_state_from_diag_A(1);
        result.chi_iom[1] = calc_chi_from_f_state(std::move(f_N), 1);
    }

    for_Int(pij, 0, n_O) {

        const Int chi_idx = 2 + pij;

        VecReal f_Oij = make_f_state_from_diag_A(chi_idx);

        result.chi_iom[chi_idx] = calc_chi_from_f_state(std::move(f_Oij), chi_idx);
    }

    if (mm) {
        PIO("==================================================");
        PIO("find_chi_Sz_N_O_nondeg finished:");
        PIO("chi_Sz   = " + STR(result.chi_iom[0][0]));
        PIO("chi_Nimp = " + STR(result.chi_iom[1][0]));

        for_Int(pij, 0, n_O) {
            const Int chi_idx = 2 + pij;

            PIO("chi_O_" + STR(O_i[pij]) + STR(O_j[pij])
                + " = " + STR(result.chi_iom[chi_idx][0]));
        }

        PIO("==================================================");
    }

    return result;
}

void CrrltFun::find_gf_greater(const Real& ge0, Green &g0) {
    find_gf(ge0, g0, true);
}

void CrrltFun::find_gf_lesser(const Real& ge0, Green &g0) {
    find_gf(ge0, g0, false);
}

void CrrltFun::find_gf_orbital(const Real& ge0, ImGreen& g0, const VecReal& vgs,
    Int set_idx, const SparseMatReal& sep_h, bool is_greater, const Str& monitor_context)
{
    if(p.g_lanczos_reorthogonalization != 0 && p.g_lanczos_reorthogonalization != 1)
        ERR("g_lanczos_reorthogonalization must be 0 or 1.");
    const bool reorthogonalize = p.g_lanczos_reorthogonalization == 1;
    const Int krylov_length = 500;
    if(new_nosp.dim == 0) return;
    const Idx max_basis = std::min<Idx>(krylov_length + 1, new_nosp.dim);
    if(mm) WRN(NAV2(set_idx, p.g_lanczos_reorthogonalization));

    const Str context = monitor_context.empty()
        ? "set=" + STR(set_idx) + " sector=" + (is_greater ? "N+1" : "N-1")
        : monitor_context;
    green_profile::Profile profile(mm,context,"contribution",p.green_krylov.solver);
    std::ostringstream checks;
    const auto log_memory=[&](const char* stage) {
        green_profile::Scope timer(&profile,green_profile::Monitor);
        log_computation_memory(mm,stage,new_nosp.dim,"GF",context);
    };
    log_memory("SOURCE_PROJECTION_BEFORE");

    green_profile::Scope source_timer(&profile,green_profile::Source);
    const VecReal source = project_uplwer_parical_space(vgs, crtorann, set_idx, 0);
    source_timer.stop();
    checks<<"GF_CHECK "<<context<<" stage=source_projection status=COMPLETE test=existing_index_guards physical_operator_mapping=NOT_CHECKED\n";
    log_memory("SOURCE_PROJECTION_AFTER");

    green_krylov::FrequencyGrid frequencies;
    Int adaptive_grid=p.green_krylov.lanczos_step==0;
    mm.Bcast(adaptive_grid);
    if(adaptive_grid) {
        frequencies.z.reset(g0.nomgs);
        for_Int(w,0,g0.nomgs) frequencies.z[w]=g0.z(w);
        frequencies.ge0=ge0;
        frequencies.is_greater=is_greater;
    }
    log_memory("LANCZOS_BEFORE");
    const auto spectra=green_krylov::solve(mm,source,sep_h,new_nosp.dim,
        max_basis,reorthogonalize,p.green_krylov,context,[&](const char* stage,Int column){
            log_computation_memory(mm,stage,new_nosp.dim,"GF",context+" column="+STR(column));
        },frequencies);
    log_memory("SPECTRAL_AFTER");
    // Commit only after every source passed checks. Preserve the existing N+/-1
    // real-Hamiltonian convention and the existing orbital indices.
    log_memory("FREQUENCY_BEFORE");
    green_krylov::accumulate_frequency(mm,spectra,g0,ge0,is_greater,&profile,checks);
    log_memory("FREQUENCY_AFTER");
    profile.finish(mm,checks.str());
}
/*
// ! only suit for the two orbital cases.
VecReal CrrltFun::find_hd_exstate(Int position, Int ex_idx) {
    VEC<VecBool> ex_VecBools;
    {
        ex_VecBools.push_back(VecBool({false, false}));// 0  , true, true
        ex_VecBools.push_back(VecBool({false,  true}));// 1  , true, true
        ex_VecBools.push_back(VecBool({ true, false}));// 2  , true, true
        ex_VecBools.push_back(VecBool({ true,  true}));// 3  , true, true
        //for special case
        ex_VecBools.push_back(VecBool());// 4
    }
#define ndiv new_nosp.ndivs
    if (p.norbs > 4) ERR(NAV(p.norbs));
    VecPartition vp(mm.np(), mm.id(), new_nosp.dim);
    
    VecReal hdex_state(ex_state);
    VecBool imp(4, false), imp_t(2, false);// the last one left for check the right.
    for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) {
        StateStatistics a(vp.bgn() + h_i, new_nosp.wherein_NocSpace(vp.bgn() + h_i), new_nosp);
        for_Int(i, 0, p.norbs) imp[i] = a.cfg.cf[(i)*ndiv].isocc(0) ? true : false;
        bool check(true);
        if (position == 0) {
            imp_t = imp.truncate(2,4);
            if (ex_idx < 4) check = (imp_t == ex_VecBools[ex_idx]) && imp.truncate(0,2) == VecBool({ true,  true}) ? true : false;
            else if (ex_idx == 4) check = ((imp_t == ex_VecBools[1]) || (imp_t == ex_VecBools[2])) && imp.truncate(0,2) == VecBool({ true,  true}) ? true : false;                            // #4 = |1> + |2>
            else if (ex_idx == 5) check = ((imp_t == ex_VecBools[0]) || (imp_t == ex_VecBools[1]) || (imp_t == ex_VecBools[2])) && imp.truncate(0,2) == VecBool({ true,  true}) ? true : false; // #5 = |0> + |1> + |2> 
            else if (ex_idx == 6) check = imp[0] && (!imp[1]) && (!imp[2]) && imp[3];// kondo peak
        }
        if (position == 2) {
            imp_t = imp.truncate(0,2);
            if (ex_idx < 4) check = (imp_t == ex_VecBools[ex_idx]) && imp.truncate(2,4) == VecBool({ true,  true}) ? true : false;
            else if (ex_idx == 4) check = ((imp_t == ex_VecBools[1]) || (imp_t == ex_VecBools[2])) && imp.truncate(2,4) == VecBool({ true,  true}) ? true : false;                            // #4 = |1> + |2>
            else if (ex_idx == 5) check = ((imp_t == ex_VecBools[0]) || (imp_t == ex_VecBools[1]) || (imp_t == ex_VecBools[2])) && imp.truncate(2,4) == VecBool({ true,  true}) ? true : false; // #5 = |0> + |1> + |2> 
        }
        hdex_state[h_i] = check ? hdex_state[h_i] : 0.;
    }
#undef ndiv

    return hdex_state;
}
*/


// ! only suit for the three orbital cases.
VecReal CrrltFun::find_hd_exstate(Int position, Int ex_idx) {
    VEC<VecBool> ex_VecBools;
    {
        //       |00> |01> |10> |11>
        //  |00>  #0   #1   #2   #3
        //  |01>  #4   #5   #6   #7
        //  |10>  #8   #9  #10  #11
        //  |11> #12  #13  #14  #15
        ex_VecBools.push_back(VecBool({false, false, false, false, true, true}));// 0
        ex_VecBools.push_back(VecBool({false, false, false,  true, true, true}));// 1
        ex_VecBools.push_back(VecBool({false, false,  true, false, true, true}));// 2
        ex_VecBools.push_back(VecBool({false, false,  true,  true, true, true}));// 3
        ex_VecBools.push_back(VecBool({false,  true, false, false, true, true}));// 4
        ex_VecBools.push_back(VecBool({false,  true, false,  true, true, true}));// 5
        ex_VecBools.push_back(VecBool({false,  true,  true, false, true, true}));// 6
        ex_VecBools.push_back(VecBool({false,  true,  true,  true, true, true}));// 7
        ex_VecBools.push_back(VecBool({ true, false, false, false, true, true}));// 8
        ex_VecBools.push_back(VecBool({ true, false, false,  true, true, true}));// 9
        ex_VecBools.push_back(VecBool({ true, false,  true, false, true, true}));// 10
        ex_VecBools.push_back(VecBool({ true, false,  true,  true, true, true}));// 11
        ex_VecBools.push_back(VecBool({ true,  true, false, false, true, true}));// 12
        ex_VecBools.push_back(VecBool({ true,  true, false,  true, true, true}));// 13
        ex_VecBools.push_back(VecBool({ true,  true,  true, false, true, true}));// 14
        ex_VecBools.push_back(VecBool({ true,  true,  true,  true, true, true}));// 15
        //for special case
        ex_VecBools.push_back(VecBool()); // 16
    }
#define ndiv new_nosp.ndivs
    if (p.norbs != 6) ERR(NAV(p.norbs));
    VecPartition vp(mm.np(), mm.id(), new_nosp.dim);
    NocSpace::GroupCursor group_cursor(new_nosp);
    
    VecReal hdex_state(ex_state);
    VecBool imp(6, false);// the last one left for check the right.
    for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) {
        StateStatistics a(vp.bgn() + h_i, group_cursor.locate(vp.bgn() + h_i), new_nosp);
        for_Int(i, 0, p.norbs) imp[i] = a.cfg.cf[(i)*ndiv].isocc(0) ? true : false;
        bool check(true);
        // if (position == 0) check = !(imp[1]) && imp[0] && imp[3] && !(imp[2]);
        // if      (position == 0) check = !(imp[2]) && !(imp[3]) && imp[0] && imp[1];
        if (position == 0)      check = (imp[0]) && (!imp[1]) && (!imp[4]) && (imp[5]); //kondo peak
        else if (position == 2) check = (!imp[0]) && (!imp[1]) && (imp[2]) && (imp[3]);//hd peak
        else if (ex_idx < 16 && position == 4) check = imp == ex_VecBools[ex_idx] ? true : false;
        else if (ex_idx == 16 && position == 4) check = ((!imp[2]) && (!imp[3]) && (imp[4]) && (imp[5])) || ((!imp[0]) && (!imp[1]) && (imp[4]) && (imp[5])); // #16 h*d+*hd
        else if (ex_idx == 17 && position == 4) check = ((imp == ex_VecBools[1]) || (imp == ex_VecBools[2]) || (imp == ex_VecBools[4]) || (imp == ex_VecBools[8])) ? true : false; // #17
        else if (ex_idx == 18 && position == 4) check = ((imp == ex_VecBools[5]) || (imp == ex_VecBools[6]) || (imp == ex_VecBools[9]) || (imp == ex_VecBools[10])) ? true : false; // #18
        else if (ex_idx == 19 && position == 4) check = ((imp == ex_VecBools[7]) || (imp == ex_VecBools[11]) || (imp == ex_VecBools[13]) || (imp == ex_VecBools[14])) ? true : false; // #19
        else if (ex_idx == 20 && position == 4) check = ((imp == ex_VecBools[3]) || (imp == ex_VecBools[12])) ? true : false; // hdd+dhd
        else if (ex_idx == 21 && position == 4) check = ((imp == ex_VecBools[1]) || (imp == ex_VecBools[2]) || (imp == ex_VecBools[4]) || (imp == ex_VecBools[8]) || (imp == ex_VecBools[5]) || (imp == ex_VecBools[6]) || (imp == ex_VecBools[9]) || (imp == ex_VecBools[10])) ? true : false; // #17 || #18
        // else if (position == 4) check = !(imp[0]) && !(imp[1]) && !(imp[2]) && !(imp[3]) && imp[4] && imp[5];
        // else if (position == 4) check = !(imp[0]) && !(imp[1]) &&  (imp[2]) &&  (imp[3]) && imp[4] && imp[5];
        // else if (position == 4) check = (!(imp[0]) && !(imp[1]) && imp[4] && imp[5]) || (!(imp[2]) && !(imp[3]) && imp[4] && imp[5]);
        // else if (position == 4) check = (!(imp[0]) && !(imp[1]) && imp[4] && imp[5] && (imp[2]) && imp[3]) || (!(imp[0]) && !(imp[1]) && imp[4] && imp[5] && !(imp[3]) && !(imp[2]));
        else {ERR("The position is wrong!" + NAV(position));}
        // if(mm && check) WRN(NAV(imp));
        hdex_state[h_i] = check ? hdex_state[h_i] : 0.;
    }
    // if (position == 2) 	for_Int(h_i, vp.bgn(), vp.end()) {
    //     StateStatistics a(h_i, new_nosp.wherein_NocSpace(h_i), new_nosp);
    //     for_Int(i, 0, p.norbs) imp[i] = a.cfg.cf[(i)*ndiv].isocc(0) ? true : false;
    //     bool check = !(imp[0]) && !(imp[1]) && imp[2] && imp[3];
    //     if(mm && check) WRN(NAV(imp));
    //     hdex_state[h_i] = check ? hdex_state[h_i] : 0.;
    // }
#undef ndiv

    return hdex_state;
}

void CrrltFun::find_hd_greater(const Real& ge0, Green &g0, Int position, Int ex_idx) 
{
    VecReal hd_exstate(find_hd_exstate(position, ex_idx));
    Real upper_fraction(mm.Allreduce(DOT(hd_exstate, hd_exstate)));
    // if (mm) WRN(NAV(upper_fraction));
    VEC<Real> ltd;	        // diagonal elements 
    VEC<Real> lt_sd;	    // sub-diagonal elements
    const SparseMatReal sep_h = find_hmlt_V2(table);
    VecReal v0(hd_exstate);
    v0 *= INV(SQRT(mm.Allreduce(DOT(v0, v0))));
    VEC<VecReal> v0_saved;
    v0_saved.push_back(v0);
    VecReal v1(sep_h * v0);
    Real a_i(0.), b_i(0.);
    a_i = mm.Allreduce(DOT(v1, v0));
    ltd.push_back(a_i);
    if(g0.type_info() == STR("ImGreen")){
        for_Int(i, 0, 200) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        ltd.push_back(a_i); lt_sd.push_back(b_i); v0_saved.push_back(v0);
        // if(v0_saved.size() > 10) v0_saved.erase(v0_saved.begin());
        }
    }
    if(g0.type_info() == STR("ReGreen")){
        for_Int(i, 0, 3000) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        ltd.push_back(a_i); lt_sd.push_back(b_i); v0_saved.push_back(v0);
        // if(v0_saved.size() > 10) v0_saved.erase(v0_saved.begin());
        }
    }
    // ImGreen last_green(1, p);
    Cmplx zero(0.,0.);
    VecCmplx green_pre(g0.nomgs, zero);
    VecCmplx green_error(g0.nomgs, zero);
    Int iter_time(0);
    while (true) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        ltd.push_back(a_i); lt_sd.push_back(b_i);
        VecReal a(ltd), b(lt_sd);
        Int n(a.size());
        //ImGreen green_i(1, p);
        for_Int(w, 0, g0.nomgs) {
            Cmplx z = g0.z(w) + ge0;
            Cmplx c(z - a[n - 2] - b[n - 2] * b[n - 2] / (z - a[n - 1]));
            for (Int i = n; i >= 3; i--) c = z - a[i - 3] - b[i - 3] * b[i - 3] / c;
            //Cmplx under_fraction(c);
            Cmplx gaz = upper_fraction / c;
            green_error[w] = gaz - green_pre[w];
            green_pre[w] = gaz;
        }
        if(iter_time % 50 == 0 && iter_time != 0){
            if(compare_error(ltd[ltd.size() - 1], ltd[ltd.size() - 2]) < 1E-6) break;
            VecCmplx check_err = g0.type_info() == STR("ImGreen") ? green_error.truncate(0,int(p.fit_num_omg/2)): green_error;
            if(mm) WRN(NAV(ABS(SUM(check_err))/check_err.size()));
            if(ABS(SUM(check_err))/check_err.size() < 1E-14) break;
        }
        iter_time++;
        // if (mm && ltd.size() > 200 && g0.type_info() == STR("ImGreen")) PIO("The size of a and b in greaer:" + NAV3(ltd.size(), lt_sd.size(), SUM(green_error)));
    }
    for_Int(w, 0, g0.nomgs) g0[w][0][0] += green_pre[w];
    if (mm) PIO("The size of a and b in greaer:" + NAV2(ltd.size(), lt_sd.size()));
    // return last_green;
}

// void CrrltFun::find_gf_greater(const Real& ge0, Green &g0, Int kind) 
// {
//     Real upper_fraction(DOT(ex_state, ex_state));
//     //VECtrdgnl trdignl(find_trdgnl_first(ex_state));
//     VEC<Real> ltd;	        // diagonal elements 
//     VEC<Real> lt_sd;	    // sub-diagonal elements
//     const SparseMatReal sep_h = find_hmlt_V2(table);
//     Cmplx zero(0.,0.);
//     VecCmplx green_pre(g0.nomgs, zero);
//     VecCmplx green_error(g0.nomgs, zero);
//     if(kind == 0){
//         VecReal v0(ex_state);
//         v0.normalize();
//         VecReal v0_saved(v0);
//         VecReal v1(sep_h * v0);
//         Real a_i(0.), b_i(0.);
//         a_i = DOT(v1, v0);
//         ltd.push_back(a_i);
//         for_Int(i, 0, 3000) {
//             find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
//             ltd.push_back(a_i); lt_sd.push_back(b_i);
//         }
//         // ImGreen last_green(1, p);
//         while (true) {
//             find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
//             ltd.push_back(a_i); lt_sd.push_back(b_i);
//             VecReal a(ltd), b(lt_sd);
//             Int n(a.size());
//             //ImGreen green_i(1, p);
//             for_Int(w, 0, g0.nomgs) {
//                 Cmplx z = g0.z(w, kind) + ge0;
//                 Cmplx c(z - a[n - 2] - b[n - 2] * b[n - 2] / (z - a[n - 1]));
//                 for (Int i = n; i >= 3; i--) c = z - a[i - 3] - b[i - 3] * b[i - 3] / c;
//                 //Cmplx under_fraction(c);
//                 Cmplx gaz = upper_fraction / c;
//                 green_error[w] = gaz - green_pre[w];
//                 green_pre[w] = gaz;
//             }
//             if (ABS(SUM(green_error)) < 1.E-10 * g0.nomgs) break;
//             a_saved.reset(a); b_saved.reset(b);
//         }
//     }
//     if(kind > 0){
//         VecReal a(a_saved), b(b_saved);
//         Int n(a.size());
//         for_Int(w, 0, g0.nomgs) {
//             Cmplx z = g0.z(w, kind) + ge0;
//             Cmplx c(z - a[n - 2] - b[n - 2] * b[n - 2] / (z - a[n - 1]));
//             for (Int i = n; i >= 3; i--) c = z - a[i - 3] - b[i - 3] * b[i - 3] / c;
//             //Cmplx under_fraction(c);
//             Cmplx gaz = upper_fraction / c;
//             green_pre[w] = gaz;
//         }
//     }
//     for_Int(w, 0, g0.nomgs) g0[w][0][0] += green_pre[w];
//     if (mm && kind == 0) PIO("The size of a and b in greaer:" + NAV2(a_saved.size(), b_saved.size()));
//     // return last_green;
// }

// void CrrltFun::find_gf_lesser(const Real& ge0, Green &g0, Int kind) 
// {
//     Real upper_fraction(DOT(ex_state, ex_state));
//     //VECtrdgnl trdignl(find_trdgnl_first(ex_state));
//     VEC<Real> ltd;	        // diagonal elements 
//     VEC<Real> lt_sd;	    // sub-diagonal elements
//     const SparseMatReal sep_h = find_hmlt_V2(table);
//     Cmplx zero(0.,0.);
//     VecCmplx green_pre(g0.nomgs, zero);
//     VecCmplx green_error(g0.nomgs, zero);
//     if(kind == 0){
//         VecReal v0(ex_state);
//         v0.normalize();
//         VecReal v0_saved(v0);
//         VecReal v1(sep_h * v0);
//         Real a_i(0.), b_i(0.);
//         a_i = DOT(v1, v0);
//         ltd.push_back(a_i);
//         for_Int(i, 0, 3000) {
//             find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
//             ltd.push_back(a_i); lt_sd.push_back(b_i);
//         }
//         // ImGreen last_green(1, p);
//         while (true) {
//             find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
//             ltd.push_back(a_i); lt_sd.push_back(b_i);
//             VecReal a(ltd), b(lt_sd);
//             Int n(a.size());
//             //ImGreen green_i(1, p);
//             for_Int(w, 0, g0.nomgs) {
//                 Cmplx z = g0.z(w, kind) - ge0;
//                 Cmplx c(z + a[n - 2] - b[n - 2] * b[n - 2] / (z + a[n - 1]));
//                 for (Int i = n; i >= 3; i--) c = z + a[i - 3] - b[i - 3] * b[i - 3] / c;
//                 //Cmplx under_fraction(c);
//                 Cmplx gaz = upper_fraction / c;
//                 green_error[w] = gaz - green_pre[w];
//                 green_pre[w] = gaz;
//             }
//             if (ABS(SUM(green_error)) < 1.E-10 * g0.nomgs) break;
//             a_saved.reset(a); b_saved.reset(b);
//         }
//     }
//     if(kind > 0){
//         VecReal a(a_saved), b(b_saved);
//         Int n(a.size());
//         for_Int(w, 0, g0.nomgs) {
//             Cmplx z = g0.z(w, kind) - ge0;
//             Cmplx c(z + a[n - 2] - b[n - 2] * b[n - 2] / (z + a[n - 1]));
//             for (Int i = n; i >= 3; i--) c = z + a[i - 3] - b[i - 3] * b[i - 3] / c;
//             //Cmplx under_fraction(c);
//             Cmplx gaz = upper_fraction / c;
//             green_pre[w] = gaz;
//         }
//     }
//     for_Int(w, 0, g0.nomgs) g0[w][0][0] += green_pre[w];
//     if (mm && kind == 0) PIO("The size of a and b in lesser:" + NAV2(a_saved.size(), b_saved.size()));
//     // return last_green;
// }

//---------------------------------------------Private function---------------------------------------------

// (Deactivate) 
CrrltFun::Vectrdgnl CrrltFun::find_trdgnl(const VecReal& initial_vector, const Int crtann) {
    // for tridiagonal matrix
    VEC<Real> ltd;	    // diagonal elements 
    VEC<Real> lt_sd;	// sub-diagonal elements
    SparseMatReal sep_h = find_hmlt_V2(table);
    VecReal v0(initial_vector);
    v0.normalize();
    VecReal v0_saved(v0);
    //DBG(NAV(v0_saved.truncate(0, 1000)));
    VecReal v1(sep_h * v0);
    Idx k = 0;
    Real a(0.), b(0.);
    while (true)
    {
        a = mm.Allreduce(DOT(v1, v0));
        ltd.push_back(a);
        if (k >= 30) if (k > v0_saved.size() - 20) break;
        k++;
        v1 -= a * v0;
        v1 -= mm.Allreduce(DOT(v1, v0_saved)) * v0_saved;
        b = v1.norm();
        lt_sd.push_back(b);
        v1 *= (1 / b);
        SWAP(v0, v1);
        v1 *= -b;
        v1 += sep_h * v0;
    }
    VecReal va_i(ltd), vb_i(lt_sd);
#ifdef _ASSERTION_
    if (va_i.size() - 1 != vb_i.size())ERR("Wrong with count orbit idx: " + NAV2(va_i.size(), vb_i.size()));
#endif
    return std::make_tuple(va_i, vb_i);
}

void CrrltFun::find_trdgnl_one_step(const VEC<VecReal>& initial_vector, VecReal& v0, VecReal& v1, Real&a, Real& b, const SparseMatReal& sep_h){
    lanczos_local::axpy(v1,-a,v0);
    b = SQRT(mm.Allreduce(lanczos_local::dot(v1, v1)));
    // v1 -= mm.Allreduce(lanczos_local::dot(v1, initial_vector)) * initial_vector;
    for_Int(i, 0, initial_vector.size() - 1) {
        lanczos_local::axpy(v1,-mm.Allreduce(lanczos_local::dot(v1,initial_vector[i])),initial_vector[i]);
        // WRN("For the iter" + NAV(i));
    }
    lanczos_local::scale(v1,INV(SQRT(mm.Allreduce(lanczos_local::dot(v1,v1)))));
    swapall(v0,v1);
    sep_h.apply(v0,v1,1.,-b);
    a = mm.Allreduce(lanczos_local::dot(v1, v0));
}

//--------------------------------------------- ex_state function part ---------------------------------------------
void CrrltFun::add_ex_state_part_in_rotation(const VecReal &initial_vector, VecReal& ex_state_part, const Int& set_n, const VecReal& rotation_coefficients, BasisIndex h_i, BasisIndex subnosp) const {
    StateStatistics a(h_i, subnosp, old_nosp);

    Real* const destination = ex_state_part.p();
    for_Int(pos, 0, rotation_coefficients.size()) {
        MatInt new_nospdiv = expand_occupations(old_nosp.div[subnosp], old_nosp.sit_mat.nrows(), old_nosp.ndivs);
        Int div_idx_in_one_set(0);
        for_Int(i, 1, p.ndiv + 1) if (SUM_0toX(old_nosp.sit_mat[set_n], i) > pos) {div_idx_in_one_set = i - 1; break; }
        if (crtorann == -1) --new_nospdiv[set_n][div_idx_in_one_set];
        if (crtorann == +1) ++new_nospdiv[set_n][div_idx_in_one_set];
        if (new_nosp.ifin_NocSpace(new_nospdiv, new_nosp.nppso)) {
            VecOnb exd_cf = a.cfg.cf;
            Int orbit_pos_in_div = pos - SUM_0toX(old_nosp.sit_mat[set_n], div_idx_in_one_set);
            if (if_in_this_orbital(exd_cf, crtorann, set_n * new_nosp.ndivs + div_idx_in_one_set, orbit_pos_in_div)) {
                if (crtorann == -1) exd_cf[set_n * new_nosp.ndivs + div_idx_in_one_set] = exd_cf[set_n * new_nosp.ndivs + div_idx_in_one_set].ann(orbit_pos_in_div);
                if (crtorann == +1) exd_cf[set_n * new_nosp.ndivs + div_idx_in_one_set] = exd_cf[set_n * new_nosp.ndivs + div_idx_in_one_set].crt(orbit_pos_in_div);
                const ComDivs b(exd_cf, new_nospdiv, new_nosp.sit_mat);
                BasisIndex begin_idx(-1);
                begin_idx = new_nosp.divs_to_idx.at(new_nospdiv.vec().string());
                Int sign  = a.cfg.sgn(-1, pos + SUM_0toX(p.nO2sets, set_n));
                if (begin_idx == -1) ERR("wrong with ex_state" + NAV2(a.cfg.ne, new_nospdiv));
                const Real amplitude = sign * rotation_coefficients[pos] * initial_vector[h_i];
                // Rotated orbitals can send several source rows to the same target.
                // One full contribution vector per rank, never one per thread.
#pragma omp atomic update
                destination[begin_idx + b.idx] += amplitude;
            }
        }
    }
}

VecReal CrrltFun::project_uplwer_parical_space(const VecReal& initial_vector,
    const Int crtann, const Int norg_set, const Int orbit_pos_in_div) const
{
    namespace par = stage2_parallel;
    VecReal ex_state_part, rotation_coefficients;
    VecPartition row_H(mm.np(), mm.id(), old_nosp.dim);
    par::collective(mm, [&] {
        if(initial_vector.size() != old_nosp.dim || (crtann != -1 && crtann != 1)
            || crtann != crtorann || norg_set < 0 || norg_set >= p.norg_sets)
            throw std::invalid_argument("Invalid source projection state/sector");
        ex_state_part.reset(new_nosp.dim, 0.);
        if(p.if_norg_imp) {
            const auto& rotation = p.rotationU[norg_set];
            if(orbit_pos_in_div < 0 || static_cast<Idx>(orbit_pos_in_div) >= rotation.ncols())
                throw std::invalid_argument("Source orbital outside rotation matrix");
            rotation_coefficients.reset(rotation.nrows());
            for(Idx pos = 0; pos < rotation.nrows(); ++pos)
                rotation_coefficients[pos] = rotation[pos][orbit_pos_in_div];
        } else if(orbit_pos_in_div < 0 || orbit_pos_in_div >= old_nosp.sit_mat[norg_set][0])
            throw std::invalid_argument("Source orbital outside impurity division");

        par::ranges(row_H.len(), [&](std::size_t first, std::size_t last, std::size_t) {
            NocSpace::GroupCursor group_cursor(old_nosp);
            for(std::size_t row = first; row < last; ++row) {
                const BasisIndex h_i = row_H.bgn() + static_cast<BasisIndex>(row);
                if(p.if_norg_imp) {
                    const BasisIndex subnosp = group_cursor.locate(h_i);
                    add_ex_state_part_in_rotation(initial_vector, ex_state_part,
                        norg_set, rotation_coefficients, h_i, subnosp);
                } else {
                    // A fixed unrotated creation/annihilation operator is injective
                    // on nonzero source configurations: disjoint writes need no atomics.
        const BasisIndex subnosp(group_cursor.locate(h_i));
        MatInt new_nospdiv = expand_occupations(old_nosp.div[subnosp], old_nosp.sit_mat.nrows(), old_nosp.ndivs);
        if (crtann == -1) --new_nospdiv[norg_set][0];
        if (crtann == +1) ++new_nospdiv[norg_set][0];
        // if (new_nosp.ifin_NocSpace(new_nospdiv)) {
        if (new_nosp.ifin_NocSpace(new_nospdiv, new_nosp.nppso)) {
            const ComDivs group(h_i - old_nosp.idx_div[subnosp], (old_nosp.div[subnosp]), (old_nosp.sit_mat), true);
            VecOnb exd_cf = group.cf;
            if (if_in_this_orbital(exd_cf, crtann, norg_set * new_nosp.ndivs, orbit_pos_in_div)) {
                if (crtann == -1) exd_cf[norg_set * new_nosp.ndivs] = exd_cf[norg_set * new_nosp.ndivs].ann(orbit_pos_in_div);
                if (crtann == +1) exd_cf[norg_set * new_nosp.ndivs] = exd_cf[norg_set * new_nosp.ndivs].crt(orbit_pos_in_div);
                const ComDivs b(exd_cf, new_nospdiv, old_nosp.sit_mat);
                BasisIndex begin_idx(-1);
                begin_idx = new_nosp.divs_to_idx.at(new_nospdiv.vec().string());
                if (begin_idx == -1) ERR("wrong with ex_state" + NAV2(group.ne, new_nospdiv));
                ex_state_part[begin_idx + b.idx] = exd_cf[norg_set * new_nosp.ndivs].sgn(orbit_pos_in_div) * initial_vector[h_i]; //! Here the "sgn(orbit_pos_in_div)" is useful only on the ↑ spin orbital.
            }
        }
    
                }
            }
        });
    });
    const VecPartition target(mm.np(), mm.id(), new_nosp.dim);
    return mm.Reduce_scatter(ex_state_part, target);
}

bool CrrltFun::if_in_this_orbital(const VecOnb &exd_cf, const Int crtann, const Int div_in_one_row_pos, const Int orbit_pos_in_div) const {
    if (crtann == -1) return exd_cf[div_in_one_row_pos].isocc(orbit_pos_in_div);
    if (crtann == +1) return exd_cf[div_in_one_row_pos].isuno(orbit_pos_in_div);
    ERR("Some thing wrong with this function, if_in_this_orbital ()!")
}
