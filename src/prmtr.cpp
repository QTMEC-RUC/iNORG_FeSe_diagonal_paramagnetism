/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "prmtr.h"

Prmtr::Prmtr(const MyMpi& mm) : np(mm.np())
{
    set_inert_values();
    set_values();
    derive();
    // if (mm) print();
    // if (mm) { OFS ofs(iox + "log.parameters.txt", std::ios::app);  print(ofs); }
    // if (mm) { OFS ofss("log.parameters.txt", std::ios::app);  print(ofss); }
}

void Prmtr::set_inert_values()
{
    nband = 3;
    norbs = nband * 2;
    project = STR(nband)+"band_KH";
    
	gauss_n_max = 512;		        // default value: 2048
	gauss_n_min = 64;		        // default value: 64
    iter_max = 200;                   // default value: 99

    dlt_freq = 0.005;
    eta_freq = 0.02;
    freq_upp = 5.;
    freq_low = -5.;

    beta = pi_Real*50;
    unit_omg = pi_Real/beta;
    nmesh = 8193;
    // unit_omg = 0.01;
}

void Prmtr::set_values() {
    //model related
    U = 3.3;
    mu = 0.0;

    //Parameter for the hhd function(arXiv:2209.14178v1)
    alpha = 1.0;
    delta = 0.3;
    chi_A = 0;
    chi_nomb = 100;
    if_n_init = 0;
    if_eigenstate_number = 0;
    eigenstate_number = 1;
    testH0 = 0;
    gs_lanczos_reorthogonalization = 0;
    g_lanczos_reorthogonalization = 0;
    green_krylov = GreenKrylovOptions();
    testDelta = 0;
    if_fix_N_subspace = 0;
    fix_start_search_N_subspace.reset();
    fix_N_subspace.clear();
    degenerate_N_subspaces.clear();
    control_divs_degenerate_N_subspaces.clear();
    occupationnumber_N_subspaces.clear();
    rotationU_N_subspaces.clear();

    // jz = 0.5;
    // Uprm = U - 2 * jz;
    jz = 0;
    Uprm = U - delta;
    bandw = 40.;                                        //SQRT(SQR(bethe_u) + SQR(bethe_u12) + SUM(t * t))
    eimp = MatReal();
    degel = 2;                                          // Degenerate energy levels
    bethe_t.reset(nband, 0.5);
	// if (nband > 1) for_Int(i, 0, nband - 1) bethe_t[i + 1] = 0.5 * bethe_t[i];
	// if (nband > 1) for_Int(i, 0, nband - 1) bethe_t[i + 1] = 0.25;
	if (nband == 2) bethe_t[1] = bethe_t[0] * 0.1;
	if (nband == 3) bethe_t[2] = bethe_t[0] * 0.1;      // set for the 3 band hhd (arXiv:2209.14178v1)//! set band 0 same as band 1.
    bsr = bethe_t * bethe_t;                            // the vector of bath sum rule.

    // fitting related
    fit_pow = 2.; // default value: 2.
    fit_rsd = 2; // default value: 2.


    // NORG parameter.
    restrain_difference.reset();
    restrain_difference_dim = 0;
    restrain_active_orbit_number.reset();
    restrain_active_orbit_number_dim = 0;
    restrain_difference_G.reset();
    restrain_active_orbit_number_G.reset();
    control_divs_G.reset();
    restrain_G_mode = 0;
    restrain_autonooc = 0;
    restrain_tight_divs.reset();
    norg_each_orbital_converg_error.reset();
    norg_each_orbital_converg_step_number.reset();
    if_norg_degenerate = 0;
    if_norg_imp = false;
    imp_backup = false;
    templet_restrain = !if_norg_imp ? VecInt{0, -0, -2, -3,  0,  3,  2,  0} : VecInt{-1, -4, -4,  4,  4,  1};
    VecInt templet_control  = !if_norg_imp ? VecInt{1,  0,  0,  3,  1,  3,  0,  0} : VecInt{ 0,  1,  1,  1,  1,  0};
    ndiv = static_cast<Int>(templet_restrain.size());
    norg_sets = norbs;                                  // default value: 1
    nI2B = SUM(templet_control) - templet_control[0];   // default value:
    nO2sets = SUM(templet_control);                     // default value:
    iter_max_norg = 99;                                 // default
    // nooc_mode = STR("nooc");
    // nooc_mode = STR("cpnooc");
    // nooc_mode = STR("cnooc");
    // nooc_mode = STR("phess");
    nooc_mode = STR("phss_v2");
    // control_divs[4] = control_divs[3] = {1,  0,  1,  2,  1,  2,  1,  0};
    // control_divs[6] = control_divs[5] = {1,  0,  1,  2,  1,  2,  1,  0};//! set band 0 same as band 1.
    // control_divs[1] = {1,  0,  3,  0,  1,  0,  3,  0};
    // control_divs[2] = {1,  0,  3,  0,  1,  0,  3,  0};
    // control_divs[3] = {1,  1,  2,  0,  1,  0,  2,  1};
    // control_divs[4] = {1,  1,  2,  0,  1,  0,  2,  1};
    // control_divs[3] = {1,  0,  3,  0,  1,  0,  3,  0};
    // control_divs[4] = {1,  0,  3,  0,  1,  0,  3,  0};
    // after_modify_prmtr();
    // recalc_partical_number();
}

Prmtr Prmtr::make_tight_subspace_parameters() const
{
    if (if_norg_imp || restrain_tight_divs.size() != static_cast<Idx>(ndiv)) {
        ERR("Tight G preconvergence requires bath-only rotation and one constraint per division")
    }
    Prmtr tight(*this);
    tight.templet_restrain.reset(restrain_tight_divs);
    // Reuse the particle-aware partition builder and restrain_autonooc.
    // Special columns, set dimensions and particle counts are preserved.
    tight.according_nppso(tight.npartical);
    return tight;
}

// we set first divison as impurity. The maximum number of cavity("-"); mean electron("+").
// void Prmtr::after_modify_prmtr() const
// {
//     mu = 0.0;
//     nI2B.reset(norg_sets, 0); nO2sets.reset(norg_sets, 0);
//     control_divs.reset(norg_sets + 1, ndiv, 0);
//     control_divs[0] = templet_restrain;
//     for_Int(i, 0, norg_sets) control_divs[i + 1] = templet_control;

//     MatInt sit_mat(control_divs.truncate_row(1,norg_sets + 1));
//     for_Int(i, 0, norg_sets) nI2B[i] = SUM(sit_mat[i]) - sit_mat[i][0];
//     for_Int(i, 0, norg_sets) nO2sets[i] = SUM(sit_mat[i]);
//     norbit = SUM(sit_mat);
//     nbath = norbit - 2 * nband;
//     if (if_norg_imp) {
//         for_Int(i, 0, norg_sets) nI2B[i] = nbath/norg_sets;
//         for_Int(i, 0, norg_sets) nO2sets[i] = norbit/norg_sets;
//     }
//     // npartical.reset(norg_sets, 0);
//     // for_Int(i, 0, norg_sets) npartical[i] = SUM(control_divs[i + 1])/2;
//     // Uprm = U - 2 * jz;
//     n_rot_orb = if_norg_imp ? norbit : nbath;
//     if(if_norg_imp) {
//         if (control_divs[0][0] == control_divs[0][ndiv / 2]) ERR("For full orbit rotation mode, The restrain is not suit for" + NAV(control_divs[0]))
//     } else {
//         if (control_divs[0][0] != control_divs[0][ndiv / 2]) ERR("For only baths rotation mode, The restrain is not suit for" + NAV(control_divs[0]))
//     }
//     derive_ImGreen();
//     rotationU = uormat_initialize();
// }

// Adjust the NORG bath orbitals numbers according to the nbaths, only suit for the one impurity for a norg_set.
void Prmtr::after_modify_prmtr(const VecInt& nbaths) const
{
    mu = 0.;
    if ((nband != 2 && nband != 3 && nband != 5) || norg_sets != 2 * nband
        || nbaths.size() != norg_sets || ndiv < 4 || ndiv % 2)
        ERR("Invalid PM orbital or division layout");
    for_Int(m, 0, nband)
        if (nbaths[2 * m] < 1 || nbaths[2 * m] != nbaths[2 * m + 1])
            ERR("PM partners must have the same positive bath size");
    nI2B.reset(norg_sets, 0); nO2sets.reset(norg_sets, 0);
    control_divs.reset(norg_sets + 1, ndiv, 0);
    control_divs[0] = templet_restrain;
    for_Int(i, 0, norg_sets) {
        const Int nb_remain = nbaths[i] - 1;
        control_divs[i + 1][0] = 1;
        control_divs[i + 1][1] = (nb_remain + 1) / 2;
        control_divs[i + 1][ndiv / 2] = 1;
        control_divs[i + 1][ndiv - 1] = nb_remain / 2;
    }

    MatInt sit_mat(control_divs.truncate_row(1,norg_sets + 1));
    for_Int(i, 0, norg_sets) nI2B[i] = SUM(sit_mat[i]) - sit_mat[i][0];
    for_Int(i, 0, norg_sets) nO2sets[i] = SUM(sit_mat[i]);
    norbit = SUM(sit_mat);
    nbath = SUM(nI2B);
    if (if_norg_imp) {
        for_Int(i, 0, norg_sets) nI2B[i] = nbath/norg_sets;
        for_Int(i, 0, norg_sets) nO2sets[i] = norbit/norg_sets;
    }
    // npartical.reset(norg_sets, 0);
    // for_Int(i, 0, norg_sets) npartical[i] = SUM(control_divs[i + 1])/2;
    // Uprm = U - 2 * jz;
    n_rot_orb = if_norg_imp ? norbit : nbath;
    if(if_norg_imp) {
        if (control_divs[0][0] == control_divs[0][ndiv / 2]) ERR("For full orbit rotation mode, The restrain is not suit for" + NAV(control_divs[0]))
    } else {
        if (control_divs[0][0] != control_divs[0][ndiv / 2]) ERR("For only baths rotation mode, The restrain is not suit for" + NAV(control_divs[0]))
    }
    derive_ImGreen();
    rotationU = uormat_initialize();
}

void Prmtr::reinit_from_nO2sets() const
{
    norbit    = SUM(nO2sets);
    nbath     = SUM(nI2B);
    n_rot_orb = if_norg_imp ? norbit : nbath;
    rotationU = uormat_initialize();
}

namespace {

void validate_autonooc_layout(const Prmtr& p, const VecInt& nppsos)
{
    // Each half contains one special column and at least one bath column.
    if (p.ndiv < 4 || p.ndiv % 2 != 0) {
        ERR("Bath-only autonooc requires an even ndiv >= 4: " + STR(p.ndiv))
    }
    if (p.restrain_autonooc < 1 || p.restrain_autonooc >= p.ndiv / 2) {
        ERR("restrain_autonooc must be in [1, " + STR(p.ndiv / 2 - 1)
            + "] for ndiv=" + STR(p.ndiv) + "; got " + STR(p.restrain_autonooc))
    }
    if (p.norg_sets < 0
        || p.control_divs.nrows() != static_cast<Idx>(p.norg_sets + 1)
        || p.control_divs.ncols() != static_cast<Idx>(p.ndiv)
        || p.templet_restrain.size() != p.ndiv
        || p.nO2sets.size() != p.norg_sets || nppsos.size() != p.norg_sets) {
        ERR("Inconsistent bath-only autonooc parameter dimensions")
    }
}

struct BathPartitionCounts {
    Int left;
    Int right;
};

BathPartitionCounts calculate_bath_partition_counts(
    const VecInt& row, Int total_orbitals, Int particles, Int set_idx)
{
    const Int half = row.size() / 2;
    const Int special_total = row[0] + row[half];
    if (row[0] < 0 || row[half] < 0 || special_total % 2 != 0
        || total_orbitals < special_total) {
        ERR("Invalid special-column counts for autonooc set " + STR(set_idx))
    }

    // Use the same special-column contribution as recalc_partical_number().
    const Int special_particles = special_total / 2;
    const Int bath_total = total_orbitals - special_total;
    if (particles < special_particles || particles > special_particles + bath_total) {
        ERR("Cannot preserve special columns for autonooc set " + STR(set_idx)
            + ": particle count " + STR(particles) + " must be in ["
            + STR(special_particles) + ", " + STR(special_particles + bath_total) + "]")
    }
    BathPartitionCounts counts;
    counts.left = particles - special_particles;
    counts.right = bath_total - counts.left;
    return counts;
}

VecInt make_bath_partition_row(
    const VecInt& previous_row, const BathPartitionCounts& counts, Int level)
{
    const Int ndiv = previous_row.size();
    VecInt row(ndiv, 0);
    row[0] = previous_row[0];
    row[ndiv / 2] = previous_row[ndiv / 2];
    row[level] = counts.left;
    row[ndiv - level] = counts.right;
    return row;
}

} // namespace

// The bath-only mode selects a mirrored partition pair via restrain_autonooc.
// The first column is the impurity; "-" limits holes and "+" limits electrons.
void Prmtr::according_nppso(const VecInt& nppsos) const
{
    if (!if_norg_imp) validate_autonooc_layout(*this, nppsos);
    control_divs[0] = templet_restrain;
    for_Int(i, 0, norg_sets) {
        if (if_norg_imp) {
            // control_divs[i + 1] = templet_control;
            control_divs[i + 1][0] = nppsos[i] - SUM(control_divs[i + 1].truncate(1, Int(ndiv / 2)));
            control_divs[i + 1][ndiv - 1] = (nO2sets[i] - nppsos[i]) - SUM(control_divs[i + 1].truncate(Int(ndiv / 2), ndiv - 1));
            // control_divs[i + 1][0] = nppsos[i] != SUM(control_divs[i + 1]) ? nppsos[i] - SUM(control_divs[i + 1].truncate(1, Int(ndiv / 2)))       : SUM(control_divs[i + 1]) - control_divs[i + 1][0];
            // control_divs[i + 1][ndiv - 1] = nppsos[i] != 0 ? (nO2sets[i] - nppsos[i]) - SUM(control_divs[i + 1].truncate(Int(ndiv / 2), ndiv - 1)) : SUM(control_divs[i + 1]) - control_divs[i + 1][0];
            // WRN(NAV2(control_divs, templet_control));
            for_Int(j, 0, ndiv / 2.) {
                if (control_divs[i + 1][j] < 0) {
                    int t = -control_divs[i + 1][j]; control_divs[i + 1][ndiv - j - 1] += t;
                    control_divs[i + 1][j + 1] -= t; control_divs[i + 1][j] = 0;
                }
                if (control_divs[i + 1][ndiv - j - 1] < 0) {
                    int t = -control_divs[i + 1][ndiv - j - 1]; control_divs[i + 1][j] += t;
                    control_divs[i + 1][ndiv - j - 2] -= t; control_divs[i + 1][ndiv - j - 1] = 0;
                }
            }
        } else {
            const BathPartitionCounts counts = calculate_bath_partition_counts(
                control_divs[i + 1], nO2sets[i], nppsos[i], i);
            control_divs[i + 1] = make_bath_partition_row(
                control_divs[i + 1], counts, restrain_autonooc);
        }
    }
}

// Apply each symmetry group's complete partition to its spin sets.
void Prmtr::according_controler(const Vec<VecInt>& controler, const VecInt& or_deg) const
{
    if (controler.size() == 0 || or_deg.size() != static_cast<Idx>(norg_sets)
        || control_divs.nrows() != static_cast<Idx>(norg_sets + 1))
        ERR("Inconsistent search partition dimensions")
    for_Idx(group, 0, controler.size()) {
        if (controler[group].size() != control_divs.ncols())
            ERR("Search partition width differs from control_divs for group " + STR(group))
    }
    for_Int(i, 0, norg_sets) {
        if (or_deg[i] <= 0 || static_cast<Idx>(or_deg[i]) >= controler.size())
            ERR("Invalid search partition group for set " + STR(i))
    }
    control_divs[0] = controler[0];
    for_Int(i, 0, norg_sets) control_divs[i+1] = controler[or_deg[i]];
}

// according the control_divs set the number of partical.
void Prmtr::recalc_partical_number() const
{
    npartical.reset(norg_sets, 0);
    for_Int(i, 0, norg_sets) if (if_norg_imp) {
        npartical[i] = SUM(control_divs[i + 1].truncate(0, Int(ndiv / 2)));

    } else {
        npartical[i] = SUM(control_divs[i + 1].truncate(1, Int(ndiv / 2))) + \
            (control_divs[i + 1][0] + control_divs[i + 1][ndiv / 2]) / 2;
    }
}


void Prmtr::derive() {

    nfreq = 1 + Int_ROUND((freq_upp - freq_low) / dlt_freq);
    Re_z.reset(nfreq);
    for_Int(n, 0, nfreq) Re_z[n] = Cmplx(freq_low + n * dlt_freq, eta_freq);

    // max_omg = 4 * SQRT(SQR(U) + DOT(t, t));
    // max_omg = 2 * (ABS(U) + 8 * SQRT(DOT(t, t) - t[0] * t[0]));
    max_omg = unit_omg * nmesh * 2;

    num_omg = Int_ROUND(max_omg / unit_omg / 2);
    Im_z.reset(num_omg);
	for_Int(n, 0, num_omg) Im_z[n] = Cmplx(0., (2 * n + 1) * unit_omg);

    fit_max_omg = bandw / 4. ;
    fit_num_omg = fit_points.empty() ? Int_ROUND(fit_max_omg / unit_omg / 2):fit_points.size(); // The default value: Int_ROUND(fit_max_omg / unit_omg / 2) change for speed reason.

    Str key_prmtr_val = STR("prmtr");
    ofx = iox + key_prmtr_val;
}

void Prmtr::derive_ImGreen() const {
    unit_omg = pi_Real/beta;
    max_omg = unit_omg * nmesh * 2;

    num_omg = Int_ROUND(max_omg / unit_omg / 2);
    Im_z.reset(num_omg);
	for_Int(n, 0, num_omg) Im_z[n] = Cmplx(0., (2 * n + 1) * unit_omg);

    fit_max_omg = bandw / 4. ;
    fit_num_omg = fit_points.empty() ? Int_ROUND(fit_max_omg / unit_omg / 2)
        : static_cast<Int>(fit_points.size());
}

void Prmtr::print(std::ostream &os) const { // ! not in used in DFT+DMFT
#define prmtr_print(var, comment) print(os, NAME(var), STR(var), comment)

    using namespace std;
    MatInt stage2(concat(stage2_restrain,stage2_control).mat(2,stage2_control.size()));    
	Str cnooc = nooc_mode;
	Str rotation_imp = if_norg_imp ? "Yes" : "NO";

    os << "// prmtr print bgn  " << present() << endl;
    prmtr_print(np, "number of mpi processes");
    prmtr_print(nbath, "number of bath orbital");

    prmtr_print(norbs, "number of spin-orbitals");
   	prmtr_print(cnooc, "Correlation nature orbital occupation constraint.");
	prmtr_print(rotation_imp, "if rotation impurity orbital is used. ");
    prmtr_print(control_divs, "to set the number of division and the shortcut restratin.");
    // prmtr_print(stage2, "to set the number of division and the shortcut restratin @ stage2.");
    prmtr_print(U, "The hubbard U interaction strength");
    prmtr_print(Uprm, "The U^' term");
    prmtr_print(jz, "The hund coupling");
    prmtr_print(mu, "The chemical potential");
    for_Int(i,0,bethe_t.size()) prmtr_print(bethe_t[i], "quarter bandwidth(0 to t.size())");
    prmtr_print(max_omg, "max_omg of imgreen");
    prmtr_print(eta_freq, "eta of omega + i * eta");
    prmtr_print(dlt_freq, "The real x-axis unit for retarded green function");
    prmtr_print(fit_max_omg, "The fitting max omg");
    prmtr_print(VecInt(fit_points), "to set the fix fitting points.");
    prmtr_print(fit_num_omg, "The max number of fitting points.");
    prmtr_print(unit_omg*2, "The image x-axis unit for Matsubara Green's function");
    // prmtr_print(imp_backup, "if you have the impurity's back up?");
    // prmtr_print(nooc_mode, "the tight mode is refer to the correlation nature orbital constraint");
    // prmtr_print(ofx, "output filename prefix");

    os << "// prmtr print end  " << present() << endl;

    os << "\n";															// blank line

#undef prmtr_print
}

