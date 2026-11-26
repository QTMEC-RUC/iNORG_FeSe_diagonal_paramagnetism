/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "crrvec.h"
using namespace std;


// Crrvec::Crrvec(const NocSpace& old_nosp_i, const Operator& main_opr, const VecReal& vgs_i, const Real& gs, const Int right_set_i, const Real omeg, const Real eta_i)
//     :mm(main_opr.mm),p(main_opr.p), sep_h(main_opr.find_hmlt(main_opr.table)),
//     ground_state(vgs_i), right_set(right_set_i), right_pos_in_div(0), opr(main_opr),
//     old_nosp(old_nosp_i), new_nosp(main_opr.scsp), crtorann(main_opr.scsp.nspa - old_nosp_i.nspa), omega(omeg), eta(eta_i), correct_green(1,cmplx(0.)),
//     ex_state(project_uplwer_parical_space(vgs_i, crtorann, right_set_i)), ground_state_energy(gs), correct_vector(ex_state.size(), 0.)
// {
//     correct_vector = find_correct_vec();
//     // Cmplx test = DOT(cmplx(ex_state), correct_vector);
//     Cmplx a = (correct_vector * (cmplx(omeg, eta) - cmplx(gs)) + cmplx(sep_h * real(correct_vector), sep_h * imag(correct_vector)) - cmplx(ex_state)).norm();
//     if(mm) WRN(NAV2(DOT(cmplx(ex_state), correct_vector), a));
//     // if(mm) WRN(NAV3(correct_vector, ex_state, a));
//     correct_vecs.reset(1,correct_vector.size());
//     correct_vecs[0] = std::move(correct_vector);
//     correct_green[0] = DOT(cmplx(ex_state), correct_vector);
// }

Crrvec::Crrvec(const NocSpace &old_nosp_i, const Operator& main_opr, const VecReal &vgs_i, const Real &gs, const Int right_set_i, const VecReal &omega_point)
    :mm(main_opr.mm),p(main_opr.p), sep_h(main_opr.find_hmlt(main_opr.table)),
    ground_state(vgs_i), right_set(right_set_i), right_pos_in_div(0), opr(main_opr),
    old_nosp(old_nosp_i), new_nosp(main_opr.scsp), crtorann(main_opr.scsp.nspa - old_nosp.nspa), omega(0), eta(0), correct_green(omega_point.size(), cmplx(0.)),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, right_set_i)), ground_state_energy(gs)
{
    krylov_update_state( I * cmplx(omega_point));
    // Cmplx test = DOT(cmplx(ex_state), correct_vecs[0]);
    // if(mm) WRN(NAV(test));
    for_Int(i, 0, omega_point.size()) correct_green[i] = DOT(cmplx(ex_state), correct_vecs[i]);
}

Crrvec::Crrvec(const NocSpace &old_nosp_i, const Operator& main_opr, const VecReal &vgs_i, const Real &gs, const Int right_set_i, const VecCmplx &omega_point)
    :mm(main_opr.mm),p(main_opr.p), sep_h(main_opr.find_hmlt(main_opr.table)),
    ground_state(vgs_i), right_set(right_set_i), right_pos_in_div(0), opr(main_opr),
    old_nosp(old_nosp_i), new_nosp(main_opr.scsp), crtorann(main_opr.scsp.nspa - old_nosp.nspa), omega(0), eta(0), correct_green(omega_point.size(), cmplx(0.)),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, right_set_i)), ground_state_energy(gs)
{
    krylov_update_state(omega_point);
    // Cmplx test = DOT(cmplx(ex_state), correct_vecs[0]);
    // Cmplx a = (correct_vecs[0] * (omega_point[0] - cmplx(gs)) + cmplx(sep_h * real(correct_vecs[0]), sep_h * imag(correct_vecs[0])) - cmplx(ex_state)).norm();
    // if(mm) WRN(NAV2(DOT(cmplx(ex_state), correct_vecs[0]), a));
    // if(mm) WRN(NAV3(correct_vecs[0], ex_state, a));
    for_Int(i, 0, omega_point.size()) correct_green[i] = DOT(cmplx(ex_state), correct_vecs[i]);
}

Crrvec::Crrvec(const NocSpace &old_nosp_i, const Operator& main_opr, const Int right_set_i, const Int right_pos_in_div_i, const VecReal &vgs_i, const Real &gs)
    :mm(main_opr.mm),p(main_opr.p), sep_h(main_opr.find_hmlt(main_opr.table)), opr(main_opr),
    ground_state(vgs_i), right_set(right_set_i), right_pos_in_div(right_pos_in_div_i),
    old_nosp(old_nosp_i), new_nosp(main_opr.scsp), crtorann(main_opr.scsp.nspa - old_nosp.nspa), omega(0), eta(0.01),
    ex_state(project_uplwer_parical_space(vgs_i, crtorann, right_set_i, right_pos_in_div_i)), ground_state_energy(gs)
{
    // WRN("test")
}

// ImGreen Crrvec::find_gf() {
// 
//     ImGreen green(1, p);
//     for_Int(w, 0, green.nomgs) {
//         omega   = real(green.z(w));
//         eta     = imag(green.z(w));
//         VecCmplx exstate(cmplx(ex_state));
//         Cmplx z = DOT(exstate, find_correct_vec());
//         green[w][0][0] = z;
//     }
//     return green;
// }

ImGreen Crrvec::find_gf_from_krylov_space() {
    VEC<Real> ltd;	        // diagonal elements 
    VEC<Real> lt_sd;	    // sub-diagonal elements
    VEC<Real> inner_m;	    // for the inner product of Krylov basis and ex_state.
    {// Test
        // WRN("TEST_Lanczos is right:::" + NAV(mm.np()));
	    // VecReal test_a(ex_state);
	    // VecReal test_b(sep_h * test_a);
	    // WRN("TEST_Lanczos[0] state" + NAV3(test_a.isnormalized(), test_b.avg_abs_elem_diff(eval[0] * test_a)));
    }
    // const SparseMatReal sep_h = find_hmlt(table);
    VecReal v0(ex_state);
    v0.normalize();
    VecReal v0_saved(v0), v1(sep_h * v0);
    Real a_i(0.), b_i(0.);
    a_i = DOT(v1, v0);
    ltd.push_back(a_i); inner_m.push_back(DOT(v0, ex_state));
    for_Int(i, 0, 80) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        lt_sd.push_back(b_i); ltd.push_back(a_i); inner_m.push_back(DOT(v0, ex_state));
    }
    Int counter(0);
    ImGreen last_green(1, p);
    while (true) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        lt_sd.push_back(b_i); ltd.push_back(a_i), inner_m.push_back(DOT(v0, ex_state));
        VecReal m(inner_m), va(ltd), vb(lt_sd);
        VecCmplx green_error(last_green.nomgs);
        MatReal ev(ltd.size(), ltd.size(), 0.);
        // if (mm) WRN(NAV3(va, vb, m));
        Int info = trd_heevr_vd(va, vb, ev); ev = ev.tr();
        // if (mm) WRN(NAV3(va, vb, m));

        if (info != 0)ERR("the i-th parameter had an illegal value." + NAV3(info, va, vb));
        // m.normalize();
        if (crtorann == +1)for_Int(w, 0, last_green.nomgs) {
            Cmplx gaz(0., 0.), z = last_green.z(w) + ground_state_energy;
            // for_Int(i, 0, va.size()) gaz += normalization_fraction * DOT(ev[i], m) * INV((z - va[i])) * DOT(m, ev[i]);
            for_Int(i, 0, va.size()) gaz += DOT(ev[i], m) * INV((z - va[i])) * DOT(m, ev[i]);
            green_error[w] = gaz - last_green[w][0][0];
            last_green[w][0][0] = gaz;
        }
        if (crtorann == -1)for_Int(w, 0, last_green.nomgs) {
            Cmplx gaz(0., 0.), z = last_green.z(w) - ground_state_energy;
            // for_Int(i, 0, va.size()) gaz += normalization_fraction * DOT(ev[i], m) * INV((z + va[i])) * DOT(m, ev[i]);
            for_Int(i, 0, va.size()) gaz += DOT(ev[i], m) * INV((z + va[i])) * DOT(m, ev[i]);
            green_error[w] = gaz - last_green[w][0][0];
            last_green[w][0][0] = gaz;
        }
        // if(mm) WRN("The size of a and b in here:" + NAV2(va.size(), vb.size()));
        break;
        
        counter++;
    }
    return last_green;
}


void Crrvec::prepare_Krylov_space() {
    VEC<Real> ltd;	        // diagonal elements 
    VEC<Real> lt_sd;	    // sub-diagonal elements
    VEC<Real> inner_m;	    // for the inner product of Krylov basis and ex_state.

    // const SparseMatReal sep_h = find_hmlt(table);
    VecReal v0(ex_state);
    v0.normalize();
    VecReal v0_saved(v0), v1(sep_h * v0);
    Real a_i(0.), b_i(0.);
    a_i = DOT(v1, v0);
    ltd.push_back(a_i); inner_m.push_back(DOT(v0, ex_state));
    for_Int(i, 0, 1000) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        ltd.push_back(a_i); lt_sd.push_back(b_i); inner_m.push_back(DOT(v0, ex_state));
    }
    a_saved.reset(ltd); b_saved.reset(lt_sd); inner_m_saved.reset(inner_m);
    // delete sep_h;
}

void Crrvec::krylov_update_state(VecCmplx omega_vec){
	// if(mm) WRN("prepare_Krylov_space() BEGIN.");
    //--------------------------------------------
    VEC<Real> ltd;	            // diagonal elements 
    VEC<Real> lt_sd;	        // sub-diagonal elements
    VEC<Real> inner_m_i;	    // for the inner product of Krylov basis and ex_state.
    Int krylov_length = 300;    // The dim of Krylov space.
    {
        VecReal v0(ex_state);
        v0.normalize();
        VecReal v0_saved(v0), v1(sep_h * v0);
        Real a_i(0.), b_i(0.);
        a_i = DOT(v1, v0);
        ltd.push_back(a_i); inner_m_i.push_back(DOT(v0, ex_state));
        for_Int(i, 0, krylov_length) {
            find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
            ltd.push_back(a_i); lt_sd.push_back(b_i); inner_m_i.push_back(DOT(v0, ex_state));
        }
    }
    VecReal va(ltd), vb(lt_sd), m(inner_m_i), v0(ex_state);
    v0.normalize();
    VecReal v0_saved(v0), v1(sep_h * v0);
    MatReal S(va.size(), va.size(), 0.);
    Int info = trd_heevr_vd(va, vb, S);
    if (info != 0)ERR("the i-th parameter had an illegal value." + NAV3(info, va, vb));
    VecCmplx inner_m(cmplx(m));
    MatCmplx first_part(omega_vec.size(), va.size()), s_mat(cmplx(S.tr()));
    // if (mm) WRN("iteration BEGIN."+NAV4(crtorann, s_mat.ct().nrows(), s_mat.nrows(), inner_m.size()));
    for_Int(i, 0, omega_vec.size()) {
        VecCmplx va_complx(va.size());
        if (crtorann == +1) for_Int(j, 0, va.size()) va_complx[j] = INV(Cmplx(ground_state_energy - va[j] + real(omega_vec[i]), imag(omega_vec[i])));
        if (crtorann == -1) for_Int(j, 0, va.size()) va_complx[j] = INV(Cmplx(-ground_state_energy + va[j] + real(omega_vec[i]), imag(omega_vec[i])));
        MatCmplx middle(dmat(va_complx));
        first_part[i] = s_mat.ct() * middle * s_mat * inner_m;
    }
    correct_vecs.reset(omega_vec.size(), ex_state.size(), cmplx(0.));

    for_Int(i, 0, omega_vec.size())  correct_vecs[i] += first_part[i][0] * cmplx(v0);
    Real a_i(0.), b_i(0.);
    a_i = DOT(v1, v0);
    for_Int(i, 0, krylov_length) {
        find_trdgnl_one_step(v0_saved, v0, v1, a_i, b_i, sep_h);
        for_Int(j, 0, omega_vec.size())  correct_vecs[j] += first_part[j][i+1] * cmplx(v0);
    }
}

SparseMatReal Crrvec::find_diagonal_sparse_matrix(Real number)
{
    VecPartition row_H(mm.np(), mm.id(), new_nosp.dim);
    SparseMatReal splited_number(row_H.len(), new_nosp.dim, mm);
    for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i)splited_number.addelement(number, h_i, h_i - row_H.bgn());
    return splited_number;
}

SparseMatReal Crrvec::add_diagonal_sparse_matrix(VEC<VecInt> h_idx, Real ge0, Real omega)
{
    VecPartition row_H(mm.np(), mm.id(), new_nosp.dim);
    SparseMatReal hmlt_splited(row_H.len(), new_nosp.dim, mm);
    bool once(true);
    Idx lastone(0);
    if (crtorann == +1) {
        Real number = -ge0 - omega;
        for (const auto& idxval : h_idx)
        {
            lastone = idxval[0];
            if (idxval[0] > lastone) once = true;
            Int coefficient_comm = idxval[2] >= 0 ? 1 : -1;
            hmlt_splited.addelement(coefficient_comm * new_nosp.coefficient[abs(idxval[2])], idxval[1], idxval[0]);
            if (once && idxval[1] == (idxval[0] + row_H.bgn())) {
                hmlt_splited.addelement(number, idxval[1], idxval[0]);
                once = false;
            }
        }
    }
    if (crtorann == -1) {
        Real number = -ge0 + omega;
        for (const auto& idxval : h_idx)
        {
            if (idxval[0] > lastone) once = true;
            Int coefficient_comm = idxval[2] >= 0 ? 1 : -1;
            hmlt_splited.addelement(coefficient_comm * new_nosp.coefficient[abs(idxval[2])], idxval[1], idxval[0]);
            if (once && idxval[1] == (idxval[0] + row_H.bgn())) {
                hmlt_splited.addelement(number, idxval[1], idxval[0]);
                once = false;
            }
        }
    }
    return hmlt_splited;
}

VecReal Crrvec::operator*(const VecReal& KetVec)
{
    VecReal x(KetVec.size(),0.);
    if (crtorann == +1) {
        x = sep_h * KetVec;
        x = sep_h * x;
        x += (omega * omega) * KetVec;
        x += (ground_state_energy * ground_state_energy) * KetVec;
        x += (eta * eta) * KetVec;
        x += (2 * omega * ground_state_energy) * KetVec;
        x -= sep_h * (2 * omega * KetVec);
        x -= sep_h * (2 * ground_state_energy * KetVec);
    }
    if (crtorann == -1) {
        x = sep_h * KetVec;
        x = sep_h * x;
        x += (omega * omega) * KetVec;
        x += (ground_state_energy * ground_state_energy) * KetVec;
        x += (eta * eta) * KetVec;
        x -= (2 * omega * ground_state_energy) * KetVec;
        x += sep_h * (2 * omega * KetVec);
        x -= sep_h * (2 * ground_state_energy * KetVec);
    }
    return x;
}

VecReal Crrvec::imag_to_real(const VecReal& v_imag)
{
    VecReal x(v_imag.size(), 0.);
    if (crtorann == +1) {
        x += (1. / eta) * (sep_h * v_imag);
        x -= ((1. / eta) * ground_state_energy) * v_imag;
        x -= ((1. / eta) * omega) * v_imag;
    }
    if (crtorann == -1) {
        x -= ((1. / eta) * omega) * v_imag;
        x += ((1. / eta) * ground_state_energy) * v_imag;
        x -= (1. / eta) * (sep_h * v_imag);
    }
    return x;
}

// VecCmplx Crrvec::find_correct_vec()
// {
//     VecReal b(ex_state.size(),0.);
//     b = -eta * ex_state;
//     VecReal imag_v(ex_state.size(), 0.);
//     conjugate_gradient_simple((*this), imag_v, b);
//     VecReal real_v (imag_to_real(imag_v));
//     return (cmplx(real_v, imag_v));
// }

void Crrvec::find_trdgnl_one_step(const VecReal& initial_vector, VecReal& v0, VecReal& v1, Real& a, Real& b, const SparseMatReal& sep_h) {
    v1 -= a * v0;
    b = SQRT(mm.Allreduce(DOT(v1, v1)));
    v1 -= mm.Allreduce(DOT(v1, initial_vector)) * initial_vector;
    v1 *= INV(SQRT(mm.Allreduce(DOT(v1, v1))));
    SWAP(v0, v1);
    v1 *= -b;
    v1 += sep_h * v0;
    a = mm.Allreduce(DOT(v1, v0));
}

void Crrvec::find_excted_state_by_ab(const VecReal& initial_vector, VecReal& v0, VecReal& v1, VecReal& vec_a, VecReal& vec_b, Int& k, const SparseMatReal& sep_h) {
    v1 -= vec_a[k-1] * v0;
    v1 -= DOT(v1, initial_vector) * initial_vector;
    v1.normalize();
    SWAP(v0, v1);
    v1 *= -vec_b[k - 1];
    v1 += sep_h * v0;
}

//--------------------------------------------- ex_state function part ---------------------------------------------

void Crrvec::add_ex_state_part_in_rotation(const VecReal &initial_vector, VecReal& ex_state_part, const Int& set_n, const Int orb_before_rot, BasisIndex h_i, BasisIndex subnosp) const {
    StateStatistics a(h_i, subnosp, old_nosp);

    VecReal rotation_coefficients = p.rotationU[set_n].tr()[orb_before_rot];
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
                ex_state_part[begin_idx + b.idx] += sign * rotation_coefficients[pos] * initial_vector[h_i];
            }
        }
    }
}

VecReal Crrvec::project_uplwer_parical_space(const VecReal& initial_vector, const Int crtann, const Int norg_set, const Int orbit_pos_in_div) const
// By using the C^+ or C on a spinless orbital(for the spinless reason ONE normal orbital has two spinless orbits).
// the norg_set only suppose for the even number.
{
    // clock_t t_find_Newstate; TIME_BGN("find_Newstate" + NAV(mm.id()), t_find_Newstate);
    VecReal ex_state_part(new_nosp.dim, 0.);
    VecPartition row_H(mm.np(), mm.id(), old_nosp.dim);
    NocSpace::GroupCursor group_cursor(old_nosp);
    if (p.if_norg_imp) for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i) {
        const BasisIndex subnosp = group_cursor.locate(h_i);
        add_ex_state_part_in_rotation(initial_vector, ex_state_part, norg_set, orbit_pos_in_div, h_i, subnosp);
    }
    else for (BasisIndex h_i = row_H.bgn(); h_i < row_H.end(); ++h_i) {
        const BasisIndex subnosp(group_cursor.locate(h_i));
        MatInt new_nospdiv = expand_occupations(old_nosp.div[subnosp], old_nosp.sit_mat.nrows(), old_nosp.ndivs);
        if (crtann == -1) --new_nospdiv[norg_set][0];
        if (crtann == +1) ++new_nospdiv[norg_set][0];
        // if (new_nosp.ifin_NocSpace(new_nospdiv)) {
        if (new_nosp.ifin_NocSpace(new_nospdiv, new_nosp.nppso)) {
            const ComDivs group(h_i - old_nosp.idx_div[subnosp], (old_nosp.div[subnosp]), (old_nosp.sit_mat), true);
            VecOnb exd_cf = group.cf;
            if (if_in_this_orbital(exd_cf, crtann, norg_set  * new_nosp.ndivs, orbit_pos_in_div)) {
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
    VecReal ex_state(mm.Allreduce(ex_state_part));
    // TIME_END("find_Newstate" + NAV(mm.id()), t_find_Newstate);
    VecPartition vp(mm.np(), mm.id(), new_nosp.dim);
    VecReal ex_state_p(vp.len());
    for (BasisIndex h_i = 0; h_i < vp.len(); ++h_i) ex_state_p[h_i] = ex_state[vp.bgn() + h_i];
    return ex_state_p;
}

bool Crrvec::if_in_this_orbital(const VecOnb &exd_cf, const Int crtann, const Int div_in_one_row_pos, const Int orbit_pos_in_div) const {
    if (crtann == -1) return exd_cf[div_in_one_row_pos].isocc(orbit_pos_in_div);
    if (crtann == +1) return exd_cf[div_in_one_row_pos].isuno(orbit_pos_in_div);
    ERR("Some thing wrong with this function, if_in_this_orbital ()!")
}
