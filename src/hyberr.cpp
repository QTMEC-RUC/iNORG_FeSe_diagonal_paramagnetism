/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

#include "hyberr.h"
#include "gen/parallel_stage2.h"


HybErr::HybErr(const Prmtr& p_i, const ImGreen& hb, const Int nb_i) :
    p(p_i), nw(p.fit_num_omg), data_n(3), nb(nb_i)
{
    if (hb.norbs != 1 || nw < 1 || nb < 1 || p.nmesh < 2 || p.nmesh > hb.nomgs
        || (!p.fit_points.empty() && p.fit_points.size() != nw))
        ERR("Invalid scalar hybridization fit dimensions");
    data_n[0] = 2 * nw;
    data_n[1] = 2;
    data_n[2] = nb;
    const Int ndata = SUM(data_n);
    x.reset(ndata);
    for_Int(i, 0, ndata) x[i] = i;
    y.reset(ndata, 0.);
    sig.reset(ndata);
    vy.reset(ndata);
    vdyda.reset(ndata);
    for_Int(i, 0, ndata) vdyda[i].reset(2 * nb, 0.);
    a_old.reset(2 * nb);
    zfit.reset(nw + 1);

    Real norm_re = 0., norm_im = 0.;
    for_Int(i, 0, nw) {
        const Int j = p.fit_points.empty() ? i : p.fit_points[i];
        if (j < 0 || j >= hb.nomgs) ERR("Hybridization fitting point outside the frequency mesh");
        zfit[i] = p.Imz(j);
        const Cmplx target = hb[j][0][0];
        y[i] = real(target);
        y[nw + i] = imag(target);
        norm_re += SQR(y[i]);
        norm_im += SQR(y[nw + i]);
    }
    const Real sigma_re = SQRT(1. + norm_re / nw) * SQRT(2. * nw);
    const Real sigma_im = SQRT(1. + norm_im / nw) * SQRT(2. * nw);
    for_Int(i, 0, nw) {
        sig[i] = sigma_re;
        sig[nw + i] = sigma_im;
    }

    // Preserve the original high-frequency anchor and bare-energy regularization.
    const Int tail = p.nmesh - 2;
    zfit[nw] = p.Imz(tail);
    y[2 * nw] = real(hb[tail][0][0]);
    y[2 * nw + 1] = imag(hb[tail][0][0]);
    for_Int(i, 2 * nw, 2 * nw + 2)
        sig[i] = 2.e-3 * SQRT(1. + SQR(y[i])) * SQRT(2.);
    const Real sigma_energy = 1.e2 * std::pow(10., 2 + (nb / 2 - 4) / 11)
        * sqrt(nb) * 4 * p.fit_max_omg;
    for_Int(k, 0, nb) {
        const Int row = 2 * nw + 2 + k;
        sig[row] = sigma_energy;
        vdyda[row][nb + k] = 1.;
    }
}

Real HybErr::operator()(const Int x, const VecReal& a) {
    if (!cache_valid || a != a_old) {
        cache_valid = false;
        a_old = a;
        get_value_mrq(a);
        cache_valid = true;
    }
    return vy[x];
}

void HybErr::operator()(const Int x, const VecReal& a, Real& y_i, VecReal& dyda_i) {
    if (!cache_valid || a != a_old) {
        cache_valid = false;
        a_old = a;
        get_value_mrq(a);
        cache_valid = true;
    }
    y_i = vy[x]; dyda_i = vdyda[x];
}

void HybErr::get_value_mrq(const VecReal& a)
{
    if (a.size() != 2 * nb) ERR("A scalar bath requires one amplitude and one energy per site");
    stage2_parallel::for_each(nw + 1, [&](std::size_t i) {
        const Int re_row = i == static_cast<std::size_t>(nw) ? 2 * nw : static_cast<Int>(i);
        const Int im_row = i == static_cast<std::size_t>(nw) ? 2 * nw + 1 : nw + static_cast<Int>(i);
        Cmplx delta(0., 0.);
        for_Int(k, 0, nb) {
            const Cmplx pole = a[k] / (zfit[i] - a[nb + k]);
            delta += a[k] * pole;
            const Cmplx dhop = 2. * pole;
            const Cmplx denergy = pole * pole;
            vdyda[re_row][k] = real(dhop);
            vdyda[im_row][k] = imag(dhop);
            vdyda[re_row][nb + k] = real(denergy);
            vdyda[im_row][nb + k] = imag(denergy);
        }
        vy[re_row] = real(delta);
        vy[im_row] = imag(delta);
    });
    for_Int(k, 0, nb) vy[2 * nw + 2 + k] = a[nb + k];
}
