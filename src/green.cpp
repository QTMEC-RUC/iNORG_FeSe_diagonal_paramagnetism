#include "green.h"

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

using namespace std;

void Green::read(const Str& file) {
	IFS ifs(file);
	if (!ifs) {
		WRN(STR("file opening failed with ") + NAV(file))
	}
	else {

		for_Int(i, 0, (norbs + 1) * 2) {
			Str strr;
			ifs >> strr;
		}
		Real omg;
		MatReal re(norbs, norbs, 0.);
		MatReal im(norbs, norbs, 0.);
		for_Int(i, 0, nomgs) {
			ifs >> omg;
			for_Int(m, 0, norbs) {

				Int n = m;
				ifs >> re[m][n];
				// }
			}
			ifs >> omg;
			for_Int(m, 0, norbs) {

				Int n = m;
				ifs >> im[m][n];
				// }
			}
			g[i] = cmplx(re, im);
		}
		if (!ifs) {
			ERR(STR("read-in error with ") + NAV(file))
		}
	}
	ifs.close();
}


Green::Green(const Prmtr& p, Int nomgs_i, const VecCmplx& z_omg_i, const Str& file) :
	nomgs(nomgs_i), norbs(p.nband), z_omg(z_omg_i), g(nomgs, MatCmplx(norbs, norbs, 0.)) {
	if (!file.empty()) {
		read(file);
	}
}

Green::Green(Int norbs_i, const Prmtr& p, Int nomgs_i, const VecCmplx& z_omg_i, const Str& file) :
	nomgs(nomgs_i), norbs(norbs_i), z_omg(z_omg_i), g(nomgs, MatCmplx(norbs, norbs, 0.)) {
	if (!file.empty()) {
		read(file);
	}
}

void ImGreen::write_occupation_info() const {
	using namespace std;
	OFS ofs; ofs.open("nmat.txt");
	VecReal counter(3);
	VecReal particals(particle_number().diagonal());

	ofs << "#   < n_i >   data:" << endl;
	for_Int(orb_i, 0, norbs * 2) {
		ofs << iofmt();
		std::string temp = (orb_i % 2) == 0 ? STR(Int(orb_i / 2) + 1) + "up" : STR(Int(orb_i / 2) + 1) + "dn";
		ofs << setw(6) << temp << setw(p_Real) << particals[Int(orb_i / 2)] << endl;
	}
	counter[0] = SUM(particals);
	counter[1] = SUM(particals);

	counter[2] = counter[0] + counter[1];
	ofs << setw(6) << "sup" << setw(p_Real) << counter[0] << endl;
	ofs << setw(6) << "sdn" << setw(p_Real) << counter[1] << endl;
	ofs << setw(6) << "sum" << setw(p_Real) << counter[2];
	ofs.close();
}

void Green::write(const Str& green_name, Int iter_cnt) const {
	OFS ofs;
	// if (iter_cnt == 999) { ofs.open(iox + green_name + ".txt"); }
	if (iter_cnt == 999) { ofs.open(green_name + ".out"); }
	else { ofs.open(iox + "zic" + prefill0(iter_cnt, 3) + ".mb." + green_name + ".txt"); }
	Str iter_str = iter_cnt == 999 ? "" : STR(green_name, iter_cnt) + "_";
	// OFS ofs(iox + "zic" + prefill0(iter_cnt, 3) + ".mb." + green_name + ".txt");
	ofs << iofmt("sci");
	// real part of green
	ofs << setw(w_Real) << "w";
	for_Int(m, 0, norbs) {

		Int n = m;
		ofs << "  " << setw(w_Real) << iter_str + "Re" + STR(m + 1) + STR(n + 1);
		// }
	}
	// imag part of green
	ofs << "  " << setw(w_Real) << "w";
	for_Int(m, 0, norbs) {

		Int n = m;
		ofs << "  " << setw(w_Real) << iter_str + "Im" + STR(m + 1) + STR(n + 1);
		// }
	}
	ofs << endl;
	for_Int(i, 0, nomgs) {
		// real part of green
		ofs << setw(w_Real) << omg(i);
		for_Int(m, 0, norbs) {

			Int n = m;
			ofs << "  " << setw(w_Real) << real(g[i][m][n]);
			// }
		}
		// imag part of green
		ofs << "  " << setw(w_Real) << omg(i);
		for_Int(m, 0, norbs) {

			Int n = m;
			ofs << "  " << setw(w_Real) << imag(g[i][m][n]);
			// }
		}
		ofs << endl;
	}
}


void Green::write(const Str& green_name, const Str& rowname, Int iter_cnt) const {
	OFS ofs;
	// if (iter_cnt == 999) { ofs.open(iox + green_name + rowname + ".txt"); }
	if (iter_cnt == 999) { ofs.open(green_name + rowname + ".txt"); }
	else { ofs.open(iox + "zic" + prefill0(iter_cnt, 3) + ".mb." + green_name + rowname + ".txt"); }
	Str iter_str = iter_cnt == 999 ? STR(rowname) + "_" : STR(rowname) + STR(iter_cnt) + "_";
	// OFS ofs(iox + "zic" + prefill0(iter_cnt, 3) + ".mb." + green_name + ".txt");
	ofs << iofmt("sci");
	// real part of green
	ofs << setw(w_Real) << "w";
	for_Int(m, 0, norbs) {
		ofs << "  " << setw(w_Real) << iter_str + "Re" + STR(m + 1) + STR(m + 1);
	}
	// imag part of green
	ofs << "  " << setw(w_Real) << "w";
	for_Int(m, 0, norbs) {
		ofs << "  " << setw(w_Real) << iter_str + "Im" + STR(m + 1) + STR(m + 1);
	}
	ofs << endl;
	for_Int(i, 0, nomgs) {
		// real part of green
		ofs << setw(w_Real) << omg(i);
		for_Int(m, 0, norbs) {
			ofs << "  " << setw(w_Real) << real(g[i][m][m]);
		}
		// imag part of green
		ofs << "  " << setw(w_Real) << omg(i);
		for_Int(m, 0, norbs) {
			ofs << "  " << setw(w_Real) << imag(g[i][m][m]);
		}
		ofs << endl;
	}
}

void Green::write_edmft(const Str& filename) const {
    OFS out(filename);
    if (!out) ERR("Cannot write Green function: " + filename)
    out << iofmt("sci");
    for_Int(n, 0, nomgs) {
        out << setw(w_Real) << omg(n);
        for_Int(m, 0, norbs)
            out << "  " << setw(w_Real) << real(g[n][m][m])
                << "  " << setw(w_Real) << imag(g[n][m][m]);
        out << std::endl;
    }
}

void Green::read_edmft(const Str& filename) {
    IFS input(filename);
    if (!input) ERR("Cannot open Green function: " + filename)
    Str line;
    Int n = 0;
    while (n < nomgs && std::getline(input, line)) {
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof() || row.peek() == '#') continue;
        Real frequency;
        if (!(row >> frequency)) ERR("Missing frequency in " + filename)
        for_Int(m, 0, norbs) {
            Real re, im;
            if (!(row >> re >> im)) ERR("Expected a complex pair per orbital in " + filename)
            g[n][m][m] = Cmplx(re, im);
        }
        row >> std::ws;
        if (!row.eof() && row.peek() != '#') ERR("Extra Green components in " + filename)
        ++n;
    }
    if (n != nomgs) ERR("Too few frequencies in " + filename)
}
Real ImGreen::error(const ImGreen& other, Real omg_rsd) const {
    VecReal weights(nomgs);
    VecReal mag_real(norbs, 0.), mag_imag(norbs, 0.);
    for_Int(n, 0, nomgs) {
        weights[n] = INV(SQR(omg(n) + omg_rsd));
        for_Int(m, 0, norbs) {
            mag_real[m] += SQR(real(g[n][m][m])) + SQR(real(other.g[n][m][m]));
            mag_imag[m] += SQR(imag(g[n][m][m])) + SQR(imag(other.g[n][m][m]));
        }
    }
    // Keep the historical normalization of the convergence diagnostic.
    weights *= INV(SUM(weights) * (2 * norbs * norbs));
    const Real scale = std::max(Real(1.e-2), std::sqrt(std::max(MAX(mag_real),MAX(mag_imag))/(2*nomgs)));
    Real sum = 0.;
    for_Int(n, 0, nomgs) for_Int(m, 0, norbs)
        sum += weights[n] * (norm(other.g[n][m][m] - g[n][m][m]) / SQR(scale));
    return std::sqrt(sum);
}

Real ImGreen::sum(const VecCmplx& gf) const
{
    const Real temp = unit_omg / pi_Real;
	Real electron_density = real(SUM(gf));
	if (nomgs <= 5) return 2 * temp * electron_density;
	Int shtct_omg(nomgs-1);				
	while(ABS(real(gf[shtct_omg]))<1.E-6 && shtct_omg > 4){
		shtct_omg--;
	}
	if(shtct_omg == 4)	return 2 * temp * electron_density;				//if gf is 0, insensitive with omg, return 0
    VecReal w = imag(z_omg.truncate(shtct_omg - 4, shtct_omg));
    VecReal gg = real(gf.truncate(shtct_omg - 4, shtct_omg));
    MatReal A(4, 4);
    MatReal B(4, 1);
    for_Int(i, 0, 4) {
        // w[i] = imag(z_omg[shtct_omg - 1 - 2 * i]);
        // gg[i] = real(gf[shtct_omg - 1 - 2 * i]);
        A[i][0] = w[i] * w[i];
        A[i][1] = 1.;
        A[i][2] = -w[i] * w[i] * gg[i];
        A[i][3] = -gg[i];
        B[i][0] = gg[i] * w[i] * w[i] * w[i] * w[i];
    }
    Int info = gaussj(A, B);
    if(info > 0)    {WRN("gaussj false"+NAV(info));	return 2 * temp * electron_density;}
    Real x = B[0][0];
    Real y = B[1][0];
    Real m = B[2][0];
    Real n = B[3][0];
	VecReal a(2,0.);
	VecReal b(2,0.);
    a[0] = 0.5 * (m + SQRT(m * m - 4 * n));
    a[1] = 0.5 * (m - SQRT(m * m - 4 * n));
    b[0] = (x * a[0] - y) / (a[0] - a[1]);
    b[1] = (y - x * a[1]) / (a[0] - a[1]);
	if(std::isnan(a[0]) || std::isnan(a[1]))	return 2 * temp * electron_density;
	for_Int(i,0,2){
		if(a[i]<0)	{electron_density += b[i] * pi_Real * tan(SQRT(-a[i]) * pi_Real / (2. * unit_omg)) / (4 * unit_omg * SQRT(-a[i]));}
		else		{electron_density += b[i] * pi_Real * tanh(SQRT(a[i]) * pi_Real / (2. * unit_omg)) / (4 * unit_omg * SQRT(a[i]));}
	}
    for_Int(n, 0, nomgs) {
		for_Int(i,0,2){
        	electron_density -= b[i] / (SQR(omg(n)) + a[i]);
		}
    }
	return 2 * temp * electron_density;
}

//gfloc.particle_number(); particle-hole transformation has been applied to the down-spin, so this cannot be used
//<c_i^+ c_j> - <c_j c_i^+> = (2/beta) sum_{n=-inf inf}<<c_j c_i^+>>_(iw_n)
//mat_ij=<c_i^+ c_j>
MatReal ImGreen::particle_number() const {
    MatReal density(norbs, norbs, 0.);
    VecCmplx channel(nomgs);
    for_Int(m, 0, norbs) {
        for_Int(n, 0, nomgs) channel[n] = g[n][m][m];
        density[m][m] = sum(channel) + 0.5;
    }
    return density;
}
