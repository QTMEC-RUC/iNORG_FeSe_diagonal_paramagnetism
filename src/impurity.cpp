#include "impurity.h"

/*
code developed and maintained by (yc045663@gmail.com, RUC, China)
*/

Impurity::Impurity(const MyMpi &mm_i, const Prmtr &prmtr_i, const Bath &bth_i, const Str& file): mm(mm_i), p(prmtr_i), bth(bth_i),
    nb(p.nbath), ni(p.norbs), ns(p.norbit), pos_imp(p.norbs), 
    h0(p.norbit, p.norbit, 0.), imp_lvl(p.norbs), ordeg(0)
{
    // find_ve();
    // if (!file.empty()) read(file);
    for_Int(i, 0, ni) imp_lvl[i] = p.eimp[i][i] - p.mu;
}

Impurity::Impurity(const MyMpi &mm_i, const Prmtr &prmtr_i, const Bath &bth_i, const VecInt or_deg): mm(mm_i), p(prmtr_i), bth(bth_i),
    nb(p.nbath), ni(p.norbs), ns(p.norbit), pos_imp(p.norbs),
    h0(p.norbit, p.norbit, 0.), imp_lvl(p.norbs, 0.), ordeg(or_deg)
{
    if (p.eimp.nrows() != ni || p.eimp.ncols() != ni || ni != 2 * p.nband)
        ERR("One impurity level is required in each PM spin-orbital set");
    for_Int(i, 0, ni) {
        for_Int(j, 0, ni)
            if (i != j && p.eimp[i][j] != 0.) ERR("Off-diagonal impurity levels are unsupported");
        if (p.eimp[i][i] != p.eimp[i - i % 2][i - i % 2])
            ERR("PM partners require equal impurity levels");
        imp_lvl[i] = p.eimp[i][i] - p.mu;
    }
}

using namespace std;

void Impurity::find_g0(Green& g0) const
{
    if (g0.norbs != p.nband) ERR("G0 requires one scalar per spatial orbital");
    for_Int(m, 0, p.nband) {
        const Int pos = pos_imp[2 * m];
        const Int size = 1 + bth.nb[m];
        if (p.testH0) {
            // A supplied spin-sector matrix can have a rotated (dense) bath.
            const MatCmplx sector = cmplx(h0.truncate(pos, pos, pos + size, pos + size));
            for_Int(n, 0, g0.nomgs)
                g0[n][m][m] = matinvlu(dmat(size, g0.z(n)) - sector)[0][0];
        } else {
            for_Int(n, 0, g0.nomgs) {
                const Cmplx z = g0.z(n);
                Cmplx inverse = z - h0[pos][pos];
                for_Int(k, 1, size)
                    inverse -= SQR(h0[pos][pos + k]) / (z - h0[pos + k][pos + k]);
                g0[n][m][m] = 1. / inverse;
            }
        }
    }
}

// rely on imp model's frame
void Impurity::find_all_g0(Green &g0) const {

    MatCmplx Z(ns, ns);
    for_Int(n, 0, g0.nomgs) {
        Z = g0.z(n);
        MatCmplx g0_z = matinvlu(Z - cmplx(h0));
        for_Int(i, 0, p.norbit) {
            g0[n][i][i] = g0_z[i][i];
        }
    }
}

void Impurity::find_hb(Green& hb) const
{
    if (hb.norbs != p.nband) ERR("Hybridization requires one scalar per spatial orbital");
    for_Int(m, 0, p.nband) {
        for_Int(n, 0, hb.nomgs) {
            const Cmplx z = hb.z(n);
            Cmplx delta(0., 0.);
            for_Int(k, 0, bth.nb[m])
                delta += SQR(bth.fvb[m][0][k]) / (z - bth.fvb[m][1][k]);
            hb[n][m][m] = delta;
        }
    }
}

MatReal Impurity::find_hop_for_test() const
{
    VecReal hop({ -0.312546, 0.159346, -0.0619612, -0.312546, 0.159346 });
    VecReal ose({ -0.474855,0.0667285, 0,           0.474855, -0.0667285 });
    MatReal h0(0, 0, 0.);
    for_Int(i, 0, p.nband) {
        const Int nb_i = p.nI2B[i * 2];
        MatReal h0_i(1 + nb_i, 1 + nb_i, 0.);
        for_Int(j, 0, nb_i) {
            h0_i[0][j + 1] = hop[j];
            h0_i[j + 1][0] = hop[j];
            h0_i[j + 1][j + 1] = ose[j];
        }
        h0_i.reset(direct_sum(h0_i, h0_i));
        h0.reset(direct_sum(h0, h0_i));
    }
    return h0;
}

void Impurity::update(Str mode) {
    if (mode.empty()) {
        set_factor();
        impH = std::make_pair(h0, set_interaction());
        // modify_Impdata_for_half_fill(impH);
    }
    else if (mode == "eDMFT") {
        set_factor();
        if(mm) WRN(NAV(h0));

        /*
        h0 = MatReal(h0.nrows(), h0.ncols(), 0.);
        if(mm) WRN(NAV(h0));
        */

        impH = std::make_pair(h0, set_edmft_interaction());

        

        // modify_Impdata_for_half_fill(impH);
    }
    else if (mode == "behte") {
        for_Int(i, 0, ni) imp_lvl[i] = p.eimp[i][i] - p.mu;
        set_factor();
        impH = std::make_pair(h0, set_interaction());
        modify_Impdata_for_half_fill(impH);
    }
    else if (mode == "behte_alpha") {
        for_Int(i, 0, ni) imp_lvl[i] = p.eimp[i][i] - p.mu;
        set_factor();
        impH = std::make_pair(h0, set_3band_interaction_withalpha());
        // modify_Impdata_for_half_fill(impH);
        modify_Impdata_for_half_fill_hhd(impH);
    }
    // if(mm) WRN(NAV(h0));
}

void Impurity::write_H0info(const Bath &b, Int ndeg, Int iter_cnt) const {
    OFS ofs;
    if(iter_cnt < 0) ofs.open("h0.txt");
    if(iter_cnt > 0) ofs.open(iox + "zic" + prefill0(iter_cnt, 3) +".h0.txt");
    
    // Write band info
    if(ndeg > 0) for_Int(i, 0, ndeg) {
        ofs << "deg-band "<< i+1  <<  " nbath: "<< b.nb[i] <<  " nmin: " << b.info[i][0] << " err: " << b.info[i][1] 
            << " err_crv: " << b.info[i][2] << " err_hight_ev: " << b.info[i][3] 
            << " err_regE: " << b.info[i][4] <<" norm: " << b.info[i][5]<< "  " << endl;
    }
    else for_Int(i, 0, p.nband) {
        ofs << "band "<< i+1  << "nmin: " << b.info[i][0] << " err: " << b.info[i][1] 
            << " err_crv: " << b.info[i][2] << " err_hight_ev: " << b.info[i][3] 
            << " err_regE: " << b.info[i][4] <<" norm: " << b.info[i][5]<< "  " << endl;
    }

    // Calculate correct matrix positions
    Int pos = 0;
    for_Int(i, 0, p.nband) {
        ofs << "band "<< i+1 << endl;
        Int begin = pos;
        const Int size = 2 * (p.nI2B[2 * i] + 1);
        Int end = begin + size;
        ofs << iofmt() << h0.truncate(begin, begin, end, end) << endl;
        pos += size;  // Update position for next band
    }
}

//---------------------------------------------Private function---------------------------------------------

//! rely on imp model's frame
void Impurity::set_factor() {
    // set hyb part and bath part
    h0 = bth.find_hop();
    Int pos = 0;
    for_Int(s, 0, ni) {
        pos_imp[s] = pos;
        h0[pos][pos] = imp_lvl[s];
        pos += 1 + p.nI2B[s];
    }

    if(p.testH0){
        const Str h0_file = "norgH0.txt";
        IFS ifs(h0_file);
        if (!ifs) ERR("Cannot open " + h0_file)

        VEC<MatReal> band_h0;
        VEC<VecReal> band_rows;
        Int current_band = -1;

        // Convert the rows collected for one band into a matrix.  The input
        // format does not state the dimensions explicitly, so infer them from
        // the row count and the number of values in each row.
        auto finish_band = [&]() {
            if (current_band < 0) return;
            if (band_rows.empty()) {
                ERR("No H0 rows were found for band " + STR(current_band))
            }

            const Int nrows = static_cast<Int>(band_rows.size());
            const Int ncols = static_cast<Int>(band_rows[0].size());
            if (ncols <= 0) {
                ERR("H0 has no columns for band " + STR(current_band))
            }
            for_Int(r, 0, nrows) {
                if (static_cast<Int>(band_rows[r].size()) != ncols) {
                    ERR("Inconsistent H0 column count in band " + STR(current_band)
                        + ", row " + STR(r))
                }
            }
            if (nrows != ncols) {
                ERR("H0 for band " + STR(current_band) + " is not square: "
                    + STR(nrows) + " x " + STR(ncols))
            }

            MatReal block(nrows, ncols, 0.);
            for_Int(r, 0, nrows) for_Int(c, 0, ncols)
                block[r][c] = band_rows[r][c];
            band_h0.push_back(block);
            band_rows.clear();
        };

        Str line;
        while (std::getline(ifs, line)) {
            std::istringstream header(line);
            Str marker;
            Int band_number = -1;
            header >> marker;
            if (marker == "band") {
                if (!(header >> band_number) || band_number <= 0) {
                    ERR("Invalid band header in " + h0_file + ": " + line)
                }
                header >> std::ws;
                if (!header.eof()) {
                    ERR("Unexpected text in band header: " + line)
                }

                finish_band();
                if (band_number != static_cast<Int>(band_h0.size()) + 1) {
                    ERR("Bands in " + h0_file
                        + " must be numbered consecutively from 1; found band "
                        + STR(band_number))
                }
                current_band = band_number;
                continue;
            }

            const std::size_t row_pos = line.find("row[");
            if (row_pos == Str::npos) continue;
            if (current_band < 0) {
                ERR("Found an H0 row before the first band header in " + h0_file)
            }

            const std::size_t left = line.find('[', row_pos);
            const std::size_t right = line.find(']', left);
            const std::size_t equal = line.find('=', right);
            if (left == Str::npos || right == Str::npos || equal == Str::npos) {
                ERR("Invalid H0 row in " + h0_file + ": " + line)
            }

            Int row_number = -1;
            std::istringstream row_index(line.substr(left + 1, right - left - 1));
            if (!(row_index >> row_number)) {
                ERR("Invalid H0 row index in " + h0_file + ": " + line)
            }
            row_index >> std::ws;
            if (!row_index.eof()
                || row_number != static_cast<Int>(band_rows.size())) {
                ERR("H0 rows for band " + STR(current_band)
                    + " must be numbered consecutively from 0; found row "
                    + STR(row_number))
            }

            VEC<Real> values;
            std::istringstream row_values(line.substr(equal + 1));
            Real value = 0.;
            while (row_values >> value) values.push_back(value);
            if (!row_values.eof() || values.empty()) {
                ERR("Invalid H0 values in " + h0_file + ": " + line)
            }
            band_rows.push_back(VecReal(values));
        }
        if (!ifs.eof()) ERR("Error while reading " + h0_file)
        finish_band();

        if (band_h0.size() != p.nband)
            ERR("Expected one H0 matrix per spatial orbital in " + h0_file);
        MatReal h0_from_file(0, 0, 0.);
        for_Int(m, 0, p.nband) {
            const MatReal& sector = band_h0[m];
            if (sector.nrows() != 1 + p.nI2B[2 * m])
                ERR("H0 size does not match the bath for orbital " + STR(m + 1));
            h0_from_file.reset(direct_sum(h0_from_file, direct_sum(sector, sector)));
        }

        if (h0_from_file.nrows() != h0.nrows()
            || h0_from_file.ncols() != h0.ncols()) {
            ERR("H0 assembled from " + h0_file + " has size "
                + STR(h0_from_file.nrows()) + " x " + STR(h0_from_file.ncols())
                + ", but set_factor() expects " + STR(h0.nrows()) + " x "
                + STR(h0.ncols()))
        }
        h0.reset(h0_from_file);
    }

}

// four-fermion operator terms for C^+_i C^+_j C_k C_l; 
// C^+_i C^+_j C_k C_l h_inter from [i][l][j][k] to [alpha][eta][beta][gamma]
VecReal Impurity::set_interaction() {
    Int n = p.norbit;
    VecReal interaction(std::pow(n, 4), 0.);
    // Mat<MatReal> imp_interact(p.norbs, p.norbs, MatReal(p.norbs, p.norbs, 0.));
    MatReal imp_dd_interact(p.norbs, p.norbs,  0.);


    for_Int(b1, 0, p.nband) {// NO double counting term for impurity
        imp_dd_interact[2 * b1][2 * b1 + 1] = p.U;
        for_Int(b2, b1, p.nband) if (b1 != b2) { // same spin orientation
            imp_dd_interact[2 * b1][2 * b2] = p.Uprm - p.jz;
            imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = p.Uprm - p.jz;
        }
        for_Int(b2, 0, p.nband) if (b1 != b2) {
            imp_dd_interact[2 * b1][2 * b2 + 1] = p.Uprm;
        }
    }

/* 
    for_Int(b1, 0, p.nband) {// with double counting term for impurity, so we need to divide by 2
        imp_dd_interact[2 * b1][2 * b1 + 1] = p.U/2.0;
        imp_dd_interact[2 * b1 + 1][2 * b1] = p.U/2.0;
        for_Int(b2, 0, p.nband) if (b1 != b2) { // same spin orientation
            imp_dd_interact[2 * b1][2 * b2] = (p.Uprm - p.jz)/2.0;
            imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = (p.Uprm - p.jz)/2.0;
        }
        for_Int(b2, 0, p.nband) if (b1 != b2) {
            imp_dd_interact[2 * b1][2 * b2 + 1] = p.Uprm/2.0;
            imp_dd_interact[2 * b1 + 1][2 * b2] = p.Uprm/2.0;
        }
    }
*/
    
    if(mm) WRN(NAV(imp_dd_interact));
    for_Int(N_i, 0, p.norbs)for_Int(N_j, 0, p.norbs) {
        // if (imp_dd_interact[N_i][N_j] != 0)
            interaction[SUM_0toX(p.nO2sets, N_i) * std::pow(n, 3) + SUM_0toX(p.nO2sets, N_j) * std::pow(n, 2) + SUM_0toX(p.nO2sets, N_j) * std::pow(n, 1) + SUM_0toX(p.nO2sets, N_i)] = imp_dd_interact[N_i][N_j];

    }
    // if (mm) WRN(NAV5(p.U, p.Uprm ,p.nO2sets, interaction.size(), SUM(interaction[0][0])));
    return interaction;
}

void Impurity::modify_Impdata_for_half_fill(Impdata& impH_i){
    //! modify the non-interacting part of impurity Hamiltonian
	MatReal& h0 = impH_i.first;
	Int norb2set = p.nO2sets[0];
	for_Int(i, 0, p.norg_sets) {
        if      (p.nband == 1)h0[i * norb2set][i * norb2set] -= p.U * 0.5;
        else if (p.nband == 2)h0[i * norb2set][i * norb2set] -= p.U * 0.5 + p.Uprm * 0.5 + (p.Uprm - p.jz) * 0.5;
        else if (p.nband == 3)h0[i * norb2set][i * norb2set] -= p.U * 0.5 + 2 * (p.Uprm * 0.5) + 2 * (p.Uprm - p.jz) * 0.5;
        else if (p.nband  > 3)h0[i * norb2set][i * norb2set] -= p.U * 0.5 + (p.nband-1) * (p.Uprm * 0.5) + (p.nband-1) * (p.Uprm - p.jz) * 0.5;
        else ERR("impurity input is wrong");
	}
}

//--------------------------------------------------------special for the hhd function(arXiv:2209.14178v1)-----------------------------------------------------------------


// four-fermion operator terms for C^+_i C^+_j C_k C_l; 
// C^+_i C^+_j C_k C_l h_inter from [i][l][j][k] to [alpha][eta][beta][gamma]
VecReal Impurity::set_3band_interaction_withalpha() {
    Int n = p.norbit;
    VecReal interaction(std::pow(n, 4), 0.);
    if (p.nband != 3) ERR("this function is only for 3-band model");
    // Mat<MatReal> imp_interact(p.norbs, p.norbs, MatReal(p.norbs, p.norbs, 0.));
    MatReal imp_dd_interact(p.norbs, p.norbs,  0.);

    for_Int(b1, 0, p.nband) {// NO double counting term for impurity
        imp_dd_interact[2 * b1][2 * b1 + 1] = p.U;
        for_Int(b2, b1, p.nband) if (b1 != b2) { // same spin orientation
            imp_dd_interact[2 * b1][2 * b2] = p.Uprm - p.jz;
            imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = p.Uprm - p.jz;
        }
        for_Int(b2, 0, p.nband) if (b1 != b2) {
            imp_dd_interact[2 * b1][2 * b2 + 1] = p.Uprm;
        }
    // setting the interation between two wide bands.
        for_Int(b2, b1, p.nband) if (b1 == 0 && b2 == 1) { // same spin orientation
            imp_dd_interact[2 * b1][2 * b2] *= p.alpha;
            imp_dd_interact[2 * b1 + 1][2 * b2 + 1] *= p.alpha;
        }
        for_Int(b2, 0, p.nband) if (b1 == 0 && b2 == 1 || b2 == 0 && b1 == 1 ) {
            imp_dd_interact[2 * b1][2 * b2 + 1] *= p.alpha;
        }
    }



    if(mm) WRN(NAV(imp_dd_interact));
    for_Int(N_i, 0, p.norbs)for_Int(N_j, 0, p.norbs) {
        // if (imp_dd_interact[N_i][N_j] != 0)
            interaction[SUM_0toX(p.nO2sets, N_i) * std::pow(n, 3) + SUM_0toX(p.nO2sets, N_j) * std::pow(n, 2) + SUM_0toX(p.nO2sets, N_j) * std::pow(n, 1) + SUM_0toX(p.nO2sets, N_i)] = imp_dd_interact[N_i][N_j];

    }
    // if (mm) WRN(NAV5(p.U, p.Uprm ,p.nO2sets, interaction.size(), SUM(interaction[0][0])));
    return interaction;
}

//! special chage for the 3 band hhd case.
void Impurity::modify_Impdata_for_half_fill_hhd(Impdata& impH_i){
    //! modify the non-interacting part of impurity Hamiltonian
	MatReal& h0 = impH_i.first;
	Int norb2set = p.nO2sets[0];
	for_Int(i, 0, p.norg_sets) {
        h0[i * norb2set][i * norb2set] -= p.U * 0.5;
        // interorbital part
            // else if (p.nband  > 3)h0[i * norb2set][i * norb2set] -= p.U * 0.5 + (p.nband-1) * (p.Uprm * 0.5) + (p.nband-1) * (p.Uprm - p.jz) * 0.5;
        //! special chage for the 3 band hhd case. (arXiv:2209.14178v1) 
        Real iner_part_per_band = ((p.nband - 1) * (p.Uprm * 0.5) + (p.nband - 1) * (p.Uprm - p.jz) * 0.5) / (p.nband - 1);
        if (i/2 < 2) h0[i * norb2set][i * norb2set] -= (iner_part_per_band * p.alpha) + iner_part_per_band * (3 - 2); //! set band 0 same as band 1.
        if (i/2 >= 2) h0[i * norb2set][i * norb2set] -= iner_part_per_band * (3 - 1);
	}
}



//---------------------------------------------------------------------------------------------------------------------------------------



//--------------------------------------------------------special for the eDMFT -----------------------------------------------------------------

// four-fermion operator terms for C^+_i C^+_j C_k C_l; 
// C^+_i C^+_j C_k C_l h_inter from [i][l][j][k] to [alpha][eta][beta][gamma]
// See SM of Physical Review B 102.16 (2020): 161118, for equation S1-S9.
VecReal Impurity::set_edmft_interaction() {
    // #---------------- # Independent components are --------------
    //                  'z^2' 'x^2-y^2' 'xz' 'yz' 'xy' 
    const Idx n = p.norbit;
    const Int n_imp_orbs = p.norbs;
    const Int n_imp_band = p.nband;
    VecReal interaction(n * n * n * n, 0.);
    if (!(n_imp_band == 5 || n_imp_band == 3 || n_imp_band == 2)) ERR("we only support for the eg, t2g and fulld mode");
    // Mat<MatReal> imp_interact(p.norbs, p.norbs, MatReal(p.norbs, p.norbs, 0.));
    MatReal imp_dd_interact(n_imp_orbs, n_imp_orbs,  0.);
    Real Uc(0.0), Jz(0.0), Up(0.0);

    /*
    p.U = 0.;
    p.jz = 0.;
    */

    if(n_imp_band == 5){
        MatReal U_interact(n_imp_band, n_imp_band,  0.);
        MatReal J_interact(n_imp_band, n_imp_band,  0.);
        VecReal S_val(9,0.0);

        // See SM of Physical Review B 102.16 (2020): 161118, for equation S1-S9.
        S_val[0] = p.U + (8.0/7.0) * p.jz;      // S1
        S_val[1] = p.U - (328.0/819.0) * p.jz;  // S2~S6
        S_val[2] = p.U + (48.0/819.0) * p.jz;   // S3~S8
        S_val[3] = p.U - (516.0/819.0) * p.jz;  // S4~S7
        S_val[4] = p.U + (236.0/819.0) * p.jz;  // S5~S9
        S_val[5] = (632.0/819.0) * p.jz;        // S6
        S_val[6] = (726.0/819.0) * p.jz;        // S7
        S_val[7] = (444.0/819.0) * p.jz;        // S8
        S_val[8] = (350.0/819.0) * p.jz;        // S9

        for_Int(b1, 0, n_imp_band) {
            // S1
            U_interact[b1][b1] = p.U + (8/7.0) * p.jz;
            J_interact[b1][b1] = 0.0;
            for_Int(b2, 0, n_imp_band) if (b1 != b2) {
                U_interact[b1][b2] = S_val[1];//S2
                J_interact[b1][b2] = S_val[5];//S6
                if ((b1 == 0 && (b2 == 1 || b2 == 4)) || \
                    (b2 == 0 && (b1 == 1 || b1 == 4))) {
                    U_interact[b1][b2] = S_val[3];//S4
                    J_interact[b1][b2] = S_val[6];//S7
                }

                if ((b1 == 0 && (b2 == 2 || b2 == 3)) || \
                    (b2 == 0 && (b1 == 2 || b1 == 3))) {
                    U_interact[b1][b2] = S_val[2];//S3
                    J_interact[b1][b2] = S_val[7];//S8
                }

                if ((b1 == 1 && b2 == 4) || \
                    (b2 == 1 && b1 == 4)) {
                    U_interact[b1][b2] = S_val[4];//S5
                    J_interact[b1][b2] = S_val[8];//S9
                }
            }
        }

        if(mm) WRN(NAV2(U_interact, J_interact));

        
        for_Int(b1, 0, n_imp_band) {// NO double counting term for impurity
            imp_dd_interact[2 * b1][2 * b1 + 1] = U_interact[b1][b1];
            for_Int(b2, b1, n_imp_band) if (b1 != b2) { // same spin orientation
                imp_dd_interact[2 * b1][2 * b2] = U_interact[b1][b2] - J_interact[b1][b2];
                imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = U_interact[b1][b2] - J_interact[b1][b2];
            }
            for_Int(b2, 0, n_imp_band) if (b1 != b2) {
                imp_dd_interact[2 * b1][2 * b2 + 1] = U_interact[b1][b2];
            }
        }
        /*
        for_Int(b1, 0, p.nband) {// NO double counting term for impurity
            imp_dd_interact[2 * b1][2 * b1 + 1] = U_interact[b1][b1];
            for_Int(b2, b1, p.nband) if (b1 != b2) { // same spin orientation
                imp_dd_interact[2 * b1][2 * b2] = U_interact[b1][b2] - 3 * J_interact[b1][b2];
                imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = U_interact[b1][b2] - 3 * J_interact[b1][b2];
            }
            for_Int(b2, 0, p.nband) if (b1 != b2) {
                imp_dd_interact[2 * b1][2 * b2 + 1] = U_interact[b1][b2] - 2 * J_interact[b1][b2];
            }
        }
        */
    }
    else if (n_imp_band == 3) {
        Uc = p.U + (8.0 / 7.0) * p.jz;      // S1
        Up = p.U - (328.0 / 819.0) * p.jz;  // S2
        Jz = (632.0 / 819.0) * p.jz;        // S6

        for_Int(b1, 0, n_imp_band) { // NO double counting term for impurity
            imp_dd_interact[2 * b1][2 * b1 + 1] = Uc;
            for_Int(b2, b1, n_imp_band) if (b1 != b2) { // same spin orientation
                imp_dd_interact[2 * b1][2 * b2] = Up - Jz;
                imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = Up - Jz;
            }
            for_Int(b2, 0, n_imp_band) if (b1 != b2) {
                imp_dd_interact[2 * b1][2 * b2 + 1] = Up;
            }
        }
    }
    else if (n_imp_band == 2) {
        // See SM of Physical Review B 102.16 (2020): 161118, for equation S1-S9.
        Uc = p.U + (8.0 / 7.0) * p.jz;      // S1
        Up = p.U - (516.0 / 819.0) * p.jz;  // S4
        Jz = (726.0 / 819.0) * p.jz;        // S7

        for_Int(b1, 0, n_imp_band) { // NO double counting term for impurity
            imp_dd_interact[2 * b1][2 * b1 + 1] = Uc;
            for_Int(b2, b1, n_imp_band) if (b1 != b2) { // same spin orientation
                imp_dd_interact[2 * b1][2 * b2] = Up - Jz;
                imp_dd_interact[2 * b1 + 1][2 * b2 + 1] = Up - Jz;
            }
            for_Int(b2, 0, n_imp_band) if (b1 != b2) {
                imp_dd_interact[2 * b1][2 * b2 + 1] = Up;
            }
        }
    }
    else {
        ERR("Now we only support for the t2g and eg orbital.")
    }

    if(mm) WRN(NAV(imp_dd_interact));
    for_Int(s, 0, ni) for_Int(t, 0, ni) {
        const Idx i = pos_imp[s], j = pos_imp[t];
        interaction[((i * n + j) * n + j) * n + i] = imp_dd_interact[s][t];
    }
    return interaction;
}

//---------------------------------------------------------------------------------------------------------------------------------------
