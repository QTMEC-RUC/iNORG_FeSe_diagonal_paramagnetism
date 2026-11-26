#pragma once
#include "mympi.h"
#include <set>
#include <sstream>

// Backup numerical options only: no adaptive state or convergence parameters.
struct GreenKrylovFixedOptions {
    Int solver=-1, max_basis=0, check_level=1, fallback=0, orth_method=1;
    Real check_tol=1.e-8, deflation_tol=64.*eps_Real, zero_tol=1.e-15;
    void validate() const {
        if(solver != -1 || max_basis < 0 || check_level < 0 ||
           check_level > 3 || (fallback!=0 && fallback!=1) ||
           (orth_method!=0 && orth_method!=1))
            throw std::invalid_argument("Invalid Green Krylov integer option");
        if(!std::isfinite(check_tol) || check_tol<=0. || check_tol>=1. ||
           !std::isfinite(deflation_tol) || deflation_tol<=0. || deflation_tol>=1. ||
           !std::isfinite(zero_tol) || zero_tol<0.)
            throw std::invalid_argument("Invalid Green Krylov tolerance");
    }
    bool parse(const Str& key, std::istringstream& in, std::set<Str>& seen) {
        Int* integer=nullptr; Real* real_value=nullptr;
        if(key=="green_solver" || key=="g_mode") integer=&solver;
        else if(key=="green_krylov_max_basis") integer=&max_basis;
        else if(key=="green_check_level") integer=&check_level;
        else if(key=="green_fallback") integer=&fallback;
        else if(key=="lanczos_orth_method") integer=&orth_method;
        else if(key=="green_check_tol") real_value=&check_tol;
        else if(key=="green_deflation_tol") real_value=&deflation_tol;
        else if(key=="green_zero_tol") real_value=&zero_tol;
        else return false;
        if(!seen.insert(key).second) throw std::invalid_argument("Duplicate "+key);
        if(integer) { if(!(in>>*integer)) throw std::invalid_argument("Expected integer "+key); }
        else if(!(in>>*real_value)) throw std::invalid_argument("Expected real "+key);
        if (integer == &solver && solver != -1)
            throw std::invalid_argument(key + " only accepts scalar Green mode -1");
        in>>std::ws;
        if(!in.eof() && in.peek()!='#') throw std::invalid_argument("Unexpected text after "+key);
        validate(); return true;
    }
    // One decision on all ranks, even if rank-local inputs differ.
    void synchronize(const MyMpi& mm) {
        mm.Bcast(solver); mm.Bcast(max_basis); mm.Bcast(check_level);
        mm.Bcast(fallback); mm.Bcast(orth_method); mm.Bcast(check_tol);
        mm.Bcast(deflation_tol); mm.Bcast(zero_tol); validate();
    }
};
