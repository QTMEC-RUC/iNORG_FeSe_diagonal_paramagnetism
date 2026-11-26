#pragma once
#include "mympi.h"
#include <set>
#include <sstream>

// Numerical policy, not machine topology. Unspecified inputs retain scalar Krylov.
struct GreenKrylovOptions {
    Int solver=-1, max_basis=0, check_level=1, fallback=0, orth_method=1;
    // Zero requests convergence; a positive value fixes the Krylov basis budget.
    Int lanczos_step=0, convergence_interval=16, convergence_stable_checks=2;
    Real check_tol=1.e-8, deflation_tol=64.*eps_Real, zero_tol=1.e-15;
    Real convergence_rtol=1.e-8, convergence_atol=1.e-10;
    void validate() const {
        if(solver != -1 || max_basis < 0 || check_level < 0 ||
           check_level > 3 || (fallback!=0 && fallback!=1) ||
           (orth_method!=0 && orth_method!=1) || lanczos_step<0 ||
           convergence_interval<1 || convergence_stable_checks<2)
            throw std::invalid_argument("Invalid Green Krylov integer option");
        if(lanczos_step>0 && max_basis>0 && max_basis<lanczos_step)
            throw std::invalid_argument("green_krylov_max_basis is smaller than fixed g_mode_lanczos_step");
        if(!std::isfinite(check_tol) || check_tol<=0. || check_tol>=1. ||
           !std::isfinite(deflation_tol) || deflation_tol<=0. || deflation_tol>=1. ||
           !std::isfinite(zero_tol) || zero_tol<0. ||
           !std::isfinite(convergence_rtol) || convergence_rtol<=0. || convergence_rtol>=1. ||
           !std::isfinite(convergence_atol) || convergence_atol<0.)
            throw std::invalid_argument("Invalid Green Krylov tolerance");
    }
    bool parse(const Str& key, std::istringstream& in, std::set<Str>& seen) {
        Int* integer=nullptr; Real* real_value=nullptr;
        if(key=="green_solver" || key=="g_mode") integer=&solver;
        else if(key=="g_mode_lanczos_step") integer=&lanczos_step;
        else if(key=="green_krylov_max_basis") integer=&max_basis;
        else if(key=="green_convergence_interval") integer=&convergence_interval;
        else if(key=="green_convergence_stable_checks") integer=&convergence_stable_checks;
        else if(key=="green_check_level") integer=&check_level;
        else if(key=="green_fallback") integer=&fallback;
        else if(key=="lanczos_orth_method") integer=&orth_method;
        else if(key=="green_convergence_rtol") real_value=&convergence_rtol;
        else if(key=="green_convergence_atol") real_value=&convergence_atol;
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
        mm.Bcast(deflation_tol); mm.Bcast(zero_tol);
        mm.Bcast(lanczos_step); mm.Bcast(convergence_interval);
        mm.Bcast(convergence_stable_checks); mm.Bcast(convergence_rtol);
        mm.Bcast(convergence_atol); validate();
    }
};
