#pragma once

// Portable serialization for one EDMFT search starting point. Particle counts
// and the matching rotation travel together; no solver/MPI types are required.
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace norg_restart {

struct Layout {
    std::vector<int> orbitals;
    std::vector<int> baths;
    int rotate_impurity;
};

struct State {
    int iteration = 0;
    std::vector<int> particles;
    std::vector<std::vector<double>> rotation; // row-major matrices by set
    bool has_rotation = false;
};

inline void require(bool ok, const std::string& message)
{
    if(!ok) throw std::runtime_error("NORG search restart: " + message);
}

inline void validate_layout(const Layout& layout)
{
    require(!layout.orbitals.empty() && layout.baths.size() == layout.orbitals.size(),
        "invalid layout");
    for(std::size_t i = 0; i < layout.orbitals.size(); ++i) {
        const int n = layout.orbitals[i];
        require(n > 0 && layout.baths[i] >= 0 && layout.baths[i] <= n,
            "invalid orbital/bath count");
    }
}

inline State identity(const Layout& layout)
{
    validate_layout(layout);
    State state;
    for(std::size_t i = 0; i < layout.orbitals.size(); ++i) {
        const int n = layout.orbitals[i];
        std::vector<double> matrix(static_cast<std::size_t>(n) * n, 0.);
        for(int j = 0; j < n; ++j) matrix[static_cast<std::size_t>(j) * n + j] = 1.;
        state.rotation.push_back(matrix);
    }
    return state;
}

inline void validate(const State& state, const Layout& layout)
{
    validate_layout(layout);
    require(state.iteration > 0, "invalid completed iteration count");
    require(state.particles.size() == layout.orbitals.size()
        && state.rotation.size() == layout.orbitals.size(), "incomplete state");
    for(std::size_t i = 0; i < layout.orbitals.size(); ++i) {
        const int n = layout.orbitals[i];
        require(state.particles[i] >= 0 && state.particles[i] <= n,
            "particle count outside the orbital range");
        require(state.rotation[i].size() == static_cast<std::size_t>(n) * n,
            "rotation dimensions do not match the layout");
        for(double value : state.rotation[i])
            require(std::isfinite(value), "non-finite rotation element");
    }
}

inline void expect(std::istream& input, const std::string& expected)
{
    std::string token;
    require(bool(input >> token) && token == expected, "expected " + expected);
}

inline State read(std::istream& input, const Layout& layout)
{
    State state = identity(layout);
    std::string token;
    require(bool(input >> token), "empty checkpoint");
    const bool legacy = token == "iter_count";
    if(!legacy) {
        require(token == "norg_search_restart", "unknown checkpoint format");
        int version;
        require(bool(input >> version) && version == 1, "unsupported checkpoint version");
        expect(input, "iter_count");
    }
    require(bool(input >> state.iteration), "missing iteration count");
    expect(input, "npartical");
    state.particles.resize(layout.orbitals.size());
    for(int& n : state.particles) require(bool(input >> n), "incomplete particle counts");
    if(!legacy) {
        expect(input, "layout");
        int count, magnetic, rotate_impurity;
        require(bool(input >> count >> magnetic >> rotate_impurity), "incomplete layout");
        require(count == static_cast<int>(layout.orbitals.size())
            && magnetic == 0 && rotate_impurity == layout.rotate_impurity,
            "checkpoint and current solver layouts differ");
        for(std::size_t i = 0; i < layout.orbitals.size(); ++i) {
            expect(input, "rotation_set");
            int index, n, baths;
            require(bool(input >> index >> n >> baths), "incomplete rotation metadata");
            require(index == static_cast<int>(i) && n == layout.orbitals[i]
                && baths == layout.baths[i], "orbital/bath dimensions have changed");
            for(double& value : state.rotation[i])
                require(bool(input >> value), "incomplete rotation matrix");
        }
        expect(input, "end_restart");
        state.has_rotation = true;
    }
    input >> std::ws;
    require(input.eof() && !input.bad(), "unexpected trailing checkpoint data");
    validate(state, layout);
    return state;
}

inline void write(std::ostream& output, const State& state, const Layout& layout)
{
    validate(state, layout);
    output << "norg_search_restart 1\niter_count " << state.iteration << "\nnpartical";
    for(int n : state.particles) output << ' ' << n;
    output << "\nlayout " << layout.orbitals.size() << ' ' << 0
        << ' ' << layout.rotate_impurity << '\n';
    output << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    for(std::size_t i = 0; i < layout.orbitals.size(); ++i) {
        const int n = layout.orbitals[i];
        output << "rotation_set " << i << ' ' << n << ' ' << layout.baths[i] << '\n';
        for(int r = 0; r < n; ++r) {
            for(int c = 0; c < n; ++c)
                output << state.rotation[i][static_cast<std::size_t>(r) * n + c] << ' ';
            output << '\n';
        }
    }
    output << "end_restart\n";
    require(bool(output), "cannot write checkpoint");
}

} // namespace norg_restart
