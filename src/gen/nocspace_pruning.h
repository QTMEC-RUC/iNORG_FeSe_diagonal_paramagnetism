#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <vector>

// Bath-only generation; column 0 remains the fastest mixed-radix digit.
namespace nocspace_pruning {
using Count = long long;
using Label = std::size_t;
struct Choice {
    std::vector<int> particles;
    Count excitations = 0;
};
struct Column {
    std::size_t layer = 0; // inner -> outer; unpaired partners share a layer
    std::vector<Choice> choices;
    std::map<Count, std::vector<Label>> allowed;
    Count largest = 0;
    const std::vector<Label>& positions(Count budget)
    {
        budget = std::max(Count(-1), std::min(budget, largest));
        auto found = allowed.find(budget);
        if (found != allowed.end()) return found->second;
        auto& result = allowed[budget];
        for (Label i = 0; i < choices.size(); ++i)
            if (choices[i].excitations <= budget) result.push_back(i);
        return result;
    }
};

// Nonnegative partial excitation counts give safe lower bounds. These become
// the exact cross-column rules at a leaf; final suit_NOOC is still retained.
inline Count budget_for(const std::string& mode, std::size_t layer,
    const std::vector<Count>& used, const std::vector<Count>& limits, Count phess_total)
{
    const Count unlimited = std::numeric_limits<Count>::max();
    if (mode == "phss_v2") {
        Count budget = unlimited, suffix = 0;
        for (std::size_t j = used.size(); j > 0;) {
            --j;
            suffix += used[j];
            if (j <= layer) budget = std::min(budget, limits[j] - suffix);
        }
        return budget;
    }
    if (mode == "cnooc" || (mode == "cpnooc" && layer != 0))
        return limits[layer] - used[layer];
    if (mode == "phess") {
        Count total = 0;
        std::size_t active_layers = 0, active_layer = layer;
        bool nonnegative_limits = true;
        for (std::size_t j = 0; j < used.size(); ++j) {
            total += used[j];
            if (used[j] != 0) { ++active_layers; active_layer = j; }
            if (limits[j] < 0) nonnegative_limits = false;
        }
        // Small-total OR single-layer branch: pair limits must NOT reject
        // configurations accepted by check_if_PHSs's small-total shortcut.
        const Count small_total = phess_total - total;
        Count single_layer = -1;
        if (nonnegative_limits && active_layers <= 1) {
            if (!active_layers || active_layer == layer)
                single_layer = limits[layer] - used[layer];
            else if (used[active_layer] <= limits[active_layer])
                single_layer = 0; // Another layer is occupied: this column may still add zero.
        }
        return std::max(small_total, single_layer);
    }
    return unlimited; // nooc has no cross-column restriction
}

template<class Visit, class Reject>
void enumerate(const std::vector<Column>& columns, const std::string& mode,
    const std::vector<Count>& limits, const std::vector<int>& target,
    const std::vector<int>& star_capacity, Count phess_total, Visit visit, Reject reject,
    Label label_begin = 0, Label label_end = std::numeric_limits<Label>::max())
{
    if (label_begin >= label_end) return;
    const std::size_t groups = target.size(), n = columns.size();
    // Only memoized positions are private; immutable occupation choices are shared.
    std::vector<std::map<Count, std::vector<Label>>> allowed(n);
    std::vector<Count> largest(n, 0);
    std::vector<Label> stride(n, 1);
    std::vector<std::vector<Count>> remaining_min(n + 1, std::vector<Count>(groups, 0));
    auto remaining_max = remaining_min;
    for (std::size_t r = 0; r < groups; ++r) remaining_max[0][r] = star_capacity[r];
    for (std::size_t k = 0; k < n; ++k) {
        if (columns[k].choices.empty()) return;
        // The caller has checked the full product for Label overflow.
        if (k) stride[k] = stride[k - 1] * columns[k - 1].choices.size();

        for (const auto& choice : columns[k].choices)
            largest[k] = std::max(largest[k], choice.excitations);
        for (std::size_t r = 0; r < groups; ++r) {
            int lo = std::numeric_limits<int>::max(), hi = 0;
            for (const auto& choice : columns[k].choices) {
                lo = std::min(lo, choice.particles[r]);
                hi = std::max(hi, choice.particles[r]);
            }
            remaining_min[k + 1][r] = remaining_min[k][r] + lo;
            remaining_max[k + 1][r] = remaining_max[k][r] + hi;
        }
    }
    const auto positions_for = [&](std::size_t k, Count budget) -> const std::vector<Label>& {
        budget = std::max(Count(-1), std::min(budget, largest[k]));
        auto found = allowed[k].find(budget);
        if (found != allowed[k].end()) return found->second;
        auto& result = allowed[k][budget];
        for (Label i = 0; i < columns[k].choices.size(); ++i)
            if (columns[k].choices[i].excitations <= budget) result.push_back(i);
        return result;
    };
    const auto reject_range = [&](Label begin, Label count) {
        const Label lo = std::max(begin, label_begin), hi = std::min(begin + count, label_end);
        if (lo < hi) reject(hi - lo);
    };
    std::vector<Count> particles(groups, 0), used(limits.size(), 0);
    const auto walk = [&](const auto& self, std::size_t digits, Label label) -> void {
        if (!digits) { if (label >= label_begin && label < label_end) visit(label); return; }
        const std::size_t k = digits - 1;
        const auto& column = columns[k];
        if (label >= label_end || label + stride[k] * column.choices.size() <= label_begin) return;
        const auto& positions = positions_for(k, budget_for(mode, column.layer, used, limits, phess_total));
        Label previous = 0;
        for (Label position : positions) {
            if (position > previous) reject_range(label + previous * stride[k], (position - previous) * stride[k]);
            previous = position + 1;
            const Label child = label + position * stride[k];
            if (child >= label_end || child + stride[k] <= label_begin) continue;
            const auto& choice = column.choices[position];
            bool possible = true;
            for (std::size_t r = 0; r < groups; ++r) {
                particles[r] += choice.particles[r];
                if (particles[r] + remaining_min[k][r] > target[r]
                    || particles[r] + remaining_max[k][r] < target[r]) possible = false;
            }
            used[column.layer] += choice.excitations;
            if (possible) self(self, k, label + position * stride[k]);
            else reject_range(child, stride[k]);
            used[column.layer] -= choice.excitations;
            for (std::size_t r = 0; r < groups; ++r) particles[r] -= choice.particles[r];
        }
        if (previous < column.choices.size())
            reject_range(label + previous * stride[k], (column.choices.size() - previous) * stride[k]);
    };
    walk(walk, n, 0);
}
} // namespace nocspace_pruning
