#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <algorithm>
#include "basis_index.h"
#include "parallel_stage2.h"
#include "tab_timing.h"

class TabBuilder;

// Immutable, shared CSR operator table. Values are signed operator IDs, not H.
// Copying a Tab/Operator shares these arrays; only TabBuilder may mutate them.
class Tab {
public:
    using Column = BasisIndex;
    using Offset = std::uint64_t;
private:
    struct Storage {
        Column global_dim = 0, row_begin = 0;
        std::vector<Offset> row_ptr{0};
        std::vector<Column> col;
        std::vector<int> op;
    };
    std::shared_ptr<const Storage> data_;
    explicit Tab(Storage&& data) : data_(std::make_shared<Storage>(std::move(data))) {}
    const Storage& data() const {
        if (data_) return *data_;
        static const Storage empty{};
        return empty;
    }
    friend class TabBuilder;
public:
    Tab() = default;
    Tab(const Tab&) noexcept = default;
    Tab& operator=(const Tab&) noexcept = default;
    Tab(Tab&&) noexcept = default;
    Tab& operator=(Tab&&) noexcept = default;

    Column global_dim() const { return data().global_dim; }
    Column row_begin() const { return data().row_begin; }
    std::size_t nrows() const { return data().row_ptr.size() - 1; }
    std::size_t nnz() const { return data().col.size(); }
    const std::vector<Offset>& row_ptr() const { return data().row_ptr; }
    const std::vector<Column>& cols() const { return data().col; }
    const std::vector<int>& ops() const { return data().op; }
    void reset() noexcept { data_.reset(); }

    // Join contiguous rank-local shards once; preserve every row and entry order.
    static Tab concatenate(std::vector<Tab>& parts, tab_timing::Merge* timing = nullptr) {
        if (parts.empty()) return Tab();
        if (parts.size() == 1) return std::move(parts.front());
        const Column dim = parts.front().global_dim(), begin = parts.front().row_begin();
        Storage result;
        const auto row_limit = result.row_ptr.max_size() - 1;
        const auto entry_limit = std::min(result.col.max_size(), result.op.max_size());
        std::vector<std::size_t> rows(parts.size() + 1, 0), offsets(parts.size() + 1, 0);
        for (std::size_t k = 0; k < parts.size(); ++k) {
            parts[k].require_layout(dim, begin + static_cast<Column>(rows[k]), parts[k].nrows());
            if (parts[k].nrows() > row_limit - rows[k] || parts[k].nnz() > entry_limit - offsets[k])
                throw std::overflow_error("Combined Tab exceeds addressable row or entry storage");
            rows[k + 1] = rows[k] + parts[k].nrows();
            offsets[k + 1] = offsets[k] + parts[k].nnz();
        }
        result.global_dim = dim; result.row_begin = begin;
        {
            tab_timing::Span phase(timing ? &timing->allocation_s : nullptr);
            result.row_ptr.resize(rows.back() + 1);
            result.col.resize(offsets.back()); result.op.resize(offsets.back());
        }
        tab_timing::Span copy_phase(timing ? &timing->copy_release_s : nullptr);
        stage2_parallel::for_each(parts.size(), [&](std::size_t k) {
            const auto& part = parts[k];
            for (std::size_t r = 0; r < part.nrows(); ++r)
                result.row_ptr[rows[k] + r] = offsets[k] + part.row_ptr()[r];
            std::copy(part.cols().begin(), part.cols().end(), result.col.begin() + offsets[k]);
            std::copy(part.ops().begin(), part.ops().end(), result.op.begin() + offsets[k]);
            parts[k].reset();
        });
        result.row_ptr.back() = offsets.back();
        return Tab(std::move(result));
    }


    void require_layout(Column dim, Column begin, std::size_t rows) const {
        if (global_dim() != dim || row_begin() != begin || nrows() != rows)
            throw std::invalid_argument("Tab basis dimension/local row partition mismatch");
    }

    // For readers: adopt arrays without a second full-sized allocation/copy.
    static Tab from_arrays(Column dim, Column begin,
        std::vector<Offset>&& ptr, std::vector<Column>&& col, std::vector<int>&& op) {
        if (ptr.empty() || begin < 0 || begin > dim || ptr.size() - 1 > static_cast<Offset>(dim - begin)
            || ptr.front() != 0 || ptr.back() != col.size() || col.size() != op.size())
            throw std::invalid_argument("Invalid CSR Tab dimensions/array lengths");
        for (std::size_t r = 1; r < ptr.size(); ++r)
            if (ptr[r] < ptr[r - 1]) throw std::invalid_argument("Nonmonotone Tab row pointers");
        for (std::size_t k = 0; k < col.size(); ++k)
            if (col[k] < 0 || col[k] >= dim
                || op[k] == 0 || op[k] == std::numeric_limits<int>::min())
                throw std::invalid_argument("Invalid Tab column/operator ID");
        Storage result;
        result.global_dim = dim;
        result.row_begin = begin;
        result.row_ptr = std::move(ptr);
        result.col = std::move(col);
        result.op = std::move(op);
        return Tab(std::move(result));
    }
};

// Append rows in nondecreasing order. Empty leading/interior/trailing rows are
// represented explicitly, including a rank that owns no rows at all.
class TabBuilder {
    Tab::Storage data_;
    std::size_t opened_rows_ = 0;
    bool finished_ = false;
public:
    TabBuilder(Tab::Column dim, Tab::Column begin, Tab::Column rows) {
        if (begin < 0 || begin > dim || rows < 0 || rows > dim - begin
            || static_cast<Tab::Offset>(rows) >= data_.row_ptr.max_size())
            throw std::invalid_argument("Invalid TabBuilder row partition");
        data_.global_dim = dim;
        data_.row_begin = begin;
        data_.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    }
    TabBuilder(const TabBuilder&) = delete;
    TabBuilder& operator=(const TabBuilder&) = delete;

    // Reserve a whole group run before appending. Geometric growth avoids
    // quadratic copying when many consecutive groups have few rows.
    void reserve_additional(std::size_t entries) {
        if (finished_) throw std::logic_error("TabBuilder already finished");
        const std::size_t limit = std::min(data_.col.max_size(), data_.op.max_size());
        if (entries > limit - data_.col.size())
            throw std::overflow_error("Tab reservation overflow");
        const std::size_t required = data_.col.size() + entries;
        const std::size_t capacity = std::min(data_.col.capacity(), data_.op.capacity());
        if (required <= capacity) return;
        const std::size_t grown = capacity > limit - capacity ? limit : capacity + capacity;
        const std::size_t next = std::max(required, grown);
        data_.col.reserve(next);
        data_.op.reserve(next);
    }

    void append(std::size_t row, Tab::Column col, int op) {
        if (finished_ || row >= data_.row_ptr.size() - 1
            || (opened_rows_ && row < opened_rows_ - 1)
            || col < 0 || col >= data_.global_dim
            || op == 0 || op == std::numeric_limits<int>::min())
            throw std::invalid_argument("Invalid or out-of-order Tab append");
        while (opened_rows_ <= row) data_.row_ptr[opened_rows_++] = data_.col.size();
        data_.col.push_back(col);
        try { data_.op.push_back(op); }
        catch (...) { data_.col.pop_back(); throw; }
    }
    Tab finish() {
        if (finished_) throw std::logic_error("TabBuilder already finished");
        while (opened_rows_ < data_.row_ptr.size())
            data_.row_ptr[opened_rows_++] = data_.col.size();
        finished_ = true;
        return Tab(std::move(data_));
    }
};
