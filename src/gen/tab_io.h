#pragma once

#include "tab.h"
#include <array>
#include <fstream>
#include <map>
#include <string>

// Each MPI rank reads/writes its own CSR shard. Legacy three-array .bdat files
// can still be read, using seeks to load only this rank's slice.
namespace tab_io {
static_assert(sizeof(int) == 4, "Tab files require 32-bit int");
using U64 = std::uint64_t;
struct Info {
    Tab::Column dim = 0;
    unsigned version = 0; // 0: three-array legacy, 1: 32-bit columns, 2: 64-bit columns
    int ranks = 0;
    U64 entries = 0;
    std::vector<U64> sizes;
};

inline U64 number(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid Tab metadata number");
    std::size_t used = 0;
    const U64 value = std::stoull(text, &used);
    if (used != text.size()) throw std::runtime_error("Invalid Tab metadata number");
    return value;
}
inline Info read_info(const std::string& name) {
    std::ifstream in(name + ".inf");
    if (!in) throw std::runtime_error("Cannot open Tab metadata: " + name);
    std::map<std::string, std::string> fields;
    std::string key, value;
    while (in >> key) {
        if (!(in >> value) || !fields.emplace(key, value).second)
            throw std::runtime_error("Incomplete/duplicate Tab metadata");
    }
    Info info;
    const U64 dim = number(fields.at("dim"));
    if (dim > static_cast<U64>(std::numeric_limits<Tab::Column>::max()))
        throw std::runtime_error("Tab dimension exceeds supported index range");
    info.dim = static_cast<Tab::Column>(dim);
    if (fields.count("format")) {
        const auto& format = fields.at("format");
        if (format == "tab_csr_v1") info.version = 1;
        else if (format == "tab_csr_v2") info.version = 2;
        else throw std::runtime_error("Unsupported Tab format");
        const U64 ranks = number(fields.at("ranks"));
        if (!ranks || ranks > static_cast<U64>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid Tab rank count");
        info.ranks = static_cast<int>(ranks);
    } else {
        info.entries = number(fields.at("size"));
        U64 total = 0;
        for (std::size_t rank = 0; fields.count("size_np" + std::to_string(rank)); ++rank) {
            const U64 size = number(fields.at("size_np" + std::to_string(rank)));
            if (size > info.entries - total) throw std::runtime_error("Invalid legacy Tab sizes");
            info.sizes.push_back(size);
            total += size;
        }
        if (total != info.entries || info.sizes.empty()
            || fields.size() != info.sizes.size() + 2)
            throw std::runtime_error("Missing/noncontiguous legacy Tab ranks");
        if (info.sizes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Too many legacy Tab ranks");
        info.ranks = static_cast<int>(info.sizes.size());
    }
    return info;
}
inline void write_info(const std::string& name, Tab::Column dim, int ranks) {
    std::ofstream out(name + ".inf", std::ios::trunc);
    out << "format tab_csr_v2\ndim " << dim << "\nranks " << ranks << '\n';
    out.close();
    if (!out) throw std::runtime_error("Cannot write Tab metadata: " + name);
}
inline std::string shard(const std::string& name, int rank) {
    return name + ".rank" + std::to_string(rank) + ".tab";
}
inline U64 file_size(std::ifstream& in) {
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) throw std::runtime_error("Cannot determine Tab file size");
    in.seekg(0);
    return static_cast<U64>(size);
}
template<class T> inline void read(std::ifstream& in, T* data, std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::runtime_error("Tab read is too large");
    if (count) in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("Truncated Tab file");
}
template<class T> inline void write(std::ofstream& out, const T* data, std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::runtime_error("Tab write is too large");
    if (count) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) throw std::runtime_error("Cannot write Tab file");
}
inline void save_shard(const std::string& name, int rank, const Tab& tab) {
    std::ofstream out(shard(name, rank), std::ios::binary | std::ios::trunc);
    // Magic, version, byte-order marker, dimension, first global row, rows, nnz.
    const U64 header[] = {0x5441424353523031ULL, 2, 0x0102030405060708ULL,
        static_cast<U64>(tab.global_dim()), static_cast<U64>(tab.row_begin()), tab.nrows(), tab.nnz()};
    write(out, header, 7);
    write(out, tab.row_ptr().data(), tab.row_ptr().size());
    write(out, tab.cols().data(), tab.nnz());
    write(out, tab.ops().data(), tab.nnz());
    out.close();
    if (!out) throw std::runtime_error("Cannot close Tab shard");
}
// Widen a legacy column slice without holding a second complete column array.
inline void read_columns32(std::ifstream& in, std::vector<Tab::Column>& columns) {
    std::array<std::int32_t, 4096> buffer;
    for (std::size_t first = 0; first < columns.size();) {
        const auto count = std::min(buffer.size(), columns.size() - first);
        read(in, buffer.data(), count);
        std::copy_n(buffer.begin(), count, columns.begin() + first);
        first += count;
    }
}
inline Tab load(const std::string& name, const Info& info, int ranks, int rank,
    Tab::Column dim, Tab::Column begin, Tab::Column rows) {
    if (info.dim != dim || info.ranks != ranks || rank < 0 || rank >= ranks
        || begin < 0 || begin > dim || rows < 0 || rows > dim - begin)
        throw std::runtime_error("Tab dimension/MPI partition mismatch");
    std::vector<Tab::Offset> ptr;
    std::vector<Tab::Column> col;
    std::vector<int> op;
    if (static_cast<U64>(rows) >= ptr.max_size())
        throw std::runtime_error("Tab row pointer array exceeds addressable storage");
    const auto n_ptr = static_cast<std::size_t>(rows) + 1;
    if (info.version != 0) {
        std::ifstream in(shard(name, rank), std::ios::binary);
        const U64 bytes = file_size(in);
        U64 h[7]; read(in, h, 7);
        if (h[0] != 0x5441424353523031ULL || h[1] != info.version ||
            (h[1] != 1 && h[1] != 2) || h[2] != 0x0102030405060708ULL)
            throw std::runtime_error("Unsupported or mismatched Tab shard version/byte order");
        if (h[3] != static_cast<U64>(dim) || h[4] != static_cast<U64>(begin) || h[5] != static_cast<U64>(rows))
            throw std::runtime_error("Tab shard row partition mismatch");
        if (bytes < sizeof(h) || n_ptr > (bytes - sizeof(h)) / sizeof(U64))
            throw std::runtime_error("Invalid Tab shard row pointer length");
        const U64 entry_bytes = bytes - sizeof(h) - n_ptr * sizeof(U64);
        const U64 width = sizeof(std::int32_t) + (info.version == 1 ? sizeof(std::int32_t) : sizeof(Tab::Column));
        if (entry_bytes % width || h[6] != entry_bytes / width || h[6] > std::min(col.max_size(), op.max_size()))
            throw std::runtime_error("Invalid Tab shard entry count");
        ptr.resize(n_ptr); col.resize(static_cast<std::size_t>(h[6])); op.resize(col.size());
        read(in, ptr.data(), ptr.size());
        if (info.version == 1) read_columns32(in, col);
        else read(in, col.data(), col.size());
        read(in, op.data(), op.size());
    } else {
        std::ifstream in(name + ".bdat", std::ios::binary);
        const U64 bytes = file_size(in);
        if (bytes % 12 || bytes / 12 != info.entries)
            throw std::runtime_error("Invalid legacy Tab file length");
        U64 offset = 0;
        for (int r = 0; r < rank; ++r) offset += info.sizes[r];
        const U64 count = info.sizes[rank];
        if (count > std::min(col.max_size(), op.max_size()))
            throw std::runtime_error("Legacy Tab slice exceeds addressable storage");
        ptr.resize(n_ptr); col.resize(static_cast<std::size_t>(count)); op.resize(col.size());
        // Stream legacy row labels directly into row_ptr; no temporary row array.
        in.seekg(static_cast<std::streamoff>(offset * sizeof(int)));
        std::size_t opened = 0;
        for (U64 k = 0; k < count; ++k) {
            int row; read(in, &row, 1);
            if (row < 0 || row >= rows
                || (opened && static_cast<std::size_t>(row) < opened - 1))
                throw std::runtime_error("Invalid legacy Tab row order");
            while (opened <= static_cast<std::size_t>(row)) ptr[opened++] = k;
        }
        while (opened < ptr.size()) ptr[opened++] = count;
        in.seekg(static_cast<std::streamoff>((info.entries + offset) * sizeof(int)));
        read_columns32(in, col);
        in.seekg(static_cast<std::streamoff>((2 * info.entries + offset) * sizeof(int)));
        read(in, op.data(), op.size());
    }
    return Tab::from_arrays(dim, begin, std::move(ptr), std::move(col), std::move(op));
}
} // namespace tab_io
