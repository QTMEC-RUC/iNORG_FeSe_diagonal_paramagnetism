#pragma once

#include "mat.h"
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <type_traits>

// Retained storage accounting, not allocator instrumentation or process RSS.
// Callers own their matrix/vector buffers. Row views are not data copies.
namespace storage_monitor {

struct Entry {
    const char* name;
    double count = 0., capacity = 0., element_bytes = 0.;
    double scalar_count = 0., scalar_bytes = 0.;
    double object_bytes = 0., payload_bytes = 0., dynamic_bytes = 0., modeled_bytes = 0.;
    std::string detail;
    explicit Entry(const char* label): name(label) {}
    double known_bytes() const { return object_bytes + dynamic_bytes; }
    double estimated_bytes() const { return known_bytes() + modeled_bytes; }
};

template<class T>
Entry matrix(const char* name, const Mat<T>& value)
{
    Entry e(name);
    e.count = e.capacity = e.scalar_count = double(value.size());
    e.element_bytes = e.scalar_bytes = sizeof(T);
    e.object_bytes = sizeof(value);
    e.payload_bytes = e.scalar_count * sizeof(T);
    const double row_bytes = double(value.nrows()) * sizeof(Vec<T>);
    e.dynamic_bytes = e.payload_bytes + row_bytes;
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << " rows=" << value.nrows() << " cols=" << value.ncols()
        << " row_views_bytes=" << row_bytes;
    e.detail = out.str();
    return e;
}

template<class T>
Entry vector(const char* name, const Vec<T>& value)
{
    Entry e(name);
    e.count = e.capacity = e.scalar_count = double(value.size());
    e.element_bytes = e.scalar_bytes = sizeof(T);
    e.object_bytes = sizeof(value);
    e.payload_bytes = e.dynamic_bytes = double(value.size()) * sizeof(T);
    return e;
}

template<class T>
Entry std_vector(const char* name, const std::vector<T>& value)
{
    Entry e(name);
    e.count = e.scalar_count = double(value.size());
    e.capacity = double(value.capacity());
    e.element_bytes = e.scalar_bytes = sizeof(T);
    e.object_bytes = sizeof(value);
    e.payload_bytes = e.count * sizeof(T);
    e.dynamic_bytes = e.capacity * sizeof(T);
    return e;
}

inline Entry matrices(const char* name, const std::vector<MatInt>& value)
{
    Entry e = std_vector(name, value);
    e.scalar_count = e.payload_bytes = 0.;
    e.scalar_bytes = sizeof(Int);
    double row_bytes = 0.;
    for(const auto& m : value) {
        e.scalar_count += double(m.size());
        row_bytes += double(m.nrows()) * sizeof(VecInt);
    }
    e.payload_bytes = e.scalar_count * sizeof(Int);
    e.dynamic_bytes += e.payload_bytes + row_bytes;
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << " first_rows=" << (value.empty() ? 0 : value.front().nrows())
        << " first_cols=" << (value.empty() ? 0 : value.front().ncols())
        << " row_views_bytes=" << row_bytes;
    e.detail = out.str();
    return e;
}

template<class T>
Entry vectors(const char* name, const std::vector<Vec<T>>& value)
{
    Entry e = std_vector(name, value);
    e.scalar_count = 0.;
    e.scalar_bytes = sizeof(T);
    for(const auto& v : value) e.scalar_count += double(v.size());
    e.payload_bytes = e.scalar_count * sizeof(T);
    e.dynamic_bytes += e.payload_bytes;
    return e;
}

template<class IndexMap>
Entry index_map(const char* name, const IndexMap& value)
{
    using Item = typename IndexMap::value_type;
    Entry e(name);
    e.count = e.capacity = double(value.size());
    e.element_bytes = sizeof(Item);
    e.object_bytes = sizeof(value);
    double keys = 0., capacities = 0., external_bytes = 0.;
    std::size_t min_len = value.empty() ? 0 : value.begin()->first.size(), max_len = 0;
    for(const auto& item : value) {
        const std::string& key = item.first;
        keys += double(key.size());
        capacities += double(key.capacity()) + 1.;
        min_len = std::min(min_len, key.size());
        max_len = std::max(max_len, key.size());
        // Inline (SSO) storage is already included in sizeof(Item).
        const std::uintptr_t object = reinterpret_cast<std::uintptr_t>(&key);
        const std::uintptr_t data = reinterpret_cast<std::uintptr_t>(key.data());
        if(!(data >= object && data - object < sizeof(key)))
            external_bytes += double(key.capacity()) + 1.;
    }
    e.payload_bytes = keys + e.count * sizeof(Idx);
    e.dynamic_bytes = e.count * sizeof(Item) + external_bytes;
    // Explicit model, NOT sizeof(the implementation's private tree node).
    const std::size_t alignment = std::max(alignof(Item), alignof(void*));
    constexpr bool tree = std::is_same<IndexMap,std::map<std::string,Idx>>::value;
    const std::size_t links = tree ? 3*sizeof(void*)+sizeof(Int)
                                   : sizeof(void*)+sizeof(std::size_t);
    const std::size_t node_extra = (links + alignment - 1) / alignment * alignment;
    e.modeled_bytes = e.count * node_extra;
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << " key_len_min=" << min_len << " key_len_max=" << max_len
        << " key_bytes=" << keys << " key_capacity_plus_nul_bytes=" << capacities
        << " external_key_buffer_bytes=" << external_bytes
        << " string_object_bytes=" << sizeof(std::string) << " index_bytes=" << sizeof(Idx)
        << " node_extra_est_per_entry_bytes=" << node_extra;
    if constexpr (!tree) {
        e.dynamic_bytes += double(value.shard_capacity()) * sizeof(typename IndexMap::Map);
        // Model bucket pointers and one cached hash per node, not allocator/RSS.
        e.modeled_bytes += double(value.bucket_count()) * sizeof(void*);
        out << " backend=sharded_hash shards=" << value.shard_count()
            << " buckets=" << value.bucket_count()
            << " bucket_and_node_overhead=modeled";
    }
    e.detail = out.str();
    return e;
}

inline Entry total(const std::vector<Entry>& entries, std::size_t object_bytes)
{
    Entry e("TOTAL");
    e.count = e.capacity = 1.;
    e.element_bytes = e.object_bytes = double(object_bytes);
    // sizeof(NocSpace) includes every inline member, padding and reference once.
    for(const auto& member : entries) {
        e.payload_bytes += member.payload_bytes;
        e.dynamic_bytes += member.dynamic_bytes;
        e.modeled_bytes += member.modeled_bytes;
    }
    return e;
}

// One bounded gather per constructor, no copies of the subspace containers.
template<class Mpi>
void report(const Mpi& mm, const char* source, std::size_t dimension,
    const std::vector<Entry>& entries)
{
    const Idx stride = entries.size() * 4;
    VecReal local(stride);
    for(Idx i = 0; i < entries.size(); ++i) {
        local[4*i] = entries[i].count;
        local[4*i+1] = entries[i].capacity;
        local[4*i+2] = entries[i].known_bytes();
        local[4*i+3] = entries[i].estimated_bytes();
    }
    const VecReal gathered = mm.Gather(local);
    if(!mm) return;
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << "NOCSPACE_STORAGE: stage=CTOR_END source=" << source
        << " detail_rank=" << mm.id() << " ranks=" << mm.np() << " dim=" << dimension
        << " sizeof_Int=" << sizeof(Int) << " sizeof_Idx=" << sizeof(Idx)
        << " sizeof_Real=" << sizeof(Real) << " sizeof_MatInt=" << sizeof(MatInt)
        << " sizeof_VecInt=" << sizeof(VecInt)
        << " local_fields=count,capacity,bytes; sum/max_fields=all_ranks"
        << " map_nodes=modeled allocator_overhead_and_retained_free_blocks=excluded"
        << " array_cookies=excluded referenced_mm_and_p=excluded not_process_RSS=1\n";
    for(Idx i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        double min_count = gathered[4*i], max_count = min_count;
        double min_capacity = gathered[4*i+1], max_capacity = min_capacity;
        double known_sum = 0., est_sum = 0., known_max = -1., est_max = -1.;
        int max_rank = -1;
        for(int rank = 0; rank < mm.np(); ++rank) {
            const Idx pos = Idx(rank) * stride + 4*i;
            min_count = std::min(min_count, gathered[pos]);
            max_count = std::max(max_count, gathered[pos]);
            min_capacity = std::min(min_capacity, gathered[pos+1]);
            max_capacity = std::max(max_capacity, gathered[pos+1]);
            known_sum += gathered[pos+2];
            est_sum += gathered[pos+3];
            known_max = std::max(known_max, gathered[pos+2]);
            if(gathered[pos+3] > est_max) { est_max = gathered[pos+3]; max_rank = rank; }
        }
        out << std::setprecision(0) << "NOCSPACE_STORAGE: member=" << e.name
            << " count=" << e.count << " capacity=" << e.capacity
            << " count_min=" << min_count << " count_max=" << max_count
            << " capacity_min=" << min_capacity << " capacity_max=" << max_capacity
            << " element_object_bytes=" << e.element_bytes
            << " scalar_count=" << e.scalar_count << " scalar_bytes=" << e.scalar_bytes
            << " container_object_bytes=" << e.object_bytes
            << " payload_bytes=" << e.payload_bytes << " known_bytes=" << e.known_bytes()
            << " modeled_extra_bytes=" << e.modeled_bytes
            << std::setprecision(2) << " avg_entry_dynamic_est_bytes="
            << (e.count ? (e.dynamic_bytes + e.modeled_bytes) / e.count : 0.)
            << std::setprecision(6) << " local_est_MiB=" << e.estimated_bytes() / 1048576.
            << " known_sum_GiB=" << known_sum / 1073741824.
            << " known_max_GiB=" << known_max / 1073741824.
            << " est_sum_GiB=" << est_sum / 1073741824.
            << " est_max_GiB=" << est_max / 1073741824.
            << " est_max_rank=" << max_rank << e.detail << '\n';
    }
    std::cout << out.str() << std::flush;
}

} // namespace storage_monitor
