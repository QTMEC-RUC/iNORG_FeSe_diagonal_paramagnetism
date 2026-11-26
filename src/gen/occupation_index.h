#pragma once
#include "parallel_stage2.h"
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

// Owned keys: copies/moves of NocSpace cannot leave dangling occupation views.
// Build/clear require exclusive access. Once built, all lookups are read-only.
// Hash order is NOT basis order; div/idx_div alone define basis numbering.
template<class Index> class OccupationIndex {
public:
    using Map = std::unordered_map<std::string, Index>;
    using value_type = typename Map::value_type;
    using Pending = std::pair<std::string, Index>;
private:
    std::vector<Map> shards_{1};
    std::size_t shard_for(const std::string& key) const {
        return std::hash<std::string>{}(key) % shards_.size();
    }
public:
    class const_iterator {
        friend class OccupationIndex;
        const OccupationIndex* owner_ = nullptr;
        std::size_t shard_ = 0;
        typename Map::const_iterator it_{};
        const_iterator(const OccupationIndex* owner, std::size_t shard,
                       typename Map::const_iterator it = {})
            : owner_(owner), shard_(shard), it_(it) {}
        void skip_empty() {
            while (shard_ < owner_->shards_.size() && it_ == owner_->shards_[shard_].end()) {
                ++shard_;
                if (shard_ < owner_->shards_.size()) it_ = owner_->shards_[shard_].begin();
            }
        }
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = typename Map::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;
        const_iterator() = default;
        reference operator*() const { return *it_; }
        pointer operator->() const { return &*it_; }
        const_iterator& operator++() { ++it_; skip_empty(); return *this; }
        const_iterator operator++(int) { auto old=*this; ++*this; return old; }
        bool operator==(const const_iterator& other) const {
            if (owner_ != other.owner_ || shard_ != other.shard_) return false;
            return !owner_ || shard_ == owner_->shards_.size() || it_ == other.it_;
        }
        bool operator!=(const const_iterator& other) const { return !(*this == other); }
    };
    const_iterator begin() const {
        if (shards_.empty()) return end();
        const_iterator it(this,0,shards_.front().begin()); it.skip_empty(); return it;
    }
    const_iterator end() const { return const_iterator(this,shards_.size()); }
    const_iterator find(const std::string& key) const {
        if (shards_.empty()) return end();
        const auto shard=shard_for(key);
        auto it=shards_[shard].find(key);
        return it==shards_[shard].end() ? end() : const_iterator(this,shard,it);
    }
    const Index& at(const std::string& key) const {
        const auto it=find(key);
        if (it==end()) throw std::out_of_range("OccupationIndex key not found");
        return it->second;
    }
    std::size_t size() const {
        std::size_t n=0; for(const auto& shard:shards_) n+=shard.size(); return n;
    }
    bool empty() const { return size()==0; }
    std::size_t shard_count() const { return shards_.size(); }
    std::size_t shard_capacity() const { return shards_.capacity(); }
    std::size_t bucket_count() const {
        std::size_t n=0; for(const auto& shard:shards_) n+=shard.bucket_count(); return n;
    }
    std::pair<const_iterator,bool> insert(Pending entry) {
        if(shards_.empty()) shards_.resize(1);
        const auto shard=shard_for(entry.first);
        auto result=shards_[shard].emplace(std::move(entry.first),entry.second);
        return {const_iterator(this,shard,result.first),result.second};
    }
    void clear() {
        stage2_parallel::for_each(shards_.size(),[&](std::size_t s){ Map{}.swap(shards_[s]); });
        // Keep only the small shard descriptor array; node/bucket storage is released.
    }
    bool operator==(const OccupationIndex& other) const {
        if(size()!=other.size()) return false;
        for(const auto& item:*this) {
            const auto found=other.find(item.first);
            if(found==other.end() || found->second!=item.second) return false;
        }
        return true;
    }
    bool operator!=(const OccupationIndex& other) const { return !(*this==other); }

    // Stable scatter of pointers, then one exclusive writer per destination shard.
    // No shared-map writes, locks, serial N-entry merge, or copied key buffers.
    // Existing keys win, then the first occurrence in original chunk/group order.
    template<class Routed> void append_chunks(std::vector<std::vector<Pending>>& chunks,
                                               Routed routed) {
        namespace par=stage2_parallel;
        std::size_t total=0;
        for(const auto& chunk:chunks) {
            if(chunk.size()>std::numeric_limits<std::size_t>::max()-total)
                throw std::overflow_error("OccupationIndex entry count overflow");
            total+=chunk.size();
        }
        if(!total) { routed(); return; }
        if(empty()) {
            std::vector<Map> fresh(static_cast<std::size_t>(par::workers(total)));
            shards_.swap(fresh);
        }
        const auto ns=shards_.size();
        std::vector<std::vector<std::size_t>> offsets(chunks.size());
        std::vector<std::vector<Pending*>> pointers(chunks.size());
        par::for_each(chunks.size(),[&](std::size_t part){
            auto& off=offsets[part]; off.assign(ns+1,0);
            for(const auto& entry:chunks[part]) ++off[shard_for(entry.first)+1];
            for(std::size_t s=0;s<ns;++s) off[s+1]+=off[s];
            auto next=off;
            auto& ptr=pointers[part]; ptr.resize(chunks[part].size());
            for(auto& entry:chunks[part]) ptr[next[shard_for(entry.first)]++]=&entry;
        });
        routed();
        par::for_each(ns,[&](std::size_t s){
            auto& target=shards_[s];
            std::size_t count=target.size();
            for(const auto& off:offsets) {
                const auto added=off[s+1]-off[s];
                if(added>target.max_size()-count)
                    throw std::length_error("OccupationIndex shard too large");
                count+=added;
            }
            target.reserve(count);
            for(std::size_t part=0;part<chunks.size();++part)
                for(auto i=offsets[part][s];i<offsets[part][s+1];++i) {
                    auto& entry=*pointers[part][i];
                    target.emplace(std::move(entry.first),entry.second);
                }
        });
    }
};
