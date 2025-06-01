#pragma once

#include "utils.hpp"
#include "static_vector.hpp"
#include "ls4patricia.hpp"
#include "key_set.hpp"
#include <sdsl/int_vector.hpp>

#include <algorithm>
#include <type_traits>
#include <vector>
#include <queue>

#include "strpool.hpp"

// #define __DEBUG_MARISA__
#ifdef __DEBUG_MARISA__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif

// #define __BENCH_MARISA__
#ifdef __BENCH_MARISA__
# define BENCH(foo) foo
#else
# define BENCH(foo)
#endif

// #define __NO_BRANCHING_LABEL__
#define __ENABLE_CACHE__


template <typename Key, bool reverse = false>
class MarisaCC : public StringPool<Key> {
 public:
  using key_type = Key;
  using label_vec = StaticVector<uint8_t>;
  using topo_t = LS4Patricia;
  using strpool_t = StringPool<key_type>;

  static constexpr bool reverse_ = reverse;
  static constexpr uint32_t link_cutoff_ = 3;  // unary paths must be at least this long to be considered for recursive compression
  static_assert(link_cutoff_ >= 2);

#ifdef __ENABLE_CACHE__
  // static constexpr uint32_t cache_ratio_ = 128;  // MARISA_HUGE_CACHE
  // static constexpr uint32_t cache_ratio_ = 256;  // MARISA_LARGE_CACHE
  static constexpr uint32_t cache_ratio_ = 512;  // MARISA_DEFAULT_CACHE
#endif

#ifdef __BENCH_MARISA__
  static uint64_t build_trie_time_;
  static uint64_t build_tail_time_;
#endif
  static void print_bench() {
  #ifdef __BENCH_MARISA__
    printf("build trie: %lf ms, build tail: %lf ms\n",
           (double)build_trie_time_/1000000, (double)build_tail_time_/1000000);
  #else
    printf("disabled\n");
  #endif
  }

  void print_space_cost_breakdown() const {
    size_t topo = 0, link = 0, data = 0;
    space_cost_breakdown(topo, link, data);
    printf("topology: %lf MB, link: %lf MB, data: %lf MB\n", (double)topo/mb_bits, (double)link/mb_bits, (double)data/mb_bits);
  }

 private:
  struct Range {
    uint32_t begin_{0};
    uint32_t end_{0};
    uint32_t depth_{0};

    Range() = default;

    Range(uint32_t begin, uint32_t end, uint32_t depth) : begin_(begin), end_(end), depth_(depth) {}
  };

#ifdef __ENABLE_CACHE__
  struct Cache {
    uint32_t parent_{0};
    uint32_t child_{0};
    union {
      uint32_t link_;
      float weight_;
    } union_;

    Cache() { set_weight(-1); }

    auto link() const -> uint32_t {
      return union_.link_;
    }

    auto weight() const -> float {
      return union_.weight_;
    }

    void set_link(uint32_t link) {
      union_.link_ = link;
    }

    void set_weight(float weight) {
      union_.weight_ = weight;
    }
  };
#endif

 public:
  MarisaCC() = default;

  ~MarisaCC() {
    delete next_;
  }

  template <typename Iterator, bool rev = reverse, typename = std::enable_if_t<!rev>>
  void build(Iterator begin, Iterator end, bool sorted = false, int max_recursion = 0, int mask = 0) {
    KeySet<key_type> key_set;
    while (begin != end) {
      key_set.emplace_back(&(*begin));
      ++begin;
    }
    assert(!key_set.empty());
    if (!sorted) {
      key_set.sort();
    }
    build(key_set, nullptr, key_set.space_cost(), max_recursion, mask);
  }

  auto size() const -> uint32_t override {
    if constexpr (!reverse_) {
      return topo_.num_leaves();
    } else {
      return links_.size();
    }
  }

  auto size_in_bytes() const -> size_t override {
    size_t ret = topo_.size_in_bytes() + labels_.size_in_bytes() + sdsl::size_in_bytes(links_) + next_->size_in_bytes();
  #ifdef __ENABLE_CACHE__
    ret += cache_.size() * sizeof(Cache);
  #endif
    return ret;
  }

  auto size_in_bits() const -> size_t override {
    return size_in_bytes() * 8;
  }

  void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const override {
    topo += topo_.size_in_bits();
  #ifdef __ENABLE_CACHE__
    topo += cache_.size() * sizeof(Cache);
  #endif
    link += sdsl::size_in_bytes(links_) * 8;
    data += labels_.size_in_bytes() * 8;
    next_->space_cost_breakdown(topo, link, data);
  }

#ifdef __NO_BRANCHING_LABEL__
  // returns leaf ID (-1 if not found)
  template <bool rev = reverse, typename = std::enable_if_t<!rev>>
  auto lookup(const key_type &key) const -> uint32_t {
    uint32_t len = key.size(), matched_len = 0;
    uint32_t pos = 0;

    while (matched_len < len) {
    #ifdef __ENABLE_CACHE__
      auto ret = search_cache(pos, key, matched_len);
      if (ret == -1) {  // mismatch
        return -1;
      } else if (ret == 1) {  // branch terminates
        return matched_len == len ? topo_.leaf_id(pos) : -1;
      } else if (ret == 2) {
        continue;
      }  // else ret == 0, i.e. not cached
    #endif
      bool found = false;
      do {  // search for label
        // _mm_prefetch(&labels_[pos], _MM_HINT_T0);
        if (topo_.is_link(pos)) {
          uint32_t link_len = next_->match(key, matched_len, topo_.link_id(pos), labels_[pos]);
          if (link_len != -1) {
            matched_len += link_len;
            found = true;
            break;
          }
        } else if (labels_[pos] == key[matched_len]) {
          matched_len++;
          found = true;
          break;
        }
        pos++;
      } while (!topo_.louds(pos));

      if (!found) {
        return -1;
      }
      if (!topo_.has_child(pos)) {  // branch terminates
        return matched_len == len ? topo_.leaf_id(pos) : -1;
      }
      pos = topo_.child_pos(pos);
    }

    if (labels_[pos] == terminator_) {  // prefix key
      return topo_.leaf_id(pos);
    }  // else early termination
    return -1;
  }
#else
  // returns leaf ID (-1 if not found)
  template <bool rev = reverse, typename = std::enable_if_t<!rev>>
  auto lookup(const key_type &key) const -> uint32_t {
    uint32_t len = key.size(), matched_len = 0;
    uint32_t pos = 0;

    while (matched_len < len) {
    #ifdef __ENABLE_CACHE__
      auto ret = search_cache(pos, key, matched_len);
      if (ret == -1) {  // mismatch
        return -1;
      } else if (ret == 1) {  // branch terminates
        return matched_len == len ? topo_.leaf_id(pos) : -1;
      } else if (ret == 2) {
        continue;
      }  // else ret == 0, i.e. not cached
    #endif
      uint32_t end = topo_.node_end(pos);
      pos = labels_.find(key[matched_len], pos, end);
      if (pos == end) {  // mismatch
        return -1;
      }
      assert(labels_[pos] == key[matched_len]);
      matched_len++;

      if (topo_.is_link(pos)) {
        uint32_t link_len = next_->match(key, matched_len, topo_.link_id(pos));
        if (link_len == -1) {
          return -1;
        }
        matched_len += link_len;
      }
      if (!topo_.has_child(pos)) {  // branch terminates
        return matched_len == len ? topo_.leaf_id(pos) : -1;
      }
      pos = topo_.child_pos(pos);
    }

    if (labels_[pos] == terminator_) {  // prefix key
      return topo_.leaf_id(pos);
    }  // else early termination
    return -1;
  }
#endif

  // returns matched length (-1 on mismatch)
  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t override {
    if constexpr (!reverse_) {
      return -1;
    }
    assert(key_id < size());
    uint32_t link = links_[key_id];
    return match_link(key, begin, link);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id, uint8_t partial_link) const -> uint32_t override {
    if constexpr (!reverse_) {
      return -1;
    }
    assert(key_id < size());
    uint32_t link = (links_[key_id] << 8) | partial_link;
    return match_link(key, begin, link);
  }

  auto match_link(const key_type &key, uint32_t begin, size_t link) const -> uint32_t {
    uint32_t pos = link;
    uint32_t matched_len = begin;
    while (true) {
      if (topo_.is_link(pos)) {
        uint32_t link_len = next_->match(key, matched_len, topo_.link_id(pos), labels_[pos]);
        if (link_len == -1) {
          return -1;
        }
        matched_len += link_len;
      } else {
        uint8_t label = labels_[pos];
        if (label != terminator_) {
          if (label != key[matched_len]) {
            return -1;
          }
          matched_len++;
        }
      }
      if (!topo_.has_parent(pos)) {
        break;
      }
      pos = topo_.parent_pos(pos);
    }
    return matched_len - begin;
  }

#ifdef __COMPARE_MARISA__
  void to_louds_marisa(std::unique_ptr<LoudsMarisa> &out) const {
    topo_.to_louds_marisa(out);
  }

  auto get_topo() const -> const topo_t * {
    return &topo_;
  }
#endif

 private:
  void build(const KeySet<key_type> &key_set, std::vector<uint8_t> *partial_links = nullptr,
             size_t original_size = 0, int max_recursion = 0, int mask = 0) override {
    KeySet<key_type> next_key_set;
    if constexpr (reverse_) {
      links_.resize(key_set.size());
    }
    if (partial_links != nullptr) {
      partial_links->resize(key_set.size());
    }
  #ifdef __ENABLE_CACHE__
    if constexpr (!reverse_) {
      reserve_cache(key_set.size());
    }
  #endif

    DEBUG( printf("build: %d\n", reverse_); )
  #ifdef __DEBUG_MARISA__
    if constexpr (reverse_) {
      for (uint32_t i = 0; i < key_set.size(); i++) {
        printf("%s\n", key_set.materialize(i).c_str());
      }
    }
  #endif
    BENCH( auto t0 = std::chrono::high_resolution_clock::now(); )
    uint32_t key_id = 0;
    uint32_t branch_id = 0;
    std::queue<Range> queue;
    queue.push(Range(0, key_set.size(), 0));
    while (!queue.empty()) {
      auto range = queue.front();
      queue.pop();
      DEBUG( printf("range (%d, %d, %d)\n", range.begin_, range.end_, range.depth_); )
      assert(range.begin_ < range.end_);

      uint64_t has_child[4]{0}, is_link[4]{0};  // each range corresponds to a node
      uint32_t node_start = branch_id, num_branches = 0;

      uint32_t begin = range.begin_, end = range.begin_;  // group common fragments

      while (end < range.end_ && range.depth_ == key_set[end].length_) {  // skip empty suffixes
        if constexpr (reverse_) {
          if (partial_links == nullptr) {
            links_[key_set[end].id_] = branch_id;
          } else {
            links_[key_set[end].id_] = branch_id >> 8;
            (*partial_links)[key_set[end].id_] = branch_id & MASK(8);
          }
        }
        end++;
      }
      if (end > begin) {
        DEBUG( printf("range (%d, %d, %d): (terminator)\n", begin, end, range.depth_); )
        DEBUG( printf("key ID %d: %s\n", key_id, key_set.materialize(begin).c_str()); )
        num_branches++;
        branch_id++;
        key_id++;
        labels_.emplace_back(terminator_);
        begin = end;
      }

      while (end < range.end_) {
        while (end < range.end_) {  // horizontal expansion
          if (key_set.get_label(end, range.depth_) != key_set.get_label(begin, range.depth_)) {
            break;
          }
          end++;
        }
        uint8_t branch_label = key_set.get_label(begin, range.depth_);
        DEBUG( printf("horizontal expansion: (%d, %d, %d, %c)\n", begin, end, range.depth_, branch_label); )
        assert(end > begin);

      #ifdef __ENABLE_CACHE__
        if constexpr (!reverse_) {
          cache_branch(node_start, num_branches, end - begin, branch_label);
        }
      #endif

        uint32_t depth;
        if (end == begin + 1) {  // single key; skip suffix
          depth = key_set[begin].length_;
        } else {
          depth = range.depth_ + 1;
          while (depth < key_set[begin].length_) {  // vertical extension
            if (key_set.get_label(begin, depth) != key_set.get_label(end - 1, depth)) {
              break;
            }
            depth++;
          }
        }
        DEBUG( printf("vertical extension: (%d, %d, %d, %d)\n", begin, end, range.depth_, depth); )

        if (depth - range.depth_ >= link_cutoff_) {  // link
          if constexpr (!reverse_) {
          #ifdef __NO_BRANCHING_LABEL__
            auto [pos, len] = key_set.substr_range(begin, range.depth_, depth - range.depth_);
            labels_.emplace_back(terminator_);
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move: %s\n", next_key_set.back().materialize(true).c_str()); )
          #else
            auto [pos, len] = key_set.substr_range(begin, range.depth_ + 1, depth - range.depth_ - 1);
            labels_.emplace_back(branch_label);  // store branching label in place for fast lookup
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move: %c:%s\n", branch_label, next_key_set.back().materialize(true).c_str()); )
          #endif
          } else {
            auto [pos, len] = key_set.substr_range(begin, range.depth_, depth - range.depth_);
            labels_.emplace_back(terminator_);
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move reverse: %s\n", next_key_set.back().materialize(true).c_str()); )
          }
          SET_BIT(is_link[num_branches / 64], num_branches % 64);
        } else {  // label
          labels_.emplace_back(branch_label);
          depth = range.depth_ + 1;
          DEBUG( printf("in place: %c\n", branch_label); )
        }

        bool empty = true;
        for (uint32_t i = begin; i < end; i++) {
          if (key_set[i].length_ > depth) {
            empty = false;
          } else if constexpr (reverse_) {
            if (partial_links == nullptr) {
              links_[key_set[i].id_] = branch_id;
            } else {
              links_[key_set[i].id_] = branch_id >> 8;
              (*partial_links)[key_set[i].id_] = branch_id & MASK(8);
            }
          }
        }
        if (!empty) {
          SET_BIT(has_child[num_branches / 64], num_branches % 64);
          queue.push(Range(begin, end, depth));
        } else {
          DEBUG( printf("key ID %d: %s\n", key_id, key_set.materialize(begin).c_str()); )
          key_id++;
        }
        num_branches++;
        branch_id++;
        begin = end;
      }
      topo_.add_node(has_child, is_link, num_branches);
    }
    topo_.build(false);
    labels_.shrink_to_fit();
    sdsl::util::bit_compress(links_);

    BENCH( auto t1 = std::chrono::high_resolution_clock::now(); )

    printf("trie size: %lf MB\n", (double)(topo_.size_in_bits() + sdsl::size_in_bytes(links_) * 8 +
           labels_.size_in_bytes() * 8) / mb_bits);

  auto build_next_with_partial_links = [&]() {
    std::vector<uint8_t> next_partial_links;
    next_ = strpool_t::build_optimal(next_key_set, &next_partial_links, original_size, max_recursion, mask);
    uint32_t pos = -1;
    for (auto partial_link : next_partial_links) {
      pos = topo_.next_link(pos + 1);
      labels_[pos] = partial_link;
    }
  };
  #ifdef __NO_BRANCHING_LABEL__
    build_next_with_partial_links();
  #else
    if constexpr (reverse_) {
      build_next_with_partial_links();
    } else {
      next_ = strpool_t::build_optimal(next_key_set, nullptr, original_size, max_recursion, mask);
    }
  #endif
    BENCH( auto t2 = std::chrono::high_resolution_clock::now(); )

  #ifdef __ENABLE_CACHE__
    if constexpr (!reverse_) {
      fill_cache();
    }
  #endif

    BENCH( build_trie_time_ += (t1 - t0).count(); )
    BENCH( build_tail_time_ += (t2 - t1).count(); )
  }

#ifdef __ENABLE_CACHE__
  void reserve_cache(uint32_t num_keys) {
    uint32_t cache_size = 256;
    while (cache_size < num_keys / cache_ratio_) {
      cache_size *= 2;
    }
    cache_.resize(cache_size);
    cache_mask_ = cache_size - 1;
  }

  void cache_branch(uint32_t parent, uint32_t child_id, float weight, uint8_t label) {
    DEBUG( printf("attempting to cache branch %d:%d:%c:%f\n", parent, child_id, label, weight); )
    auto cache_id = get_cache_id(parent, label);
    auto &cache = cache_[cache_id];
    DEBUG( printf("cache ID: %d, original: %d:%d:%c:%f\n", cache_id, cache.parent_, cache.child_,
                  restore_label(cache_id), cache.weight()); )
    if (weight > cache.weight()) {
      DEBUG( printf("replaced\n"); )
      cache.parent_ = parent;
      cache.child_ = child_id;
      cache.set_weight(weight);
    }
  }

  auto get_cache_id(uint32_t pos, uint8_t label) const -> uint32_t {
    return (pos ^ (pos << 5) ^ label) & cache_mask_;
  }

  auto restore_label(uint32_t cache_id) const -> uint8_t {
    return restore_label(cache_id, cache_[cache_id].parent_);
  }

  auto restore_label(uint32_t cache_id, uint32_t pos) const -> uint8_t {
    return (cache_id ^ pos ^ (pos << 5)) & cache_mask_;
  }

  void fill_cache() {
    uint32_t cache_size = cache_.size();
    DEBUG( uint32_t invalid = 0; )
    for (uint32_t i = 0; i < cache_size; i++) {
      auto &cache = cache_[i];
      auto parent = cache.parent_, branch_pos= cache.parent_ + cache.child_;
      DEBUG( printf("filling cache %d: %d, %d, %c, %f\n", i, parent, cache.child_, restore_label(i), cache.weight()); )
      if (cache.weight() < 0) {
        DEBUG( printf("invalid\n"); invalid++; )
        cache.parent_ = -1;
        cache.child_ = -1;
      } else {
        cache.child_ = (topo_.has_child(branch_pos) ? topo_.child_pos(branch_pos) : 0);
        if (!topo_.is_link(branch_pos)) {
          assert(restore_label(i) == labels_[branch_pos]);
          cache.set_link(-1);
          DEBUG( printf("regular branch, child: %d\n", cache.child_); )
        } else {
        #ifdef __NO_BRANCHING_LABEL__
          assert(topo_.num_links() < (1 << 24));
          cache.set_link((topo_.link_id(branch_pos) << 8) | labels_[branch_pos]);
        #else
          cache.set_link(topo_.link_id(branch_pos));
        #endif
          DEBUG( printf("link, child: %d, link: %d\n", cache.child_, cache.link()); )
        }
      }
    }
    DEBUG( printf("cache size: %d, invalid: %d, valid: %d\n", cache_size, invalid, cache_size - invalid); )
  }

  auto search_cache(uint32_t &pos, const key_type &key, uint32_t &matched_len) const -> uint32_t {
    auto cache_id = get_cache_id(pos, key[matched_len]);
    const auto &cache = cache_[cache_id];
    if (cache.parent_ != pos) {  // not cached
      return 0;
    }

    if (cache.link() == -1) {
      DEBUG( printf("cached: %d, %c, %d, regular branch\n", pos, key[matched_len], cache.child_); )
      matched_len++;
    } else {
      DEBUG( printf("cached: %d, %c, %d, link: %d\n", pos, key[matched_len], cache.child_, cache.link()); )
    #ifdef __NO_BRANCHING_LABEL__
      uint32_t link_len = next_->match(key, matched_len, cache.link() >> 8, cache.link() & MASK(8));
    #else
      matched_len++;
      uint32_t link_len = next_->match(key, matched_len, cache.link());
    #endif
      if (link_len == -1) {  // mismatch
        return -1;
      }
      matched_len += link_len;
    }

    if (cache.child_ == 0) {  // no child
      return 1;
    } else {
      pos = cache.child_;
      return 2;
    }
  }
#endif

  topo_t topo_;
  label_vec labels_;
  sdsl::int_vector<> links_;
  strpool_t *next_{nullptr};

#ifdef __ENABLE_CACHE__
  std::vector<Cache> cache_;
  uint32_t cache_mask_{0};
#endif

  template <typename K> friend class StringPool;
};

#ifdef __BENCH_MARISA__
template <typename K, bool r> uint64_t MarisaCC<K, r>::build_trie_time_ = 0;
template <typename K, bool r> uint64_t MarisaCC<K, r>::build_tail_time_ = 0;
#endif


#undef DEBUG