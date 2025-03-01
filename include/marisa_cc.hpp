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

#define __BENCH_MARISA__
#ifdef __BENCH_MARISA__
# define BENCH(foo) foo
#else
# define BENCH(foo)
#endif


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
    return ret;
  }

  auto size_in_bits() const -> size_t override {
    return size_in_bytes() * 8;
  }

  void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const {
    topo += topo_.size_in_bits();
    link += sdsl::size_in_bytes(links_) * 8;
    data += labels_.size_in_bytes() * 8;
    next_->space_cost_breakdown(topo, link, data);
  }

  // returns leaf ID (-1 if not found)
  template <bool rev = reverse, typename = std::enable_if_t<!rev>>
  auto lookup(const key_type &key) const -> uint32_t {
    uint32_t len = key.size(), matched_len = 0;
    uint32_t pos = 0;

    while (matched_len < len) {
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
    }
    return -1;
  }

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
      uint32_t num_branches = 0;

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
        DEBUG( printf("horizontal expansion: (%d, %d, %d, %c)\n", begin, end, range.depth_, key_set.get_label(begin, range.depth_)); )
        assert(end > begin);

        uint32_t depth;
        if (end == begin + 1) {
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
            auto [pos, len] = key_set.substr_range(begin, range.depth_ + 1, depth - range.depth_ - 1);
            labels_.emplace_back(key_set.get_label(begin, range.depth_));  // store branching label in place for fast lookup
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move: %c:%s\n", key_set.get_label(begin, range.depth_), next_key_set.back().materialize(true).c_str()); )
          } else {
            auto [pos, len] = key_set.substr_range(begin, range.depth_, depth - range.depth_);
            labels_.emplace_back(terminator_);
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move reverse: %s\n", next_key_set.back().materialize(true).c_str()); )
          }
          SET_BIT(is_link[num_branches / 64], num_branches % 64);
        } else {  // label
          labels_.emplace_back(key_set.get_label(begin, range.depth_));
          depth = range.depth_ + 1;
          DEBUG( printf("in place: %c\n", key_set.get_label(begin, range.depth_)); )
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

    if constexpr (reverse_) {
      std::vector<uint8_t> next_partial_links;
      next_ = strpool_t::build_optimal(next_key_set, &next_partial_links, original_size, max_recursion, mask);
      uint32_t pos = -1;
      for (auto partial_link : next_partial_links) {
        pos = topo_.next_link(pos + 1);
        labels_[pos] = partial_link;
      }
    } else {
      next_ = strpool_t::build_optimal(next_key_set, nullptr, original_size, max_recursion, mask);
    }
    BENCH( auto t2 = std::chrono::high_resolution_clock::now(); )

    BENCH( build_trie_time_ += (t1 - t0).count(); )
    BENCH( build_tail_time_ += (t2 - t1).count(); )
  }

  topo_t topo_;
  label_vec labels_;
  sdsl::int_vector<> links_;
  strpool_t *next_{nullptr};

  template <typename K> friend class StringPool;
};

#ifdef __BENCH_MARISA__
template <typename K, bool r> uint64_t MarisaCC<K, r>::build_trie_time_ = 0;
template <typename K, bool r> uint64_t MarisaCC<K, r>::build_tail_time_ = 0;
#endif


#undef DEBUG