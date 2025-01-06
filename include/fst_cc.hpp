#pragma once

#include "utils.hpp"
#include "static_vector.hpp"
#include "louds_sparse_cc.hpp"
#include "bit_vector.hpp"
#include "key_set.hpp"
#include "marisa_cc.hpp"

#include <limits>
#include <vector>
#include <queue>


// #define __DEBUG_FST__
#ifdef __DEBUG_FST__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


// louds-sparse trie used for building CoCo-trie
template <typename Key>
class FstCC {
 public:
  using key_type = Key;
  using strpool_t = StringPool<key_type>;
  using label_vec = StaticVector<uint8_t>;
  using topo_t = LoudsSparseCC;
  using bitvec_t = BitVector;

  static constexpr uint32_t link_cutoff_ = 4;  // suffixes of length above this value will be moved to string pool

 public:
  // helper class for walking down the trie and traversing macro-node keys; used by the optimizer
  struct walker {
    using trie_t = FstCC;

    key_type key_;
    const trie_t *trie_;
    uint32_t pos_;
    uint32_t level_;  // number of walked levels

    walker(const trie_t *trie, uint32_t pos) : trie_(trie), pos_(pos), level_(1) {
      uint8_t label = trie_->get_label(pos_);
      if (label != terminator_) {
        key_.push_back(label);
      }
    }

    walker(const walker &other) : key_(other.key_), trie_(other.trie_), pos_(other.pos_), level_(other.level_) {}

    // is the key legitimately terminated?
    auto valid() const -> bool {
      return !trie_->topo_.has_child(pos_);
    }

    // is the current key a prefix key?
    auto prefix_key() const -> bool {
      return trie_->get_label(pos_) == terminator_;
    }

    auto key() const -> const key_type & {
      return key_;
    }

    // move to the leftmost label in node
    void move_to_front() {
      uint32_t front = trie_->topo_.node_start(pos_);
      if (trie_->get_label(front) == terminator_) {
        pos_ = front;
        key_.pop_back();
      } else {
        pos_ = front;
        key_.back() = trie_->get_label(pos_);
      }
    }

    // move to the rightmost label in node
    void move_to_back() {
      uint32_t back = trie_->topo_.node_end(pos_) - 1;
      if (trie_->get_label(pos_) == terminator_) {
        pos_ = back;
        key_.push_back(trie_->get_label(pos_));
      } else {
        pos_ = back;
        key_.back() = trie_->get_label(pos_);
      }
    }

    // does NOT regress if current level is already greater than `max_level`
    void get_min_key(uint32_t max_level = std::numeric_limits<uint32_t>::max()) {
      while (level_ < max_level && trie_->topo_.has_child(pos_)) {
        pos_ = trie_->topo_.child_pos(pos_);  // keep taking the leftmost branch
        uint8_t label = trie_->get_label(pos_);
        if (label != terminator_) {
          key_.push_back(label);
        }
        level_++;
      }
    }

    // does NOT regress if current level is already greater than `max_level`
    void get_max_key(uint32_t max_level = std::numeric_limits<uint32_t>::max()) {
      while (level_ < max_level && trie_->topo_.has_child(pos_)) {
        pos_ = trie_->topo_.child_pos(pos_);
        pos_ = trie_->topo_.node_end(pos_) - 1;  // keep taking the rightmost branch
        uint8_t label = trie_->get_label(pos_);
        if (label != terminator_) {
          key_.push_back(label);
        }
        level_++;
      }
    }

    // make sure to call `get_min_key(max_level)` before calling this
    // return true on success and false if there is no more key
    auto next(uint32_t max_level = std::numeric_limits<uint32_t>::max()) -> bool {
      while (level_ > 0) {
        if (trie_->get_label(pos_) == terminator_) {  // terminator is never the last label
          pos_++;
          key_.push_back(trie_->get_label(pos_));
          get_min_key(max_level);
          return true;
        } else if (pos_ + 1 < trie_->topo_.size() && !trie_->topo_.louds(pos_ + 1)) {  // not the last label in node
          pos_++;
          key_.back() = trie_->get_label(pos_);
          get_min_key(max_level);
          return true;
        }
        // last label in node; regress to parent
        key_.pop_back();
        pos_ = trie_->topo_.parent_pos(pos_);
        level_--;
      }
      return false;
    }

    // move to the leftmost next-level node in subtrie
    // return true if found and false otherwise
    // calling any of the `move_down` functions after false is returned is undefined behavior
    auto move_down_one_level_left() -> bool {
      uint32_t next_level = level_ + 1;

      get_min_key(next_level);
      while (level_ < next_level) {
        assert(!trie_->topo_.has_child(pos_));  // key terminates before `next_level`
        while (true) {
          if (trie_->get_label(pos_) != terminator_) {
            key_.pop_back();
          }
          uint32_t end = trie_->topo_.node_end(pos_);
          uint32_t next = trie_->topo_.next_child(pos_ + 1);
          if (next < end) {  // trace next branch
            pos_ = next;
            key_.push_back(trie_->get_label(pos_));
            break;
          }
          // regress to parent
          pos_ = trie_->topo_.parent_pos(pos_);
          level_--;
          if (level_ == 0) {
            return false;
          }
        }
        get_min_key(next_level);
      }
      return true;
    }

    // move to the rightmost next-level node in subtrie
    // return true if found and false otherwise
    // calling any of the `move_down` functions after false is returned is undefined behavior
    auto move_down_one_level_right() -> bool {
      uint32_t next_level = level_ + 1;

      get_max_key(next_level);
      while (level_ < next_level) {
        assert(!trie_->topo_.has_child(pos_));  // key terminates before `next_level`
        while (true) {
          if (trie_->get_label(pos_) != terminator_) {
            key_.pop_back();
          }
          uint32_t start = trie_->topo_.node_start(pos_);
          uint32_t prev = trie_->topo_.prev_child(pos_ - 1);
          if (prev >= start) {  // trace previous branch
            pos_ = prev;
            key_.push_back(trie_->get_label(pos_));
            break;
          }
          // regress to parent
          pos_ = trie_->topo_.parent_pos(pos_);
          level_--;
          if (level_ == 0) {
            return false;
          }
        }
        get_max_key(next_level);
      }
      return true;
    }
  };

 private:
  class TempStringPool : public StringPool<key_type> {
   public:
    using strpool_t = typename succinct::tries::compressed_string_pool<uint8_t>;

    TempStringPool() = default;

    ~TempStringPool() = default;

    void build(const KeySet<key_type> &key_set, size_t original_size = 0,
               int max_recursion = 0, int mask = 0) {
      keys_ = key_set;
    }

    void build(KeySet<key_type> &&key_set) {
      keys_ = key_set;
    }

    auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t override {
      return -1;  // unused
    }

    auto size() const -> uint32_t override {
      return keys_.size();
    }

    auto size_in_bytes() const -> size_t override {
      return 0;  // unused
    }

    auto size_in_bits() const -> size_t override {
      return 0; // unused
    }
   private:
    KeySet<key_type> keys_;

    template <typename K> friend class FstCC;
    template <typename K> friend class CoCoCC;
  };

 private:
  struct Range {
    uint32_t begin_{0};
    uint32_t end_{0};
    uint32_t depth_{0};
    uint32_t lcp_{0};

    Range() = default;

    Range(uint32_t begin, uint32_t end, uint32_t depth, uint32_t lcp)
        : begin_(begin), end_(end), depth_(depth), lcp_(lcp) {}
  };

 public:
  FstCC() = default;

  ~FstCC() {
    delete next_;
  }

  template <typename Iterator>
  void build(Iterator begin, Iterator end, bool sorted = false,
             int max_recursion = 0, int mask = 0) {
    KeySet<key_type> key_set;
    while (begin != end) {
      key_set.emplace_back(&(*begin));
      ++begin;
    }
    assert(!key_set.empty());
    if (!sorted) {
      key_set.sort();
    }
    build(key_set, false, max_recursion, mask);
  }

  void clear() {
    labels_.clear();
    topo_.clear();
  }

  auto get_label(uint32_t idx) const -> uint8_t {
    assert(idx < labels_.size());
    return labels_.at(idx);
  }

  auto lookup(const key_type &key) const -> uint32_t {
    uint16_t len = key.size(), matched_len = 0;
    uint32_t pos = 0;

    while (matched_len < len) {
      uint32_t end = topo_.node_end(pos);
      pos = labels_.find(key[matched_len], pos, end);
      if (pos == end) {  // mismatch
        return -1;
      }
      assert(get_label(pos) == key[matched_len]);
      matched_len++;

      if (!topo_.has_child(pos)) {
        auto leaf_id = topo_.leaf_id(pos);
        if (!is_link_.get(leaf_id)) {
          return matched_len == len ? leaf_id : -1;
        } else {
          uint32_t link = is_link_.rank1(leaf_id);
          return next_->match(key, matched_len, link) == len - matched_len ? leaf_id : -1;
        }
      }
      pos = topo_.child_pos(pos);
    }

    if (get_label(pos) == terminator_) {  // prefix key
      return topo_.leaf_id(pos);
    }
    return -1;
  }

  // return the cutoffs between levels
  auto get_level_boundaries() const -> std::vector<uint32_t> {
    return topo_.get_level_boundaries();
  }

  auto size_in_bytes() const -> size_t {
    return labels_.size_in_bytes() + topo_.size_in_bytes() + next_->size_in_bytes() + sizeof(uint32_t);
  }

  auto size_in_bits() const -> size_t {
    return size_in_bytes() * 8;
  }

 private:
  void build(const KeySet<key_type> &key_set, bool temp = false,
             int max_recursion = 0, int mask = 0) {
    KeySet<key_type> suffixes;

    auto lcp = [&](uint32_t begin, uint32_t end, uint32_t depth) -> uint32_t {
      assert(end > begin);
      assert(depth <= key_set[begin].length_ && depth <= key_set[end - 1].length_);
      if (end == begin + 1) {
        return key_set[begin].length_ - depth;
      }
      uint32_t len = std::min(key_set[begin].length_, key_set[end - 1].length_) - depth;
      uint32_t ret = 0;
      while (ret < len) {
        if (key_set.get_label(begin, depth + ret) != key_set.get_label(end - 1, depth + ret)) {
          break;
        }
        ret++;
      }
      return ret;
    };

    auto is_same_key = [&](const Range &range) -> bool {
      return range.lcp_ == key_set[range.end_ - 1].length_ - range.depth_;
    };

    std::queue<Range> queue;
    queue.push(Range(0, key_set.size(), 0, lcp(0, key_set.size(), 0)));
    while (!queue.empty()) {
      auto range = queue.front();
      queue.pop();
      DEBUG( printf("range (%d, %d, %d)\n", range.begin_, range.end_, range.depth_); )
      assert(range.begin_ < range.end_);

      uint64_t has_child[4]{0};    // each range corresponds to a node
      uint32_t num_branches = 0;

      if (range.lcp_ > 0) {
        labels_.emplace_back(key_set.get_label(range.begin_, range.depth_));
        if (!is_same_key(range)) {  // not a suffix
          SET_BIT(has_child[0], 0);
          topo_.add_node(has_child, 1);
          queue.push(Range(range.begin_, range.end_, range.depth_ + 1, range.lcp_ - 1));
        } else if (range.lcp_ < link_cutoff_) {  // in place
          if (range.lcp_ == 1) {  // last label
            topo_.add_node(has_child, 1);
            is_link_.append0();
          } else {  // not last label
            SET_BIT(has_child[0], 0);
            topo_.add_node(has_child, 1);
            queue.push(Range(range.begin_, range.end_, range.depth_ + 1, range.lcp_ - 1));
          }
        } else {  // link
          topo_.add_node(has_child, 1);
          is_link_.append1();
          suffixes.emplace_back(key_set[range.begin_].key_, range.depth_ + 1, range.lcp_ - 1);
        }
        continue;
      }

      // group common fragments
      uint32_t begin = range.begin_, end = range.begin_;

      while (end < range.end_ && range.depth_ == key_set[end].length_) {  // skip empty suffixes
        end++;
      }
      if (end > begin) {
        DEBUG( printf("range (%d, %d, %d): (terminator)\n", begin, end, range.depth_); )
        DEBUG( printf("key ID %d: %s\n", is_link_.size(), key_set.materialize(begin).c_str()); )
        num_branches++;
        is_link_.append0();
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
        DEBUG( printf("horizontal expansion (%d, %d, %d, %c)\n", begin, end, range.depth_, key_set.get_label(begin, range.depth_)); )
        assert(end > begin);

        labels_.emplace_back(key_set.get_label(begin, range.depth_));

        uint32_t depth = range.depth_ + 1;
        if (key_set[end - 1].length_ > depth) {  // subtree not empty
          SET_BIT(has_child[num_branches / 64], num_branches % 64);
          queue.push(Range(begin, end, depth, lcp(begin, end, depth)));
        } else {
          DEBUG( printf("key ID %d: %s\n", is_link_.size(), key_set.materialize(begin).c_str()); )
          is_link_.append0();
        }
        num_branches++;
        begin = end;
      }
      topo_.add_node(has_child, num_branches);
    }
    topo_.build();
    is_link_.build();
    labels_.shrink_to_fit();

    if (!temp) {
      next_ = strpool_t::build_optimal(suffixes, key_set.space_cost(), max_recursion, mask);
    } else {
      auto temp = new TempStringPool();
      temp->build(std::move(suffixes));
      next_ = temp;
    }
  }

  label_vec labels_;

  topo_t topo_;

  bitvec_t is_link_;

  strpool_t *next_{nullptr};

  friend class walker;
  template <typename K> friend class CoCoOptimizer;
  template <typename K> friend class CoCoCC;
};


#undef DEBUG