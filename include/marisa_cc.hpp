#pragma once

#include "static_vector.hpp"
#include "ls4patricia.hpp"
#include "key_set.hpp"
#include <sdsl/int_vector.hpp>

#include <algorithm>
#include <type_traits>
#include <vector>

// #define __DEBUG__
#ifdef __DEBUG__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


template <typename Key, typename Container, bool reverse = false>
class MarisaCC {
 public:
  using key_type = Key;
  using container_t = Container;
  using label_vec = StaticVector<uint8_t>;
  using topo_t = LS4Patricia;
  using next_trie_t = MarisaCC<key_type, container_t, true>;  // all secondary tries store reversed key fragments

  static constexpr bool reverse_ = reverse;
  static constexpr uint8_t terminator_ = 0;
  static constexpr uint32_t link_cutoff_ = 4;  // unary paths must be at least this long to be considered for recursive compression
  static_assert(link_cutoff_ >= 2);
  // stop building the next trie when the total size of unary paths is below this percentage of the original key set size
  static constexpr int size_percentage_cutoff_ = 10;

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
    delete dict_;
    delete next_trie_;
  }

  template <typename Iterator, bool rev = reverse, typename = std::enable_if_t<!rev>>
  void build(Iterator begin, Iterator end, bool sorted = false, int max_recursion = 0) {
    KeySet<key_type> key_set;
    while (begin != end) {
      key_set.emplace_back(&(*begin));
      ++begin;
    }
    assert(!key_set.empty());
    if (!sorted) {
      key_set.sort();
    }
    build(key_set, max_recursion, key_set.space_cost(), nullptr);
  }

  auto size_in_bits() const -> size_t {
    size_t ret = topo_.size_in_bits() + labels_.size_in_bits() + sdsl::size_in_bytes(links_)*8 + sizeof(void *)*2*8;
    if (next_trie_ != nullptr) {
      ret += next_trie_->size_in_bits();
    }
    if (dict_ != nullptr) {
      ret += dict_->size_in_bits();
    }
    return ret;
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
      assert(labels_.at(pos) == key[matched_len]);
      matched_len++;

      if (topo_.is_link(pos)) {
        if (dict_ != nullptr) {
          assert(next_trie_ == nullptr);
          uint32_t link_len = dict_->match(key, matched_len, topo_.link_id(pos));
          if (link_len == -1) {
            return -1;
          }
          matched_len += link_len;
        } else if (next_trie_ != nullptr) {
          uint32_t link_len = next_trie_->match_rev(key, matched_len, links_[topo_.link_id(pos)]);
          if (link_len == -1) {
            return -1;
          }
          matched_len += link_len;
        }
      }
      if (!topo_.has_child(pos)) {  // branch terminates
        return matched_len == len ? topo_.leaf_id(pos) : -1;
      }
      pos = topo_.child_pos(pos);
    }

    if (labels_.at(pos) == terminator_) {  // prefix key
      return topo_.leaf_id(pos);
    }
    return -1;
  }

  // returns matched length (-1 on mismatch)
  template <bool rev = reverse, typename = std::enable_if_t<rev>>
  auto match_rev(const key_type &key, uint32_t begin, uint32_t leaf_id) const -> uint32_t {
    assert(leaf_id < topo_.num_leaves());
    uint32_t len = key.size(), matched_len = begin;
    uint32_t pos = topo_.select_leaf(leaf_id);

    while (true) {
      if (topo_.is_link(pos)) {
        if (next_trie_ != nullptr) {
          assert(dict_ == nullptr);
          uint32_t link = links_[topo_.link_id(pos)];
          uint32_t link_len = next_trie_->match_rev(key, matched_len, link);
          if (link_len == -1) {
            return -1;
          }
          matched_len += link_len;
        } else {
          assert(dict_ != nullptr);
          uint32_t link_len = dict_->match(key, matched_len, topo_.link_id(pos));
          if (link_len == -1) {
            return -1;
          }
          matched_len += link_len;
        }
      } else {
        uint8_t label = labels_.at(topo_.label_id(pos));
        if (label != terminator_) {
          if (label != key[matched_len]) {
            return -1;
          }
          matched_len++;
        }  // else terminator, do nothing
      }
      if (!topo_.has_parent(pos)) {
        break;
      }
      pos = topo_.parent_pos(pos);
    }
    return matched_len - begin;
  }

 private:
  void build(KeySet<key_type> &key_set, int max_recursion = 0, size_t original_size = 0, sdsl::int_vector<> *links = nullptr) {
    KeySet<key_type> next_key_set;
    if (links != nullptr) {
      links->resize(key_set.size());
    }

    DEBUG( printf("build: %d\n", reverse_); )
   #ifdef __DEBUG__
    if constexpr (reverse_) {
      for (uint32_t i = 0; i < key_set.size(); i++) {
        printf("%s\n", key_set.materialize(i).c_str());
      }
    }
   #endif
    uint32_t key_id = 0;
    std::queue<Range> queue;
    queue.push(Range(0, key_set.size(), 0));
    while (!queue.empty()) {
      auto range = queue.front();
      queue.pop();
      DEBUG( printf("range (%d, %d, %d)\n", range.begin_, range.end_, range.depth_); )
      assert(range.begin_ < range.end_);

      uint64_t has_child[4]{0}, is_link[4]{0};    // each range corresponds to a node
      uint32_t num_branches = 0;

      // group common fragments
      uint32_t begin = range.begin_, end = range.begin_;

      while (end < range.end_ && range.depth_ == key_set[end].length_) {  // skip empty suffixes
        if (links != nullptr) {
          (*links)[key_set[end].id_] = key_id;
        }
        end++;
      }
      if (end > begin) {
        DEBUG( printf("range (%d, %d, %d): (terminator)\n", begin, end, range.depth_); )
        DEBUG( printf("key ID %d: %s\n", key_id, key_set.materialize(begin).c_str()); )
        num_branches++;
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
        DEBUG( printf("range (%d, %d, %d): %c\n", begin, end, range.depth_, key_set.get_label(begin, range.depth_)); )
        assert(end > begin);

        uint32_t depth = range.depth_ + 1;
        while (depth < key_set[begin].length_) {  // vertical expansion
          bool extend = true;
          for (uint32_t i = begin + 1; i < end; i++) {
            assert(depth < key_set[i].length_);
            if (key_set.get_label(i, depth) != key_set.get_label(begin, depth)) {
              extend = false;
              break;
            }
          }
          if (!extend) {
            break;
          }
          DEBUG( printf("extend range (%d, %d, %d): %c\n", begin, end, depth, key_set.get_label(begin, depth)); )
          depth++;
        }
        if (depth - range.depth_ >= link_cutoff_) {  // link
          if constexpr (!reverse_) {
            auto [pos, len] = key_set.substr_range(begin, range.depth_ + 1, depth - range.depth_ - 1);
            labels_.emplace_back(key_set.get_label(begin, range.depth_));  // store branching label in place for fast lookup
            next_key_set.emplace_back(key_set[begin].key_, pos, len);
            DEBUG( printf("move: %c:%s\n", key_set.get_label(begin, range.depth_), next_key_set.back().materialize(true).c_str()); )
          } else {
            auto [pos, len] = key_set.substr_range(begin, range.depth_, depth - range.depth_);
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
          } else if (links != nullptr) {
            (*links)[key_set[i].id_] = key_id;
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
        begin = end;
      }
      topo_.add_node(has_child, is_link, num_branches);
    }
    topo_.build();
    labels_.shrink_to_fit();

    if (max_recursion > 0 && 100 * next_key_set.space_cost() >= size_percentage_cutoff_ * original_size) {
      next_key_set.reverse();
      next_key_set.sort();
      next_trie_ = new next_trie_t();
      next_trie_->build(next_key_set, max_recursion - 1, original_size, &links_);
      sdsl::util::bit_compress(links_);
    } else {
      dict_ = new container_t();
      dict_->build(next_key_set);
    }
  }

  topo_t topo_;
  label_vec labels_;
  sdsl::int_vector<> links_;
  container_t *dict_{nullptr};
  next_trie_t *next_trie_{nullptr};

  friend class MarisaCC<key_type, container_t, false>;
};


#undef __DEBUG__
#undef DEBUG