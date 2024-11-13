#pragma once

#include "utils.hpp"
#include "bit_vector_builder.hpp"

#include <vector>
#include <limits>


template<typename Key, int fanout = 256, int cutoff = 0>
class FstBuilder {
 public:
  using key_type = Key;
  using bv_builder = BitVectorBuilder;

  static constexpr int fanout_ = fanout;
  static constexpr int cutoff_ = cutoff;
  static constexpr uint8_t terminator_ = 0;

  FstBuilder() = default;

  // keys must be sorted and unique
  template<typename Iterator>
  void build(Iterator begin, Iterator end) {
    // initialize an empty root node
    if constexpr (cutoff_ == 0) {
      expand_louds_sparse();
    } else {
      expand_louds_dense();
      d_labels_.back().reserve(fanout_);
      d_has_child_.back().reserve(fanout_);
    }
    // insert keys
    while (begin != end) {
      add_key(*begin);
      ++begin;
    }
  }

  auto rank_first_ls_level() const -> uint32_t {
    if (s_labels_.size() == 0) {
      return 0;
    }
    uint32_t ret = 0;
    for (auto elt : s_louds_[0].bits_) {
      ret += __builtin_popcountll(elt);
    }
  }
 private:
  void add_key(const key_type &key) {
    uint16_t len = key.size();
    if (len == 0) {
      insert_empty_key();
      return;
    }

    uint16_t level = 0;
    while (has_child(key, level)) {  // skip prefix already in trie
      level++;
    }
    if (has_branch(key, level)) {  // previous key is a strict prefix; convert to prefix key
      insert_prefix_node(key, level);
    }
    diverge(key, level);
    while (level < len - 1) {
      insert_new_node(key, level);
    }
  }

  void insert_empty_key() {
    if constexpr (cutoff_ > 0) {
      d_labels_[0].set1_rev(fanout_ - 1);
    } else {
      s_labels_[0].emplace_back(terminator_);
      s_has_child_[0].append0();
      s_louds_[0].append1();
    }
  }

  auto has_child(const key_type &key, uint16_t level) const -> bool {
    if (level < cutoff_) {
      assert(level < d_labels_.size());
      bool ret = d_has_child_[level].get_rev(fanout_ - key[level] - 1);
      return ret;
    } else {
      assert(level < cutoff_ + s_labels_.size());
      auto s_level = level - cutoff_;
      bool ret = (s_labels_[s_level].size() > 0) &&
                  (s_labels_[s_level].back() == key[level]) && (s_has_child_[s_level].back());
      return ret;
    }
  }

  auto has_branch(const key_type &key, uint16_t level) const -> bool {
    if (level < cutoff_) {
      return d_labels_[level].get_rev(fanout_ - key[level] - 1);
    } else {
      auto s_level = level - cutoff_;
      return s_labels_[s_level].size() > 0 && s_labels_[s_level].back() == key[level];
    }
  }

  void diverge(const key_type &key, uint16_t &level) {
    if (level < cutoff_) {
      assert(!d_labels_[level].get_rev(fanout_ - key[level] - 1));  // must be new label
      d_labels_[level].set1_rev(fanout_ - key[level] - 1);
    } else {
      auto s_level = level - cutoff_;
      if (s_labels_[s_level].size() == 0) {  // empty level
        s_labels_[s_level].emplace_back(key[level]);
        s_has_child_[s_level].append0();
        s_louds_[s_level].append1();
      } else {  // new label
        assert(s_labels_[s_level].back() != key[level]);
        s_labels_[s_level].emplace_back(key[level]);
        s_has_child_[s_level].append0();
        s_louds_[s_level].append0();
      }
    }
  }

  void insert_new_node(const key_type &key, uint16_t &level) {
    if (level < cutoff_) {  // mark branch as child
      d_has_child_[level].set1_rev(fanout_ - key[level] - 1);
    } else {
      s_has_child_[level - cutoff_].set1_rev(0);
    }

    // insert next byte
    level++;
    if (level < cutoff_) {
      if (level >= d_labels_.size()) {
        expand_louds_dense();
      }
      d_labels_[level].reserve(fanout_);
      d_labels_[level].set1_rev(fanout_ - key[level] - 1);
      d_has_child_[level].reserve(fanout_);
    } else {
      if (level >= cutoff_ + s_labels_.size()) {
        expand_louds_sparse();
      }
      auto s_level = level - cutoff_;
      s_labels_[s_level].emplace_back(key[level]);
      s_has_child_[s_level].append0();
      s_louds_[s_level].append1();
    }
  }

  void insert_prefix_node(const key_type &key, uint16_t &level) {
    if (level < cutoff_) {  // mark branch as child
      d_has_child_[level].set1_rev(fanout_ - key[level] - 1);
    } else {
      s_has_child_[level - cutoff_].set1_rev(0);
    }

    ++level;
    if (level < cutoff_) {
      if (level >= d_labels_.size()) {
        expand_louds_dense();
      }
      d_labels_[level].reserve(fanout_);
      d_labels_[level].set1_rev(fanout_ - 1);  // null terminator
      d_has_child_[level].reserve(fanout_);
    } else {
      if (level >= cutoff_ + s_labels_.size()) {
        expand_louds_sparse();
      }
      auto s_level = level - cutoff_;
      s_labels_[s_level].emplace_back(terminator_);
      s_has_child_[s_level].append0();
      s_louds_[s_level].append1();
    }
  }

  void expand_louds_dense() {
    assert(d_labels_.size() < cutoff_);
    assert(s_labels_.size() == 0);

    d_labels_.emplace_back();
    d_has_child_.emplace_back();
  }

  void expand_louds_sparse() {
    assert(d_labels_.size() == cutoff_);

    s_labels_.emplace_back();
    s_has_child_.emplace_back();
    s_louds_.emplace_back();
  }

  std::vector<bv_builder> d_labels_;
  std::vector<bv_builder> d_has_child_;

  std::vector<std::vector<uint8_t>> s_labels_;
  std::vector<bv_builder> s_has_child_;
  std::vector<bv_builder> s_louds_;

  template<typename K, int c, int f> friend class FstCC;
  template<typename K, int c, int f> friend class LoudsDenseCC;
  template<typename K, int c> friend class LoudsSparseCC;
  template<typename K> friend class LS4CoCo;
  template<typename K> friend class LS4CoCoRecursive; 
};