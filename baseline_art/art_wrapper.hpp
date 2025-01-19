#pragma once

#include "include/art.hpp"
#include "../include/utils.hpp"

#include <vector>
#include <string>


// Did not pass our correctness test because of inappropriate handling of prefix keys
class ArtWrapper {  // unified API
 public:
  using trie_t = ART;

  ArtWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
             int max_recursion = 0, int mask = 0) {
    max_key_len_ = 0;
    size_t total_size = 0;
    for (const auto &key : keys) {
      max_key_len_ = std::max(max_key_len_, static_cast<uint32_t>(key.size()));
      total_size += key.size();
    }
    total_size += max_key_len_ - keys.back().size();  // make sure the last key does not overflow
    trie_ = new ART(max_key_len_);
    db_ = new char[total_size];

    std::vector<uint64_t> values;
    values.reserve(keys.size());
    values.push_back(reinterpret_cast<uint64_t>(db_));
    size_t padded_size = 0;
    for (const auto &key : keys) {
      memcpy(db_ + padded_size, key.c_str(), key.size());
      padded_size += key.size();
      values.push_back(reinterpret_cast<uint64_t>(db_ + padded_size));
    }
    assert(padded_size <= total_size);
    trie_->load(const_cast<std::vector<std::string> &>(keys), values, max_key_len_);
  }

  ~ArtWrapper() {
    delete trie_;
    delete[] db_;
  }

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    uint32_t ret = const_cast<trie_t *>(trie_)->lookup(reinterpret_cast<uint8_t *>(const_cast<char *>(key.c_str())),
                                                       key.size(), max_key_len_);
    return ret == 0 ? -1 : ret;
  }

  auto space_cost() const -> size_t {  // only the filter size
    return const_cast<trie_t *>(trie_)->getMemory() * 8;
  }

  static void print_bench() {
    printf("not implemented\n");
  }

  void print_space_cost_breakdown() const {
    printf("not implemented\n");
  }
 private:
  trie_t *trie_{nullptr};
  char *db_{nullptr};  // the actual keys; this is needed because ART is a filter and may return false positives
  uint32_t max_key_len_{0};
};