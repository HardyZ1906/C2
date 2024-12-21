#pragma once

#include "fst.hpp"

#include <vector>
#include <string>


class FstWrapper {  // unified API
 public:
  using trie_t = fst::Trie;

  FstWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
             uint32_t pattern_len = 0, uint32_t min_occur = 0) : trie_(keys) {}

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.exactSearch(key);
  }

  auto space_cost() const -> size_t {
    return trie_.getMemoryUsage() * 8;
  }
 private:
  trie_t trie_;
};