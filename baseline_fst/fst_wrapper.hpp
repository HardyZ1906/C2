#pragma once

#include "include/fst.hpp"

#include <vector>
#include <string>


class FstWrapper {  // unified API
 public:
  using trie_t = fst::Trie;

  FstWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
             int max_recursion = 0, int mask = 0) : trie_(keys) {}

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.exactSearch(key);
  }

  auto space_cost() const -> size_t {
    return trie_.getMemoryUsage() * 8;
  }

  static void print_bench() {
    printf("not implemented\n");
  }

  void print_space_cost_breakdown() const {
    printf("not implemented\n");
  }
 private:
  trie_t trie_;
};