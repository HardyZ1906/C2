#pragma once

#define BIG_ALPHABET

#include "include/utils.hpp"
#include "include/uncompacted_trie.hpp"
#include "include/CoCo-trie_v2.hpp"
#include "../include/utils.hpp"

#include <vector>
#include <string>


class CoCoWrapper {  // unified API
 public:
  using trie_t = CoCo_v2<1, uint128_t, MAX_L_THRS, 5>;

  __NOINLINE_IF_PROFILE CoCoWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
                                    int max_recursion = 0, int mask = 0)
                                    : trie_([&keys]() {
                                        datasetStats ds = dataset_stats_from_vector(keys);
                                        MIN_CHAR = ds.get_min_char();
                                        ALPHABET_SIZE = ds.get_alphabet_size();
                                        return trie_t(keys);
                                      }()) {}

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    return trie_.look_up(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
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