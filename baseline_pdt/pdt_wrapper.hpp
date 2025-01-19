#pragma once

#include "../include/compressed_string_pool.hpp"
#include "include/vbyte_string_pool.hpp"
#include "include/path_decomposed_trie.hpp"
#include "../lib/ds2i/succinct/mapper.hpp"
#include "../include/utils.hpp"

#include <vector>
#include <string>

class PdtWrapper {  // unified API
 public:
  using trie_t = succinct::tries::path_decomposed_trie<succinct::tries::compressed_string_pool<uint16_t>>;
  // using trie_t = succinct::tries::path_decomposed_trie<succinct::tries::vbyte_string_pool>;

  PdtWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
             uint32_t max_recursion = 0, int mask = 0) : trie_(keys) {}

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    return trie_.index(key);
  }

  auto space_cost() const -> size_t {
    return succinct::mapper::size_of(const_cast<trie_t &>(trie_)) * 8;
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