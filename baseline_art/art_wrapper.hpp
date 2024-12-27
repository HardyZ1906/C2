#pragma once

#include "art/art.hpp"

#include <vector>
#include <string>


class ArtWrapper {  // unified API
 public:
  using trie_t = art::art<bool>;

  ArtWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0, uint32_t max_recursion = 0) {
    for (const auto &key : keys) {
      trie_.set(key.c_str(), true);
    }
  }

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.get(key.c_str());
  }

  auto space_cost() const -> size_t {
    return 0;  // not implemented
  }
 private:
  trie_t trie_;
};