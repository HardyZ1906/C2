#pragma once

#include <marisa.h>

#include <string>
#include <vector>


class MarisaWrapper {  // unified API
 public:
  using trie_t = marisa::Trie;

  MarisaWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0, uint32_t max_recursion = 0) {
    marisa::Keyset keyset;
    for (const auto &key : keys) {
      keyset.push_back(key.c_str());
    }
    trie_.build(keyset, max_recursion + 1);
  }

  auto lookup(const std::string &key) const -> uint32_t {
    marisa::Agent agent;
    agent.set_query(key.c_str());
    if (trie_.lookup(agent)) {
      return agent.key().id();
    }
    return -1;
  }

  auto space_cost() const -> size_t {
    return trie_.total_size() * 8;
  }
 private:
  trie_t trie_;
};