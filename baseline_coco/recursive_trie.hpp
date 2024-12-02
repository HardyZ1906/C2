#pragma once

#include "defs.hpp"
#include "uncompacted_trie.hpp"
#include <vector>
#include <string>


#define __DEBUG__


// a list of uncompacted tries recursively compressing key suffixes
// max_recursion: the maximum level of recursion; disabled if set to 0
// cutoff: don't insert into next trie if suffix length is below this level
// estimate: the estimated size ratio of the resulting trie to the original dataset, in percentage
template<uint8_t MIN_L = 1, typename code_type = uint128_t, uint8_t MAX_L_THRS = MAX_L_THRS,
         uint8_t space_relaxation = 0, uint32_t max_recursion = 0, uint32_t cutoff = 6, uint8_t estimate = 20>
class RecursiveTrie {
 public:
  static_assert(estimate <= 100);

  using BaseTrie = Trie_lw<MIN_L, code_type, MAX_L_THRS, space_relaxation>;

  std::vector<BaseTrie> tries_;

  RecursiveTrie() = default;

  RecursiveTrie(const std::vector<std::string> &strs, size_t l_fixed = 0) {
    build(strs, l_fixed);
  }

  ~RecursiveTrie() = default;

  // build tries from a string dataset; keys don't have to be sorted beforehand
  void build(const std::vector<std::string> &strs, size_t l_fixed = 0) {
    uint32_t num_strs = strs.size();
    std::vector<uint32_t> inserted_len(num_strs, 0);
    std::vector<std::pair<std::string, uint32_t>> suffixes(num_strs);  // {string, index in the original `strs` vector}
    std::vector<uint32_t> suffix_inserted_len(num_strs, 0);

    auto build_last_trie = [&]() {
      tries_.emplace_back();
      tries_.back().insert(suffixes[0].first);
      for (size_t i = 1; i < suffixes.size(); i++) {
        if (suffixes[i].first != suffixes[i-1].first) {
          tries_.back().insert(suffixes[i].first);
        }
      }
    };

    uint32_t num_recursion = 0;
    while (true) {
      suffixes.clear();
      suffix_inserted_len.clear();

      // sort key suffixes
      for (size_t i = 0; i < strs.size(); i++) {
        if (inserted_len[i] < strs[i].size()) {
          suffixes.emplace_back(std::make_pair<std::string, uint32_t>(strs[i].substr(inserted_len[i]), i));
          suffix_inserted_len.emplace_back(0);
        }
      }
      printf("%d suffixes\n", suffixes.size());
      std::sort(suffixes.begin(), suffixes.end());

      if (num_recursion == max_recursion - 1) {
        build_last_trie();
        break;
      }
      num_recursion++;

      // extract the shortest unique prefix from each key
      uint32_t pfx_len = 0;
      uint32_t i = 0, j = 0;
      uint32_t num_unique_suffixes = 0, total_size = 0;
      while (i < suffixes.size()) {
        while (j < suffixes.size() && suffixes[j].first == suffixes[i].first) {  // deduplicate
          j++;
        }

        // extract longest common prefix
        uint32_t next_pfx_len = 0;
        if (j < suffixes.size()) {
          uint32_t len = std::min<size_t>(suffixes[i].first.size(), suffixes[j].first.size());
          uint32_t k = 0;
          while (k < len && suffixes[i].first[k] == suffixes[j].first[k]) {
            k++;
          }
          next_pfx_len = k;
        }

        pfx_len = std::max<uint32_t>(pfx_len, next_pfx_len);
        if (suffixes[i].first.size() - pfx_len <= cutoff) {  // insert entire suffix
          while (i < j) {
            suffix_inserted_len[i] = suffixes[i].first.size();
            i++;
          }
        } else {  // only insert to the first distinguishing character
          while (i < j) {
            suffix_inserted_len[i] = pfx_len + 1;
            i++;
          }
        }
        pfx_len = next_pfx_len;

        num_unique_suffixes++;
        total_size += suffixes[i].first.size();
      }
      printf("%d unique suffixes, %d bytes\n", num_unique_suffixes, total_size);

      // s <= s*e + n + mlogm
      if (total_size*log2_ceil(ALPHABET_SIZE)*(100 - estimate) <=
          (suffixes.size() + num_unique_suffixes*log2_ceil(num_unique_suffixes))*100) {
        build_last_trie();
        break;
      }
      tries_.emplace_back();
      tries_.back().insert(suffixes[0].first, 0, suffix_inserted_len[0]);
      inserted_len[suffixes[0].second] += suffix_inserted_len[0];
      for (size_t i = 1; i < suffixes.size(); i++) {
        if (suffixes[i].first != suffixes[i-1].first) {
          printf("%d: %s\n", i, suffixes[i].first.c_str());
          tries_.back().insert(suffixes[i].first, 0, suffix_inserted_len[i]);
          inserted_len[suffixes[i].second] += suffix_inserted_len[i];
        }
      }
    }

    for (auto &trie : tries_) {
      trie.space_cost_all_nodes(l_fixed);
      trie.build_actual_CoCo_children();
    }
  }

  auto get_trie(int i) -> BaseTrie * {
    return &tries_[i];
  }

  auto num_tries() const -> uint32_t {
    return tries_.size();
  }
};
