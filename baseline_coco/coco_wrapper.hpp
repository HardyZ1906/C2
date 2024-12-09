#define BIG_ALPHABET

#include "utils.hpp"
#include "uncompacted_trie.hpp"
#include "CoCo-trie_v2.hpp"

#include <vector>
#include <string>


class CoCoWrapper {  // unified API
 public:
  using trie_t = CoCo_v2<>;

  CoCoWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
              uint32_t pattern_len = 0, uint32_t min_occur = 0)
              : trie_([&keys]() {
                  datasetStats ds = dataset_stats_from_vector(keys);
                  MIN_CHAR = ds.get_min_char();
                  ALPHABET_SIZE = ds.get_alphabet_size();
                  return CoCo_v2<>(keys);
                }()) {}

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.look_up(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }
 private:
  trie_t trie_;
};