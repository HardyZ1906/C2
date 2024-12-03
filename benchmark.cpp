#include "include/ls4coco.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"
#include "baseline_coco/coco_wrapper.hpp"
#include "baseline_fst/fst_wrapper.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>


class FstCCWrapper {
 public:
  FstCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
               uint32_t pattern_len = 0, uint32_t min_occur = 0) {
    trie_.build(keys.begin(), keys.end());
  }

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }
 private:
  LS4CoCo<std::string> trie_;
};

class CoCoCCWrapper {  // unified API
 public:
  CoCoCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
                uint32_t pattern_len = 0, uint32_t min_occur = 0)
                : trie_(keys.begin(), keys.end(), space_relaxation, pattern_len, min_occur) {}

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }
 private:
  CoCoCC<std::string> trie_;
};

// using trie_t = FstWrapper;
// using trie_t = FstCCWrapper;
// using trie_t = CoCoWrapper;
using trie_t = CoCoCCWrapper;


template<typename trie_t>
void __attribute__((noinline)) query_trie(const std::vector<std::string> &keys, const trie_t &trie) {
  for (const auto &key : keys) {
    volatile uint32_t key_id = trie.lookup(key);
  }
}

int main(int argc, char *argv[]) {
  assert(argc > 1);

  uint32_t space_relaxation = argc >= 3 ? std::atoi(argv[2]) : 0;
  uint32_t pattern_len = argc >= 4 ? std::atoi(argv[3]) : 0;
  uint32_t min_occur = argc >= 5 ? std::atoi(argv[4]) : 0;
  uint32_t positive_percent = argc >= 6 ? std::atoi(argv[5]) : 100;
  positive_percent = std::min(positive_percent, 100u);
  positive_percent = std::max(positive_percent, 10u);

  printf("Processing dataset...\n");
  std::string filename(argv[1]);
  std::ifstream file(filename);
  std::vector<std::string> keys;
  std::string key;
  while (std::getline(file, key)) {
    keys.emplace_back(key);
  }
  std::sort(keys.begin(), keys.end());
  auto new_end = std::unique(keys.begin(), keys.end());
  keys.erase(new_end, keys.end());

  std::shuffle(keys.begin(), keys.end(), std::mt19937{1});
  size_t query_size = keys.size() * positive_percent / 100;
  std::vector<std::string> trie_keys(keys.begin(), keys.begin() + query_size);
  std::sort(trie_keys.begin(), trie_keys.end());
  printf("Done!\n");

  printf("Building trie...\n");
  auto start = std::chrono::high_resolution_clock::now();
  trie_t trie(trie_keys, space_relaxation, pattern_len, min_occur);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  printf("Done!\n");
  printf("build time: %lf ms\n", (double)duration/1000000);

  constexpr int mb_bits = 1024*1024*8;
  size_t space_cost = trie.space_cost();
  printf("space cost: %lf MB\n", (double)space_cost / mb_bits);

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  std::unordered_set<uint32_t> key_ids;
  printf("Querying trie...\n");
  start = std::chrono::high_resolution_clock::now();
  query_trie<trie_t>(keys, trie);
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("Done!\n");
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, (double)duration/keys.size());
  printf("[PASSED]\n");
}