#include "include/fst_cc.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"
#include "include/marisa_cc.hpp"
#include "include/repair_strpool.hpp"
#include "baseline_pdt/pdt_wrapper.hpp"
#include "baseline_coco/coco_wrapper.hpp"
#include "baseline_fst/fst_wrapper.hpp"
#include "baseline_marisa/marisa_wrapper.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>


// #define __DEBUG__


class FstCCWrapper {  // unified API
 public:
  using trie_t = FstCC<std::string>;

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
  trie_t trie_;
};

class CoCoCCWrapper {  // unified API
 public:
  using trie_t = CoCoCC<std::string>;

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
  trie_t trie_;
};

class MarisaCCWrapper {  // unified API
 public:
  using trie_t = MarisaCC<std::string, RepairStringPool<std::string>>;

  MarisaCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
                  uint32_t pattern_len = 0, uint32_t min_occur = 0) {
    trie_.build(keys.begin(), keys.end(), false, 3);
  }

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }
 private:
  trie_t trie_;
};

template <typename trie_t>
void __attribute__((noinline)) query_trie(const std::vector<std::string> &keys, const trie_t &trie) {
 #ifdef __DEBUG__
  std::unordered_set<uint32_t> key_ids;
 #endif
  // int counter = 0;
  for (const auto &key : keys) {
    // printf("lookup %d:%s\n", counter++, key.c_str());
    volatile uint32_t key_id = trie.lookup(key);
   #ifdef __DEBUG__
    uint32_t id = key_id;
    assert(id != -1);
    assert(id < keys.size());
    assert(key_ids.count(id) == 0);
    key_ids.insert(id);
   #endif
  }
}

template <typename trie_t>
void __attribute__((noinline)) test_trie(const char *filename, uint32_t space_relaxation, uint32_t pattern_len,
                                         uint32_t min_occur, uint32_t positive_percent) {
  printf("Processing dataset...\n");
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
  printf("Querying trie...\n");
  start = std::chrono::high_resolution_clock::now();
  query_trie<trie_t>(keys, trie);
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("Done!\n");
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, (double)duration/keys.size());
  printf("[PASSED]\n");
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);

  int choice = argc >= 3 ? std::atoi(argv[2]) : 0;
  uint32_t space_relaxation = argc >= 4 ? std::atoi(argv[3]) : 0;
  uint32_t pattern_len = argc >= 5 ? std::atoi(argv[4]) : 0;
  uint32_t min_occur = argc >= 6 ? std::atoi(argv[5]) : 0;
  uint32_t positive_percent = argc >= 7 ? std::atoi(argv[6]) : 100;
  positive_percent = std::min(positive_percent, 100u);
  positive_percent = std::max(positive_percent, 10u);

  switch (choice) {
   case 0:
    test_trie<FstCCWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 1:
    test_trie<CoCoCCWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 2:
    test_trie<MarisaCCWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 3:
    test_trie<FstWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 4:
    test_trie<CoCoWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 5:
    test_trie<MarisaWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
   case 6:
    test_trie<PdtWrapper>(argv[1], space_relaxation, pattern_len, min_occur, positive_percent);
    break;
  }
}

#undef __DEBUG__