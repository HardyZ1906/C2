#include "include/fst_cc.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"
#include "include/marisa_cc.hpp"
#include "baseline_pdt/pdt_wrapper.hpp"
#include "baseline_coco/coco_wrapper.hpp"
#include "baseline_fst/fst_wrapper.hpp"
#include "baseline_marisa/marisa_wrapper.hpp"
#include "baseline_art/art_wrapper.hpp"
#include "baseline_art/cart_wrapper.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>


// #define __CORRECTNESS_TEST__


class FstCCWrapper {  // unified API
 public:
  using trie_t = FstCC<std::string>;

  FstCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
               int max_recursion = 0, int mask = 0) {
    trie_.build(keys.begin(), keys.end(), true, max_recursion, mask);
  }

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }
 private:
  trie_t trie_;
};

class CoCoCCWrapper {  // unified API
 public:
  using trie_t = CoCoCC<std::string>;

  CoCoCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
                int max_recursion = 0, int mask = 0)
                : trie_(keys.begin(), keys.end(), true, space_relaxation, max_recursion, mask) {}

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }
 private:
  trie_t trie_;
};

class MarisaCCWrapper {  // unified API
 public:
  using trie_t = MarisaCC<std::string>;

  MarisaCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
                  int max_recursion = 0, int mask = 0) {
    trie_.build(keys.begin(), keys.end(), true, max_recursion, mask);
  }

  auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }
 private:
  trie_t trie_;
};

template <typename trie_t>
void __attribute__((noinline)) query_trie(const std::vector<std::string> &keys, const trie_t &trie,
                                          const std::unordered_set<std::string> positive_queries) {
#ifdef __CORRECTNESS_TEST__
  std::unordered_set<uint32_t> key_ids;
#endif
  for (uint32_t i = 0; i < keys.size(); i++) {
  #ifdef __CORRECTNESS_TEST__
    bool positive = positive_queries.count(keys[i]);
    // printf("lookup %d: %s %s\n", i, keys[i].c_str(), positive ? "(positive)" : "(negative)");
  #endif
    volatile uint32_t key_id = trie.lookup(keys[i]);
  #ifdef __CORRECTNESS_TEST__
    uint32_t id = key_id;
    // printf("id = %d\n", id);
    if (positive) {
      EXPECT(id != -1);
      if constexpr (std::is_same_v<trie_t, FstCCWrapper> || std::is_same_v<trie_t, CoCoCCWrapper> ||
                    std::is_same_v<trie_t, MarisaCCWrapper>) {  // key IDs must be in range [0, n-1] and unique
        EXPECT(id < keys.size());
        EXPECT(key_ids.count(id) == 0);
        key_ids.insert(id);
      }
    } else {
      EXPECT(id == -1);
    }
  #endif
  }
}

template <typename trie_t>
void __attribute__((noinline)) test_trie(const char *filename, uint32_t space_relaxation, int max_recursion,
                                         int mask, uint32_t positive_percentage) {
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
  size_t query_size = keys.size() * positive_percentage / 100;
  std::vector<std::string> trie_keys(keys.begin(), keys.begin() + query_size);
  std::unordered_set<std::string> trie_key_set(trie_keys.begin(), trie_keys.end());
  std::sort(trie_keys.begin(), trie_keys.end());
  printf("Done!\n");

  printf("Building trie...\n");
  auto start = std::chrono::high_resolution_clock::now();
  trie_t trie(trie_keys, space_relaxation, max_recursion, mask);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  printf("Done!\n");
  printf("build time: %lf ms\n", (double)duration/1000000);

  size_t space_cost = trie.space_cost();
  printf("space cost: %lf MB\n", (double)space_cost / mb_bits);

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  printf("Querying trie...\n");
  start = std::chrono::high_resolution_clock::now();
  query_trie<trie_t>(keys, trie, trie_key_set);
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("Done!\n");
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, (double)duration/keys.size());
  trie_t::print_bench();
  printf("[PASSED]\n");
}

void benchmark_bv() {
  BitVector bv;
  static constexpr int size = 1000000;
  uint64_t bits[size];
  std::mt19937 gen{1};
  std::uniform_int_distribution<uint64_t> dist;
  for (int i = 0; i < size; i++) {
    bits[i] = dist(gen);
  }
  bv.load_bits(bits, 0, 1000000*8);
  bv.build();

  std::vector<int> queries(size);
  for (int i = 0; i < size; i++) {
    queries[i] = i;
  }
  std::shuffle(queries.begin(), queries.end(), std::mt19937{2});

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : queries) {
    volatile auto j = bv.rank1(i);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  printf("%ld ranks take %lf ms\n", (double)duration/1000000);

  start = std::chrono::high_resolution_clock::now();
  for (auto i : queries) {
    volatile auto j = bv.get(i);
  }
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("%ld gets take %lf ms\n", (double)duration/1000000);

  start = std::chrono::high_resolution_clock::now();
  for (auto i : queries) {
    asm("");
  }
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("%ld empty loops take %lf ms\n", (double)duration/1000000);
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);

  int choice = argc >= 3 ? std::atoi(argv[2]) : 0;
  uint32_t space_relaxation = argc >= 4 ? std::atoi(argv[3]) : 0;
  int max_recursion = argc >= 5 ? std::atoi(argv[4]) : 0;
  int mask = argc >= 6 ? std::atoi(argv[5]) : 0;
  uint32_t positive_percentage = argc >= 7 ? std::atoi(argv[6]) : 100;
  positive_percentage = std::min(positive_percentage, 100u);
  positive_percentage = std::max(positive_percentage, 10u);

  switch (choice) {
   case 0:
    printf("[TEST FST CC]\n");
    test_trie<FstCCWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 1:
    printf("[TEST COCO CC]\n");
    test_trie<CoCoCCWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 2:
    printf("[TEST MARISA CC]\n");
    test_trie<MarisaCCWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 3:
    printf("[TEST FST]\n");
    test_trie<FstWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 4:
    printf("[TEST COCO]\n");
    test_trie<CoCoWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 5:
    printf("[TEST MARISA]\n");
    test_trie<MarisaWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 6:
    printf("[TEST PDT]\n");
    test_trie<PdtWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 7:
    printf("[TEST ART]\n");
    test_trie<ArtWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   case 8:
    printf("[TEST CART]\n");
    test_trie<CArtWrapper>(argv[1], space_relaxation, max_recursion, mask, positive_percentage);
    break;
   default:
    printf("unrecognized index; stopped\n");
  }
}
