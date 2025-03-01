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


#define __CORRECTNESS_TEST__


class FstCCWrapper {  // unified API
 public:
  using trie_t = FstCC<std::string>;

  FstCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
               int max_recursion = 0, int mask = 0) {
    trie_.build(keys.begin(), keys.end(), true, max_recursion, mask);
  }

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }

  void print_space_cost_breakdown() const {
    trie_.print_space_cost_breakdown();
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

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }

  void print_space_cost_breakdown() const {
    trie_.print_space_cost_breakdown();
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

  __NOINLINE_IF_PROFILE auto lookup(const std::string &key) const -> uint32_t {
    return trie_.lookup(key);
  }

  auto space_cost() const -> size_t {
    return trie_.size_in_bits();
  }

  static void print_bench() {
    trie_t::print_bench();
  }

  void print_space_cost_breakdown() const {
    trie_.print_space_cost_breakdown();
  }
 private:
  trie_t trie_;
};

template <typename trie_t>
void __attribute__((noinline)) query_trie(const std::vector<std::string> &keys, const trie_t &trie) {
#ifdef __CORRECTNESS_TEST__
  std::unordered_set<uint32_t> key_ids;
#endif
  for (uint32_t i = 0; i < keys.size(); i++) {
    volatile uint32_t key_id = trie.lookup(keys[i]);
  #ifdef __CORRECTNESS_TEST__
    uint32_t id = key_id;
    printf("id = %d\n", id);
    EXPECT(id != -1);
    if constexpr (std::is_same_v<trie_t, FstCCWrapper> || std::is_same_v<trie_t, CoCoCCWrapper> ||
                  std::is_same_v<trie_t, MarisaCCWrapper>) {  // key IDs must be in range [0, n-1] and unique
      EXPECT(id < keys.size());
      EXPECT(key_ids.count(id) == 0);
      key_ids.insert(id);
    }
  #endif
  }
}

template <typename trie_t>
void __attribute__((noinline)) test_trie(const char *filename, uint32_t space_relaxation, int max_recursion, int mask) {
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
  printf("Done!\n");

  printf("Building trie...\n");
  auto start = std::chrono::high_resolution_clock::now();
  trie_t trie(keys, space_relaxation, max_recursion, mask);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  double build_time = (double)duration/1000000;
  printf("Done!\n");
  printf("build time: %lf ms\n", build_time);

  size_t space_cost = trie.space_cost();
  double size_in_mb = (double)space_cost/mb_bits;
  printf("space cost: %lf MB\n", size_in_mb);

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  printf("Querying trie...\n");
  start = std::chrono::high_resolution_clock::now();
  query_trie<trie_t>(keys, trie);
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  double avg_latency = (double)duration/keys.size();
  printf("Done!\n");
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, avg_latency);
  trie_t::print_bench();
  trie.print_space_cost_breakdown();

  printf("%lf,%lf,%lf\n", build_time, size_in_mb, avg_latency);
  printf("[PASSED]\n");
}

void test_repair(const std::string &filename, size_t trim) {
  std::ifstream file(filename);
  std::vector<std::string> keys;
  std::string line;
  while (std::getline(file, line)) {
    keys.emplace_back(line);
  }

  size_t size_before = 0, count = 0;
  std::vector<uint8_t> concat;
  for (const auto &key : keys) {
    if (key.size() > trim) {
      size_before += key.size() - trim;
      concat.insert(concat.end(), key.begin() + trim, key.end());
      concat.insert(concat.end(), terminator_);
      count++;
    }
  }
  double avg_key_len = 1.*size_before / count;
  size_before *= 8;
  RepairStringPool<std::string> pool;
  pool.build(concat);
  size_t size_after = pool.size_in_bits();
  printf("avg key length: %lf, size before: %lf MB, size after: %lf MB\n", avg_key_len,
         (double)size_before/mb_bits, (double)size_after/mb_bits);
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

  switch (choice) {
   case 0:
    printf("[TEST FST CC]\n");
    test_trie<FstCCWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 1:
    printf("[TEST COCO CC]\n");
    test_trie<CoCoCCWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 2:
    printf("[TEST MARISA CC]\n");
    test_trie<MarisaCCWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 3:
    printf("[TEST FST]\n");
    test_trie<FstWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 4:
    printf("[TEST COCO]\n");
    test_trie<CoCoWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 5:
    printf("[TEST MARISA]\n");
    test_trie<MarisaWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 6:
    printf("[TEST PDT]\n");
    test_trie<PdtWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 7:
    printf("[TEST ART]\n");
    test_trie<ArtWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 8:
    printf("[TEST CART]\n");
    test_trie<CArtWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 9:
    printf("[TEST REPAIR]\n");
    test_repair(argv[1], space_relaxation);
    break;
   default:
    printf("unrecognized index; stopped\n");
  }
}
