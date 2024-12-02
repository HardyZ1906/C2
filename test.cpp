#include "include/ls4coco.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>


// #define __TEST_REV__


int main(int argc, char *argv[]) {
  assert(argc > 1);

  uint32_t space_relaxation = argc >= 3 ? std::atoi(argv[2]) : 0;
  uint32_t pattern_len = argc >= 4 ? std::atoi(argv[3]) : 0;
  uint32_t min_occur = argc >= 5 ? std::atoi(argv[4]) : 0;
  uint32_t positive_percent = argc >= 6 ? std::atoi(argv[5]) : 100;
  positive_percent = std::min(positive_percent, 100u);
  positive_percent = std::max(positive_percent, 10u);

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

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  size_t query_size = keys.size() * positive_percent / 100;
  std::sort(keys.begin(), keys.begin() + query_size);
  printf("processed dataset\n");

  auto start = std::chrono::high_resolution_clock::now();
  LS4CoCo<std::string> trie;
  trie.build(keys.begin(), keys.begin() + query_size);
  printf("built uncompacted trie\n");

  CoCoOptimizer<std::string> optimizer(&trie);
  optimizer.optimize(space_relaxation, pattern_len, min_occur);
  printf("optimized trie\n");
// #ifdef __DEBUG_OPTIMIZER__
//   optimizer.print_optimal();
// #endif
  // fflush(stdout);
  // exit(0);

#ifndef __TEST_REV__
  CoCoCC<std::string> coco(optimizer);
#else
  CoCoCC<std::string, true> coco(optimizer);
#endif
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  printf("build time: %lf ms\n", (double)duration/1000000);

  constexpr int mb_bits = 1024*1024*8;
  auto [enc_cost, total_cost] = optimizer.get_final_cost();
  auto [expected_macros, expected_leaves] = optimizer.get_num_nodes();
  auto [actual_macros, actual_leaves] = coco.get_num_nodes();
  printf("expected encoding cost: %lf MB, expected total cost: %lf MB\n", (double)enc_cost/mb_bits, (double)total_cost/mb_bits);
  printf("expected #macros: %d, expected #leaves: %d\n", expected_macros, expected_leaves);
  printf("actual encoding cost: %lf MB, actual total cost: %lf MB\n", (double)coco.encoding_size()/mb_bits,
         (double)coco.size_in_bits()/mb_bits);
  printf("actual #macros: %d, actual #leaves: %d\n", actual_macros, actual_leaves);

  // fflush(stdout);
  // exit(0);

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  std::unordered_set<uint32_t> key_ids;
  printf("test query\n");
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < keys.size(); i += 1000) {
    for (size_t j = i; j < keys.size() && j < i + 1000; j++) {
      // printf("%ld: get %s\n", j, keys[j].c_str());
      volatile uint32_t key_id = coco.lookup(keys[j]);
      uint32_t id = key_id;
      // printf("id = %d\n", id);
      assert(id != -1);
      assert(id < keys.size());
      assert(key_ids.count(id) == 0);
      key_ids.insert(id);

    #ifdef __TEST_REV__
      std::string rev = keys[j];
      std::reverse(rev.begin(), rev.end());
      auto len = coco.match_rev(rev, 0, id);
      assert(len == keys[j].size());
    #endif
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, (double)duration/keys.size());
  printf("[PASSED]\n");
}