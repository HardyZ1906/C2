#include "include/fst_cc.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"
#include "include/basic_strpool.hpp"
#include "include/repair_strpool.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>


// #define __TEST_UNCOMPACTED__


int main(int argc, char *argv[]) {
  assert(argc > 1);

  uint32_t space_relaxation = argc >= 3 ? std::atoi(argv[2]) : 0;

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
  std::sort(keys.begin(), keys.end());
  printf("processed dataset\n");

  FstCC<std::string, BasicStringPool<std::string>> trie;
  trie.build(keys.begin(), keys.end());
  printf("built uncompacted trie\n");

  std::shuffle(keys.begin(), keys.end(), std::mt19937{1});
  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  std::unordered_set<uint32_t> key_ids;
#ifdef __TEST_UNCOMPACTED__
  printf("test uncompacted trie\n");
  int counter = 0;
  for (size_t i = 0; i < keys.size(); i++) {
    printf("get %d: %s\n", counter++, keys[i].c_str());
    volatile uint32_t key_id = trie.lookup(keys[i]);
    uint32_t id = key_id;
    assert(id != -1);
    assert(id < keys.size());
    assert(key_ids.count(id) == 0);
    key_ids.insert(id);
  }
  printf("[PASSED]\n");
  key_ids.clear();
#endif

  CoCoOptimizer<std::string> optimizer(&trie);
  optimizer.optimize(space_relaxation);
  printf("optimized trie\n");
#ifdef __DEBUG_OPTIMIZER__
  optimizer.print_optimal();
#endif
  CoCoCC<std::string, RepairStringPool<std::string>> coco(optimizer);

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

  printf("test coco\n");
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
    }
  }
  printf("[PASSED]\n");
}