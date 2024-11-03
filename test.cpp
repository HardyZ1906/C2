#include "include/ls4coco.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <random>


int main(int argc, char *argv[]) {
  assert(argc > 1);

  uint32_t space_relaxation = argc == 3 ? std::atoi(argv[2]) : 0;

  std::string filename(argv[1]);
  std::ifstream file(filename);
  std::vector<std::string> keys;
  std::string key;
  while (file >> key) {
    keys.emplace_back(key);
  }
  std::sort(keys.begin(), keys.end());
  auto new_end = std::unique(keys.begin(), keys.end());
  keys.erase(new_end, keys.end());

  LS4CoCo<std::string> trie;
  trie.build(keys.begin(), keys.end());

  auto start = std::chrono::high_resolution_clock::now();
  CoCoOptimizer<std::string> optimizer(&trie);
  optimizer.optimize(space_relaxation);
#ifdef __DEBUG_OPTIMIZER__
  optimizer.print_optimal();
#endif
  CoCoCC<std::string> coco(optimizer);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = (end - start).count();
  printf("build time: %lf ms\n", (double)duration/1000000);

  constexpr int mb_bits = 1024*1024*8;
  auto [enc_cost, total_cost] = optimizer.get_final_cost();
  printf("expected encoding cost: %lf MB, expected total cost: %lf MB\n", (double)enc_cost/mb_bits, (double)total_cost/mb_bits);
  printf("actual encoding cost: %lf MB, actual total cost: %lf MB\n", (double)coco.encoding_size()/mb_bits,
         (double)coco.size_in_bits()/mb_bits);

  std::shuffle(keys.begin(), keys.end(), std::mt19937{2});
  std::set<uint32_t> key_ids;
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < keys.size(); i += 1000) {
    for (size_t j = i; j < keys.size() && j < i + 1000; j++) {
      // printf("%ld: get %s", j, keys[j].c_str());
      auto key_id = coco.lookup(keys[j]);
      // printf(", id = %d\n", key_id);
      // assert(key_id != -1);
      // assert(key_id < keys.size());
      // assert(key_ids.count(key_id) == 0);
      // key_ids.insert(key_id);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = (end - start).count();
  printf("total time: %lf ms, avg latency: %lf ns\n", (double)duration/1000000, (double)duration/keys.size());
  printf("[PASSED]\n");
}