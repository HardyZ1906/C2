#include "include/fst_cc.hpp"
#include "include/coco_optimizer.hpp"
#include "include/coco_cc.hpp"
#include "include/marisa_cc.hpp"
#include "include/louds_cc.hpp"
#include "include/louds_sparse_cc.hpp"
#include "include/louds_sux.hpp"
#include "include/louds_marisa.hpp"

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

  __NOINLINE_IF_PROFILE FstCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
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

  __NOINLINE_IF_PROFILE CoCoCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
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

class CoCoLSWrapper {  // unified API
 public:
  using trie_t = CoCoCC<std::string, LoudsSparseCC>;

  __NOINLINE_IF_PROFILE CoCoLSWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
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

class CoCoSuxWrapper {  // unified API
 public:
  using trie_t = CoCoCC<std::string, LoudsSux<>>;

  __NOINLINE_IF_PROFILE CoCoSuxWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
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

  __NOINLINE_IF_PROFILE MarisaCCWrapper(const std::vector<std::string> &keys, uint32_t space_relaxation = 0,
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
    printf("%d:%s, id = %d\n", i, keys[i].c_str(), id);
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
  size_t original_size = 0;
  while (std::getline(file, key)) {
    keys.emplace_back(key);
    original_size += key.size();
  }
  double original_size_in_mb = (double)original_size/mb_bytes;
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
  printf("build time: %lf ms (%lf ns per key)\n", build_time, (double)duration/keys.size());

  size_t space_cost = trie.space_cost();
  double size_in_mb = (double)space_cost/mb_bits;
  printf("space cost: %lf MB (%lf%% of original size %lf MB)\n", size_in_mb,
         size_in_mb/original_size_in_mb*100, original_size_in_mb);

#ifdef __PROFILE__
  size_t min_queries = 10000000, n = keys.size();  // make the test loop last longer for more accurate results
  while (keys.size() < min_queries) {
    for (size_t j = 0; j < n; j++) {  // replicate original dataset
      keys.push_back(keys[j]);
    }
  }
  printf("replicated dataset size: %ld\n", keys.size());
#endif
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
  auto start = std::chrono::high_resolution_clock::now();
  RepairStringPool<std::string> pool;
  pool.build(concat);
  auto end = std::chrono::high_resolution_clock::now();
  size_t size_after = pool.size_in_bits();
  printf("build time: %lf ms, avg key length: %lf, size before: %lf MB, size after: %lf MB\n",
         (double)(end - start).count()/1000000, avg_key_len, (double)size_before/mb_bits, (double)size_after/mb_bits);
}

template <typename T>
void test_compression(const std::string &filename, bool sort_rev = false) {
  std::ifstream file(filename);
  std::vector<std::string> keys;
  KeySet<std::string> key_set;
  std::string line;

  while (std::getline(file, line)) {
    keys.emplace_back(std::move(line));
  }
  printf("Processing dataset...\n");
  for (const auto &key : keys) {
    key_set.emplace_back(&key);
  }
  if (sort_rev) {
    auto t0 = std::chrono::high_resolution_clock::now();
    key_set.reverse();
    key_set.sort();
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("sort time: %lf ms\n", (double)(t1 - t0).count()/1000000);
  }
  printf("Done!\n");

  printf("Compressing...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  T dict;
  dict.build(key_set);
  auto t1 = std::chrono::high_resolution_clock::now();
  double original_size = (double)key_set.space_cost()/mb_bytes;
  double compressed_size = (double)dict.size_in_bytes()/mb_bytes;
  printf("Done!\n");
  printf("build time: %lf ms\n", (double)(t1 - t0).count()/1000000);
  printf("original size: %lf MB, compressed size: %lf MB, compression ratio: %lf%%\n",
         original_size, compressed_size, 100*compressed_size/original_size);

  std::vector<bool> sampled(key_set.size(), false);
  KeySet<std::string> sample;
  size_t sample_size = 8*1024*1024, cur_size = 0;
  double original_sample_size, compressed_sample_size;
  if (key_set.space_cost() <= sample_size) {
    original_sample_size = key_set.space_cost();
  } else {
    printf("Creating sample...\n");
    while (cur_size < sample_size) {
      size_t i = rand() % key_set.size();
      if (!sampled[i]) {
        sampled[i] = true;
        sample.push_back(key_set[i], true);
        cur_size += key_set[i].size();
      }
    }
    if (sort_rev) {
      sample.reverse();
      sample.sort();
    }
    printf("Done!\n");

    printf("Compressing sample...\n");
    T dict_sample;
    dict_sample.build(sample);
    original_sample_size = (double)sample.space_cost() / mb_bytes;
    compressed_sample_size = (double)dict_sample.size_in_bytes() / mb_bytes;
    printf("Done!\n");
  }
  printf("sample size: %lf MB, compressed sample size: %lf MB, sample compression ratio: %lf%%\n",
         original_sample_size, compressed_sample_size, 100*compressed_sample_size/original_sample_size);
}

#ifdef __COMPARE_COCO__
void compare_louds_coco(const std::string &filename, uint32_t space_relaxation, int max_recursion, int mask) {
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
  CoCoCC<std::string> trie(keys.begin(), keys.end(), true, space_relaxation, max_recursion, mask);
  CoCoCC<std::string, LoudsSparseCC> trie_ls(keys.begin(), keys.end(), true, space_relaxation, max_recursion, mask);
  printf("Done!\n");

  printf("Converting to standard LOUDS...\n");
  std::unique_ptr<LoudsSux<>> louds;
  trie.to_louds_sux(louds);
  printf("Done!\n");

  printf("Comparing LOUDS performance...\n");
  auto topo = trie.get_topo();
  auto topo_ls = trie_ls.get_topo();
  uint32_t bv_size = topo->num_nodes() * 2 - 1;
  std::vector<uint32_t> leaves[2], internals[2], child_query[2];
  leaves[0].reserve(topo->num_leaves());
  leaves[1].reserve(topo->num_leaves());
  internals[0].reserve(topo->num_internals());
  internals[1].reserve(topo->num_internals());
  child_query[0].reserve(topo->num_nodes() - 1);
  child_query[1].reserve(topo_ls->num_children());
  for (uint32_t i = 0; i < bv_size; i++) {
    assert(topo->get(i) == louds->get(i));
    if (i == 0 || !topo->get(i - 1)) {
      if (topo->get(i)) {
        internals[0].push_back(i);
      } else {
        leaves[0].push_back(i);
      }
    }
    if (topo->get(i)) {
      child_query[0].push_back(i);
    }
  }

  for (uint32_t i = 0; i < topo_ls->size(); i++) {
    if (topo_ls->louds(i)) {
      internals[1].push_back(i);
    }
    if (topo_ls->has_child(i)) {
      child_query[1].push_back(i);
    } else {
      leaves[1].push_back(i);
    }
  }

  std::shuffle(leaves[0].begin(), leaves[0].end(), std::mt19937{1});
  std::shuffle(leaves[1].begin(), leaves[1].end(), std::mt19937{1});
  std::shuffle(internals[0].begin(), internals[0].end(), std::mt19937{2});
  std::shuffle(internals[1].begin(), internals[1].end(), std::mt19937{2});
  std::shuffle(child_query[0].begin(), child_query[0].end(), std::mt19937{3});
  std::shuffle(child_query[1].begin(), child_query[1].end(), std::mt19937{3});

  size_t leaf_id_time[3], internal_id_time[3], degree_time[3], child_pos_time[3];

  printf("[LEAF ID]...\n");
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : leaves[0]) {
    volatile auto res = topo->leaf_id(i);
  }
  auto end = std::chrono::high_resolution_clock::now();
  leaf_id_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : leaves[0]) {
    volatile auto res = louds->leaf_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  leaf_id_time[1] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : leaves[1]) {
    volatile auto res = topo_ls->leaf_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  leaf_id_time[2] = (end - start).count();

  printf("[INTERNAL ID]...\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[0]) {
    volatile auto res = topo->internal_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  internal_id_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[0]) {
    volatile auto res = louds->internal_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  internal_id_time[1] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[1]) {
    volatile auto res = topo_ls->node_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  internal_id_time[2] = (end - start).count();

  printf("[DEGREE]...\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[0]) {
    volatile auto res = topo->node_degree(i);
  }
  end = std::chrono::high_resolution_clock::now();
  degree_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[0]) {
    volatile auto res = louds->node_degree(i);
  }
  end = std::chrono::high_resolution_clock::now();
  degree_time[1] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : internals[1]) {
    volatile auto res = topo_ls->node_degree(i);
  }
  end = std::chrono::high_resolution_clock::now();
  degree_time[2] = (end - start).count();

  printf("[CHILD_POS]...\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : child_query[0]) {
    volatile auto res = topo->child_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  child_pos_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : child_query[0]) {
    volatile auto res = louds->child_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  child_pos_time[1] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : child_query[1]) {
    volatile auto res = topo_ls->child_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  child_pos_time[2] = (end - start).count();

  printf("LEAF_ID(ns): %lf vs %lf vs %lf\n", (double)leaf_id_time[0]/leaves[0].size(),
         (double)leaf_id_time[1]/leaves[0].size(), (double)leaf_id_time[2]/leaves[1].size());
  printf("INTERNAL_ID(ns): %lf vs %lf vs %lf\n", (double)internal_id_time[0]/internals[0].size(),
         (double)internal_id_time[1]/internals[0].size(), (double)internal_id_time[2]/internals[1].size());
  printf("DEGREE(ns): %lf vs %lf vs %lf\n", (double)degree_time[0]/internals[0].size(),
         (double)degree_time[1]/internals[0].size(), (double)degree_time[2]/internals[1].size());
  printf("CHILD_POS(ns): %lf vs %lf vs %lf\n", (double)child_pos_time[0]/child_query[0].size(),
         (double)child_pos_time[1]/child_query[0].size(), (double)child_pos_time[2]/child_query[1].size());

  printf("Done!\n");
}
#endif

#ifdef __COMPARE_MARISA__
void compare_louds_marisa(const std::string &filename, int max_recursion, int mask) {
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
  MarisaCC<std::string, false> trie;
  trie.build(keys.begin(), keys.end(), true, max_recursion, mask);
  printf("Done!\n");

  printf("Converting to standard LOUDS...\n");
  std::unique_ptr<LoudsMarisa> louds;
  trie.to_louds_marisa(louds);
  printf("Done!\n");

  printf("Comparing LOUDS performance...\n");
  auto topo = trie.get_topo();
  std::vector<uint32_t> bv_pos[2], link[2], term[2];
  std::vector<uint32_t> child_query[2], parent_query[2];
  bv_pos[0].reserve(topo->size());
  bv_pos[1].reserve(topo->size());
  link[0].reserve(topo->num_links());
  link[1].reserve(topo->num_links());
  term[0].reserve(topo->num_leaves());
  term[1].reserve(topo->num_leaves());
  child_query[0].reserve(topo->num_children());
  child_query[1].reserve(topo->num_children());
  parent_query[0].reserve(topo->num_nodes() - 1);
  parent_query[1].reserve(topo->num_nodes() - 1);
  for (uint32_t i = 0; i < topo->size(); i++) {
    bv_pos[0].push_back(i);
    bv_pos[1].push_back(i);
    if (topo->has_child(i)) {
      child_query[0].push_back(i);
    } else {
      term[0].push_back(i);
      term[1].push_back(i);
    }
    if (topo->louds(i) && topo->has_parent(i)) {
      parent_query[0].push_back(i);
    }
    if (topo->is_link(i)) {
      link[0].push_back(i);
      link[1].push_back(i);
    }
  }
  bool is_root = true;
  for (uint32_t i = 0; i < louds->size(); i++) {
    if (louds->louds(i)) {
      child_query[1].push_back(i);
    } else if (is_root) {
      is_root = false;
    } else {
      parent_query[1].push_back(i);
    }
  }

  std::shuffle(bv_pos[0].begin(), bv_pos[0].end(), std::mt19937{1});
  std::shuffle(bv_pos[1].begin(), bv_pos[1].end(), std::mt19937{1});
  std::shuffle(link[0].begin(), link[0].end(), std::mt19937{2});
  std::shuffle(link[1].begin(), link[1].end(), std::mt19937{2});
  std::shuffle(term[0].begin(), term[0].end(), std::mt19937{3});
  std::shuffle(term[1].begin(), term[1].end(), std::mt19937{3});
  std::shuffle(child_query[0].begin(), child_query[0].end(), std::mt19937{4});
  std::shuffle(child_query[1].begin(), child_query[1].end(), std::mt19937{4});
  std::shuffle(parent_query[0].begin(), parent_query[0].end(), std::mt19937{5});
  std::shuffle(parent_query[1].begin(), parent_query[1].end(), std::mt19937{5});

  printf("get: %ld vs %ld, link: %ld vs %ld, term: %ld vs %ld, child: %ld vs %ld, parent: %ld vs %ld\n",
         bv_pos[0].size(), bv_pos[1].size(), link[0].size(), link[1].size(), term[0].size(), term[1].size(),
         child_query[0].size(), child_query[1].size(), parent_query[0].size(), parent_query[1].size());

  size_t get_time[2], leaf_id_time[2], link_id_time[2];
  size_t child_pos_time[2], parent_pos_time[2];

  printf("[GET]...\n");
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : bv_pos[0]) {
    // topo->prefetch_block(i);
    volatile auto res0 = topo->is_link(i);
    volatile auto res1 = topo->has_child(i);
  }
  auto end = std::chrono::high_resolution_clock::now();
  get_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : bv_pos[1]) {
    volatile auto res0 = louds->is_link(i);
    volatile auto res1 = louds->is_term(i);
  }
  end = std::chrono::high_resolution_clock::now();
  get_time[1] = (end - start).count();

  printf("[LINK_ID]...\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : link[0]) {
    // topo->prefetch_block(i);
    volatile auto res = topo->link_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  link_id_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : link[1]) {
    volatile auto res = louds->link_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  link_id_time[1] = (end - start).count();

  printf("[LEAF_ID]...\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : term[0]) {
    // topo->prefetch_block(i);
    volatile auto res = topo->leaf_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  leaf_id_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : term[1]) {
    volatile auto res = louds->leaf_id(i);
  }
  end = std::chrono::high_resolution_clock::now();
  leaf_id_time[1] = (end - start).count();

  printf("[CHILD_POS]\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : child_query[0]) {
    // topo->prefetch_block(i);
    volatile auto res = topo->child_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  child_pos_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : child_query[1]) {
    volatile auto res = louds->child_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  child_pos_time[1] = (end - start).count();

  printf("[PARENT_POS]\n");
  start = std::chrono::high_resolution_clock::now();
  for (auto i : parent_query[0]) {
    // topo->prefetch_block(i);
    volatile auto res = topo->parent_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  parent_pos_time[0] = (end - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (auto i : parent_query[1]) {
    volatile auto res = louds->parent_pos(i);
  }
  end = std::chrono::high_resolution_clock::now();
  parent_pos_time[1] = (end - start).count();

  printf("GET(ns): %lf vs %lf\n", (double)get_time[0]/bv_pos[0].size(), (double)get_time[1]/bv_pos[0].size());
  printf("LINK_ID(ns): %lf vs %lf, LEAF_ID(ns): %lf vs %lf\n", (double)link_id_time[0]/link[1].size(),
         (double)link_id_time[1]/link[1].size(), (double)leaf_id_time[0]/term[0].size(), (double)leaf_id_time[1]/term[1].size());
  printf("CHILD_POS(ns): %lf vs %lf, PARENT_POS(ns): %lf vs %lf\n", (double)child_pos_time[0]/child_query[0].size(),
         (double)child_pos_time[1]/child_query[1].size(), (double)parent_pos_time[0]/parent_query[0].size(),
         (double)parent_pos_time[1]/parent_query[1].size());

  printf("Done!\n");
}
#endif

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
    printf("[TEST CoCo(LOUDS-Sparse)]\n");
    test_trie<CoCoLSWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 10:
    printf("[TEST CoCo(Sux)]\n");
    test_trie<CoCoSuxWrapper>(argv[1], space_relaxation, max_recursion, mask);
    break;
   case 11:
    printf("[TEST REPAIR]\n");
    test_repair(argv[1], space_relaxation);
    break;
  #ifdef __COMPARE_COCO__
   case 12:
    printf("[COMPARE LOUDS COCO]\n");
    compare_louds_coco(argv[1], space_relaxation, max_recursion, mask);
    break;
  #endif
  #ifdef __COMPARE_MARISA__
   case 13:
    printf("[COMPARE LOUDS MARISA]\n");
    compare_louds_marisa(argv[1], max_recursion, mask);
    break;
  #endif
   case 14:
    printf("[REPAIR]\n");
    test_compression<RepairStringPool<std::string>>(argv[1], false);
    break;
   case 15:
    printf("[FSST]\n");
    test_compression<FsstStringPool<std::string>>(argv[1], true);
    break;
   case 16:
    printf("[SORTED]\n");
    test_compression<SortedStringPool<std::string>>(argv[1], false);
    break;
   default:
    printf("unrecognized index; stopped\n");
  }
}
