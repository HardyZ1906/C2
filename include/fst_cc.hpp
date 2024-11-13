#pragma once

#include "utils.hpp"
#include "fst_builder.hpp"

#include <vector>
#include <limits>

#ifdef __MICROBENCHMARK__
# define BENCHMARK(foo) foo
# include <chrono>
#else
# define BENCHMARK(foo)
#endif


// similar to FST, but with StaticBitVector
template<typename Key, int cutoff = 0, int fanout = 256>
class FstCC {
 public:
  using key_type = Key;

  using bitvec = StaticBitVector;
  using bv_builder = BitVectorBuilder;
  using label_vec = StaticVector<uint8_t>;

  static constexpr int cutoff_ = cutoff;
  static constexpr int fanout_ = fanout;
  static constexpr uint8_t terminator_ = 0;
  static constexpr size_t invalid_offset_ = std::numeric_limits<size_t>::max();

  template<typename Iterator>
  void build(Iterator begin, Iterator end) {
    BENCHMARK( auto start_time = std::chrono::high_resolution_clock::now(); )

    Builder builder;
    builder.build(begin, end, overflow_);
    overflow_.shrink_to_fit();

    BENCHMARK(
      auto end_time = std::chrono::high_resolution_clock::now();
      builder_init_time_ += (end_time - start_time).count();
    )

    BENCHMARK( start_time = std::chrono::high_resolution_clock::now(); )

    louds_dense_.build(builder);
    louds_sparse_.build(builder);
    if constexpr (cutoff_ > 0) {
      louds_sparse_.offset_ = louds_dense_.has_child_.rank1() - louds_dense_.has_child_.size() / fanout_;
    } else {
      louds_sparse_.offset_ = 0;
    }

    BENCHMARK(
      end_time = std::chrono::high_resolution_clock::now();
      louds_build_time_ += (end_time - start_time).count();
    )

  #ifdef __MICROBENCHMARK__
    print_memory_usage();
  #endif
  }

  auto depth() const -> uint16_t {
    return louds_dense_.depth_ + louds_sparse_.depth_;
  }

  auto depth_d() const -> uint16_t {
    return louds_dense_.depth_;
  }

  auto depth_s() const -> uint16_t {
    return louds_sparse_.depth_;
  }

  auto get(const key_type &key) const -> value_type {
    value_type value;
    get(key, value);
    return value;
  }

  auto get(const key_type &key, value_type &value) const -> bool {
    size_t offset = 0;

    if constexpr (cutoff_ > 0) {
      if (louds_dense_.get(key, value, offset, overflow_)) {
        return true;
      } else if (offset == invalid_offset_) {
        return false;
      }
    }
    return louds_sparse_.get(key, value, offset - louds_dense_.labels_.size() / fanout_, overflow_);
  }

  auto begin() const -> iterator {
    iterator iter(this);
    iter.move_to_min_key();
    return iter;
  }

  auto end() const -> iterator {
    iterator iter(this);
    iter.invalidate();
    return iter;
  }

#ifdef __DEBUG__
  void print() const {
    printf("[LOUDS DENSE]\n");
    louds_dense_.print(overflow_);
    printf("[LOUDS SPARSE]\n");
    louds_sparse_.print(overflow_);
  }
#endif

#ifdef __MICROBENCHMARK__
  static void print_microbenchmark() {
    printf("[STATIC FST2 WITH SUFFIX COMPRESSION MICROBENCHMARK]\n");
    printf("init builder: %lf ms; build louds: %lf ms\n",
           (double)builder_init_time_/1000000, (double)louds_build_time_/1000000);
    bitvec::print_microbenchmark();
  }

  static void clear_microbenchmark() {
    bitvec::clear_microbenchmark();
  }

  void print_memory_usage() {
    size_t suffix_size = ((louds_dense_.elts_.size() + louds_sparse_.elts_.size()) * sizeof(Suffix) + overflow_.size()) * 8;
    size_t bv_size = (louds_dense_.labels_.size() + louds_dense_.has_child_.size() + louds_dense_.is_prefix_key_.size()) +
                     (louds_sparse_.has_child_.size() + louds_sparse_.louds_.size()) + louds_sparse_.labels_.size() * 8;
    printf("bit vector size: %ld bits, suffix size: %ld bits\n", bv_size, suffix_size);
  }
#endif
 private:
  LoudsDense louds_dense_;
  LoudsSparse louds_sparse_;

  // overflow suffix bytes
  std::vector<uint8_t> overflow_;

  friend class iterator;

#ifdef __MICROBENCHMARK__
  static size_t builder_init_time_;
  static size_t louds_build_time_;
#endif

#ifdef __DEBUG__
  friend class FstCCTest;
#endif
};

#ifdef __MICROBENCHMARK__
# define FST_TMPL_ARGS template<typename Key, int cutoff, int fanout>
# define FST_TMPL FstCC<Key, Value, cutoff>

FST_TMPL_ARGS size_t FST_TMPL::builder_init_time_ = 0;
FST_TMPL_ARGS size_t FST_TMPL::louds_build_time_ = 0;

# undef FST_TMPL_ARGS
# undef FST_TMPL
#endif

#ifdef __DEBUG__

#include <map>
#include <fstream>

class FstCCTest {
 public:
  static void test_all() {
    // test_small();
    test_large("words.txt");
  }

  static void test_small() {
    std::vector<std::pair<std::string, uint32_t>> kvs{
      {"f", {0}}, {"far", {1}}, {"fas", {2}}, {"fast", {3}}, {"fat", {4}},
      {"s", {5}}, {"top", {6}}, {"toy", {7}}, {"trie", {8}}, {"trip", {9}},
      {"try", {10}},
    };
    FstCC<std::string, uint32_t> fst;

    printf("[TEST BUILD]\n");
    fst.build<decltype(kvs)::iterator>(kvs.begin(), kvs.end());
    printf("[PASSED]\n");
    // fst.print();

    printf("[TEST GET]\n");
    for (const auto &kv : kvs) {
      // printf("get %s\n", kv.first.c_str());
      auto value1 = kv.second;
      auto value2 = fst.get(kv.first);
      EXPECT(value1, value2, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST ITERATOR]\n");
    auto iter1 = kvs.begin(), end1 = kvs.end();
    auto iter2 = fst.begin(), end2 = fst.end();
    while (iter1 != end1) {
      auto key1 = iter1->first;
      auto key2 = iter2.key();
      assert(key1 == key2);

      auto value1 = iter1->second;
      auto value2 = iter2.value();
      assert(value1 == value2);

      ++iter1;
      ++iter2;
    }
    assert(iter2 == end2);
    printf("[PASSED]\n");
  }

  static void test_large(const std::string &filename) {
    std::ifstream file(filename);
    size_t size = 1000000;

    std::map<std::string, uint32_t> map;
    FstCC<std::string, uint32_t> fst;

    std::string key;
    for (size_t i = 0; i < size; i++) {
      if (!file.good()) {
        break;
      }
      file >> key;
      map[key] = i;
    }

    printf("[TEST BUILD]\n");
    fst.build<decltype(map)::iterator>(map.begin(), map.end());
    printf("[PASSED]\n");
    // fst.print();

    printf("[TEST READ]\n");
    for (auto i = map.begin(); i != map.end(); ++i) {
      // printf("get %s\n", i->first.c_str());
      auto value1 = i->second;
      uint32_t value2;
      bool found = fst.get(i->first, value2);
      assert(found);
      EXPECT(value1, value2, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST ITERATOR]\n");
    auto iter1 = map.begin(), end1 = map.end();
    auto iter2 = fst.begin(), end2 = fst.end();
    // int count = 0;
    while (iter1 != end1) {
      auto key1 = iter1->first;
      // printf("%d:%s\n", count++, key1.c_str());
      auto key2 = iter2.key();
      assert(key1 == key2);

      auto value1 = iter1->second;
      auto value2 = iter2.value();
      assert(value1 == value2);

      ++iter1;
      ++iter2;
    }
    assert(iter2 == end2);
    printf("[PASSED]\n");
  }
};
#endif

#undef BENCHMARK