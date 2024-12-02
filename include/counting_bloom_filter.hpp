#pragma once

#include "../lib/wyhash/wyhash.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>


template<typename ElementType>
class ArrayCounter {
 public:
  using elt_t = ElementType;

  ArrayCounter() = default;

  ArrayCounter(size_t size) : size_(size) {
    assert(size > 0);
    elts_ = new elt_t[size];
    std::memset(elts_, 0, sizeof(elt_t)*size_);
  }

  ~ArrayCounter() {
    delete[] elts_;
  }

  void resize(size_t size) {
    delete[] elts_;
    elts_ = new elt_t[size];
    std::memset(elts_, 0, sizeof(elt_t)*size_);
    size_ = size;
  }

  void increment(size_t pos) {
    assert(pos < size_);
    if (elts_[pos] != std::numeric_limits<elt_t>::max()) {
      ++elts_[pos];
    }
  }

  auto get(size_t pos) const -> elt_t {
    assert(pos < size_);
    return elts_[pos];
  }

  auto size() const -> size_t {
    return size_;
  }
 private:
  elt_t *elts_{nullptr};
  size_t size_{0};

  template<typename E, typename C> friend class CountingBloomFilter;
};


template<typename ElementType = uint8_t, typename Counter = ArrayCounter<ElementType>>
class CountingBloomFilter {
 public:
  using counter_type = Counter;
  using elt_t = ElementType;

  CountingBloomFilter() = default;

  CountingBloomFilter(size_t n, size_t k = 4) : counter_(n), k_(k) {
    assert(k <= 16);
  }

  void resize(size_t n) {
    counter_.resize(n);
  }

  auto insert(const void *key, size_t size) -> elt_t {
    elt_t ret = std::numeric_limits<elt_t>::max();
    for (size_t i = 0; i < k_; i++) {  // increment all k counters
      // TODO: maybe handle collision
      size_t pos = wyhash(key, size, seeds_[i], _wyp) % counter_.size();
      counter_.increment(pos);
      ret = std::min(ret, counter_.get(pos));
    }
    return ret;
  }

  auto count(const void *key, size_t size) const -> elt_t {
    elt_t ret = std::numeric_limits<elt_t>::max();
    for (size_t i = 0; i < k_; i++) {  // return the minimum of the k counters
      size_t pos = wyhash(key, size, seeds_[i], _wyp) % counter_.size();
      ret = std::min(ret, counter_.get(pos));
    }
    return ret;
  }
 private:
  static constexpr uint64_t seeds_[16] {
    0x07573FF16174F6FBull, 0x0D4BB65C481927D2ull, 0x74128F9D939FE7BCull, 0xAD801EC1EE497255ull,
    0x06213A55699611DDull, 0x5CE416D81888B79Bull, 0xFB6C7222FC149A98ull, 0x931FAA76DBB3606Aull,
    0x424A9A6930691CEDull, 0xE1C43E6A4CB5F384ull, 0xF820D100EE9BB35Aull, 0x4931D80D7C26FBC6ull,
    0x71DB8A80B6AE8CBFull, 0x3CA7852D9018A59Cull, 0x7B0F6C30F102C8FEull, 0x287964BCB5E3359Aull,
  };
  counter_type counter_;
  size_t k_{4};  // number of hash functions
};