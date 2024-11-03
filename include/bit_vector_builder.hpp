#pragma once

#include "utils.hpp"
#include <vector>
#include <cassert>


class BitVectorBuilder {
 public:
  std::vector<uint64_t> bits_;

  size_t size_;

  BitVectorBuilder() = default;

  auto bits() const -> const uint64_t * {
    return bits_.data();
  }

  auto size() const -> size_t {
    return size_;
  }

  void reserve(size_t size) {
    size_t leftover = (size_ + 63) / 64 * 64 - size_;
    if (size <= leftover) {
      size_ += size;
      return;
    }

    size -= leftover;
    size_ += leftover;
    while (size > 64) {
      bits_.emplace_back(0);
      size -= 64;
      size_ += 64;
    }
    if (size > 0) {
      bits_.emplace_back(0);
      size_ += size;
    }
  }

  void append1() {
    if (size_ == bits_.size() * 64) {
      bits_.emplace_back(1);
    } else {
      SET_BIT(bits_.back(), size_ % 64);
    }
    size_++;
  }

  void append0() {
    if (size_ == bits_.size() * 64) {
      bits_.emplace_back(0);
    }  // otherwise unused bits are 0 by default
    size_++;
  }

  auto back() const -> bool {
    return get_rev(0);
  }

  auto get_rev(size_t pos) const -> bool {
    assert(pos < size_);
    size_t bit_idx = size_ - pos - 1;
    return GET_BIT(bits_[bit_idx / 64], bit_idx % 64);
  }

  auto set1_rev(size_t pos) -> bool {
    assert(pos < size_);
    size_t bit_idx = size_ - pos - 1;
    return SET_BIT(bits_[bit_idx / 64], bit_idx % 64);
  }

  auto set0_rev(size_t pos) -> bool {
    assert(pos < size_);
    size_t bit_idx = size_ - pos - 1;
    return CLEAR_BIT(bits_[bit_idx / 64], bit_idx % 64);
  }
};