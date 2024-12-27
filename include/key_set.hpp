#pragma once

#include "utils.hpp"


template <typename Key>
struct KeySet {
  using key_type = Key;

  struct Fragment {
    const key_type *key_;
    uint32_t offset_;
    uint32_t length_;
    uint32_t id_;

    Fragment() = default;

    Fragment(const key_type *key, uint32_t offset, uint32_t length, uint32_t id)
        : key_(key), offset_(offset), length_(length), id_(id) {}

    Fragment(const key_type *key, uint32_t id)
        : key_(key), offset_(0), length_(key->size()), id_(id) {}

    auto get_label(uint32_t idx, bool reverse = false) const -> uint8_t {
      assert(idx < length_);
      return reverse ? (*key_)[offset_ + length_ - idx - 1] : (*key_)[offset_ + idx];
    }

    auto size() const -> uint32_t {
      return length_;
    }

    auto lt(const Fragment &rhs, bool reverse) const -> bool {
      int len = std::min(length_, rhs.length_);
      for (int i = 0; i < len; i++) {
        if (get_label(i, reverse) < rhs.get_label(i, reverse)) {
          return true;
        } else if (get_label(i, reverse) > rhs.get_label(i, reverse)) {
          return false;
        }
      }
      return length_ < rhs.length_;
    }

    auto materialize(bool reverse = false) const -> key_type {
      key_type ret = key_->substr(offset_, length_);
      if (reverse) {
        std::reverse(ret.begin(), ret.end());
      }
      return ret;
    }

    void append_to(std::vector<uint8_t> &vec, bool terminator = true) const {
      vec.insert(vec.end(), key_->begin() + offset_, key_->begin() + offset_ + length_);
      if (terminator) {
        vec.emplace_back(0);
      }
    }

    auto substr_range(uint32_t idx, uint32_t length, bool reverse) const -> std::pair<uint32_t, uint32_t> {
      assert(idx + length <= length_);
      if (reverse) {
        return std::make_pair(offset_ + length_ - idx - length, length);
      } else {
        return std::make_pair(offset_ + idx, length);
      }
    }
  };

  std::vector<Fragment> fragments_;
  size_t space_cost_{0};
  bool reverse_{false};

  KeySet() = default;

  KeySet(bool reverse) : reverse_(reverse) {}

  void emplace_back(const key_type *key) {
    fragments_.emplace_back(key, fragments_.size());
    space_cost_ += key->size();
  }

  void emplace_back(const key_type *key, uint32_t offset, uint32_t length) {
    assert(offset + length <= key->size());
    fragments_.emplace_back(key, offset, length, fragments_.size());
    space_cost_ += length;
  }

  auto operator[](int idx) const -> const Fragment & {
    return get(idx);
  }

  auto get(int idx) const -> const Fragment & {
    assert(idx >=0 && idx < size());
    return fragments_[idx];
  }

  auto get_label(int idx, int depth) const -> uint8_t {
    assert(idx >= 0 && idx < size());
    return fragments_[idx].get_label(depth, reverse_);
  }

  auto materialize(int idx) const -> key_type {
    assert(idx >= 0 && idx < size());
    return fragments_[idx].materialize(reverse_);
  }

  auto substr_range(uint32_t idx, uint32_t offset, uint32_t length) const -> std::pair<uint32_t, uint32_t> {
    assert(idx >= 0 && idx < size());
    return fragments_[idx].substr_range(offset, length, reverse_);
  }

  auto front() const -> const Fragment & {
    return get(0);
  }

  auto back() const -> const Fragment & {
    return get(size() - 1);
  }

  auto empty() const -> bool {
    return fragments_.empty();
  }

  auto size() const -> size_t {
    return fragments_.size();
  }

  auto space_cost() const -> size_t {
    return space_cost_;
  }

  void sort() {
    std::sort(fragments_.begin(), fragments_.end(),
              [&](const Fragment &f1, const Fragment &f2) -> bool {
                return f1.lt(f2, reverse_);
              });
  }

  void reverse() {
    reverse_ = !reverse_;
  }
};