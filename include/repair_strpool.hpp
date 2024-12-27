#pragma once

#include "utils.hpp"
#include "key_set.hpp"
#include "compressed_string_pool.hpp"
#include "../lib/ds2i/succinct/mapper.hpp"

#include <vector>


template <typename Key>
class RepairStringPool {
 public:
  static constexpr uint8_t terminator_ = 0;

  using key_type = Key;
  using strpool_t = typename succinct::tries::compressed_string_pool<uint8_t>;

  RepairStringPool() = default;

  void build(const KeySet<key_type> &key_set) {
    std::vector<uint8_t> keys;
    for (const auto &frag : key_set.fragments_) {
      frag.append_to(keys);
    }
    build(keys);
  }

  void build(std::vector<uint8_t> keys) {
    if (!keys.empty()) {
      strpool_t strpool(keys);
      strpool_.swap(strpool);
    } else {
      strpool_t strpool;
      strpool_.swap(strpool);
    }
  }

  void swap(RepairStringPool &rhs) {
    strpool_.swap(rhs.strpool_);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t {
    assert(key_id < size());

    auto enu = strpool_.get_string_enumerator(key_id);
    uint32_t matched_len = 0;
    uint8_t label;
    while ((label = enu.next()) != terminator_) {
      if (begin + matched_len >= key.size() || key[begin + matched_len] != label) {
        return -1;
      }
      matched_len++;
    }
    return matched_len;
  }

  auto get(uint32_t key_id) const -> key_type {
    assert(key_id < size());

    key_type ret;
    auto enu = strpool_.get_string_enumerator(key_id);
    uint8_t label;
    while ((label = enu.next()) != terminator_) {
      ret.push_back(label);
    }
    return ret;
  }

  void append_to(std::vector<uint8_t> &vec, uint32_t key_id, bool terminator = true) const {
    assert(key_id < size());

    auto enu = strpool_.get_string_enumerator(key_id);
    uint8_t label;
    while ((label = enu.next()) != terminator_) {
      vec.push_back(label);
    }
    if (terminator) {
      vec.push_back(terminator_);
    }
  }

  auto size() const -> uint32_t {
    return strpool_.size();
  }

  auto size_in_bytes() const -> size_t {
    return succinct::mapper::size_of(const_cast<strpool_t &>(strpool_));
  }

  auto size_in_bits() const -> size_t {
    return size_in_bytes() * 8;
  }
 private:
  strpool_t strpool_;
};
