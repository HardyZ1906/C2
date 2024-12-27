#pragma once

#include "utils.hpp"
#include "key_set.hpp"

#include "../lib/ds2i/succinct/elias_fano.hpp"
#include "../lib/ds2i/succinct/mapper.hpp"


template <typename Key>
class BasicStringPool {
 public:
  static constexpr uint8_t terminator_ = 0;

  using key_type = Key;

  BasicStringPool() = default;

  void build(const KeySet<key_type> &keys) {
    std::vector<uint32_t> positions;
    positions.reserve(keys.size() + 1);
    positions.push_back(0);
    labels_.reserve(keys.space_cost());
    for (uint32_t i = 0; i < keys.size(); i++) {
      const auto &frag = keys.get(i);
      frag.append_to(labels_, false);
      positions.emplace_back(labels_.size());
    }
    labels_.shrink_to_fit();
    typename succinct::elias_fano::elias_fano_builder builder(positions.back() + 1, positions.size());
    for (uint32_t i = 0; i < positions.size(); i++) {
      builder.push_back(positions[i]);
    }
    typename succinct::elias_fano(&builder, false).swap(positions_);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t {
    assert(key_id < size());

    auto [pos, end] = positions_.select_range(key_id);
    for (uint32_t i = 0; i < end - pos; i++) {
      if (begin + i >= key.size() || key[begin + i] != labels_[pos + i]) {
        return -1;
      }
    }
    return end - pos;
  }

  auto get(uint32_t key_id) const -> key_type {
    assert(key_id < size());

    auto [pos, end] = positions_.select_range(key_id);
    return key_type(reinterpret_cast<const char *>(labels_.data() + pos), end - pos);
  }

  void append_to(std::vector<uint8_t> &vec, uint32_t key_id, bool terminator = true) const {
    assert(key_id < size());

    auto [pos, end] = positions_.select_range(key_id);
    while (pos < end) {
      vec.push_back(labels_[pos++]);
    }
    if (terminator) {
      vec.push_back(terminator_);
    }
  }

  auto size() const -> uint32_t {
    return positions_.num_ones();
  }

  auto size_in_bytes() const -> size_t {
    return succinct::mapper::size_of(const_cast<succinct::elias_fano &>(positions_)) +
           labels_.size() * sizeof(uint8_t) + sizeof(labels_);
  }

  auto size_in_bits() const -> size_t {
    return size_in_bytes() * 8;
  }
 private:
  std::vector<uint8_t> labels_;
  succinct::elias_fano positions_;
};