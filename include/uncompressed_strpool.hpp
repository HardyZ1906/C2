#pragma once

#include "utils.hpp"
#include "key_set.hpp"

#include "../lib/ds2i/succinct/elias_fano.hpp"


template <typename Key>
class UncompressedStringPool {
 public:
  using key_type = Key;

  UncompressedStringPool() = default;

  void build(const KeySet<key_type> &keys) {
    std::vector<size_t> labels_;
    positions.reserve(keys.size() + 1);
    labels_.reserve(keys.space_cost());
    for (uint32_t i = 0; i < keys.size(); i++) {
      positions.emplace_back(labels_.size());
      const auto &frag = keys.get(i);
      for (uint32_t j = 0; j < frag.size(); j++) {
        labels_.emplace_back(frag.get_label(j));
      }
    }
    typename succinct::elias_fano::elias_fano_builder builder(positions.back() + 1, positions.size());
    for (uint32_t i = 0; i < positions.size(); i++) {
      builder.push_back(positions[i]);
    }
    positions_.swap(typename succinct::elias_fano(&builder, false));
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

  auto size() const -> uint32_t {
    return positions_.num_ones();
  }
 private:
  std::vector<uint8_t> labels_;
  succinct::elias_fano positions_;
};