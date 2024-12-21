#pragma once

#include "utils.hpp"
#include "key_set.hpp"
#include "compressed_string_pool.hpp"
#include "../lib/ds2i/succinct/mapper.hpp"

#include <vector>

#define __DEBUG__
#ifdef __DEBUG__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


template <typename Key>
class RepairStringPool {
 public:
  using key_type = Key;
  using strpool_t = typename succinct::tries::compressed_string_pool<uint8_t>;

  RepairStringPool() = default;

  void build(const KeySet<key_type> &key_set) {
    std::vector<uint8_t> keys;
    for (const auto &frag : key_set.fragments_) {
      frag.append_to(keys);
    }
    strpool_t strpool(keys);
    strpool_.swap(strpool);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t {
    assert(key_id < size());

    auto enu = strpool_.get_string_enumerator(key_id);
    uint32_t matched_len = 0;
    uint8_t label;
    while ((label = enu.next()) != 0) {
      if (begin + matched_len >= key.size() || key[begin + matched_len] != label) {
        return -1;
      }
      matched_len++;
    }
    return matched_len;
  }

  auto size() const -> uint32_t {
    return strpool_.size();
  }

  auto size_in_bits() const -> size_t {
    return succinct::mapper::size_of(const_cast<strpool_t &>(strpool_)) * 8;
  }
 private:
  strpool_t strpool_;
};


#undef __DEBUG__
#undef DEBUG