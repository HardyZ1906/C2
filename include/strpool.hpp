#pragma once

#include "utils.hpp"
#include "key_set.hpp"
#include "compressed_string_pool.hpp"
#include "../lib/ds2i/succinct/mapper.hpp"

#include <sdsl/int_vector.hpp>
#include <vector>


// #define __DEBUG_RECURSION__

#define __DEBUG_STRPOOL__
#ifdef __DEBUG_STRPOOL__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


template <typename Key, bool reverse>
class MarisaCC;

template <typename Key>
class SortedStringPool;

template <typename Key>
class UnsortedStringPool;

template <typename Key>
class RepairStringPool;

template <typename Key>
class StringPool {
 public:
  using key_type = Key;
  using trie_t = MarisaCC<key_type, true>;
  using repair_t = RepairStringPool<key_type>;
  using unsorted_t = UnsortedStringPool<key_type>;
  using sorted_t = SortedStringPool<key_type>;

  enum class Type {
    UNSORTED = 0, SORTED = 1, REPAIR = 2, TRIE = 3
  };

  static constexpr int UNSORTED_FLAG = BIT(static_cast<int>(Type::UNSORTED));
  static constexpr int SORTED_FLAG   = BIT(static_cast<int>(Type::SORTED));
  static constexpr int REPAIR_FLAG   = BIT(static_cast<int>(Type::REPAIR));

  // stop recursion when the total size of unary paths is below this percentage of the original key set size
  static constexpr float size_percentage_cutoff_ = 2.;
  // stop recursion when the average lcp is below this value
  static constexpr float avg_lcp_cutoff_ = 7.;
  // use SORTED if `avg_lcp/key_len` is above this value
  static constexpr float lcp_percentage_cutoff_ = 95;

  StringPool() = default;

  virtual ~StringPool() = default;

  // mask: which tail string pools are enabled? 0 means everything is enabled
  static auto get_optimal_type(const KeySet<key_type> &keys, const KeySet<key_type> &sorted_rev_keys,
                               size_t original_size, int max_recursion, int mask) -> Type {
  #ifdef __DEBUG_RECURSION__
    if (max_recursion > 0) {
      return Type::TRIE;
    } else if (mask & UNSORTED_FLAG) {
      return Type::UNSORTED;
    } else if (mask & SORTED_FLAG) {
      return Type::SORTED;
    } else {
      return Type::REPAIR;
    }
  #else
    if ((mask & MASK(3)) == 0) {
      mask = MASK(3);
    }
    auto [lcp_size, sorted_size] = sorted_rev_keys.lcp_size();
    float avg_lcp = 1.*lcp_size/sorted_rev_keys.size();
    float avg_key_len = 1.*keys.space_cost()/keys.size();
    DEBUG(
      printf("original: %f MB, current: %f MB, sorted: %f MB, avg lcp = %f, avg key len = %f, num keys = %ld\n",
             (float)original_size*8/mb_bits, (float)keys.space_cost()*8/mb_bits,
             (float)sorted_size*8/mb_bits, avg_lcp, avg_key_len, keys.size());
    )
    if ((max_recursion > 0)/* && (avg_lcp > avg_lcp_cutoff_) &&
        (keys.space_cost()*100 > original_size*size_percentage_cutoff_)*/) {
      return Type::TRIE;
    } else if ((mask & SORTED_FLAG) && (avg_lcp*100 > avg_key_len*lcp_percentage_cutoff_)) {
      return Type::SORTED;
    } else if ((mask & REPAIR_FLAG) && (keys.space_cost()*100 > original_size*size_percentage_cutoff_) ||
               !(mask & UNSORTED_FLAG) && !(mask & SORTED_FLAG)) {
      return Type::REPAIR;
    } else {
      size_t unsorted_cost = (mask & UNSORTED_FLAG) ? unsorted_t::estimate_space_cost(keys) : std::numeric_limits<uint64_t>::max();
      size_t sorted_cost = (mask & SORTED_FLAG) ? sorted_t::estimate_space_cost(sorted_size, keys.size()) :
                           std::numeric_limits<uint64_t>::max();
      DEBUG( printf("unsorted: %f MB, sorted: %f MB\n", (float)unsorted_cost/mb_bits, (float)sorted_cost/mb_bits); )
      return sorted_cost <= unsorted_cost ? Type::SORTED : Type::UNSORTED;
    }
  #endif
  }

  static auto build_optimal(const KeySet<key_type> &keys, std::vector<uint8_t> *partial_links = nullptr,
                            size_t original_size = 0, int max_recursion = 0, int mask = 0) -> StringPool * {
    auto sorted_rev_keys = keys;
    sorted_rev_keys.reverse();
    sorted_rev_keys.sort();

    auto optimal_type = get_optimal_type(keys, sorted_rev_keys, original_size, max_recursion, mask);
    StringPool *ret;
    switch (optimal_type) {
     case Type::SORTED:
      DEBUG( printf("max recursion = %d, type = SORTED\n", max_recursion); )
      ret = new sorted_t();
      ret->build(sorted_rev_keys, partial_links, original_size, max_recursion - 1, mask);
      DEBUG( printf("max recursion = %d, type = SORTED, cost = %f MB\n",
                    max_recursion, (float)ret->size_in_bits()/mb_bits); )
      return ret;
     case Type::UNSORTED:
      DEBUG( printf("max recursion = %d, type = UNSORTED\n", max_recursion); )
      ret = new unsorted_t();
      ret->build(keys, partial_links, original_size, max_recursion - 1, mask);
      DEBUG( printf("max recursion = %d, type = UNSORTED, cost = %f MB\n",
                    max_recursion, (float)ret->size_in_bits()/mb_bits); )
      return ret;
     case Type::TRIE:
      assert(max_recursion > 0);
      DEBUG( printf("max recursion = %d, type = TRIE\n", max_recursion); )
      ret = new trie_t();
      ret->build(sorted_rev_keys, partial_links, original_size, max_recursion - 1, mask);
      DEBUG( printf("max recursion = %d, type = TRIE, cost = %f MB\n",
                    max_recursion, (float)ret->size_in_bits()/mb_bits); )
      return ret;
     case Type::REPAIR:
      DEBUG( printf("max recursion = %d, type = REPAIR\n", max_recursion); )
      ret = new repair_t();
      ret->build(keys, partial_links, original_size, max_recursion - 1, mask);
      DEBUG( printf("max recursion = %d, type = REPAIR, cost = %f MB\n",
                    max_recursion, (float)ret->size_in_bits()/mb_bits); )
      return ret;
    }
  }

  virtual void build(const KeySet<key_type> &keys, std::vector<uint8_t> *partial_links = nullptr,
                     size_t original_size = 0, int max_recursion = 0, int mask = 0) = 0;

  virtual auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t = 0;

  virtual auto match(const key_type &key, uint32_t begin, uint32_t key_id,
                     uint8_t partial_link) const -> uint32_t = 0;

  virtual auto size() const -> uint32_t = 0;

  virtual auto size_in_bytes() const -> size_t = 0;

  virtual auto size_in_bits() const -> size_t = 0;

  virtual void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const = 0;
};


template <typename Key>
class SortedStringPool : public StringPool<Key> {
 public:
  using key_type = Key;

  SortedStringPool() = default;

  ~SortedStringPool() = default;

  // partial link: 8 low bits of the actual link
  void build(const KeySet<key_type> &sorted_rev_keys, std::vector<uint8_t> *partial_links = nullptr,
             size_t original_size = 0, int max_recursion = 0, int mask = 0) override {
    links_.resize(sorted_rev_keys.size());
    if (partial_links != nullptr) {
      partial_links->resize(sorted_rev_keys.size());
    }

    const typename KeySet<key_type>::Fragment *next = nullptr;
    for (size_t i = sorted_rev_keys.size(); i > 0; i--) {
      auto &cur = sorted_rev_keys[i - 1];
      if (next != nullptr) {
        uint32_t len = std::min(cur.size(), next->size());
        uint32_t match = 0;
        while (match < len && cur.get_label(match, true) == next->get_label(match, true)) {
          match++;
        }
        if (match == cur.size()) {  // deduplicate prefix key
          size_t link = labels_.size() - match - 1;
          if (partial_links == nullptr) {
            links_[cur.id_] = link;
          } else {
            links_[cur.id_] = link >> 8;
            (*partial_links)[cur.id_] = link & MASK(8);
          }
          continue;
        }
      }
      if (partial_links == nullptr) {
        links_[cur.id_] = labels_.size();
      } else {
        links_[cur.id_] = labels_.size() >> 8;
        (*partial_links)[cur.id_] = labels_.size() & MASK(8);
      }
      cur.append_to(labels_, true);
      next = &cur;
    }
    sdsl::util::bit_compress(links_);
    labels_.shrink_to_fit();
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t override {
    assert(key_id < size());
    size_t link = links_[key_id];
    return match_link(key, begin, link);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id, uint8_t partial_link) const -> uint32_t override {
    assert(key_id < size());
    size_t link = (links_[key_id] << 8) | partial_link;
    return match_link(key, begin, link);
  }

  auto match_link(const key_type &key, uint32_t begin, size_t link) const -> uint32_t {
    uint32_t matched_len = 0;
    while (labels_[link + matched_len] != terminator_) {
      if (begin + matched_len >= key.size() || labels_[link + matched_len] != key[begin + matched_len]) {
        return -1;
      }
      matched_len++;
    }
    return matched_len;
  }

  auto size() const -> uint32_t override {
    return links_.size();
  }

  auto size_in_bytes() const -> size_t override {
    return sdsl::size_in_bytes(links_) + labels_.size() * sizeof(uint8_t) + sizeof(labels_);
  }

  auto size_in_bits() const -> size_t override {
    return size_in_bytes() * 8;
  }

  void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const {
    link += sdsl::size_in_bytes(links_) * 8;
    data += labels_.size() * sizeof(uint8_t) * 8;
  }

  static auto estimate_space_cost(const KeySet<key_type> &sorted_keys) -> size_t {
    auto [lcp_size, sorted_size] = sorted_keys.lcp_size();
    return estimate_space_cost(sorted_size, sorted_keys.size());
  }

  static auto estimate_space_cost(size_t sorted_size, size_t num_keys) -> size_t {
    return sorted_size * 8 + (64 - __builtin_clzll(sorted_size)) * num_keys;
  }
 private:
  std::vector<uint8_t> labels_;
  sdsl::int_vector<> links_;
};

template <typename Key>
class UnsortedStringPool : public StringPool<Key> {
 public:
  using key_type = Key;

  UnsortedStringPool() = default;

  ~UnsortedStringPool() = default;

  // partial links: (4 low bits of this link) | (4 low bits of next link)
  void build(const KeySet<key_type> &keys, std::vector<uint8_t> *partial_links = nullptr,
             size_t original_size = 0, int max_recursion = 0, int mask = 0) override {
    std::vector<uint32_t> links;
    links.reserve(keys.size() + 1);
    links.push_back(0);
    labels_.reserve(keys.space_cost());
    if (partial_links != nullptr) {
      partial_links->reserve(keys.size());
    }
    uint8_t prev = 0;
    for (uint32_t i = 0; i < keys.size(); i++) {
      const auto &frag = keys.get(i);
      frag.append_to(labels_, false);
      if (partial_links == nullptr) {
        links.emplace_back(labels_.size());
      } else {
        links.emplace_back(labels_.size() >> 4);
        partial_links->push_back(prev | (labels_.size() & MASK(4)));
        prev = (labels_.size() & MASK(4)) << 4;
      }
    }
    labels_.shrink_to_fit();
    typename succinct::elias_fano::elias_fano_builder builder(links.back() + 1, links.size());
    for (uint32_t i = 0; i < links.size(); i++) {
      builder.push_back(links[i]);
    }
    typename succinct::elias_fano(&builder, false).swap(links_);
    labels_.shrink_to_fit();
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t override {
    assert(key_id < size());
    auto [link, end] = links_.select_range(key_id);
    return match_link(key, begin, link, end);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id, uint8_t partial_link) const -> uint32_t override {
    assert(key_id < size());
    auto [link, end] = links_.select_range(key_id);
    link = (link << 4) | (partial_link >> 4);
    end = (end << 4) | (partial_link & MASK(4));
    return match_link(key, begin, link, end);
  }

  auto match_link(const key_type &key, uint32_t begin, size_t link, size_t end) const -> uint32_t {
    for (uint32_t i = 0; i < end - link; i++) {
      if (begin + i >= key.size() || key[begin + i] != labels_[link + i]) {
        return -1;
      }
    }
    return end - link;
  }

  auto size() const -> uint32_t override {
    return links_.num_ones();
  }

  auto size_in_bytes() const -> size_t override {
    return succinct::mapper::size_of(const_cast<succinct::elias_fano &>(links_)) +
           labels_.size() * sizeof(uint8_t) + sizeof(labels_);
  }

  auto size_in_bits() const -> size_t override {
    return size_in_bytes() * 8;
  }

  void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const {
    link += succinct::mapper::size_of(const_cast<succinct::elias_fano &>(links_)) * 8;
    data += labels_.size() * sizeof(uint8_t) * 8;
  }

  static auto estimate_elias_fano_cost(size_t n, size_t m) -> size_t {
    constexpr uint32_t block_size = 1024, subblock_size = 32;
    int low_bits = (n >= m & m > 0) ? (64 - __builtin_clzll(n / m)) : 0;
    size_t values_cost = ((m + 1) + (n >> low_bits) + 1) + low_bits * m;  // elias fano
    size_t index_cost = (m / block_size) * (64 + 16*block_size/subblock_size);  // select index, assuming no overflows
    return values_cost + index_cost;
  }

  static auto estimate_space_cost(const KeySet<key_type> &keys) -> size_t {
    size_t num_labels = keys.space_cost();
    size_t n = num_labels + 1;
    size_t m = keys.size();
    size_t links_cost = estimate_elias_fano_cost(n, m);
    return links_cost + num_labels*8;
  }
 private:
  std::vector<uint8_t> labels_;
  succinct::elias_fano links_;
};

template <typename Key>
class RepairStringPool : public StringPool<Key> {
 public:
  using key_type = Key;
  using strpool_t = typename succinct::tries::compressed_string_pool<uint8_t>;

  RepairStringPool() = default;

  ~RepairStringPool() = default;

  // partial links: (4 low bits of this link) | (4 low bits of next link)
  void build(const KeySet<key_type> &key_set, std::vector<uint8_t> *partial_links = nullptr,
             size_t original_size = 0, int max_recursion = 0, int mask = 0) override {
    std::vector<uint8_t> keys;
    for (const auto &frag : key_set.fragments_) {
      frag.append_to(keys);
    }
    build(keys, partial_links);
  }

  void build(const std::vector<uint8_t> &keys, std::vector<uint8_t> *partial_links = nullptr) {
    if (!keys.empty()) {
      strpool_t strpool(keys);
      strpool_.swap(strpool);
      if (partial_links == nullptr) {
        links_.swap(strpool_.m_positions);
      } else {
        auto enu = typename succinct::elias_fano::select_enumerator(strpool_.m_positions, 1);

        size_t n = (strpool_.m_positions.select(strpool_.size()) >> 4) + 1;
        size_t m = strpool_.size() + 1;
        typename succinct::elias_fano::elias_fano_builder builder(n, m);
        uint8_t prev = 0;
        builder.push_back(0);
        partial_links->reserve(m - 1);
        for (size_t i = 0; i < m - 1; i++) {
          size_t link = enu.next();
          builder.push_back(link >> 4);
          partial_links->push_back(prev | link & MASK(4));
          prev = (link & MASK(4)) << 4;
        }
        strpool_.release_positions();
        typename succinct::elias_fano(&builder, false).swap(links_);
      }
    } else {
      strpool_t strpool;
      strpool_.swap(strpool);
    }
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id) const -> uint32_t override {
    assert(key_id < size());
    auto [link, end] = links_.select_range(key_id);
    return match_link(key, begin, link, end);
  }

  auto match(const key_type &key, uint32_t begin, uint32_t key_id, uint8_t partial_link) const -> uint32_t override {
    assert(key_id < size());
    auto [link, end] = links_.select_range(key_id);
    link = (link << 4) | (partial_link >> 4);
    end = (end << 4) | (partial_link & MASK(4));
    return match_link(key, begin, link, end);
  }

  auto match_link(const key_type &key, uint32_t begin, size_t link, size_t end) const -> uint32_t {
    auto enu = strpool_.get_string_enumerator(link, end);
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

  auto size() const -> uint32_t override {
    return strpool_.size();
  }

  auto size_in_bytes() const -> size_t override {
    return succinct::mapper::size_of(const_cast<strpool_t &>(strpool_)) +
           succinct::mapper::size_of(const_cast<succinct::elias_fano &>(links_));
  }

  auto size_in_bits() const -> size_t override {
    return size_in_bytes() * 8;
  }

  void space_cost_breakdown(size_t &topo, size_t &link, size_t &data) const {
    link += succinct::mapper::size_of(const_cast<succinct::elias_fano &>(links_)) * 8;
    data += succinct::mapper::size_of(const_cast<strpool_t &>(strpool_)) * 8;
  }
 private:
  strpool_t strpool_;
  succinct::elias_fano links_;
};


#undef DEBUG