#pragma once

#include "utils.hpp"
#include "alphabet.hpp"
#include "louds_cc.hpp"
#include "louds_cc2.hpp"
#include "ls4coco.hpp"
#include "coco_optimizer.hpp"
#include "ls4coco.hpp"

#include "../lib/ds2i/succinct/bit_vector.hpp"
#include "../lib/ds2i/succinct/bp_vector.hpp"
#include <sdsl/int_vector.hpp>

#include <vector>
#include <queue>


// #define __DEBUG_COCO__
#ifdef __DEBUG_COCO__
# include <iostream>
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


template<typename Key, bool reverse = false>
class CoCoCC {
 public:
  using key_type = Key;
  using optimizer = CoCoOptimizer<key_type>;
  using encoding_t = optimizer::encoding_t;
  using next_trie_t = CoCoCC<Key, true>;
  using topo_t = std::conditional<reverse, LoudsCC2, LoudsCC>::type;

  static constexpr bool reverse_ = reverse;
  static constexpr uint8_t encoding_bits_ = optimizer::encoding_bits_;
  static constexpr uint32_t max_depth_ = optimizer::max_depth_;
  static constexpr uint8_t depth_bits_ = optimizer::depth_bits_;
  static constexpr uint32_t degree_threshold_ = optimizer::degree_threshold_;
  static constexpr uint32_t bv_block_sz_ = optimizer::bv_block_sz_;
  static constexpr uint8_t terminator_ = optimizer::terminator_;

  CoCoCC() = default;

  CoCoCC(const optimizer &opt) : next_trie_{nullptr} {
    build(opt);
  }

  template<typename Iterator>
  CoCoCC(Iterator begin, Iterator end, uint32_t space_relaxation = 0,
         uint32_t pattern_len = 10, uint32_t min_occur = 20) {
    build(begin, end, space_relaxation, pattern_len, min_occur);
  }

  template<typename Iterator>
  void build(Iterator begin, Iterator end, uint32_t space_relaxation = 0,
             uint32_t pattern_len = 10, uint32_t min_occur = 20) {
    if constexpr (reverse_) {
      pattern_len = 0;
      min_occur = 0;
    }
    typename optimizer::trie_t trie;
    trie.build(begin, end);
    optimizer opt(&trie);
    opt.optimize(space_relaxation, pattern_len, min_occur);
    build(opt);
  }

  void build(const optimizer &opt) {
    alphabet_ = opt.alphabet_;
    topo_.reserve(opt.states_[0].num_macros_ + opt.states_[0].num_leaves_);
    ptrs_.resize(opt.states_[0].num_macros_ + 1);

    succinct::bit_vector_builder bv;

    std::queue<uint32_t> queue;
    queue.push(0);

    if constexpr (!reverse_) {
      std::vector<key_type> patterns;
      opt.extract_patterns(patterns);
      if (!patterns.empty()) {
        next_trie_ = new next_trie_t();
        next_trie_->build(patterns.begin(), patterns.end(), 0, 0, 0);
        link_bits_ = std::max(log2_ceil(patterns.size()), 1);
      }
    }

    auto push_child = [&](uint32_t pos) {
      if (opt.trie_->has_child(pos)) {
        queue.push(opt.trie_->child_pos(pos));
      } else {
        queue.push(-1);
      }
    };

    uint32_t macro_id = 0;
    while (!queue.empty()) {
      uint32_t pos = queue.front();
      queue.pop();
      if (pos == -1) {  // leaf node
        // DEBUG( printf("leaf node\n"); )
        topo_.add_node(0);
        continue;
      }  // else macro node
      ptrs_[macro_id++] = bv.size();  // set pointer

      uint32_t node_id = opt.trie_->node_id(pos);
      const typename optimizer::state_t &state = opt.states_[node_id];

      bool prefix_key = false;
      if (opt.trie_->get_label(pos) == terminator_) {  // mark key as null instead of encoding it
        prefix_key = true;
        pos++;
        queue.push(-1);
      }
      encoding_t encoding = state.encoding_;
      uint32_t depth = state.depth_;

      DEBUG(
        printf("macro node %d: pos = %d, depth = %d, encoding = %d, ptr = %ld, expected cost = %lf, total cost = %lf\n",
               macro_id - 1, pos, depth, encoding, bv.size(), state.self_enc_cost_, state.enc_cost_);
        uint32_t key_idx = 0;
        if (prefix_key) {
          printf("key %d: (null)\n", key_idx++);
        }
      )

      // traverse and encode all keys
      typename optimizer::trie_t::walker walker(opt.trie_, pos);
      walker.get_min_key(depth);
      push_child(walker.pos_);

      if (encoding == encoding_t::UNARY_PATH || encoding == encoding_t::UP_REMAP) {
        assert(!prefix_key);
        DEBUG( printf("key %d: %s\n", key_idx++, walker.key().c_str()); )
        bv.append_bits(static_cast<uint64_t>(encoding), encoding_bits_);
        if (encoding == encoding_t::UNARY_PATH) {
          encode_unary_path(bv, opt, pos);
        } else {
          write_alphabet(bv, state.remap_);
          encode_unary_path_remap(bv, opt, pos, state.remap_);
        }
        topo_.add_node(1);
        continue;
      }

      code_t first_code;
      std::vector<code_t> codes;
      if (!(static_cast<int>(encoding) & 0x4)) {  // no remap
        DEBUG( printf("key %d: %s\n", key_idx++, walker.key().c_str()); )
        first_code = encode(walker.key(), 0, depth);
        while (walker.next(depth)) {
          DEBUG( printf("key %d: %s\n", key_idx++, walker.key().c_str()); )
          push_child(walker.pos_);
          code_t code = encode(walker.key(), 0, depth);
          codes.emplace_back(code - first_code - 1);
        }
        // write metadata
        write_metadata(bv, encoding, prefix_key, depth, codes.size() + 1 + prefix_key);
        // write first code
        uint32_t width = optimizer::code_len(alphabet_, depth);
        bv.append_bits(first_code, width);
      } else {  // remap
        DEBUG( printf("key %d: %s\n", key_idx++, walker.key().c_str()); )
        first_code = encode(walker.key(), 0, depth, state.remap_);
        while (walker.next(depth)) {
          DEBUG( printf("key %d: %s\n", key_idx++, walker.key().c_str()); )
          push_child(walker.pos_);
          code_t code = encode(walker.key(), 0, depth, state.remap_);
          codes.emplace_back(code - first_code - 1);
        }
        //write metadata & local alphabet
        write_metadata(bv, encoding, prefix_key, depth, codes.size() + 1 + prefix_key);
        write_alphabet(bv, state.remap_);
        // write first code
        uint32_t width = optimizer::code_len(state.remap_, depth);
        bv.append_bits(first_code, width);
      }
      assert(std::is_sorted(codes.begin(), codes.end()));
      assert(std::unique(codes.begin(), codes.end()) == codes.end());
      // encode keys
      switch (state.encoding_) {
       case encoding_t::ELIAS_FANO:
       case encoding_t::EF_REMAP:
        encode_elias_fano(bv, codes);
        break;
       case encoding_t::PACKED:
       case encoding_t::PA_REMAP:
        encode_packed(bv, codes);
        break;
        break;
       case encoding_t::BITVECTOR:
       case encoding_t::BV_REMAP:
        encode_bitvector(bv, codes);
        break;
       case encoding_t::DENSE:
       case encoding_t::DE_REMAP:
        // do nothing, as encoding is implied in metadata and the first code
        break;
       default: assert(false); // should not happen
      }
      topo_.add_node(codes.size() + 1 + prefix_key);  // update topology
    }
    ptrs_[macro_id] = bv.size();  // sentinel
    DEBUG( printf("final encoding cost: %ld\n", bv.size()); )
    topo_.build();
    sdsl::util::bit_compress(ptrs_);
    new (&macros_) succinct::bit_vector(&bv);

    if (!reverse_ && next_trie_ != nullptr) {
      next_trie_->finish();
    }
  }

  auto lookup(const key_type &key) const -> uint32_t {
    return lookup(key, 0, key.size());
  }

  // return leaf id (-1 if not found)
  auto lookup(const key_type &key, uint32_t begin, uint32_t len) const -> uint32_t {
    assert(begin + len <= key.size());
    uint32_t end = begin + len, scanned_chars = begin;
    uint32_t pos = 0;
    uint32_t idx = begin;
    while (true) {
      if (topo_.is_leaf(pos)) {
        return idx >= end ? topo_.leaf_id(pos) : -1;
      }
      uint32_t macro_id = topo_.internal_id(pos);
      succinct::bit_vector::enumerator it(macros_, ptrs_[macro_id]);
      size_t next = ptrs_[macro_id + 1];
      encoding_t encoding = static_cast<encoding_t>(it.take(encoding_bits_));
      Alphabet remap;

      if (encoding == encoding_t::UNARY_PATH || encoding == encoding_t::UP_REMAP) {
        assert(topo_.degree(pos) == 1);
        uint32_t depth;
        if (encoding == encoding_t::UNARY_PATH) {
          if ((depth = match_unary_path(it, key, idx, next)) == -1) {
            return -1;
          }
        } else {  // UP_REMAP
          read_alphabet(it, remap);
          if ((depth = match_unary_path_remap(it, key, remap, idx, next)) == -1) {
            return -1;
          }
        }
        // DEBUG(
        //   printf("matched unary path, key: %s, pos %d, macro node %d, encoding = %d, depth = %d\n",
        //          key.substr(idx, depth).c_str(), pos, macro_id, encoding, depth);
        // )
        idx += depth;
        pos = topo_.child_pos(pos);
        continue;
      }

      // read metadata
      bool prefix_key = it.take(1);
      if (idx >= end) {
        return prefix_key ? topo_.leaf_id(topo_.child_pos(pos)) : -1;
      }
      uint32_t depth = it.take(depth_bits_) + 1;
    #ifdef __DEGREE_IN_PLACE__
      bool degree_in_place = it.take(1);
      uint32_t degree = (degree_in_place ? ds2i::read_delta(it) : topo_.degree(pos));
    #else
      uint32_t degree = topo_.degree(pos);
    #endif
      assert(degree >= 1 + prefix_key);
      // read first code
      code_t first_code, code;
      if (!(static_cast<int>(encoding) & 0x4)) {  // no remap
        if (!is_legal(key, idx, depth)) {
          return -1;
        }
        code = encode(key, idx, depth);
        first_code = it.take(optimizer::code_len(alphabet_, depth));
      } else {
        read_alphabet(it, remap);
        if (!is_legal(key, idx, depth, remap)) {
          return -1;
        }
        code = encode(key, idx, depth, remap);
        first_code = it.take(optimizer::code_len(remap, depth));
      }
      uint32_t child_id;
      // compare against first code
      if (code < first_code) {
        child_id = -1;
      } else if (code == first_code) {
        child_id = prefix_key;
      } else {  // search encoding
        code_t target = code - first_code - 1;
        switch (encoding) {
         case encoding_t::ELIAS_FANO:
         case encoding_t::EF_REMAP:
          child_id = match_elias_fano(it, prefix_key, degree - prefix_key - 1, target);
          break;
         case encoding_t::PACKED:
         case encoding_t::PA_REMAP:
          child_id = match_packed(it, prefix_key, degree - prefix_key - 1, next, target);
          break;
         case encoding_t::BITVECTOR:
         case encoding_t::BV_REMAP:
          child_id = match_bitvector(it, prefix_key, degree - prefix_key - 1, next, target);
          break;
         case encoding_t::DENSE:
         case encoding_t::DE_REMAP:
          child_id = match_dense(it, prefix_key, degree - prefix_key - 1, target);
          break;
         default: assert(false); // should not happen
        }
      }
      if (child_id == -1) {
        return -1;
      }
      assert(child_id < degree);
      // DEBUG(
      //   printf("key: %s, code = %ld, pos %d, macro node %d, child %d, encoding = %d, depth = %d, degree = %d, prefix key = %d\n",
      //          key.substr(idx, depth).c_str(), code, pos + child_id, macro_id, child_id, encoding, depth, degree, prefix_key);
      // )
      idx += depth;
      pos = topo_.child_pos(pos + child_id);
    }
  }

  // check if `key[begin:]` is able to reach the root starting from leaf `leaf_id`
  // returns the number of labels matched on match and -1 on mismatch
  template <bool rev = reverse, typename = std::enable_if_t<rev>>
  auto match_rev(const key_type &key, uint32_t begin, uint32_t leaf_id) const -> uint32_t {
    uint32_t pos = topo_.select_leaf(leaf_id);
    uint32_t idx = begin;

    while (pos > 0) {
      pos = topo_.parent_pos(pos);
      uint32_t macro_id = topo_.internal_id(pos);
      succinct::bit_vector::enumerator it(macros_, ptrs_[macro_id]);
      size_t next = ptrs_[macro_id + 1];
      encoding_t encoding = static_cast<encoding_t>(it.take(encoding_bits_));
      Alphabet remap;

      if (encoding == encoding_t::UNARY_PATH || encoding == encoding_t::UP_REMAP) {
        assert(topo_.degree(pos) == 1);
        uint32_t depth;
        if (encoding == encoding_t::UNARY_PATH) {
          if ((depth = match_unary_path_rev(it, key, idx, next)) == -1) {
            return -1;
          }
        } else {  // UP_REMAP
          read_alphabet(it, remap);
          if ((depth = match_unary_path_remap_rev(it, key, remap, idx, next)) == -1) {
            return -1;
          }
        }
        // DEBUG(
        //   printf("matched unary path, key: %s, pos %d, macro node %d, encoding = %d, depth = %d\n",
        //          key.substr(idx, depth).c_str(), pos, macro_id, encoding, depth);
        // )
        idx += depth;
        continue;
      }

      bool prefix_key = it.take(1);
      uint32_t node_start = topo_.node_start(pos);
      if (prefix_key && pos == node_start) {
        continue;
      }
      uint32_t depth = it.take(depth_bits_) + 1;
    #ifdef __DEGREE_IN_PLACE__
      bool degree_in_place = it.take(1);
      uint32_t degree = (degree_in_place ? ds2i::read_delta(it) : topo_.degree(node_start));
    #else
      uint32_t degree = topo_.degree(node_start);
    #endif
      assert(degree >= 1 + prefix_key);
      uint32_t num_codes = degree - prefix_key - 1;
      // read first code
      code_t code;
      if (!(static_cast<int>(encoding) & 0x4)) {  // no remap
        code = it.take(optimizer::code_len(alphabet_, depth));
      } else {
        read_alphabet(it, remap);
        code = it.take(optimizer::code_len(remap, depth));
      }
      // get target code
      uint32_t key_id = pos - node_start - prefix_key;
      if (key_id > 0) {
        switch (encoding) {
         case encoding_t::ELIAS_FANO:
         case encoding_t::EF_REMAP:
          code += get_elias_fano(it, key_id - 1, num_codes) + 1;
          break;
         case encoding_t::PACKED:
         case encoding_t::PA_REMAP:
          code += get_packed(it, key_id - 1, num_codes, next) + 1;
          break;
         case encoding_t::BITVECTOR:
         case encoding_t::BV_REMAP:
          code += get_bitvector(it, key_id - 1, num_codes, next) + 1;
          break;
         case encoding_t::DENSE:
         case encoding_t::DE_REMAP:
          code += get_dense(it, key_id - 1) + 1;
          break;
         default: assert(false);  // should not be reachable
        }
      }
      // DEBUG(
      //   printf("key: %s, code = %ld, pos %d, macro node %d, child %d, encoding = %d, depth = %d, degree = %d, prefix key = %d\n",
      //          key.substr(idx, depth).c_str(), code, pos, macro_id, key_id + prefix_key, encoding, depth, degree, prefix_key);
      // )
      // match
      depth = (static_cast<int>(encoding) & 0x4) ? match_code_rev_remap(key, idx, code, remap)
                                                 : match_code_rev(key, idx, code);
      if (depth == -1) {
        return -1;
      }
      idx += depth;
      pos = node_start;
    }
    return idx - begin;
  }

  auto encoding_size() const -> size_t {
    if (next_trie_ != nullptr) {
      return macros_.size() + next_trie_->encoding_size();
    }
    return macros_.size();
  }

  auto size_in_bits() const -> size_t {
    size_t next_trie_size = next_trie_ == nullptr ? 0 : next_trie_->size_in_bits();
    return next_trie_size + topo_.size_in_bits() + sdsl::size_in_bytes(ptrs_)*8 + macros_.size();
  }

  auto get_num_nodes() const -> std::pair<uint32_t, uint32_t> {
    return {topo_.num_internals(), topo_.num_leaves()};
  }
 private:
  void encode_elias_fano(succinct::bit_vector_builder &bv, const std::vector<code_t> &codes) {
    assert(!codes.empty());
    // write universe
    code_t universe = codes.back() + 1;
    ds2i::write_delta(bv, universe);
    // write codes
    ds2i::compact_elias_fano<code_t>::write(bv, codes.begin(), universe, codes.size(), params);
  }

  void encode_packed(succinct::bit_vector_builder &bv, const std::vector<code_t> &codes) {
    assert(!codes.empty());
    // write codes
    uint32_t width = width_in_bits(codes.back());
    for (auto code : codes) {
      bv.append_bits(code, width);
    }
  }

  void encode_bitvector(succinct::bit_vector_builder &bv, const std::vector<code_t> &codes) {
    assert(!codes.empty());
    // write rank index; rank is at most `codes.size()`
    uint32_t idx_width = width_in_bits(codes.size());
    uint32_t num_blocks = (codes.back() + 1) / bv_block_sz_;
    size_t j = 0;
    for (uint32_t i = 1; i <= num_blocks; i++) {
      while (j < codes.size() && codes[j] < i*bv_block_sz_) {
        j++;
      }
      bv.append_bits(j, idx_width);
    }
    // write codes
    size_t start = bv.size();
    bv.zero_extend(codes.back() + 1);
    for (auto code : codes) {
      bv.set(start + code, 1);
    }
  }

  void encode_unary_path(succinct::bit_vector_builder &bv, const optimizer &opt, uint32_t pos) {
    if constexpr (reverse_) {
      // we don't encode the terminator as the second trie does not support double trie encoding
      uint32_t label_width = std::max(1, log2_ceil(alphabet_.alphabet_size() - 1));
      const typename optimizer::state_t &state = opt.states_[opt.trie_->node_id(pos)];
      uint32_t depth = state.depth_;

      for (uint32_t i = 0; i < depth; i++) {
        const typename optimizer::state_t &s = opt.states_[opt.trie_->node_id(pos)];
        assert(s.encoding_ == encoding_t::UNARY_PATH || s.encoding_ == encoding_t::UP_REMAP);
        assert(s.depth_ == s.path_len_);
        uint8_t label = encode(opt.trie_->get_label(pos));
        assert(label > 0);
        bv.append_bits(static_cast<uint64_t>(label - 1), label_width);
        if (opt.trie_->has_child(pos)) {
          pos = opt.trie_->child_pos(pos);
        }
      }
    } else {
      uint32_t label_width = log2_ceil(alphabet_.alphabet_size());
      const typename optimizer::state_t &state = opt.states_[opt.trie_->node_id(pos)];
      uint32_t depth = state.depth_;

      uint32_t i = 0;
      while (i < depth) {
        const typename optimizer::state_t &s = opt.states_[opt.trie_->node_id(pos)];
        assert(s.encoding_ == encoding_t::UNARY_PATH || s.encoding_ == encoding_t::UP_REMAP);
        assert(s.depth_ == s.path_len_);
        if (s.pattern_len_ == 0) {
          bv.append_bits(static_cast<uint64_t>(encode(opt.trie_->get_label(pos))), label_width);
          if (opt.trie_->has_child(pos)) {
            pos = opt.trie_->child_pos(pos);
          } else {
            pos = -1;
          }
          i++;
          continue;
        }

        key_type pattern;
        for (uint32_t j = 0; j < s.pattern_len_; j++) {
          pattern.push_back(opt.trie_->get_label(pos));
          if (opt.trie_->has_child(pos)) {
            pos = opt.trie_->child_pos(pos);
          } else {
            pos = -1;
          }
        }
        std::reverse(pattern.begin(), pattern.end());
        bv.append_bits(static_cast<uint64_t>(terminator_), label_width);  // insert a terminator to mark link
        auto link = next_trie_->lookup(pattern);
        assert(link != -1);
        bv.append_bits(static_cast<uint64_t>(link), link_bits_);
        i += s.pattern_len_;
      }
    }
  }

  void encode_unary_path_remap(succinct::bit_vector_builder &bv, const optimizer &opt,
                               uint32_t pos, const Alphabet &remap) {
    if constexpr (reverse) {
      // we don't encode the terminator as the second trie does not support double trie encoding
      uint32_t label_width = std::max(1, log2_ceil(remap.alphabet_size() - 1));
      const typename optimizer::state_t &state = opt.states_[opt.trie_->node_id(pos)];
      uint32_t depth = state.depth_;

      for (uint32_t i = 0; i < depth; i++) {
        const typename optimizer::state_t &s = opt.states_[opt.trie_->node_id(pos)];
        assert(s.encoding_ == encoding_t::UNARY_PATH || s.encoding_ == encoding_t::UP_REMAP);
        assert(s.depth_ == s.path_len_);
        uint8_t label = encode(opt.trie_->get_label(pos), remap);
        assert(label > 0);
        bv.append_bits(static_cast<uint64_t>(label - 1), label_width);
        if (opt.trie_->has_child(pos)) {
          pos = opt.trie_->child_pos(pos);
        }
      }
    } else {
      uint32_t label_width = log2_ceil(remap.alphabet_size());
      const typename optimizer::state_t &state = opt.states_[opt.trie_->node_id(pos)];
      uint32_t depth = state.depth_;

      uint32_t i = 0;
      while (i < depth) {
        const typename optimizer::state_t &s = opt.states_[opt.trie_->node_id(pos)];
        assert(s.encoding_ == encoding_t::UNARY_PATH || s.encoding_ == encoding_t::UP_REMAP);
        assert(s.depth_ == s.path_len_);
        if (reverse || s.pattern_len_ == 0) {
          bv.append_bits(static_cast<uint64_t>(encode(opt.trie_->get_label(pos), remap)), label_width);
          if (opt.trie_->has_child(pos)) {
            pos = opt.trie_->child_pos(pos);
          }
          i++;
          continue;
        }

        key_type pattern;
        for (uint32_t j = 0; j < s.pattern_len_; j++) {
          pattern.push_back(opt.trie_->get_label(pos));
          if (opt.trie_->has_child(pos)) {
            pos = opt.trie_->child_pos(pos);
          }
        }
        bv.append_bits(static_cast<uint64_t>(terminator_), label_width);  // insert a terminator to mark link
        auto link = next_trie_->lookup(pattern);
        assert(link != -1);
        bv.append_bits(static_cast<uint64_t>(link), link_bits_);
        i += s.pattern_len_;
      }
    }
  }

  auto match_elias_fano(succinct::bit_vector::enumerator &it, bool prefix_key,
                        uint32_t num_codes, code_t target) const -> uint32_t {
    code_t universe = ds2i::read_delta(it);
    typename ds2i::compact_elias_fano<code_t>::enumerator enu(macros_, it.position(), universe, num_codes, params);
    auto result = enu.next_geq(target);
    if (result.second != universe && result.second == target) {  // exact match
      return result.first + 1 + prefix_key;
    }
    return -1;
  }

  auto match_packed(succinct::bit_vector::enumerator &it, bool prefix_key, uint32_t num_codes,
                    size_t next, code_t target) const -> uint32_t {
    assert((next - it.position()) % num_codes == 0);
    size_t pos = it.position();
    uint32_t width = (next - pos) / num_codes;
    uint32_t cutoff = 64*8*8 / width;  // 8 cachelines
    uint32_t left = 0, right = (next - pos) / width - 1;
    while (left + cutoff < right) {
      uint32_t mid = (left + right + 1) / 2;
      if (macros_.get_bits(pos + mid*width, width) < target) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }
    code_t val;
    while ((val = macros_.get_bits(pos + left*width, width)) < target) {
      left++;
    }
    if (val == target) {
      return left + 1 + prefix_key;
    }
    return -1;
  }

  auto match_bitvector(succinct::bit_vector::enumerator &it, bool prefix_key, uint32_t num_codes,
                       size_t next, code_t target) const -> uint32_t {
    size_t pos = it.position();
    uint32_t idx_width = width_in_bits(num_codes);
    size_t len = next - pos;
    size_t num_blocks = len / (idx_width + bv_block_sz_);
    size_t bv_start = pos + idx_width*num_blocks;
    code_t universe = next - bv_start;
    if (target >= universe) {
      return -1;
    }
    size_t target_pos = bv_start + target;
    uint32_t block_idx = target / bv_block_sz_;
    if (macros_[target_pos]) {
      uint32_t rank = block_idx == 0 ? 0 : macros_.get_bits(pos + (block_idx - 1)*idx_width, idx_width);;
      it.move(bv_start + block_idx*bv_block_sz_);
      while (target_pos - it.position() + 1 > 64) {
        rank += __builtin_popcountll(it.take(64));
      }
      rank += __builtin_popcountll(it.take(target_pos - it.position() + 1));
      return rank + prefix_key;
    }
    return -1;
  }

  auto match_dense(succinct::bit_vector::enumerator &it, bool prefix_key,
                   uint32_t num_codes, code_t target) const -> uint32_t {
    if (target < num_codes) {
      return target + 1 + prefix_key;
    }
    return -1;
  }

  // returns the number of matched labels on success and -1 on failure
  auto match_unary_path(succinct::bit_vector::enumerator &it, const key_type &key,
                        uint32_t begin, size_t next) const -> uint32_t {
    uint32_t matched_len = 0;
    if constexpr (reverse_) {
      uint32_t label_width = std::max(1, log2_ceil(alphabet_.alphabet_size() - 1));
      assert((next - it.position()) % label_width == 0);
      while (it.position() < next) {
        uint8_t label = it.take(label_width) + 1;
        if (!is_legal(key[begin + matched_len]) || label != encode(key[begin + matched_len])) {
          return -1;
        } else {
          matched_len++;
        }
      }
    } else {
      uint32_t label_width = log2_ceil(alphabet_.alphabet_size());
      while (it.position() < next) {
        uint8_t label = it.take(label_width);
        if (label == terminator_) {  // double trie encoding
          uint32_t link = it.take(link_bits_);
          uint32_t depth = next_trie_->match_rev(key, begin + matched_len, link);
          if (depth == -1) {
            return -1;
          }
          matched_len += depth;
          continue;
        } else if (!is_legal(key[begin + matched_len]) || label != encode(key[begin + matched_len])) {
          return -1;
        } else {
          matched_len++;
        }
      }
    }
    return matched_len;
  }

  // returns the number of matched labels; if not matched, returns -1
  auto match_unary_path_remap(succinct::bit_vector::enumerator &it, const key_type &key,
                              const Alphabet &remap, uint32_t begin, size_t next) const -> uint32_t {
    uint32_t matched_len = 0;
    if constexpr (reverse_) {
      uint32_t label_width = std::max(1, log2_ceil(remap.alphabet_size() - 1));
      assert((next - it.position()) % label_width == 0);
      while (it.position() < next) {
        uint8_t label = it.take(label_width) + 1;
        if (!is_legal(key[begin + matched_len], remap) || label != encode(key[begin + matched_len], remap)) {
          return -1;
        } else {
          matched_len++;
        }
      }
    } else {
      uint32_t label_width = log2_ceil(alphabet_.alphabet_size());
      while (it.position() < next) {
        uint8_t label = it.take(label_width);
        if (label == terminator_) {  // double trie encoding
          uint32_t link = it.take(link_bits_);
          uint32_t depth = next_trie_->match_rev(key, begin + matched_len, link);
          if (depth == -1) {
            return -1;
          }
          matched_len += depth;
          continue;
        } else if (!is_legal(key[begin + matched_len], remap) || label != encode(key[begin + matched_len], remap)) {
          return -1;
        } else {
          matched_len++;
        }
      }
    }
    return matched_len;
  }

  auto get_elias_fano(succinct::bit_vector::enumerator &it, uint32_t key_id, uint32_t num_codes) const -> code_t {
    code_t universe = ds2i::read_delta(it);
    typename ds2i::compact_elias_fano<code_t>::enumerator enu(macros_, it.position(), universe, num_codes, params);
    auto ret = enu.move(key_id);
    assert(ret.first == key_id);
    return ret.second;
  }

  auto get_packed(succinct::bit_vector::enumerator &it, uint32_t key_id, uint32_t num_codes, size_t next) const -> code_t {
    assert((next - it.position()) % num_codes == 0);
    size_t pos = it.position();
    uint32_t width = (next - pos) / num_codes;
    code_t ret = macros_.get_bits(pos + key_id*width, width);
    return ret;
  }

  auto get_bitvector(succinct::bit_vector::enumerator &it, uint32_t key_id, uint32_t num_codes, size_t next) const -> code_t {
    size_t pos = it.position();
    uint32_t idx_width = width_in_bits(num_codes);
    size_t len = next - pos;
    size_t num_blocks = len / (idx_width + bv_block_sz_);
    size_t bv_start = pos + idx_width*num_blocks;

    uint32_t target_rank = key_id + 1;
    uint32_t block_idx;
    if (num_blocks == 0 || target_rank <= macros_.get_bits(pos, idx_width)) {
      block_idx = 0;
    } else {
      uint32_t left = 0, right = num_blocks - 1;
      uint32_t cutoff = 64*8*8 / idx_width;  // 8 cachelines
      while (left + cutoff < right) {
        uint32_t mid = (left + right + 1) / 2;
        if (macros_.get_bits(pos + mid*idx_width, idx_width) < target_rank) {
          left = mid;
        } else {
          right = mid - 1;
        }
      }
      while (left + 1 < num_blocks && macros_.get_bits(pos + (left+1)*idx_width, idx_width) < target_rank) {
        left++;
      }
      target_rank -= macros_.get_bits(pos + left*idx_width, idx_width);
      block_idx = left + 1;
    }
    it.move(bv_start + block_idx*bv_block_sz_);
    while (it.position() + 64 < next) {
      uint64_t bits = it.take(64);
      uint32_t rank = static_cast<uint32_t>(__builtin_popcountll(bits));
      if (target_rank <= rank) {
        return it.position() + selectll(bits, target_rank) - 64 - bv_start;
      }
      target_rank -= rank;
    }
    size_t p = it.position();
    uint64_t bits = it.take(next - p);
    assert(static_cast<uint32_t>(__builtin_popcountll(bits)) >= target_rank);
    return p + selectll(bits, target_rank) - bv_start;
  }

  auto get_dense(succinct::bit_vector::enumerator &it, uint32_t key_id) const -> code_t {
    return key_id;
  }

  auto match_code_rev(const key_type &key, uint32_t begin, code_t code) const -> uint32_t {
    assert(code > 0);

    uint32_t base = alphabet_.alphabet_size();
    uint32_t matched_len = 0;
    while (code % base == 0) {
      code /= base;
    }
    while (code > 0) {
      uint8_t label = code % base;
      if (begin + matched_len >= key.size() || !is_legal(key[begin + matched_len]) ||
          encode(key[begin + matched_len]) != label) {
        return -1;
      }
      code /= base;
      matched_len++;
    }
    return matched_len;
  }

  auto match_code_rev_remap(const key_type &key, uint32_t begin, code_t code, const Alphabet &remap) const -> uint32_t {
    assert(code > 0);

    uint32_t base = remap.alphabet_size();
    uint32_t matched_len = 0;
    while (code % base == 0) {
      code /= base;
    }
    while (code > 0) {
      uint8_t label = code % base;
      if (begin + matched_len >= key.size() || !is_legal(key[begin + matched_len], remap) ||
          encode(key[begin + matched_len], remap) != label) {
        return -1;
      }
      code /= base;
      matched_len++;
    }
    return matched_len;
  }

  auto match_unary_path_rev(succinct::bit_vector::enumerator &it, const key_type &key,
                            uint32_t begin, size_t next) const -> uint32_t {
    uint32_t label_width = log2_ceil(alphabet_.alphabet_size() - 1);
    assert((next - it.position()) % label_width == 0);
    size_t start = it.position();
    uint32_t depth = (next - start) / label_width;
    for (uint32_t i = 0; i < depth; i++) {
      uint8_t label = macros_.get_bits(start + (depth - 1 - i)*label_width, label_width) + 1;
      if (!is_legal(key[begin + i]) || label != encode(key[begin + i])) {
        return -1;
      }
    }
    return depth;
  }

  auto match_unary_path_remap_rev(succinct::bit_vector::enumerator &it, const key_type &key,
                                  const Alphabet &remap, uint32_t begin, size_t next) const -> uint32_t {
    uint32_t label_width = log2_ceil(remap.alphabet_size() - 1);
    assert((next - it.position()) % label_width == 0);
    size_t start = it.position();
    uint32_t depth = (next - start) / label_width;
    for (uint32_t i = 0; i < depth; i++) {
      uint8_t label = macros_.get_bits(start + (depth - 1 - i)*label_width, label_width) + 1;
      if (!is_legal(key[begin + i], remap) || label != encode(key[begin + i], remap)) {
        return -1;
      }
    }
    return depth;
  }

  void write_metadata(succinct::bit_vector_builder &bv, encoding_t encoding, bool prefix_key,
                      uint32_t depth, size_t degree) {
    assert(depth >= 1 && depth <= max_depth_);
    bv.append_bits(static_cast<uint64_t>(encoding), encoding_bits_);  // write encoding
    bv.append_bits(static_cast<uint64_t>(prefix_key), 1);  // write prefix key indicator
    bv.append_bits(static_cast<uint64_t>(depth - 1), depth_bits_);  // skip 0
  #ifdef __DEGREE_IN_PLACE__
    if (degree < degree_threshold_) {  // search in topology
      bv.append_bits(static_cast<uint64_t>(0), 1);
    } else {  // store degree in place
      bv.append_bits(static_cast<uint64_t>(0), 1);
      ds2i::write_delta(bv, degree);
    }
  #endif
  }

  // we don't write bit 0 as it is always reserved for terminator
  void write_alphabet(succinct::bit_vector_builder &bv, const Alphabet &alphabet) {
    uint32_t alphabet_size = alphabet_.alphabet_size();
    if (alphabet_size <= 63) {
      bv.append_bits(alphabet.bits_[0] >> 1, alphabet_size);
    } else {
      bv.append_bits(alphabet.bits_[0] >> 1, 63);
      alphabet_size -= 63;
      int i = 1;
      while (alphabet_size > 64) {
        bv.append_bits(alphabet.bits_[i], 64);
        i++;
        alphabet_size -= 64;
      }
      bv.append_bits(alphabet.bits_[i], alphabet_size);
    }
  }

  void read_alphabet(succinct::bit_vector::enumerator &it, Alphabet &remap) const {
    uint32_t alphabet_size = alphabet_.alphabet_size();
    if (alphabet_size <= 63) {
      remap.bits_[0] = (it.take(alphabet_size) << 1) | 1;
    } else {
      remap.bits_[0] = (it.take(63) << 1) | 1;
      alphabet_size -= 63;
      int i = 1;
      while (alphabet_size > 64) {
        remap.bits_[i] = it.take(64);
        i++;
        alphabet_size -= 64;
      }
      remap.bits_[i] = it.take(alphabet_size);
    }
    remap.build_index();
  }

  auto encode(uint8_t label) const -> uint8_t {
    return alphabet_.encode(label);
  }

  auto encode(uint8_t label, const Alphabet &remap) const -> uint8_t {
    return remap.encode(alphabet_.encode(label));
  }

  auto is_legal(uint8_t label) const -> bool {
    return alphabet_.get(label);
  }

  auto is_legal(uint8_t label, const Alphabet &remap) const -> bool {
    return alphabet_.get(label) && remap.get(alphabet_.encode(label));
  }

  auto is_legal(const key_type &key, uint32_t begin, uint32_t size) const -> bool {
    uint32_t end = std::min(begin + size, static_cast<uint32_t>(key.size()));
    for (uint32_t i = begin; i < end; i++) {
      if (!is_legal(key[i])) {
        return false;
      }
    }
    return true;
  }

  auto is_legal(const key_type &key, uint32_t begin, uint32_t size, const Alphabet &remap) const -> bool {
    uint32_t end = std::min(begin + size, static_cast<uint32_t>(key.size()));
    for (uint32_t i = begin; i < end; i++) {
      if (!is_legal(key[i], remap)) {
        return false;
      }
    }
    return true;
  }

  // encode key using the global alphabet
  auto encode(const key_type &key, uint32_t begin, uint32_t size) const -> code_t {
    uint32_t base = alphabet_.alphabet_size();
    uint32_t end = begin + size;
    code_t ret = 0;
    if (end <= key.size()) {
      for (uint32_t i = begin; i < end; i++) {
        ret = ret*base + alphabet_.encode(key[i]);
      }
    } else {
      for (uint32_t i = begin; i < key.size(); i++) {
        ret = ret*base + alphabet_.encode(key[i]);
      }
      for (uint32_t i = key.size(); i < end; i++) {  // pad 0 till aligned
        ret *= base;
      }
    }
    return ret;
  }

  // encode key using the local alphabet
  auto encode(const key_type &key, uint32_t begin, uint32_t size, const Alphabet &remap) const -> code_t {
    uint32_t base = remap.alphabet_size();
    uint32_t end = begin + size;
    code_t ret = 0;
    if (end <= key.size()) {
      for (uint32_t i = begin; i < end; i++) {
        ret = ret*base + remap.encode(alphabet_.encode(key[i]));
      }
    } else {
      for (uint32_t i = begin; i < key.size(); i++) {
        ret = ret*base + remap.encode(alphabet_.encode(key[i]));
      }
      for (uint32_t i = key.size(); i < end; i++) {  // pad 0 till aligned
        ret *= base;
      }
    }
    return ret;
  }

  static auto actual_key_len(code_t code, uint32_t depth, const Alphabet &alphabet) -> uint32_t {
    assert(code != 0);
    uint32_t base = alphabet.alphabet_size();
    uint32_t padded_len = 0;
    while (code % base == 0) {
      code /= base;
      padded_len++;
    }
    return depth - padded_len;
  }

  // checks if `code1` is a strict prefix of `code2`
  // if yes, return the length of prefix; otherwise return -1
  static auto is_prefix(code_t code1, code_t code2, uint32_t depth, const Alphabet &alphabet) -> uint32_t {
    assert(code1 != 0 && code2 != 0);
    uint32_t base = alphabet.alphabet_size();
    uint32_t padded_len = 0;
    while (code1 % base == 0) {
      code1 /= base;
      code2 /= base;
      padded_len++;
    }
    if (code1 == code2) {
      return depth - padded_len;
    }
    return -1;
  }

  template <bool rev = reverse, typename = std::enable_if_t<rev>>
  void finish() {
    topo_.finish();
  }

  template<typename K, bool r> friend class CoCoCC;

  Alphabet alphabet_;
  topo_t topo_;
  next_trie_t *next_trie_;
  sdsl::int_vector<> ptrs_;
  succinct::bit_vector macros_;  // macro node encoding
  uint32_t link_bits_{0};  // #bits consumed by each link pointer
};

#undef DEBUG