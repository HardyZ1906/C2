#pragma once

#include "utils.hpp"
#include "key_set.hpp"
#include "alphabet.hpp"
#include "louds_cc.hpp"
#include "fst_cc.hpp"
#include "marisa_cc.hpp"
#include "coco_optimizer.hpp"
#include "bit_vector.hpp"

#include "../lib/ds2i/succinct/bit_vector.hpp"
#include "../lib/ds2i/succinct/bp_vector.hpp"
#include <sdsl/int_vector.hpp>

#include <vector>
#include <queue>


// #define __DEBUG_COCO__
#ifdef __DEBUG_COCO__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif

#define __BENCH_COCO__
#ifdef __BENCH_COCO__
# define BENCH(foo) foo
#else
# define BENCH(foo)
#endif


template <typename Key>
class CoCoCC {
 public:
  using key_type = Key;
  using strpool_t = StringPool<key_type>;
  using optimizer_t = CoCoOptimizer<key_type>;
  using encoding_t = optimizer_t::encoding_t;
  using topo_t = LoudsCC;
  using bitvec_t = BitVector;

  static constexpr uint8_t encoding_bits_ = optimizer_t::encoding_bits_;
  static constexpr uint32_t max_depth_ = optimizer_t::max_depth_;
  static constexpr uint8_t depth_bits_ = optimizer_t::depth_bits_;
  static constexpr uint32_t degree_threshold_ = optimizer_t::degree_threshold_;
  static constexpr uint32_t bv_block_sz_ = optimizer_t::bv_block_sz_;

#ifdef __BENCH_COCO__
  static uint64_t build_fst_time_;
  static uint64_t optimize_time_;
  static uint64_t build_trie_time_;
  static uint64_t build_tail_time_;
#endif
  static void print_bench() {
  #ifdef __BENCH_COCO__
    printf("build FST: %lf ms, optimize: %lf ms, build trie: %lf ms, build tail: %lf ms\n",
           (double)build_fst_time_/1000000, (double)optimize_time_/1000000,
           (double)build_trie_time_/1000000, (double)build_tail_time_/1000000);
  #else
    printf("disabled\n");
  #endif
  }

  void print_space_cost_breakdown() const {
    size_t topo = topo_.size_in_bits();
    size_t link = is_link_.size_in_bits() + sdsl::size_in_bytes(ptrs_) * 8;
    size_t data = macros_.size();
    next_->space_cost_breakdown(topo, link, data);
    printf("topology: %lf MB, link: %lf MB, data: %lf MB\n", (double)topo/mb_bits, (double)link/mb_bits, (double)data/mb_bits);
  }

  CoCoCC() = default;

  CoCoCC(optimizer_t &opt, int max_recursion = 0, int mask = 0) {
    build(opt, max_recursion, mask);
  }

  template <typename Iterator>
  CoCoCC(Iterator begin, Iterator end, bool sorted = false,
         uint32_t space_relaxation = 0, int max_recursion = 0, int mask = 0) {
    build(begin, end, sorted, space_relaxation, max_recursion, mask);
  }

  ~CoCoCC() {
    delete next_;
  }

  template <typename Iterator>
  void build(Iterator begin, Iterator end, bool sorted = false,
             uint32_t space_relaxation = 0, int max_recursion = 0, int mask = 0) {
    KeySet<key_type> key_set;
    while (begin != end) {
      key_set.emplace_back(&(*begin));
      ++begin;
    }
    if (!sorted) {
      key_set.sort();
    }

    BENCH( auto t0 = std::chrono::high_resolution_clock::now(); )
    typename optimizer_t::trie_t trie;
    trie.build(key_set, true, max_recursion);
    BENCH( auto t1 = std::chrono::high_resolution_clock::now(); )
    optimizer_t opt(&trie);
    opt.optimize(space_relaxation);
    BENCH( auto t2 = std::chrono::high_resolution_clock::now(); )
    build(opt, key_set.space_cost(), max_recursion, mask);
    BENCH( build_fst_time_ += (t1 - t0).count(); )
    BENCH( optimize_time_ += (t2 - t1).count(); )
  }

  void build(optimizer_t &opt, size_t original_size = 0, int max_recursion = 0, int mask = 0) {
    size_t total_depth = 0;
    size_t nef = 0, npa = 0, nbv = 0, nde = 0, nefr = 0, npar = 0, nbvr = 0, nder = 0;

    alphabet_ = opt.alphabet_;
    topo_.reserve(opt.states_[0].num_macros_ + opt.states_[0].num_leaves_);
    ptrs_.resize(opt.states_[0].num_macros_ + 1);

    succinct::bit_vector_builder bv;

    auto &old_suffixes = reinterpret_cast<optimizer_t::trie_t::TempStringPool *>(opt.trie_->next_)->keys_;

    std::queue<std::pair<uint32_t, uint32_t>> queue;
    KeySet<key_type> suffixes;
    auto push_to_queue = [&](uint32_t x, uint32_t y) {
      queue.push(std::make_pair(x, y));
    };
    auto push_child = [&](uint32_t pos) {
      if (opt.trie_->topo_.has_child(pos)) {
        push_to_queue(opt.trie_->topo_.child_pos(pos), -1);
      } else {
        auto leaf_id = opt.trie_->topo_.leaf_id(pos);
        if (!opt.trie_->is_link_.get(leaf_id)) {
          push_to_queue(-1, -1);
        } else {
          push_to_queue(-1, opt.trie_->is_link_.rank1(leaf_id));
        }
      }
    };

    BENCH( auto t0 = std::chrono::high_resolution_clock::now(); )
    push_to_queue(0, -1);
    uint32_t macro_id = 0;
    while (!queue.empty()) {
      auto [pos, link_id] = queue.front();
      queue.pop();
      if (pos == -1) {  // leaf node
        // DEBUG( printf("leaf node\n"); )
        topo_.add_node(0);
        if (link_id != -1) {
          DEBUG( printf("link leaf node, %d: %s\n", link_id, old_suffixes[link_id].materialize().c_str()); )
          suffixes.push_back(old_suffixes[link_id]);
          is_link_.append1();
        } else {
          DEBUG( printf("regular leaf node\n"); )
          is_link_.append0();
        }
        continue;
      }  // else macro node
      ptrs_[macro_id++] = bv.size();  // set pointer

      uint32_t node_id = opt.trie_->topo_.node_id(pos);
      const typename optimizer_t::state_t &state = opt.states_[node_id];

      bool prefix_key = false;
      if (opt.trie_->get_label(pos) == terminator_) {  // mark key as null instead of encoding it
        prefix_key = true;
        pos++;
        push_to_queue(-1, -1);
      }
      encoding_t encoding = state.encoding_;
      uint32_t depth = state.depth_;

      total_depth += depth;
      switch (encoding) {
      case encoding_t::ELIAS_FANO:
        nef++;
        break;
      case encoding_t::EF_REMAP:
        nefr++;
        break;
      case encoding_t::PACKED:
        npa++;
        break;
      case encoding_t::PA_REMAP:
        npar++;
        break;
      case encoding_t::BITVECTOR:
        nbv++;
        break;
      case encoding_t::BV_REMAP:
        nbvr++;
        break;
      case encoding_t::DENSE:
        nde++;
        break;
      case encoding_t::DE_REMAP:
        nder++;
      }

      DEBUG(
        printf("macro node %d: pos = %d, depth = %d, encoding = %d, ptr = %ld, total cost = %lf\n",
               macro_id - 1, pos, depth, encoding, bv.size(), state.enc_cost_);
        uint32_t key_idx = 0;
        if (prefix_key) {
          printf("key %d: (null)\n", key_idx++);
        }
      )

      // traverse and encode all keys
      typename optimizer_t::trie_t::walker walker(opt.trie_, pos);
      walker.get_min_key(depth);
      push_child(walker.pos_);

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
        uint32_t width = optimizer_t::code_len(alphabet_, depth);
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
        uint32_t width = optimizer_t::code_len(state.remap_, depth);
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
    is_link_.build();
    sdsl::util::bit_compress(ptrs_);
    new (&macros_) succinct::bit_vector(&bv);
    BENCH( auto t1 = std::chrono::high_resolution_clock::now(); )

    next_ = strpool_t::build_optimal(suffixes, nullptr, original_size, max_recursion, mask);
    BENCH( auto t2 = std::chrono::high_resolution_clock::now(); )
    BENCH( build_trie_time_ += (t1 - t0).count(); )
    BENCH( build_tail_time_ += (t2 - t1).count(); )

    size_t n = macro_id;
    printf("num nodes = %ld, average depth = %lf\n", n, (double)total_depth/n);
    printf("ef = %lf, pa = %lf, bv = %lf, de = %lf\n", (double)nef/n, (double)npa/n, (double)nbv/n, (double)nde/n);
    printf("efr = %lf, par = %lf, bvr = %lf, der = %lf\n", (double)nefr/n, (double)npar/n, (double)nbvr/n, (double)nder/n);
  }

  auto lookup(const key_type &key) const -> uint32_t {
    return lookup(key, 0, key.size());
  }

  // return leaf id (-1 if not found)
  auto lookup(const key_type &key, uint32_t begin, uint32_t len) const -> uint32_t {
    assert(begin + len <= key.size());
    uint32_t end = begin + len;
    uint32_t pos = 0;
    assert(!topo_.is_leaf(pos));  // root node must not be empty
    uint32_t matched_len = begin;

    while (true) {
      uint32_t macro_id = topo_.internal_id(pos);
      succinct::bit_vector::enumerator it(macros_, ptrs_[macro_id]);
      size_t next = ptrs_[macro_id + 1];
      encoding_t encoding = static_cast<encoding_t>(it.take(encoding_bits_));
      Alphabet remap;

      // read metadata
      bool prefix_key = it.take(1);
      if (matched_len >= end) {
        return prefix_key ? topo_.leaf_id(topo_.child_pos(pos)) : -1;
      }
      uint32_t depth = it.take(depth_bits_) + 1;
      if (!is_legal(key, matched_len, depth)) {  // contains illegal label
        return -1;
      }
    #ifdef __DEGREE_IN_PLACE__
      bool degree_in_place = it.take(1);
      uint32_t degree = (degree_in_place ? ds2i::read_delta(it) : topo_.degree(pos));
    #else
      uint32_t degree = topo_.degree(pos);
    #endif
      assert(degree >= 1 + prefix_key);
      // read first code
      code_t first_code, code;
      bool is_remap = (static_cast<int>(encoding) & 0x4);
      if (!is_remap) {
        code = encode(key, matched_len, depth);
        first_code = it.take(optimizer_t::code_len(alphabet_, depth));
      } else {
        read_alphabet(it, remap);
        code = encode_safe(key, matched_len, depth, remap);
        first_code = it.take(optimizer_t::code_len(remap, depth));
      }

      uint32_t child_id;
      code_t lower_bound;
      // compare against first code
      if (code < first_code) {
        return -1;
      } else if (code == first_code) {
        child_id = prefix_key;
        lower_bound = first_code;
      } else {  // search encoding
        code_t target = code - first_code - 1;
        std::pair<uint32_t, code_t> lb;
        switch (encoding) {
         case encoding_t::ELIAS_FANO:
         case encoding_t::EF_REMAP:
          lb = lower_bound_elias_fano(it, next, degree - prefix_key - 1, target);
          break;
         case encoding_t::PACKED:
         case encoding_t::PA_REMAP:
          lb = lower_bound_packed(it, next, degree - prefix_key - 1, target);
          break;
         case encoding_t::BITVECTOR:
         case encoding_t::BV_REMAP:
          lb = lower_bound_bitvector(it, next, degree - prefix_key - 1, target);
          break;
         case encoding_t::DENSE:
         case encoding_t::DE_REMAP:
          lb = lower_bound_dense(it, next, degree - prefix_key - 1, target);
          break;
         default: assert(false); // should not happen
        }
        child_id = lb.first + 1 + prefix_key;
        lower_bound = lb.second + 1 + first_code;
      }
      assert(child_id < degree);
      DEBUG(
        printf("key: %s, code = %ld, pos %d, macro node %d, child %d, encoding = %d, depth = %d, degree = %d, prefix key = %d\n",
               key.substr(matched_len, depth).c_str(), code, pos + child_id, macro_id, child_id, encoding, depth, degree, prefix_key);
      )
      pos = topo_.child_pos(pos + child_id);

      if (!topo_.is_leaf(pos)) {
        if (code != lower_bound) {
          return -1;
        }
        matched_len += depth;  // must be full match
      } else {
        uint32_t prefix_len = is_prefix(lower_bound, code, depth, is_remap ? remap : alphabet_);
        if (prefix_len == -1) {
          return -1;
        }
        matched_len += prefix_len;

        uint32_t leaf_id = topo_.leaf_id(pos);
        if (!is_link_.get(leaf_id) && matched_len != key.size()) {  // unmatched suffixes
          return -1;
        }
        if (is_link_.get(leaf_id) &&
            next_->match(key, matched_len, is_link_.rank1(leaf_id)) != key.size() - matched_len) {  // suffix mismatch          
          return -1;
        }
        return leaf_id;
      }
    }
  }

  auto encoding_size() const -> size_t {
    return macros_.size();
  }

  auto size_in_bits() const -> size_t {
    return topo_.size_in_bits() + sdsl::size_in_bytes(ptrs_)*8 + macros_.size() + next_->size_in_bits() + is_link_.size_in_bits();
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

  // all the `lower_bound_xx` functions return {index, value}
  auto lower_bound_elias_fano(succinct::bit_vector::enumerator &it, size_t next,
                              uint32_t num_codes, code_t target) const -> std::pair<uint32_t, code_t> {
    code_t universe = ds2i::read_delta(it);
    typename ds2i::compact_elias_fano<code_t>::enumerator enu(macros_, it.position(), universe, num_codes, params);
    if (target >= universe) {
      auto last = enu.move(num_codes - 1);
      assert(last.first == num_codes - 1);
      return {last.first, last.second};
    }
    auto result = enu.next_geq(target + 1);
    if (result.first == 0) {
      return {-1, -1};
    }
    return {result.first - 1, enu.prev_value()};
  }

  auto lower_bound_packed(succinct::bit_vector::enumerator &it, size_t next,
                          uint32_t num_codes, code_t target) const -> std::pair<uint32_t, code_t> {
    assert((next - it.position()) % num_codes == 0);

    size_t pos = it.position();
    uint32_t width = (next - pos) / num_codes;
    uint32_t cutoff = 64*8*8 / width;  // 8 cachelines
    uint32_t left = 0, right = (next - pos) / width - 1;
    while (left + cutoff < right) {
      uint32_t mid = (left + right + 1) / 2;
      if (macros_.get_bits(pos + mid*width, width) <= target) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }
    while (pos + (left+1)*width < next && macros_.get_bits(pos + (left+1)*width, width) <= target) {
      left++;
    }
    code_t val = macros_.get_bits(pos + left*width, width);
    if (val > target) {
      return {-1, -1};
    }
    return {left, val};
  }

  auto lower_bound_bitvector(succinct::bit_vector::enumerator &it, size_t next,
                             uint32_t num_codes, code_t target) const -> std::pair<uint32_t, code_t> {
    size_t pos = it.position();
    uint32_t idx_width = width_in_bits(num_codes);
    size_t len = next - pos;
    size_t num_blocks = len / (idx_width + bv_block_sz_);
    size_t bv_start = pos + idx_width*num_blocks;
    code_t universe = next - bv_start;

    auto rank1 = [&](size_t target_pos) -> uint32_t {  // exclusive
      uint32_t block_idx = (target_pos - bv_start) / bv_block_sz_;
      uint32_t rank = block_idx == 0 ? 0 : macros_.get_bits(pos + (block_idx - 1)*idx_width, idx_width);
      it.move(bv_start + block_idx*bv_block_sz_);
      while (target_pos - it.position() > 64) {
        rank += __builtin_popcountll(it.take(64));
      }
      rank += __builtin_popcountll(it.take(target_pos - it.position()));
      return rank;
    };

    if (target >= universe) {
      target = universe - 1;
    }
    size_t target_pos = bv_start + target;
    if (!macros_[target_pos] && (target_pos = macros_.predecessor1(target_pos)) < bv_start) {
      return {-1, -1};
    }
    return {rank1(target_pos), target_pos - bv_start};
  }

  auto lower_bound_dense(succinct::bit_vector::enumerator &it, size_t next,
                         uint32_t num_codes, code_t target) const -> std::pair<uint32_t, code_t> {
    if (target < num_codes) {
      return {target, target};
    } else {
      return {num_codes - 1, num_codes - 1};
    }
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

  // encode key using the global alphabet
  auto encode(const key_type &key, uint32_t begin, uint32_t len) const -> code_t {
    uint32_t base = alphabet_.alphabet_size();
    uint32_t end = begin + len;
    code_t ret = 0;
    if (end <= key.size()) {
      for (uint32_t i = begin; i < end; i++) {
        ret = ret*base + encode(key[i]);
      }
    } else {
      for (uint32_t i = begin; i < key.size(); i++) {
        ret = ret*base + encode(key[i]);
      }
      for (uint32_t i = key.size(); i < end; i++) {  // pad 0 till aligned
        ret *= base;
      }
    }
    return ret;
  }

  // encode key using the local alphabet
  auto encode(const key_type &key, uint32_t begin, uint32_t len, const Alphabet &remap) const -> code_t {
    uint32_t base = remap.alphabet_size();
    uint32_t end = begin + len;
    code_t ret = 0;
    if (end <= key.size()) {
      for (uint32_t i = begin; i < end; i++) {
        ret = ret*base + encode(key[i], remap);
      }
    } else {
      for (uint32_t i = begin; i < key.size(); i++) {
        ret = ret*base + encode(key[i], remap);
      }
      for (uint32_t i = key.size(); i < end; i++) {  // pad 0 till aligned
        ret *= base;
      }
    }
    return ret;
  }

  // keep encoding until we meet the first illegal label, after which zeros are padded till aligned
  auto encode_safe(const key_type &key, uint32_t begin, uint32_t len, const Alphabet &remap) const -> code_t {
    uint32_t base = remap.alphabet_size();
    uint32_t end = std::min(begin + len, static_cast<uint32_t>(key.size()));
    code_t ret = 0;
    uint32_t i = begin;
    while (i < end) {
      if (!is_legal(key[i], remap)) {
        break;
      }
      ret = ret*base + encode(key[i], remap);
      i++;
    }
    while (i < begin + len) {
      ret *= base;
      i++;
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

  Alphabet alphabet_;
  topo_t topo_;
  bitvec_t is_link_;
  sdsl::int_vector<> ptrs_;
  succinct::bit_vector macros_;  // macro node encoding
  strpool_t *next_{nullptr};
};

#ifdef __BENCH_COCO__
template <typename K> uint64_t CoCoCC<K>::build_fst_time_ = 0;
template <typename K> uint64_t CoCoCC<K>::optimize_time_ = 0;
template <typename K> uint64_t CoCoCC<K>::build_trie_time_ = 0;
template <typename K> uint64_t CoCoCC<K>::build_tail_time_ = 0;
#endif

#undef DEBUG
#undef BENCH