#pragma once

#include "utils.hpp"
#include "ls4coco.hpp"
#include "alphabet.hpp"
#include "../lib/ds2i/compact_elias_fano.hpp"
#include "../lib/ds2i/integer_codes.hpp"

#include <vector>

// #define __DEBUG_OPTIMIZER__
#ifdef __DEBUG_OPTIMIZER__
# include <queue>
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


template<typename Key>
class CoCoOptimizer {
 public:
  enum class encoding_t {
    ELIAS_FANO = 0, PACKED = 1, BITVECTOR = 2, DENSE = 3,
    EF_REMAP = 4, PA_REMAP = 5, BV_REMAP = 6, DE_REMAP = 7,
    RECURSIVE = 8, COUNT = 9,
  };
  static constexpr size_t encoding_bits_ = log2_ceil(static_cast<uint32_t>(encoding_t::COUNT));
  static constexpr uint32_t max_depth_ = 32;  // max # of collapsed levels
  static constexpr uint8_t depth_bits_ = log2_ceil(max_depth_);
  static constexpr uint8_t terminator_ = 0;
  static constexpr uint32_t bv_block_sz_ = 1024;
  static constexpr uint32_t degree_threshold_ = 1 << 12;

  using key_type = Key;
  using alphabet_t = Alphabet;
  using trie_t = LS4CoCo<key_type>;

  struct state_s {
    alphabet_t remap_;  // local alphabet of optimal macro node
    size_t enc_cost_{0};   // optimal space cost of encoding
    uint32_t depth_{0};    // depth of optimal macro node
    uint32_t num_macros_{0};  // # of macro nodes in optimal subtrie
    uint32_t num_leaves_{0};  // # of leaf nodes in optimal subtrie
    encoding_t encoding_{encoding_t::COUNT};
  };

  CoCoOptimizer(const trie_t *trie) : trie_(trie) {
    init();
  }

  void init() {
    assert(trie_ != nullptr);
    states_.resize(trie_->num_nodes());
    get_global_alphabet();
  }

  void optimize() {
    assert(trie_ != nullptr);

    std::vector<uint32_t> levels = trie_->get_level_boundaries();
    // bottom-up optimization
    for (size_t i = levels.size() - 1; i > 0; i--) {
      uint32_t pos = levels[i-1];
      DEBUG( printf("optimize level %d(%d:%d)\n", i - 1, levels[i-1], levels[i]); )
      uint32_t count = 0;
      while (pos < levels[i]) {  // optimize level
        DEBUG( printf("optimize node %d\n", pos); )
        optimize_node(pos);
        pos = trie_->node_end(pos);
        count++;
      }
    }
  }

  auto get_final_cost() const -> std::pair<size_t, size_t> {
    return {states_[0].enc_cost_, space_cost_total(states_[0])};
  }

  // optimize node at position pos
  // @require: all descendents must have been optimized
  void optimize_node(uint32_t pos) {
    uint32_t left = trie_->node_start(pos), right = trie_->node_end(pos) - 1;  // level boundaries
    uint32_t root_id = trie_->node_id(left);
    state_s &state = states_[root_id];
    // printf("optimize node %d:%d, %d\n", left, right, root_id);

    key_type min_key, max_key;  // universe
    bool min_key_found = false, max_key_found = false;
    uint32_t num_full_keys = 0;

    alphabet_t remap;  // local alphabet
    remap.set1(0);  // 0 is reserved for terminator

    size_t best_cost = std::numeric_limits<size_t>::max();
    uint32_t depth = 1;
    uint32_t max_depth_no_remap = max_key_len(alphabet_);
    bool prefix_key = false;
    if (trie_->get_label(left) == terminator_) {  // we don't encode null key but would create a leaf node for it
      left++;
      prefix_key = true;
    }
    while (depth <= max_depth_) {
      // update min/max keys(universe)
      if (!min_key_found) {
        uint8_t label = trie_->get_label(left);
        if (label != terminator_) {
          min_key.push_back(label);
        }
        min_key_found = !trie_->bv_.template get<0>(left);  // has_child.get(left)
      }
      if (!max_key_found) {
        uint8_t label = trie_->get_label(right);
        if (label != terminator_) {
          max_key.push_back(label);
        }
        max_key_found = !trie_->bv_.template get<0>(right);  // has_child.get(right)
      }

      // compute number of keys, local alphabet and descendent costs
      uint32_t next_level_left = std::numeric_limits<uint32_t>::max(), next_level_right = 0;
      uint32_t num_partial_keys = 0;
      size_t desc_enc_cost = 0;
      uint32_t desc_num_macros = 0, desc_num_leaves = 0;
      for (uint32_t i = left; i <= right; i++) {
        remap.set1(alphabet_.encode(trie_->get_label(i)));
        if (!trie_->bv_.template get<0>(i)) {  // !has_child.get(i)
          num_full_keys++;
        } else {
          num_partial_keys++;

          uint32_t desc_pos = trie_->child_pos(i);
          uint32_t desc_id = trie_->node_id(desc_pos);
          state_s &desc_state = states_[desc_id];
          assert(desc_state.encoding_ != encoding_t::COUNT);
          desc_enc_cost += desc_state.enc_cost_;
          desc_num_macros += desc_state.num_macros_;
          desc_num_leaves += desc_state.num_leaves_;

          next_level_left = std::min<uint32_t>(next_level_left, desc_pos);
          next_level_right = std::max<uint32_t>(next_level_right, trie_->node_end(desc_pos) - 1); 
        }
      }

      // determine optimal encoding
      remap.build_index();
      if (depth > max_key_len(remap)) {
        break;
      }
      encoding_t best_encoding = encoding_t::COUNT;
      size_t best_enc_cost = std::numeric_limits<size_t>::max();
      uint32_t num_keys = num_full_keys + num_partial_keys;
      assert(num_keys >= 1);
      if (depth <= max_depth_no_remap) {  // using global alphabet
        uint32_t width = code_len(alphabet_, depth);
        code_t min_code = encode(min_key, depth), max_code = encode(max_key, depth);
        code_t universe = max_code - min_code;
        if (code_t(num_keys - 1) == universe) {
          size_t enc_cost = space_cost_dense(universe, width, num_keys, prefix_key);
          if (enc_cost < best_enc_cost) {
            best_encoding = encoding_t::DENSE;
            best_enc_cost = space_cost_dense(universe, width, num_keys, prefix_key);
          }
          // TODO: recursive encoding
        } else {
          assert(num_keys > 1);
          size_t enc_cost = space_cost_elias_fano(universe, width, num_keys, prefix_key);
          if (enc_cost < best_enc_cost) {
            best_encoding = encoding_t::ELIAS_FANO;
            best_enc_cost = enc_cost;
          }
          enc_cost = space_cost_packed(universe, width, num_keys, prefix_key);
          if (enc_cost < best_enc_cost) {
            best_encoding = encoding_t::PACKED;
            best_enc_cost = enc_cost;
          }
          enc_cost = space_cost_bitvector(universe, width, num_keys, prefix_key);
          if (enc_cost < best_enc_cost) {
            best_encoding = encoding_t::BITVECTOR;
            best_enc_cost = enc_cost;
          }
        }
      }
      // using local alphabet
      uint32_t width = code_len(remap, depth);
      code_t min_code = encode(min_key, depth, remap), max_code = encode(max_key, depth, remap);
      code_t universe = max_code - min_code;
      if (code_t(num_keys - 1) == universe) {
        size_t enc_cost = space_cost_dense_remap(universe, width, num_keys, prefix_key);
        if (enc_cost < best_enc_cost) {
          best_encoding = encoding_t::DE_REMAP;
          best_enc_cost = enc_cost;
        }
        // TODO: recursive encoding
      } else {
        assert(num_keys > 1);
        size_t enc_cost = space_cost_elias_fano_remap(universe, width, num_keys, prefix_key);
        if (enc_cost < best_enc_cost) {
          best_encoding = encoding_t::EF_REMAP;
          best_enc_cost = enc_cost;
        }
        enc_cost = space_cost_packed_remap(universe, width, num_keys, prefix_key);
        if (enc_cost < best_enc_cost) {
          best_encoding = encoding_t::PA_REMAP;
          best_enc_cost = enc_cost;
        }
        enc_cost = space_cost_bitvector_remap(universe, width, num_keys, prefix_key);
        if (enc_cost < best_enc_cost) {
          best_encoding = encoding_t::BV_REMAP;
          best_enc_cost = enc_cost;
        }
      }
      // TODO: space relaxation
      assert(best_encoding != encoding_t::COUNT);
      size_t enc_cost = best_enc_cost + desc_enc_cost;
      uint32_t num_macros = 1 + desc_num_macros;
      uint32_t num_leaves = num_full_keys + prefix_key + desc_num_leaves;
      size_t total_cost = space_cost_total(enc_cost, num_macros, num_leaves);
      if (total_cost <= best_cost) {
        best_cost = total_cost;
        state.encoding_ = best_encoding;
        state.depth_ = depth;
        state.enc_cost_ = enc_cost;
        state.num_macros_ = num_macros;
        state.num_leaves_ = num_leaves;
        state.remap_ = remap;
      }

      // move to next level
      if (next_level_left == std::numeric_limits<uint32_t>::max()) {
        break;
      }
      left = next_level_left;
      right = next_level_right;
      depth++;
    }
    assert(state.encoding_ != encoding_t::COUNT);
  }

#ifdef __DEBUG_OPTIMIZER__
  void print_optimal() {
    std::queue<uint32_t> queue;
    queue.push(0);
    uint32_t macro_node_id = 0;
    while (!queue.empty()) {
      uint32_t pos = queue.front();
      queue.pop();
      uint32_t node_id = trie_->node_id(pos);
      state_s &state = states_[node_id];
      printf("macro node %d: encoding = %d, depth = %d, cost = %ld\n", macro_node_id, state.encoding_,
             state.depth_, space_cost_total(state));
      macro_node_id++;
      typename trie_t::walker left_walker(trie_, pos), right_walker(trie_, pos);
      right_walker.move_to_back();
      for (uint32_t i = 1; i < state.depth_; i++) {
        bool ok = left_walker.move_down_one_level_left();
        assert(ok);
        ok = right_walker.move_down_one_level_right();
        assert(ok);
      }
      // printf("%d %d\n", left_walker.pos_, right_walker.pos_);
      pos = left_walker.pos_;
      while (pos <= right_walker.pos_) {
        if (trie_->bv_.template get<0>(pos)) {  // has_child.get(pos)
          queue.push(trie_->child_pos(pos));
        }
        pos++;
      }
    }
  }
#endif

  void get_global_alphabet() {
    alphabet_.set1(0);  // 0 is always reserved for terminator
    for (size_t i = 0; i < trie_->labels_.size(); i++) {
      alphabet_.set1(trie_->get_label(i));
    }
    alphabet_.build_index();
  }

  static auto max_key_len(const alphabet_t &alphabet) -> uint32_t {
    uint32_t base = alphabet.alphabet_size();
    code_t code = 1;
    uint32_t len = 0;
    while (true) {
      if (code >= std::numeric_limits<code_t>::max() / base) {
        return len;
      }
      code *= base;
      len++;
    }
    return len;
  }

  static auto key_universe(const alphabet_t &alphabet, uint32_t key_len) -> code_t {
    assert(key_len <= max_key_len(alphabet));

    uint32_t base = alphabet.alphabet_size();
    code_t code = 0;
    for (uint32_t i = 0; i < key_len; i++) {
      code = code*base + base - 1;
    }
    return code;
  }

  static auto code_len(const alphabet_t &alphabet, uint32_t key_len) -> uint32_t {
    return width_in_bits(key_universe(alphabet, key_len));
  }

  // encode key using the global alphabet
  auto encode(const key_type &key, uint32_t alignment) const -> code_t {
    assert(key.size() <= alignment);

    uint32_t base = alphabet_.alphabet_size();
    code_t ret = 0;
    for (uint32_t i = 0; i < key.size(); i++) {
      ret = ret*base + alphabet_.encode(key[i]);
    }
    for (uint32_t i = key.size(); i < alignment; i++) {  // pad 0 till aligned
      ret *= base;
    }
    return ret;
  }

  // encode key using the local alphabet
  auto encode(const key_type &key, uint32_t alignment, const alphabet_t &remap) const -> code_t {
    assert(key.size() <= alignment);

    uint32_t base = remap.alphabet_size();
    code_t ret = 0;
    for (uint32_t i = 0; i < key.size(); i++) {
      ret = ret*base + remap.encode(alphabet_.encode(key[i]));
    }
    for (uint32_t i = key.size(); i < alignment; i++) {  // pad 0 till aligned
      ret *= base;
    }
    return ret;
  }

  // space cost of encoding + pointer + louds
  auto space_cost_total(size_t enc_cost, uint32_t num_macros, uint32_t num_leaves) const -> size_t {
    uint32_t ptr_size = log2_ceil(trie_->size_in_bits());  // estimate pointer size using the uncollapsed trie size
    size_t bv_size = (2*num_leaves + 255) / 256 * sizeof(typename trie_t::bitvec::Block);
    return enc_cost + ptr_size*num_macros + 2*num_leaves;
  }

  auto space_cost_total(const state_s &state) const -> size_t {
    return space_cost_total(state.enc_cost_, state.num_macros_, state.num_leaves_);
  }

  // space cost of elias-delta encoding `code`
  static auto space_cost_delta(uint32_t code) -> uint32_t {
    uint32_t width = 32 - __builtin_clz(code + 1);
    return 2*(32 - __builtin_clz(width + 1)) + 1 + width;
  }

  // space cost of elias-delta encoding `code`
  static auto space_cost_delta(code_t code) -> uint32_t {
    uint32_t width = 64 - __builtin_clzll(code + 1);
    return 2*(64 - __builtin_clzll(width + 1)) + 1 + width;
  }

  // type + prefix key indicator + depth (+ degree) + first code + universe + encoding
  auto space_cost_elias_fano(code_t universe, uint32_t code_len,
                             uint32_t num_keys, bool prefix_key) const -> size_t {
    assert(num_keys > 1);
    uint32_t degree = num_keys + prefix_key;
    if (degree < degree_threshold_) {
      return encoding_bits_ + depth_bits_ + 2 + code_len + space_cost_delta(universe) +
             ds2i::compact_elias_fano<code_t>::bitsize(params, universe, num_keys - 1);
    }  // otherwise there are too many keys; we store degree in place to avoid slow `next1`
    return encoding_bits_ + depth_bits_ + 2 + space_cost_delta(degree) + code_len +
           space_cost_delta(universe) + ds2i::compact_elias_fano<code_t>::bitsize(params, universe, num_keys);
  }

  // elias fano encoding + alphabet size
  auto space_cost_elias_fano_remap(code_t universe, uint32_t code_len,
                                   uint32_t num_keys, bool prefix_key) const -> size_t {
    return space_cost_elias_fano(universe, code_len, num_keys, prefix_key) + alphabet_.alphabet_size() - 1;
  }

  // type + prefix key indicator + depth (+ num codes) + first code + encoding
  auto space_cost_packed(code_t universe, uint32_t code_len,
                         uint32_t num_keys, bool prefix_key) const -> size_t {
    assert(num_keys > 1);
    uint32_t width = log2_ceil(universe);  // skip 0
    uint32_t degree = num_keys + prefix_key;
    if (degree < degree_threshold_) {
      return encoding_bits_ + depth_bits_ + 2 + code_len + width*(num_keys - 1);
    }
    return encoding_bits_ + depth_bits_ + 2 + space_cost_delta(degree) + code_len + width*(num_keys - 1);
  }

  // packed encoding + local alphabet
  auto space_cost_packed_remap(code_t universe, uint32_t code_len,
                               uint32_t num_keys, bool prefix_key) const -> size_t {
    return space_cost_packed(universe, code_len, num_keys, prefix_key) + alphabet_.alphabet_size() - 1;
  }

  // type + prefix key indicator + depth (+ num codes) + first code + rank index + encoding
  auto space_cost_bitvector(code_t universe, uint32_t code_len,
                            uint32_t num_keys, bool prefix_key) const -> size_t {
    assert(num_keys > 1);
    if (universe > std::numeric_limits<uint32_t>::max()) {
      return std::numeric_limits<size_t>::max();
    }
    size_t index_sz = log2_ceil(num_keys - 1) * (universe / bv_block_sz_);  // skip 0
    uint32_t degree = num_keys + prefix_key;
    if (degree < degree_threshold_) {
      return encoding_bits_ + depth_bits_ + 2 + code_len + index_sz + static_cast<size_t>(universe);
    }
    return encoding_bits_ + depth_bits_ + 2 + space_cost_delta(degree) + code_len + index_sz + static_cast<size_t>(universe);
  }

  // bitvector encoding + local alphabet
  auto space_cost_bitvector_remap(code_t universe, uint32_t code_len,
                                  uint32_t num_keys, bool prefix_key) const -> size_t {
    size_t enc_cost = space_cost_bitvector(universe, code_len, num_keys, prefix_key);
    if (enc_cost == std::numeric_limits<size_t>::max()) {
      return std::numeric_limits<size_t>::max();
    }
    return enc_cost + alphabet_.alphabet_size() - 1;
  }

  // type + prefix key indicator + depth (+ num codes) + first code
  auto space_cost_dense(code_t universe, uint32_t code_len,
                        uint32_t num_keys, bool prefix_key) const -> size_t {
    if (universe > std::numeric_limits<uint32_t>::max() || static_cast<uint32_t>(universe) != num_keys) {
      return std::numeric_limits<size_t>::max();
    }
    uint32_t degree = num_keys + prefix_key;
    if (degree < degree_threshold_) {
      return encoding_bits_ + depth_bits_ + 1 + code_len;
    }  // otherwise there are too many keys; we store degree in place to avoid slow `next1`
    return encoding_bits_ + depth_bits_ + 1 + space_cost_delta(degree) + code_len;
  }

  // dense encoding + local alphabet
  auto space_cost_dense_remap(code_t universe, uint32_t code_len,
                              uint32_t num_keys, bool prefix_key) const -> size_t {
    return space_cost_dense(universe, code_len, num_keys, prefix_key) + alphabet_.alphabet_size() - 1;
  }

 private:
  std::vector<state_s> states_;

  alphabet_t alphabet_;  // global alphabet

  const trie_t *trie_{nullptr};

  template<typename K> friend class CoCoCC;
};

#undef DEBUG