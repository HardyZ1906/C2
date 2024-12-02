// This file is part of CoCo-trie <https://github.com/aboffa/CoCo-trie>.
// Copyright (c) 2022 Antonio Boffa.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once


#include <queue>
#include "louds_cc.hpp"
#include "CoCo-trie_v1.hpp"


// #define __DEBUG__
#ifdef __DEBUG__
# define DEBUG(foo) foo
#else
# define DEBUG(foo)
#endif


// #define __BENCH_CC__
#ifdef __BENCH_CC__
# include <chrono>
# define BENCH_CC(foo) foo
#else
# define BENCH_CC(foo)
#endif


template<uint8_t MIN_L = 1,
         typename code_type = uint128_t,
         uint8_t MAX_L_THRS = MAX_L_THRS,
         uint8_t space_relaxation = 0,
         int spill_threshold = 128>
class CoCoCC {
public:
  using utrie_t = Trie_lw<MIN_L, code_type, MAX_L_THRS, space_relaxation>;
  using CoCo_v1_t = CoCo_v1<MIN_L, code_type, MAX_L_THRS, space_relaxation>;

  // MATCHED: perfect match; PARTIAL: strict prefix match, with extraneous suffix; FAILED: mismatch
  enum class State { MATCHED, PARTIAL, FAILED, };

  static constexpr uint8_t space_relaxation_ = space_relaxation;
  static constexpr int spill_threshold_ = spill_threshold;

  std::unique_ptr<succinct::bit_vector> internal_variable;

  std::unique_ptr<sdsl::int_vector<>> pointers_to_encoding;

  std::unique_ptr<LoudsCC<spill_threshold_>> topology_;

  size_t num_child_root = 0;

  size_t bits_for_L = log_universe((uint64_t) MAX_L_THRS);

  size_t log_sigma = log_universe(ALPHABET_SIZE);

#ifdef __BENCH_CC__
  static size_t node_nav_time_;
  static size_t key_search_time_;
  static size_t child_nav_time_;
  static size_t is_leaf_time_;
  static size_t internal_rank_time_;
  static size_t num_child_time_;

  static size_t ef_keys_;
  static size_t ef_remap_keys_;
  static size_t packed_keys_;
  static size_t packed_remap_keys_;
  static size_t bv_keys_;
  static size_t bv_remap_keys_;
  static size_t dense_keys_;
  static size_t dense_remap_keys_;

  static void print_microbenchmark() {
    printf("elias-fano: %ld, elias-fano remapped: %ld, packed: %ld, packed remapped: %ld, bitvector: %ld, bitvector remapped: %ld, "
           "dense: %ld, dense_remapped: %ld\n", ef_keys_, ef_remap_keys_, packed_keys_, packed_remap_keys_, bv_keys_, bv_remap_keys_,
           dense_keys_, dense_remap_keys_);
    printf("node navigation: %lf ms; key search: %lf ms\n", (double)node_nav_time_/1000000, (double)key_search_time_/1000000);
    printf("child navigation: %lf ms; is leaf: %lf ms; internal rank: %lf ms; num child: %lf ms\n", (double)child_nav_time_/1000000, 
           (double)is_leaf_time_/1000000, (double)internal_rank_time_/1000000, (double)num_child_time_/1000000);
    LoudsCC<spill_threshold_>::print_microbenchmark();
  }

  static void clear_microbenchmark() {
    node_nav_time_ = key_search_time_ = 0;
    child_nav_time_ = is_leaf_time_ = internal_rank_time_ = num_child_time_ = 0;
    ef_keys_ = ef_remap_keys_ = packed_keys_ = packed_remap_keys_ = bv_keys_ = bv_remap_keys_ = dense_keys_ = dense_remap_keys_ = 0;
    LoudsCC<spill_threshold_>::clear_microbenchmark();
  }
#else
  static void print_microbenchmark() {
    LoudsCC<spill_threshold_>::print_microbenchmark();
  }
  static void clear_microbenchmark() {
    LoudsCC<spill_threshold_>::clear_microbenchmark();
  }
#endif

  CoCoCC() = default;

  CoCoCC(const std::vector<std::string> &dataset, size_t l_fixed = 0) {
    build(dataset, l_fixed);
  }

  CoCoCC(utrie_t &uncompacted) {
    build(uncompacted);
  }

  void build(const std::vector<std::string> &dataset, size_t l_fixed = 0) {
    utrie_t uncompacted;
    // filling the uncompacted trie
    for (int i = 0; i < dataset.size(); i++)
      uncompacted.insert(dataset[i]);

    // computing the best number of levels to collapse into the nodes
    uncompacted.space_cost_all_nodes(l_fixed);
    uncompacted.build_actual_CoCo_children();

    topology_ = std::make_unique<LoudsCC<spill_threshold_>>(uncompacted.global_number_nodes_CoCo);
    pointers_to_encoding = std::make_unique<sdsl::int_vector<>>(uncompacted.global_number_nodes_CoCo);

    num_child_root = 1 + uncompacted.root->n_vec[uncompacted.root->l_idx];

    bits_for_L = log_universe((uint64_t) uncompacted.max_l_idx);
    build_CoCo_from_uncompacted_trie(uncompacted.root);
  }

  void build(utrie_t &uncompacted) {
    topology_ = std::make_unique<LoudsCC<spill_threshold_>>(uncompacted.global_number_nodes_CoCo);
    pointers_to_encoding = std::make_unique<sdsl::int_vector<>>(uncompacted.global_number_nodes_CoCo);

    num_child_root = 1 + uncompacted.root->n_vec[uncompacted.root->l_idx];
    bits_for_L = log_universe((uint64_t) uncompacted.max_l_idx);
    build_CoCo_from_uncompacted_trie(uncompacted.root);
  }

  // returns the LOUDS ID of `to_search`; if not found, return -1
  auto look_up(const std::string &to_search) const -> size_t {
    size_t scanned_chars = 0, node_rank;
    if (look_up(to_search, scanned_chars, node_rank) == State::MATCHED) {
      return node_rank;
    }
    return -1;
  }

  // similar the default version, but starts searching from the `scanned_chars` label
  // returns the state of matching (MATCHED, PARTIAL, FAILED)
  // also returns the total number of labels matched in `scanned_chars` and the node_id in `node_rank`
  auto look_up(const std::string &to_search, size_t &scanned_chars, size_t &node_rank) const -> State {
    size_t internal_rank = 0; // number of internal nodes before bv_index (initially refers to the root)
    size_t bv_index = louds_sux<>::root_idx; // position in the bv (initially refers to the root)
    node_rank = 0; // number of nodes before  bv_index (initially refers to the root)
    std::string_view substr;
    size_t n = num_child_root;
    std::string_view to_search_view(to_search);
    while (true) {
      assert(!topology_->is_leaf(bv_index));

      // printf("%d %d %d %d %d\n", scanned_chars, node_rank, bv_index, internal_rank, n);

      BENCH_CC( auto start = std::chrono::high_resolution_clock::now(); )

      succinct::bit_vector::enumerator it(*internal_variable, (*pointers_to_encoding)[internal_rank]);
      size_t l = it.take(bits_for_L);
      auto nt = (node_type) it.take(NUM_BIT_TYPE);
      bool is_end_of_world = it.take(1);
      if (to_search.size() == scanned_chars) {
        BENCH_CC(
          auto end = std::chrono::high_resolution_clock::now();
          key_search_time_ += (end - start).count();
        )
        return is_end_of_world ? State::MATCHED : State::FAILED;  // perfect match/early termination
      }

      const bool is_remapped = (nt >= elias_fano_amap);
      code_type first_code = 0;
      code_type to_search_code = 0;
      uint128_t alphamap_uint128 = 0;
      substr = to_search_view.substr(scanned_chars, std::min(l + MIN_L, to_search_view.size()));
      if (is_remapped) { // remap
        alphamap_uint128 = it.take128(ALPHABET_SIZE);
        assert(alphamap_uint128 != 0);
        alphamap am(alphamap_uint128);
        // check if all the characters in the wanted string are in the subtrie rooted in the current node
        if (!am.check_chars(substr)) {
          BENCH_CC(
            auto end = std::chrono::high_resolution_clock::now();
            key_search_time_ += (end - start).count();
          )
          return State::FAILED;  // illegal label
        }
        size_t bits_first_code_local_as = bits_first_code<MIN_L, code_type>(am.rankmax(), l);
        first_code = it.take128(bits_first_code_local_as);
        to_search_code = utrie_t::enc_real(substr, l, am);
      } else { // no remap
        first_code = it.take128(log_sigma * (l + MIN_L));
        to_search_code = utrie_t::enc(substr, l);
      }
      assert(first_code != code_type(0));
      if (to_search_code < first_code) {
        BENCH_CC(
          auto end = std::chrono::high_resolution_clock::now();
          key_search_time_ += (end - start).count();
        )
        return State::FAILED;  // illegal range
      }
      assert(to_search_code >= first_code);
      to_search_code -= first_code;
      // index where we arrived to search in the queried string to_search
      size_t child_to_continue = 1;
      // if to_search_code == 0 then to_search_code was equal to first_code so continue on first child
      if (to_search_code != code_type(0)) {
        to_search_code--;
        it.move(it.position());
        switch (nt) {
          case (elias_fano) : {
            child_to_continue = CoCo_v1_t::read_and_search_elias_fano(
                    *internal_variable, it, n - 1, to_search_code);
            break;
          }
          case (bitvector) : {
            size_t next_pointer = (internal_rank == pointers_to_encoding->size() - 1)
                                  ? internal_variable->size() :
                                  (*pointers_to_encoding)[internal_rank + 1];
            child_to_continue = CoCo_v1_t::read_and_search_bitvector(
                    *internal_variable, it, to_search_code,
                    next_pointer - it.position());
            break;
          }
          case (packed) : {
            // in order to pass the encoding length to read_and_search_packed()
            size_t next_pointer = (internal_rank == pointers_to_encoding->size() - 1)
                                  ? internal_variable->size() :
                                  (*pointers_to_encoding)[internal_rank + 1];
            child_to_continue = CoCo_v1_t::read_and_search_packed(
                    *internal_variable, it, n - 1, to_search_code,
                    next_pointer - it.position());
            break;
          }
          case (all_ones) : {
            child_to_continue = (to_search_code < code_type(n - 1)) ? size_t(to_search_code) + 2 : -1;
            break;
          }
          case (elias_fano_amap) : {
            child_to_continue = CoCo_v1_t::read_and_search_elias_fano(
                    *internal_variable, it, n - 1,
                    to_search_code);
            break;
          }
          case (bitvector_amap) : {
            size_t next_pointer = (internal_rank == pointers_to_encoding->size() - 1)
                                  ? internal_variable->size() :
                                  (*pointers_to_encoding)[internal_rank + 1];
            child_to_continue = CoCo_v1_t::read_and_search_bitvector(
                    *internal_variable, it, to_search_code,
                    next_pointer - it.position());
            break;
          }
          case (packed_amap) : {
            size_t next_pointer = (internal_rank == pointers_to_encoding->size() - 1)
                                  ? internal_variable->size() :
                                  (*pointers_to_encoding)[internal_rank + 1];
            child_to_continue = CoCo_v1_t::read_and_search_packed(
                    *internal_variable, it, n - 1, to_search_code,
                    next_pointer - it.position());
            break;
          }
          case (all_ones_amap) : {
            child_to_continue = (to_search_code < code_type(n - 1)) ? size_t(to_search_code) + 2 : -1;
            break;
          }
          default :
            assert(0);
        }
      }

      BENCH_CC(
        auto end = std::chrono::high_resolution_clock::now();
        key_search_time_ += (end - start).count();
      )

      if (child_to_continue == -1) {
        return State::FAILED;  // mismatch  // TODO: need to handle substring; could be PARTIAL
      }

      scanned_chars += l + MIN_L;
      BENCH_CC( auto t1 = std::chrono::high_resolution_clock::now(); )
      // number of nodes before  bv_index (initially refers to the root
      std::tie(node_rank, bv_index) = topology_->move_to_child(bv_index, child_to_continue - 1);
      --node_rank;
      BENCH_CC( auto t2 = std::chrono::high_resolution_clock::now(); )
      if (topology_->is_leaf(bv_index)) {
        BENCH_CC(
          auto t3 = std::chrono::high_resolution_clock::now();
          child_nav_time_ += (t2 - t1).count();
          is_leaf_time_ += (t3 - t2).count();
          node_nav_time_ += (t3 - t1).count();
        )
        return (scanned_chars < to_search.size()) ? State::PARTIAL : State::MATCHED;  // prefix/perfect match
      }
      BENCH_CC( auto t3 = std::chrono::high_resolution_clock::now(); )
      // number of internal nodes before bv_index (initially refers to the root)
      internal_rank = topology_->internal_rank(bv_index, node_rank);
      BENCH_CC( auto t4 = std::chrono::high_resolution_clock::now(); )
      n = topology_->num_child(bv_index);
      BENCH_CC( auto t5 = std::chrono::high_resolution_clock::now(); )

      BENCH_CC(
        child_nav_time_ += (t2 - t1).count();
        is_leaf_time_ += (t3 - t2).count();
        internal_rank_time_ += (t4 - t3).count();
        num_child_time_ += (t5 - t4).count();
        node_nav_time_ += (t5 - t1).count();
      )
    }
  }

  size_t size_in_bits() const {
    size_t to_return = 0;
    to_return += topology_->size_in_bits();
    to_return += internal_variable->size();
    to_return += sdsl::size_in_bytes(*pointers_to_encoding) * CHAR_BIT;
    to_return += sizeof(*this) * CHAR_BIT;
    return to_return;
  }

  auto num_leaves() const -> size_t {
    return topology_->bv_.rank00();
  }

  template<typename root_type>
  void build_CoCo_from_uncompacted_trie(root_type root) {
    succinct::bit_vector_builder bvb;
    size_t num_built_nodes = 0;
    std::queue<root_type> q;
    q.push(root);
    while (!q.empty()) {
      typename utrie_t::TrieNode_lw *node = q.front();
      q.pop();

      if (node != nullptr && node->children.size() > 0) {
        printf("macro node %ld: encoding = %d, depth = %d, cost = %ld\n", num_built_nodes, node->node_type_vec[node->l_idx],
               node->l_idx, node->spacecost[node->l_idx]);
      }

      if (node == nullptr) {
        topology_->add_node(0);
        continue;
      }
      const bool is_leaf = (node->children.size() == 0);

      if (is_leaf) {
        topology_->add_node(0);
      } else {
        for (auto &child: node->actual_CoCo_children) {
          q.push(child.second);
        }
        node_type nt = node->node_type_vec[node->l_idx];
        std::vector<code_type> codes;
        if (nt != all_ones and nt != all_ones_amap) {
          codes.reserve(node->actual_CoCo_children.size());
        }
        bool is_first = true;
        code_type first_code = 0;
        const bool is_remapped = (nt >= elias_fano_amap);
        size_t this_node_starting_point = bvb.size();
        int count = 0;
        for (auto &child: node->actual_CoCo_children) {
          std::string child_string(child.first);
          printf("key %d: %s\n", count++, child_string.c_str());
          if (is_first) {
            is_first = false;
            bvb.append_bits((uint64_t) node->l_idx, bits_for_L);
            bvb.append_bits((uint64_t) node->node_type_vec[node->l_idx], NUM_BIT_TYPE);
            bvb.append_bits((uint64_t) node->isEndOfWord, 1);
            if (is_remapped) {
              first_code = utrie_t::enc_real(child_string,
                                              node->l_idx,
                                              node->alphamaps[node->l_idx]);
              bvb.append_bits(node->alphamaps[node->l_idx].bitmap, ALPHABET_SIZE);
              size_t bit_first_code_val = bits_first_code<MIN_L, code_type>(
                      node->alphamaps[node->l_idx].rankmax(),
                      node->l_idx);
              bvb.append_bits(first_code, bit_first_code_val);
            } else {
              first_code = utrie_t::enc(child_string, node->l_idx);
              bvb.append_bits(first_code, log_sigma * (node->l_idx + MIN_L));
            }
          } else {
            if (nt != all_ones and nt != all_ones_amap) {
              code_type new_code = 0;
              if (is_remapped) {
                new_code = utrie_t::enc_real(child_string,
                                             node->l_idx,
                                             node->alphamaps[node->l_idx]);
                assert(new_code > first_code);
                // we use delta come with respect first_code
                new_code -= first_code;
                // we don't encode 0
                new_code--;
                assert(new_code <= node->u_vec_real[node->l_idx]);

              } else {
                new_code = utrie_t::enc(child_string, node->l_idx);
                assert(new_code > first_code);
                // we use delta come with respect first_code
                new_code -= first_code;
                // we don't encode 0
                new_code--;
                assert(new_code <= node->u_vec[node->l_idx]);
              }
              codes.push_back(new_code);
            }
          }
        }
        assert(std::is_sorted(codes.begin(), codes.end()));
        assert(!(std::unique(codes.begin(), codes.end()) != codes.end()));
        size_t bvb_prev_size = bvb.size();
        switch (nt) {
          case (elias_fano) : {
            BENCH_CC( ef_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_elias_fano(node->u_vec[node->l_idx], codes, bvb);
            break;
          }
          case (elias_fano_amap) : {
            BENCH_CC( ef_remap_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_elias_fano(node->u_vec_real[node->l_idx], codes, bvb);
            break;
          }
          case (bitvector) : {
            BENCH_CC( bv_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_bitvector(node->u_vec[node->l_idx], codes, bvb);
            break;
          }
          case (bitvector_amap) : {
            BENCH_CC( bv_remap_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_bitvector(node->u_vec_real[node->l_idx], codes,
                                       bvb);
            break;
          }
          case (packed) : {
            BENCH_CC( packed_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_packed(node->u_vec[node->l_idx], codes, bvb);
            break;
          }
          case (packed_amap) : {
            BENCH_CC( packed_remap_keys_ += codes.size() + 1; )
            CoCo_v1_t::write_packed(node->u_vec_real[node->l_idx], codes,
                                    bvb);
            break;
          }
          case (all_ones) : {
            BENCH_CC( dense_keys_ += codes.size() + 1; )
            break;
          }
          case (all_ones_amap) : {
            BENCH_CC( dense_remap_keys_ += codes.size() + 1; )
            break;
          }
          default :
            assert(0);
        }
        // add this node to the vector of nodes
        assert(node->node_type_vec[node->l_idx] < index_types);
        (*pointers_to_encoding)[num_built_nodes] = this_node_starting_point;
        num_built_nodes++;
        assert(bvb.size() != 0);
        assert(node->actual_CoCo_children.size() == (1 + node->n_vec[node->l_idx]));
        // add the node to louds representation
        // +1 because n_vec is number of children - 1
        topology_->add_node(1 + node->n_vec[node->l_idx]);
      }
    }
    topology_->build_rank_select_ds();
    pointers_to_encoding->resize(num_built_nodes);
    sdsl::util::bit_compress(*pointers_to_encoding);
    internal_variable = std::make_unique<succinct::bit_vector>(&bvb);
    assert(internal_variable->size() > 0);
  }
};


#ifdef __BENCH_CC__

#define COCO_TMPL_ARGS template<uint8_t MIN_L, typename code_type, uint8_t MAX_L_THRS, uint8_t space_relaxation, int spill_threshold>

#define COCO_TMPL CoCoCC<MIN_L, code_type, MAX_L_THRS, space_relaxation, spill_threshold>

COCO_TMPL_ARGS size_t COCO_TMPL::node_nav_time_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::key_search_time_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::child_nav_time_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::is_leaf_time_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::internal_rank_time_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::num_child_time_ = 0;

COCO_TMPL_ARGS size_t COCO_TMPL::ef_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::ef_remap_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::packed_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::packed_remap_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::bv_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::bv_remap_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::dense_keys_ = 0;
COCO_TMPL_ARGS size_t COCO_TMPL::dense_remap_keys_ = 0;

#undef COCO_TMPL_ARGS
#undef COCO_TMPL

#endif

#ifdef __DEBUG__

#include "utils.hpp"
#include "CoCo-trie_v2.hpp"

class CoCoCCTest {
 public:
  static void test() {
    // load data
    std::string filename = "../dataset/words-230k.txt";
    std::vector<std::string> dataset;
    datasetStats ds = load_data_from_file(dataset, filename);

    // global variables
    MIN_CHAR = ds.get_min_char();
    ALPHABET_SIZE = ds.get_alphabet_size();
    assert(ALPHABET_SIZE < 127);

    Trie_lw<1, uint128_t, MAX_L_THRS, 0> trie;
    for (const auto &s : dataset) {
      trie.insert(s);
    }

    // computing the best number of levels to collapse into the nodes
    trie.space_cost_all_nodes();
    trie.build_actual_CoCo_children();

    CoCo_v2<> coco0(trie);
    CoCoCC<> coco1(trie);
    for (const auto &s : dataset) {
      // printf("look up %s\n", s.c_str());
      auto ret0 = coco0.look_up(s);
      auto ret1 = coco1.look_up(s);
      assert(ret0 == ret1);
    }
  }
};

#endif

#undef __BENCH_CC__
#undef BENCH_CC