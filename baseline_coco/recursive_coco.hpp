#pragma once

#include "recursive_trie.hpp"
#include "coco_cc.hpp"
#include "static_bit_vector.hpp"


template<uint8_t MIN_L = 1, typename code_type = uint128_t, uint8_t MAX_L_THRS = MAX_L_THRS,
         uint8_t space_relaxation = 0, int spill_threshold = 128, uint32_t max_recursion = 0,
         uint32_t cutoff = 6, uint8_t estimate = 20>
class RecursiveCoCo {
 public:
  using utrie_t = RecursiveTrie<MIN_L, code_type, MAX_L_THRS, space_relaxation, max_recursion, cutoff, estimate>;
  using coco_t = CoCoCC<MIN_L, code_type, MAX_L_THRS, space_relaxation, spill_threshold>;

  struct CoCoTrie {
    coco_t coco_;
    StaticBitVector next_;
    sdsl::int_vector<> link_;

    CoCoTrie() = default;

    CoCoTrie(const utrie_t &trie, size_t l_fixed = 0) : coco_(trie, l_fixed) {}

    ~CoCoTrie() = default;
  };

  RecursiveCoCo(const std::vector<std::string> &strs, size_t l_fixed = 0) {
    utrie_t utries(strs, l_fixed);

    // build coco tries
    num_tries_ = utries.num_tries();
    tries_ = new CoCoTrie[num_tries_];
    for (size_t i = 0; i < num_tries_ - 1; i++) {
      auto utrie = utries.get_trie(i);
      tries_[i].coco_.build(*utrie);
      tries_[i].next_.resize(tries_[i].coco_.num_leaves());
    }
    tries_[num_tries_-1].coco_.build(*utries.get_trie(num_tries_ - 1));

    // now set up `next_` and `link_` for each trie
    std::vector<std::vector<uint32_t>> node_ids(strs.size()); // node IDs in each trie
    // first query all keys and fill in `next_`, keeping track of node IDs
    for (size_t i = 0; i < strs.size(); i++) {
      size_t matched_len = 0;
      for (size_t j = 0; j < num_tries_; j++) {
        printf("%s %d\n", strs[i].c_str(), matched_len);
        size_t node_id;
        auto state = tries_[j].coco_.look_up(strs[i], matched_len, node_id);
        assert(state != coco_t::State::FAILED);
        node_ids[i].emplace_back(node_id);  // record node ID
        // set `next_`
        if (state == coco_t::State::PARTIAL) {
          assert(j != num_tries_ - 1);
          tries_[j].next_.set1(node_id);
        } else {  // state == coco_t::State::MATCHED
          if (j != num_tries_ - 1) {
            tries_[j].next_.set0(node_id);
          }
          break;
        }
      }
    }
    // then query again and fill in `link_`
    for (uint32_t i = 0; i < num_tries_ - 1; i++) {
      tries_[i].link_.resize(tries_[i].next_.rank1());
    }
    for (uint32_t i = 0; i < strs.size(); i++) {
      const auto &links = node_ids[i];
      uint32_t j = 0;
      while (j < links.size() - 1) {
        auto link_idx = tries_[j].next_.rank1(links[j]);
        tries_[j].link_[link_idx] = links[j+1];
      }
    }
    for (uint32_t i = 0; i < num_tries_ - 1; i++) {
      sdsl::util::bit_compress(tries_[i].link_);
    }
  }

  ~RecursiveCoCo() {
    for (uint32_t i = 0; i < num_tries_; i++) {
      tries_[i].~CoCoTrie();
    }
    free(tries_);
  }

  auto look_up(const std::string &key) const -> size_t {
    size_t ret, node_id, link_id;
    size_t matched_len;
    for (uint32_t i = 0; i < num_tries_; i++) {
      auto state = tries_[0].coco_.look_up(key, matched_len, node_id);
      if (i == 0) {
        ret = node_id;
      }
      switch (state) {
       case coco_t::State::FAILED:
        return -1;
       case coco_t::State::MATCHED:
        if (i != num_tries_ - 1 && tries_[i].next_.get(node_id)) {  // early termination
          return -1;
        } else if (i != 0 && link_id != node_id) {  // incorrect link
          return -1;
        } else {
          assert(ret != -1);
          return ret;
        }
       case coco_t::State::PARTIAL:
        if (i == num_tries_ - 1 || !tries_[i].next_.get(node_id)) {  // unmatched suffix
          return -1;
        }
        if (i > 0 && link_id != node_id) {  // incorrect link
          return -1;
        }
        link_id = tries_[i].link_[tries_[i].next_.rank1(node_id)];
      }
    }
    return -1;  // should not be reachable
  }

  CoCoTrie *tries_{nullptr};
  uint32_t num_tries_{0};
};


#ifdef __DEBUG__

#include "utils.hpp"

class RecursiveCoCoTest {
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

    CoCo_v2<> coco0(dataset);
    RecursiveCoCo<1, uint128_t, MAX_L_THRS, 0, 128, 0, 0, 0> coco1(dataset);
    for (const auto &s : dataset) {
      // printf("look up %s\n", s.c_str());
      auto ret0 = coco0.look_up(s);
      auto ret1 = coco1.look_up(s);
      assert(ret0 == ret1);
    }
    printf("[PASSED]\n");
  }
};

#endif