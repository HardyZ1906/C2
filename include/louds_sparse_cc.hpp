#pragma once

#include "utils.hpp"
#include "fst_cc_builder.hpp"
#include "static_vector.hpp"

#include <limits>


#ifdef __BENCH_LS__
# define BENCH_LS(foo) foo
# include <chrono>
#else
# define BENCH_LS(foo)
#endif


// louds-sparse encoding, but with optimized layout
// 8 bits per label
template<typename Key, int cutoff>
class LoudsSparseCC {
 public:
  struct BitVector;

  using key_type = Key;
  using bitvec = BitVector;
  using label_vec = StaticVector<uint8_t>;

  static constexpr uint8_t terminator_ = 0;
  static constexpr int cutoff_ = cutoff;
  static constexpr size_t invalid_offset_ = std::numeric_limits<size_t>::max();

  LoudsSparseCC() = default;

  void build(FstCCBuilder &builder) {
    clear();

    depth_ = builder.s_labels_.size();
    if (depth_ == 0) {
      clear();
      return;
    }

    size_t size = 0;
    for (const auto &labels : builder.s_labels_) {
      size += labels.size();
    }

    labels_.reserve(size);
    bv_.reserve(size);

    for (const auto &labels : builder.s_labels_) {
      for (auto label : labels) {
        labels_.emplace_back(label);
      }
    }

    for (int i = 0; i < builder.s_has_child_.size(); i++) {
      bv_.load_bits(builder.s_has_child_[i].bits(), builder.s_louds_[i].bits(), 0, builder.s_has_child_[i].size());
    }
    bv_.build(256, builder.rank_first_ls_level());
  }

  void clear() {
    depth_ = offset_ = 0;
    labels_.clear();
    bv_.clear();
  }

  auto get(const key_type &key, size_t start_node_num) const -> size_t {
    if (depth_ == 0) {
      return invalid_offset_;
    }

    uint16_t len = key.size(), matched_len = cutoff_;
    size_t pos = bv_.select1(start_node_num + 1);

    while (matched_len < len) {
      // locate the end of current node
      size_t end = node_end(pos);
      // search for label
      pos = labels_.find(key[matched_len], pos, end);
      if (pos == end) {  // mismatch
        return invalid_offset_;
      }
      assert(labels_.at(pos) == key[matched_len]);

      if (!bv_.get<0>(pos)) {  // branch terminates
        return value_pos(pos);
      }

      // trace down child pointer
      pos = child_pos(pos);
      matched_len++;
    }

    if (labels_.at(pos) == terminator_) {  // matched as a prefix key
      return value_pos(pos);
    }
    return invalid_offset_;
  }

 private:
  auto node_start(size_t pos) const -> size_t {
    return bv_.prev1<1>(pos);
  }

  auto node_end(size_t pos) const -> size_t {
    return bv_.next1<1>(pos + 1);
  }

  auto value_pos(size_t pos) const -> size_t {
    return pos - bv_.rank1<0>(pos);
  }

  auto child_pos(size_t pos) const -> size_t {
    return bv_.rs<1>(pos);
  }

  // 8 bits per label; indicates valid branches at each node, in level order
  // labels belonging to the same node are sorted in ascending order
  label_vec labels_;

  // 2 bits per label; packed has-child and louds
  bitvec bv_;

  // since LOUDS-Sparse may not start from the first level, we need to add an offset of `# missing children - # missing nodes`
  size_t offset_;

  uint16_t depth_;

  friend class FstCC;

 private:
  struct BitVector {
    struct Block {  // packed has-child and louds bitvectors
      uint32_t rank0_{0};    // cumulative block ranks of bv<0> (i.e. has-child); bv<1> (i.e. louds) doesn't need to support rank
      uint32_t select1_{0};  // precomputed result of bv<1>.select1(rank0_)
      uint8_t subrank0_[4]{0};
      uint8_t subrank1_[4]{0};
      uint64_t bits0_[4]{0};  // has-child
      uint64_t bits1_[4]{0};  // louds

      template<int bvnum>
      auto build_index() -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);

        uint32_t rank = 0;
        if constexpr (bvnum == 0) {
          for (int i = 0; i < 4; i++) {
            subrank0_[i] = rank;
            rank += __builtin_popcountll(bits0_[i]);
          }
          return rank;
        } else {
          for (int i = 0; i < 4; i++) {
            subrank1_[i] = rank;
            rank += __builtin_popcountll(bits1_[i]);
          }
          return rank;
        }
      }

      template<int bvnum>
      auto get(uint32_t pos) const -> bool {
        if constexpr (bvnum == 0) {
          return GET_BIT(bits0_[pos/64], pos % 64);
        } else {
          return GET_BIT(bits1_[pos/64], pos % 64);
        }
      }

      template<int bvnum>
      auto rank1(uint32_t size) const -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);
        if constexpr (bvnum == 0) {
          return subrank0_[size/64] + __builtin_popcountll(bits0_[size/64] & MASK(size%64));
        } else {
          return subrank1_[size/64] + __builtin_popcountll(bits1_[size/64] & MASK(size%64));
        }
      }

      template<int bvnum>
      auto select1(uint32_t rank) const -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);
        if constexpr (bvnum == 0) {
          assert(rank <= rank1());
          if (rank <= subrank0_[2]) {
            if (rank <= subrank0_[1]) {
              return 0*64 + selectll(bits0_[0], rank);
            } else {
              return 1*64 + selectll(bits0_[1], rank - subrank0_[1]);
            }
          } else {
            if (rank <= subrank0_[3]) {
              return 2*64 + selectll(bits0_[2], rank - subrank0_[2]);
            } else {
              return 3*64 + selectll(bits0_[3], rank - subrank0_[3]);
            }
          }
        } else {
          assert(rank <= rank1());
          if (rank <= subrank1_[2]) {
            if (rank <= subrank1_[1]) {
              return 0*64 + selectll(bits1_[0], rank);
            } else {
              return 1*64 + selectll(bits1_[1], rank - subrank1_[1]);
            }
          } else {
            if (rank <= subrank1_[3]) {
              return 2*64 + selectll(bits1_[2], rank - subrank1_[2]);
            } else {
              return 3*64 + selectll(bits1_[3], rank - subrank1_[3]);
            }
          }
        }
      }

      template<int bvnum>
      auto rank1() const -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);
        if constexpr (bvnum == 0) {
          return subrank0_[3] + __builtin_popcountll(bits0_[3]);
        } else {
          return subrank1_[3] + __builtin_popcountll(bits1_[3]);
        }
      }
    };  // 80 bytes

    Block *blocks_{nullptr};

    // precomputed select results for the first level of bv<1> (i.e. louds)
    uint32_t *sample_{nullptr};

    uint32_t capacity_{0};

    uint32_t size_{0};

    uint32_t has_child_rank_;

    uint32_t louds_l1_rank_;

    uint32_t offset_{0};

    uint32_t sample_rate_{0};

    BitVector() = default;

    ~BitVector() {
      free(blocks_);
      free(sample_);
    }

    auto size() const -> size_t {
      return size_;
    }

    template<int bvnum>
    auto rank1() const -> size_t {
      static_assert(bvnum == 0 || bvnum == 1);
      return rank_[bvnum];
    }

    auto is_empty() const -> bool {
      return size_ == 0;
    }

    void clear() {
      free(blocks_);
      free(sample_);
      blocks_ = nullptr;
      sample_ = nullptr;
    }

    void init_sample(uint32_t sample_rate, uint32_t max_rank) {
      free(sample_);

      sample_rate_ = sample_rate;
      if (sample_rate == 0) {
        sample_ = nullptr;
        return;
      }

      uint32_t sample_size = max_rank / sample_rate_;
      sample_ = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * (sample_size + 1)));

      uint32_t j = 0, rank = 0;
      uint32_t block_rank = blocks_[0].rank1<1>();
      sample_[0] = 0;
      for (uint32_t i = 1; i < sample_size; i++) {
        while (rank + block_rank < i * sample_rate_) {
          rank += block_rank;
          j++;
          block_rank = blocks_[j].rank1<1>();
        }
        assert(i == 0 || rank < i * sample_rate_);
        sample_[i] = j*256 + blocks_[j].select1(i*sample_rate_ - rank);
      }
    }

    void init_select() {
      int num_blocks = (size_ + 255) / 256;

      // build select index for bv<1>
      uint32_t i = 0, j = 0;
      uint32_t block_rank = blocks_[0].rank1<1>();
      uint32_t rank = 0;
      while (blocks_[j].rank0_ + offset_ + 1 <= rank_[1]) {
        while (i < num_blocks && rank + block_rank < blocks_[j].rank0_ + offset_ + 1) {
          rank += block_rank;
          i++;
          block_rank = blocks_[i].rank1<1>();
        }
        blocks_[j].select1_ = i*256 + blocks_[j].select1<1>(blocks_[j].rank0_ + offset_ + 1 - rank);
        j++;
      }
      while (j < num_blocks) {
        blocks_[j].select1_ = size_;
        j++;
      }
    }

    void reserve(uint32_t size) {
      if (size > capacity_) {
        capacity_ = (size + 255) / 256 * 256;
        blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
      }
    }

    void load_bits(const uint64_t *bits0, const uint64_t *bits1, size_t start, size_t size) {
      if (size_ + size > capacity_) {
        capacity_ = std::max<uint32_t>(((size_ + size) * 2 + 255) / 256 * 256, 256 * 8);
        blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
      }

      uint32_t leftover = (size_ + 255) / 256 * 256 - size_;
      if (size <= leftover) {
        copy_bits(blocks_[size_/256].bits0_, size_ % 256, bits0, start, size);
        copy_bits(blocks_[size_/256].bits1_, size_ % 256, bits1, start, size);
        size_ += size;
        return;
      }

      size_t end = start + size;
      copy_bits(blocks_[size_/256].bits0_, size_ % 256, bits0, start, leftover);
      copy_bits(blocks_[size_/256].bits1_, size_ % 256, bits1, start, leftover);
      size_ += leftover;
      start += leftover;
      while (start + 256 <= end) {
        copy_bits(blocks_[size_/256].bits0_, size_ % 256, bits0, start, 256);
        copy_bits(blocks_[size_/256].bits1_, size_ % 256, bits1, start, 256);
        size_ += 256;
        start += 256;
      }
      copy_bits(blocks_[size_/256].bits0_, size_ % 256, bits0, start, end - start);
      copy_bits(blocks_[size_/256].bits1_, size_ % 256, bits1, start, end - start);
      size_ += end - start;
    }

    void build(uint32_t sample_rate, uint32_t louds_first_level_rank) {
      if (capacity_ - size_ >= 256) {
        capacity_ = (size_ + 255) / 256 * 256;
        blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
      }
      sample_rate_ = sample_rate;

      if (blocks_ == nullptr) {
        return;
      }

      // clear trailing bits
      uint32_t remainder = size_ % 256;
      for (uint32_t i = (remainder + 63) / 64; i < 4; i++) {
        blocks_[size_ / 256].bits0_[i] = 0;
        blocks_[size_ / 256].bits1_[i] = 0;
      }
      for (uint32_t i = 0; i < 4; i++) {
        blocks_[capacity_ / 256].subrank0_[i] = 0;
        blocks_[capacity_ / 256].subrank1_[i] = 0;
        blocks_[capacity_ / 256].bits0_[i] = 0;
        blocks_[capacity_ / 256].bits1_[i] = 0;
      }

      louds_l1_rank_ = louds_first_level_rank;
      has_child_rank_ = 0;
      for (uint32_t i = 0; i < capacity_ / 256; i++) {
        auto block_rank0 = blocks_[i].build_index<0>();
        blocks_[i].build_index<1>();
        blocks_[i].rank0_ = has_child_rank_;
        has_child_rank_ += block_rank0;
      }
      blocks_[capacity_ / 256].rank0_ = rank_[0];

      init_sample(sample_rate, louds_first_level_rank);
      init_select();
    }

    // get the `pos`-th bit
    template<int bvnum>
    auto get(size_t pos) const -> bool {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(pos < size_);

      BENCH_LS( auto start_time = std::chrono::high_resolution_clock::now(); )
      bool ret = blocks_[pos/256].get(pos % 256);
      BENCH_LS(
        auto end_time = std::chrono::high_resolution_clock::now();
        get_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // `bvnum` must be 0; compute the rank of the first `size` bits of the bv<0>
    template<int bvnum = 0>
    auto rank1(size_t size) const -> size_t {
      static_assert(bvnum == 0);
      assert(size <= size_);

      BENCH_LS( auto start_time = std::chrono::high_resolution_clock::now(); )
      const auto &block = blocks_[size/256];
      size_t ret = block.rank0_ + block.rank1<bvnum>(size % 256);
      BENCH_LS(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // `bvnum` must be 1; select the `rank`-th 1 bit from the second bit vector
    // `rank` must not exceed `louds_l1_rank_` as only selects in the first level are supported
    template<int bvnum = 1>
    auto select1(size_t rank) const -> size_t {
      static_assert(bvnum == 1);
      assert(rank <= louds_l1_rank_);

      BENCH_LS( auto start_time = std::chrono::high_resolution_clock::now(); )
      uint32_t block_idx;
      uint32_t sample_idx = rank / sample_size_;
      uint32_t pos = sample_[sample_idx];
      block_idx = pos / 256;
      // offset to start of block
      rank = rank - sample_idx*sample_rate_ + blocks_[block_idx].rank1<1>(pos % 256) + (sample_idx > 0);
      // linear scan
      uint32_t cumulative_rank = 0;
      uint32_t block_rank = blocks_[block_idx].rank1<1>();
      while (cumulative_rank + block_rank < rank) {
        cumulative_rank += block_rank;
        block_idx++;
        block_rank = blocks_[block_idx].rank1<1>();
      }
      size_t ret = block_idx*256 + blocks_[block_idx].select1<1>(rank - cumulative_rank);
      BENCH_LS(
        auto end_time = std::chrono::high_resolution_clock::now();
        select_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // return the position of the closest 1 bit before position `pos` (inclusive)
    // if not found, return 0
    template<int bvnum>
    auto prev1(size_t pos) const -> size_t {
      static_assert(bvnum == 0 || bvnum == 1);

      if (pos >= size_) {
        return size_;
      }
      uint32_t block_idx = pos / 64;
      uint32_t remainder = pos % 64;
      uint32_t dist = 0;
      if constexpr (bvnum == 0) {
        const uint64_t &block = blocks_[block_idx / 4].bits0_[block_idx % 4];
        if (block << (63 - remainder)) {
          return pos - __builtin_clzll(block << (63 - remainder));
        }
        dist += remainder + 1;
        while (pos - dist >= 0) {
          block_idx = (pos - dist) / 64;
          const uint64_t &block = blocks_[block_idx / 4].bits0_[block_idx % 4];
          if (block) {
            return pos - dist - __builtin_clzll(block);
          }
          dist += 64;
        }
      } else {
        const uint64_t &block = blocks_[block_idx / 4].bits1_[block_idx % 4];
        if (block << (63 - remainder)) {
          return pos - __builtin_clzll(block << (63 - remainder));
        }
        dist += remainder + 1;
        while (pos - dist >= 0) {
          block_idx = (pos - dist) / 64;
          const uint64_t &block = blocks_[block_idx / 4].bits1_[block_idx % 4];
          if (block) {
            return pos - dist - __builtin_clzll(block);
          }
          dist += 64;
        }
      }
    }

    // return the position of the closest 1 bit after position `pos` (inclusive)
    // if not found, return `size_`
    template<int bvnum>
    auto next1(size_t pos) const -> size_t {
      static_assert(bvnum == 0 || bvnum == 1);

      if (pos >= size_) {
        return size_;
      }
      uint32_t block_idx = pos / 64;
      uint32_t remainder = pos % 64;
      uint32_t dist = 0;
      if constexpr (bvnum == 0) {
        const uint64_t &block = blocks_[block_idx / 4].bits0_[block_idx % 4];
        if (block >> remainder) {
          return pos + __builtin_ctzll(block >> remainder);
        }
        dist += 64 - remainder;
        while (pos + dist < size_) {
          block_idx = (pos + dist) / 64;
          const uint64_t &block = blocks_[block_idx / 4].bits0_[block_idx % 4];
          if (block) {
            return pos + dist + __builtin_ctzll(block);
          }
          dist += 64;
        }
      } else {
        const uint64_t &block = blocks_[block_idx / 4].bits1_[block_idx % 4];
        if (block >> remainder) {
          return pos + __builtin_ctzll(block >> remainder);
        }
        dist += 64 - remainder;
        while (pos + dist < size_) {
          block_idx = (pos + dist) / 64;
          const uint64_t &block = blocks_[block_idx / 4].bits1_[block_idx % 4];
          if (block) {
            return pos + dist + __builtin_ctzll(block);
          }
          dist += 64;
        }
      }
      return size_;
    }

    // bvnum must be 1; returns bv<1>.select1(bv<0>.rank1(pos + 1) + offset_ + 1); used for child navigation
    template<int bvnum = 1>
    auto rs(size_t pos) const -> size_t {
      static_assert(bvnum == 1);
      assert(pos < size_);

      BENCH_LS( auto start_time = std::chrono::high_resolution_clock::now(); )
      Block &block = blocks_[(pos+1)/256];
      uint32_t rank = block.rank0_ + block.rank1<0>((pos + 1) % 256);
      BENCH_LS(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )

      BENCH_LS( start_time = std::chrono::high_resolution_clock::now(); )
      uint32_t target = block.select1_;
      uint32_t block_idx = target / 256;
      // offset to start of block
      rank = rank - block.rank0_ + blocks_[block_idx].rank1<1>(target % 256) + ((pos + 1) / 256 == 0);
      // linear scan
      uint32_t cumulative_rank = 0;
      uint32_t block_rank = blocks_[block_idx].rank1<1>();
      while (cumulative_rank + block_rank < rank) {
        cumulative_rank += block_rank;
        block_idx++;
        block_rank = blocks_[block_idx].rank1<1>();
      }
      size_t ret = block_idx*256 + blocks_[block_idx].select1(rank - cumulative_rank);
      BENCH_LS(
        end_time = std::chrono::high_resolution_clock::now();
        select_time_ += (end_time - start_time).count();
      )
      return ret;
    }
  };
};

#undef BENCH_LS