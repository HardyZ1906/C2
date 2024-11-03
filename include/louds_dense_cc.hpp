#pragma once

#include "utils.hpp"
#include "bit_vector_builder.hpp"


#ifdef __BENCH_LD__
# define BENCH_LD(foo) foo
# include <chrono>
#else
# define BENCH_LD(foo)
#endif


// louds dense encoding, but with optimized layout
// @require: label 0 is reserved for terminator
template<typename Key, int cutoff, int fanout = 256>
class LoudsDenseCC {
 public:
  struct BitVector;

  using key_type = Key;
  using bitvec = BitVector;

  static constexpr int fanout_ = fanout;
  static constexpr int cutoff_ = cutoff;
  static constexpr int node_blocks_ = (fanout_ + 63) / 64;
  static constexpr size_t invalid_offset_ = std::numeric_limits<size_t>::max();

  LoudsDense() = default;

  void build(FstCCBuilder &builder) {
    clear();

    depth_ = builder.d_labels_.size();
    if (depth_ == 0) {
      return;
    }

    size_t size = 0;
    for (const auto &labels : builder.d_labels_) {
      size += labels.size();
    }
    bv_.reserve(size);

    for (size_t i = 0; i < builder.d_labels_.size(); i++) {
      bv_.load_bits(builder.d_labels_[i].bits(), builder.d_has_child_[i].bits(), 0, builder.d_labels_[i].size());
    }
    bv_.build();
  }

  void clear() {
    depth_ = 0;
    bv_.clear();
  }

  // try getting a key; if matched, `pos` returns the value position (LOUDS ID)
  // if partially matched, returns the node ID in the next level;
  // else returns `invalid_offset_`
  auto get(const key_type &key, size_t &pos) const -> bool {
    uint16_t len = key.size(), matched_len = 0;
    pos = 0;

    while (matched_len < len && matched_len < depth_) {
      assert(pos + fanout_ <= bv_.size());

      pos += key[matched_len];
      if (!bv_.get<0>(pos)) {  // mismatch
        return false;
      }

      if (!bv_.get<1>(pos)) {
        if (matched_len == len - 1) {  // matched
          pos = value_pos(pos);
          return true;
        }
        pos = invalid_offset_;
        return false;  // early termination
      }

      // trace down child pointer
      pos = child_pos(pos);
      matched_len++;
    }

    if (matched_len == depth_) {
      pos /= fanout_;
      return false;
    }

    if (bv_.get<0>(pos)) {  // matched as prefix key
      pos = value_pos(pos + 1);
      return true;
    }  // otherwise, prefix in tree but not a valid key
    pos = invalid_offset_;
    return false;
  }

 private:
  // given a node position, return position of the associated value
  // @require: has_child_.get(pos) == 0
  auto value_pos(size_t pos) const -> size_t {
    return bv_.rank1<0>(pos) - bv_.rank1<1>(pos);
  }

  auto child_pos(size_t pos) const -> size_t {
    return bv_.rank1<1>(pos + 1) * fanout_;
  }

  uint16_t depth_{0};

  // packed labels and has-child bitvectors; bv<0>: labels; bv<1>: has-child
  bitvec bv_;

  friend class FstCC;

 private:
  struct BitVector {  // labels and has-child are packed, is-prefix-key is removed
    struct Block {
      uint32_t rank0_{0};  // accumulated block ranks of bv<0> (i.e. labels)
      uint32_t rank1_{0};  // accumulated block ranks of bv<1> (i.e. has-child)
      uint8_t subrank0_[4]{0};
      uint8_t subrank1_[4]{0};
      uint64_t bits0_[4]{0};  // labels
      uint64_t bits1_[4]{0};  // has-child

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
    };

    // Louds-Dense doesn't need to support select
    Block *blocks_{nullptr};

    uint32_t capacity_{0};

    uint32_t size_{0};

    uint32_t rank_[2]{0};

  #ifdef __MICROBENCH_LD__
    static size_t get_time_;
    static size_t rank_time_;

    static void print_microBENCH_LD() {
      printf("[STATIC BITVECTOR MICROBENCH_LD]\n");
      printf("get: %lf ms; rank: %lf ms\n",c(double)get_time_/1000000, (double)rank_time_/1000000);
    }

    static void clear_microBENCH_LD() {
      get_time_ = rank_time_ = 0;
    }
  #endif

    BitVector() = default;

    ~BitVector() {
      free(blocks_);
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
      blocks_ = nullptr;
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

    void build() {
      if (capacity_ - size_ >= 256) {
        capacity_ = (size_ + 255) / 256 * 256;
        blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
      }

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

      rank_[0] = rank_[1] = 0;
      for (uint32_t i = 0; i < capacity_ / 256; i++) {
        auto block_rank0 = blocks_[i].build_index<0>();
        auto block_rank1 = blocks_[i].build_index<1>();
        blocks_[i].rank0_ = rank_[0];
        blocks_[i].rank1_ = rank_[1];
        rank_[0] += block_rank0;
        rank_[1] += block_rank1;
      }
      blocks_[capacity_ / 256].rank0_ = rank_[0];
      blocks_[capacity_ / 256].rank1_ = rank_[1];
    }

    // get the `pos`-th bit
    template<int bvnum>
    auto get(size_t pos) const -> bool {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(pos < size_);

      BENCH_LD( auto start_time = std::chrono::high_resolution_clock::now(); )
      bool ret;
      if constexpr (bvnum == 0) {
        ret = GET_BIT(blocks_[pos/256].bits0_[(pos%256)/64], pos%64);
      } else {
        ret = GET_BIT(blocks_[pos/256].bits1_[(pos%256)/64], pos%64);
      }
      BENCH_LD(
        auto end_time = std::chrono::high_resolution_clock::now();
        get_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // compute the rank of the first `size` bits
    template<int bvnum>
    auto rank1(size_t size) const -> size_t {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(size <= size_);

      BENCH_LD( auto start_time = std::chrono::high_resolution_clock::now(); )
      const auto &block = blocks_[size / 256];
      size_t ret;
      if constexpr (bvnum == 0) {
        ret = block.rank0_ + block.subrank0_[(size%256)/64] +
              __builtin_popcountll(block.bits0_[(size%256)/64] & MASK(size%64));
      } else {
        ret = block.rank1_ + block.subrank1_[(size%256)/64] +
              __builtin_popcountll(block.bits1_[(size%256)/64] & MASK(size%64));
      }
      BENCH_LD(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )
      return ret;
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
  };
};

#undef BENCH_LD