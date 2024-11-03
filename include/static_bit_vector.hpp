#pragma once

// #define __DEBUG__

#include "utils.hpp"

#include <cstring>
#include <cassert>

#ifdef __MICROBENCHMARK__
# define BENCHMARK(foo) foo
# include <chrono>
#else
# define BENCHMARK(foo)
#endif


/**
 * Static bit vector that only supports read operation (i.e. get, rank & select)
 * Design adopted from https://github.com/krareT/terark-base/tree/master/src/terark/succinct
*/
class StaticBitVector {
 public:
  struct Block {
    uint32_t rank1_;           // layer 2 index: the number of 1 bits before this block
    uint8_t  subranks_[4]{0};  // layer 1 index: rank1_[i] = popcount(bits_[0...i]) exclusive
    uint64_t bits_[4]{0};      // the actual bits; interleave bits and index so as to reduce cache misses

    Block() = default;

    // read 256 bits starting from the `start`-th bit of `bits` into this block
    // returns the rank of the bits read
    // does NOT set `rank1_` - it is the caller's responsibility
    auto init(const uint64_t *bits, size_t start) -> int {
      int elt_idx = start / 64;
      int shift_amt = start % 64;
      int curr_rank = 0;
      if (shift_amt == 0) {
        for (int i = 0; i < 4; i++) {
          bits_[i] = bits[elt_idx + i];
          subranks_[i] = curr_rank;
          curr_rank += __builtin_popcountll(bits_[i]);
        }
      } else {
        int shift_amt = start % 64;
        for (int i = 0; i < 4; i++) {
          bits_[i] = (bits[elt_idx+i] >> shift_amt) | (bits[elt_idx+i+1] << (64-shift_amt));
          subranks_[i] = curr_rank;
          curr_rank += __builtin_popcountll(bits_[i]);
        }
      }
      return curr_rank;
    }

    // read `size` bits starting from the `start`-th bit of `bits` into this block
    // returns the rank of the bits read
    // does NOT set `rank1_` - it is the caller's responsibility
    auto init(const uint64_t *bits, size_t start, size_t size) -> int {
      assert(size <= 256);

      int elt_idx = start / 64, num_elts = (size + 63) / 64;
      int shift_amt = start % 64;
      int curr_rank = 0;
      if (shift_amt == 0) {
        for (int i = 0; i < num_elts - 1; i++) {
          bits_[i] = bits[elt_idx + i];
          subranks_[i] = curr_rank;
          curr_rank += __builtin_popcountll(bits_[i]);
        }

        if (size % 64) {
          bits_[num_elts-1] = bits[elt_idx+num_elts-1] & MASK(size%64);
        } else {
          bits_[num_elts-1] = bits[elt_idx+num_elts-1];
        }
        subranks_[num_elts-1] = curr_rank;
        curr_rank += __builtin_popcountll(bits_[num_elts-1]);

        for (int i = num_elts; i < 4; i++) {
          bits_[i] = 0;
          subranks_[i] = curr_rank;
        }
      } else {
        for (int i = 0; i < num_elts - 1; i++) {
          bits_[i] = (bits[elt_idx+i] >> shift_amt) | (bits[elt_idx+i+1] << (64-shift_amt));
          subranks_[i] = curr_rank;
          curr_rank += __builtin_popcountll(bits_[i]);
        }

        if (shift_amt + size % 64 < 64) {
          size_t remainder_bits = (shift_amt + size) % 64;
          uint64_t last_elt = bits[elt_idx+num_elts-1] & MASK(remainder_bits);
          bits_[num_elts-1] = last_elt >> shift_amt;
        } else {
          size_t remainder_bits = (shift_amt + size) % 64;
          uint64_t last_elt = bits[elt_idx+num_elts] & MASK(remainder_bits);
          bits_[num_elts-1] = (bits[elt_idx+num_elts-1] >> shift_amt) | (last_elt << (64-shift_amt));
        }

        for (int i = num_elts; i < 4; i++) {
          bits_[i] = 0;
          subranks_[i] = curr_rank;
        }
      }
      return curr_rank;
    }

    auto build_index() -> uint32_t {
      uint32_t rank = 0;
      for (int i = 0; i < 4; i++) {
        subranks_[i] = rank;
        rank += __builtin_popcountll(bits_[i]);
      }
      return rank;
    }

    void serialize(uint64_t *dst, size_t start) const {
      size_t shift_amt = start % 64;
      if (shift_amt == 0) {
        dst[start/64 + 0] = bits_[0];
        dst[start/64 + 1] = bits_[1];
        dst[start/64 + 2] = bits_[2];
        dst[start/64 + 3] = bits_[3];
      } else {
        dst[start/64 + 0] = (bits_[0] << shift_amt) | (dst[start/64 + 0] & (64 - shift_amt));
        dst[start/64 + 1] = (bits_[1] << shift_amt) | (bits_[0] >> (64-shift_amt));
        dst[start/64 + 2] = (bits_[2] << shift_amt) | (bits_[1] >> (64-shift_amt));
        dst[start/64 + 3] = (bits_[3] << shift_amt) | (bits_[2] >> (64-shift_amt));
        dst[start/64 + 4] = (bits_[3] >> (64-shift_amt));
      }
    }

    __HOT __ALWAYS_INLINE auto rank1(uint32_t size) const -> uint32_t {
      return subranks_[size/64] + __builtin_popcountll(bits_[size/64] & MASK(size%64));
    }

    __HOT __ALWAYS_INLINE auto select1(uint32_t rank) const -> uint32_t {
      if (rank <= subranks_[2]) {
        if (rank <= subranks_[1]) {
          return 0*64 + selectll(bits_[0], rank);
        } else {
          return 1*64 + selectll(bits_[1], rank - subranks_[1]);
        }
      } else {
        if (rank <= subranks_[3]) {
          return 2*64 + selectll(bits_[2], rank - subranks_[2]);
        } else {
          return 3*64 + selectll(bits_[3], rank - subranks_[3]);
        }
      }
    }
  };  // 40 bytes

  StaticBitVector() = default;

  StaticBitVector(const uint64_t *bits, size_t start, size_t size, uint32_t sample_rate = 0) {
    init(bits, start, size, sample_rate);
  }

  ~StaticBitVector() {
    free(blocks_);
    free(sample_);
  }

  auto size() const -> size_t {
    return size_;
  }

  auto rank1() const -> size_t {
    return rank_;
  }

  auto is_empty() const -> bool {
    return size_ == 0;
  }

  void reserve(size_t size) {
    if (size > capacity_) {
      capacity_ = (size + 255) / 256 * 256;
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }
  }

  void resize(size_t size) {
    reserve(size);
    size_ = size;
  }

  void clear() {
    free(blocks_);
    blocks_ = nullptr;
  }

  void append1() {
    if (size_ == capacity_) {
      capacity_ = std::max<uint32_t>((size_ * 2 + 255) / 256 * 256, 256 * 8);
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }
    SET_BIT(blocks_[size_/256].bits_[(size_%256)/64], size_ % 64);
    size_++;
  }

  void append0() {
    if (size_ == capacity_) {
      capacity_ = std::max<uint32_t>((size_ * 2 + 255) / 256 * 256, 256 * 8);
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }
    CLEAR_BIT(blocks_[size_/256].bits_[(size_%256)/64], size_ % 64);
    size_++;
  }

  void set1(size_t pos) {
    assert(pos < size_);
    SET_BIT(blocks_[pos/256].bits_[(pos%256)/64], pos % 64);
  }

  void set0(size_t pos) {
    assert(pos < size_);
    CLEAR_BIT(blocks_[pos/256].bits_[(pos%256)/64], pos % 64);
  }

  void init(const uint64_t *bits, size_t start, size_t size, uint32_t sample_rate = 0) {
    int num_blocks = (size + 255) / 256;

    size_ = size;
    capacity_ = num_blocks * 256;

    clear();
    blocks_ = reinterpret_cast<Block *>(malloc(sizeof(Block) * (num_blocks + 1)));  // the last elt acts as a sentinel

    size_t total_rank = 0;
    int shift_amt = start % 64;
    size_t i = 0;
    auto ptr = bits;
    while (i < size_ / 256) {  // deal with input in batches of 4
      size_t curr_rank = blocks_[i].init(bits, i*256);
      blocks_[i].rank1_ = total_rank;
      total_rank += curr_rank;
      i++;
    }

    if (size_ % 256) {  // deal with remained input, if any
      size_t curr_rank = blocks_[size_/256].init(bits, size_ - size_%256, size_%256);
      blocks_[size_/256].rank1_ = total_rank;
      total_rank += curr_rank;
    }

    // set sentinel
    memset(&blocks_[num_blocks], 0, sizeof(Block));
    blocks_[num_blocks].rank1_ = rank_ = total_rank;

    init_sample(sample_rate);
  }

  void init_sample(uint32_t sample_rate) {
    free(sample_);

    sample_rate_ = sample_rate;
    if (sample_rate == 0) {
      sample_ = nullptr;
      return;
    }

    uint32_t sample_size = rank_ / sample_rate_ + 1;
    sample_ = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * (sample_size + 1)));

    sample_[0] = 0;
    for (uint32_t i = 1; i < sample_size; i++) {
      uint32_t j = sample_[i-1];
      assert(blocks_[j].rank1_ < i*sample_rate_);
      while (blocks_[j+1].rank1_ < i*sample_rate_) {
        j++;
      }
      sample_[i] = j;  // `sample_[i]` points to the rightmost block with `rank1_ < i * 256`
    }
    sample_[sample_size] = (size_ + 255) / 256;
  }

  void reserve(uint32_t size) {
    if (size > capacity_) {
      capacity_ = (size + 255) / 256 * 256;
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }
  }

  void load_bits(const uint64_t *bits, size_t start, size_t size) {
    if (size_ + size > capacity_) {
      capacity_ = std::max<uint32_t>(((size_ + size) * 3 / 2 + 255) / 256 * 256, 256 * 8);
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }

    uint32_t leftover = (size_ + 255) / 256 * 256 - size_;
    if (size <= leftover) {
      copy_bits(blocks_[size_/256].bits_, size_ % 256, bits, start, size);
      size_ += size;
      return;
    }

    size_t end = start + size;
    copy_bits(blocks_[size_/256].bits_, size_ % 256, bits, start, leftover);
    size_ += leftover;
    start += leftover;
    while (start + 256 <= end) {
      copy_bits(blocks_[size_/256].bits_, size_ % 256, bits, start, 256);
      size_ += 256;
      start += 256;
    }
    copy_bits(blocks_[size_/256].bits_, size_ % 256, bits, start, end - start);
    size_ += end - start;
  }

  void build(uint32_t sample_rate) {
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
      blocks_[size_ / 256].bits_[i] = 0;
    }
    for (uint32_t i = 0; i < 4; i++) {
      blocks_[capacity_ / 256].subranks_[i] = 0;
      blocks_[capacity_ / 256].bits_[i] = 0;
    }

    rank_ = 0;
    for (uint32_t i = 0; i < capacity_ / 256; i++) {
      auto block_rank = blocks_[i].build_index();
      blocks_[i].rank1_ = rank_;
      rank_ += block_rank;
    }
    blocks_[capacity_ / 256].rank1_ = rank_;

    init_sample(sample_rate);
  }

  // get the `pos`-th bit
  auto get(size_t pos) const -> bool {
    assert(pos < size_);

    BENCHMARK( auto start_time = std::chrono::high_resolution_clock::now(); )
    auto ret = static_cast<bool>(GET_BIT(blocks_[pos/256].bits_[(pos%256)/64], pos%64));
    BENCHMARK(
      auto end_time = std::chrono::high_resolution_clock::now();
      get_time_ += (end_time - start_time).count();
    )
    return ret;
  }

  // compute the rank of the first `size` bits
  auto rank1(size_t size) const -> size_t {
    assert(size <= size_);

    BENCHMARK( auto start_time = std::chrono::high_resolution_clock::now(); )
    const auto &block = blocks_[size / 256];
    auto ret = block.rank1_ + block.rank1(size % 256);
    BENCHMARK(
      auto end_time = std::chrono::high_resolution_clock::now();
      rank_time_ += (end_time - start_time).count();
    )
    return ret;
  }

  // select the `rank`-th 1 bit
  auto select1(size_t rank) const -> size_t {
    assert(rank <= rank_);

    BENCHMARK( auto start_time = std::chrono::high_resolution_clock::now(); )

    int block_idx;

    int left, right;
    if (sample_ == nullptr) {
      left = 0;
      right = size_ / 256;
    } else {
      left = sample_[rank/256];
      right = sample_[(rank+255)/256] + 1;
    }

    while (left + 8 < right) {  // binary search when the gap is large
      int mid = (left + right + 1) / 2;
      if (blocks_[mid].rank1_ < rank) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }
    block_idx = left;  // switch to linear search
    while (blocks_[block_idx + 1].rank1_ < rank) {
      block_idx++;
    }

    assert((rank == 0 || blocks_[block_idx].rank1_ < rank) && blocks_[block_idx+1].rank1_ >= rank);

    auto &block = blocks_[block_idx];
    size_t remainder = rank - block.rank1_;
    size_t ret = block_idx*256 + block.select1(remainder);

    BENCHMARK(
      auto end_time = std::chrono::high_resolution_clock::now();
      select_time_ += (end_time - start_time).count();
    )
    return ret;
  }

  // return the position of the first 1 bit starting from position `pos` (inclusive)
  // if not found, return `size_`
  auto next1(size_t pos) const -> size_t {
    if (pos >= size_) {
      return size_;
    }
    uint32_t elt_idx = pos / 64;
    uint32_t remainder = pos % 64;
    uint32_t dist = 0;
    const uint64_t &elt = blocks_[elt_idx / 4].bits_[elt_idx % 4];
    if (elt >> remainder) {
      return pos + __builtin_ctzll(elt >> remainder);
    }
    dist += 64 - remainder;
    while (pos + dist < size_) {
      elt_idx = (pos + dist) / 64;
      const uint64_t &elt = blocks_[elt_idx / 4].bits_[elt_idx % 4];
      if (elt) {
        return pos + dist + __builtin_ctzll(elt);
      }
      dist += 64;
    }
    return size_;
  }

  void serialize(uint64_t *dst, size_t start) {
    int num_blocks = (size_ + 255) / 256;
    for (int i = 0; i < num_blocks; i++) {
      blocks_[i].serialize(dst, start);
      start += 256;
    }
  }

#ifdef __TRACK_MEMORY__
  void print_memory_usage() const {
    printf("[STATIC BITVECTOR MEMORY USAGE]\n");
    size_t actual = size();
    size_t total = sizeof(Block) * (size_ + 255) / 256 + sizeof(StaticBitVector);
    if (sample_ != nullptr) {
      total += sizeof(uint32_t) * (rank_ + sample_rate_ - 1) / sample_rate_;
    }
    total *= 8;
    printf("bits stored: %lu, total bits used: %lu, usage: %lf\n", actual, total, (double)actual/total);
  }
#endif

#ifdef __DEBUG__
  __NOINLINE void print() const {
    printf("size: %u, rank: %u\nbits: ", size_, rank_);
    for (size_t i = 0; i < (size_ + 255) / 256; i++) {
      printf("%016lx,%016lx,%016lx,%016lx,", blocks_[i].bits_[0], blocks_[i].bits_[1],
                                             blocks_[i].bits_[2], blocks_[i].bits_[3]);
    }
    printf("\n");
  }
#endif

#ifdef __MICROBENCHMARK__
  static void print_microbenchmark() {
    printf("[STATIC BITVECTOR MICROBENCHMARK]\n");
    printf("get: %lf ms; rank: %lf ms; select: %lf ms\n",
           (double)get_time_/1000000, (double)rank_time_/1000000, (double)select_time_/1000000);
  }

  static void clear_microbenchmark() {
    get_time_ = rank_time_ = select_time_ = 0;
  }
#endif
 private:
  Block *blocks_{nullptr};

  uint32_t *sample_{nullptr};

  uint32_t capacity_{0};

  uint32_t size_{0};

  uint32_t rank_{0};

  uint32_t sample_rate_{0};

#ifdef __MICROBENCHMARK__
  static size_t get_time_;
  static size_t rank_time_;
  static size_t select_time_;
#endif

#ifdef __DEBUG__
  friend class StaticBitVectorTest;
#endif
};

#ifdef __MICROBENCHMARK__
size_t StaticBitVector::get_time_ = 0;
size_t StaticBitVector::rank_time_ = 0;
size_t StaticBitVector::select_time_ = 0;
#endif

#undef BENCHMARK