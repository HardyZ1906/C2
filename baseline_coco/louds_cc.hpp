#pragma once

#include "defs.hpp"

// #define __BENCH_BV__
#ifdef __BENCH_BV__
# define BENCH_BV(foo) foo
# include <chrono>
#else
# define BENCH_BV(foo)
#endif

// #define __BENCH_LOUDS__
#ifdef __BENCH_LOUDS__
# define BENCH_LOUDS(foo) foo
#else
# define BENCH_LOUDS(foo)
#endif


template<int spill_threshold = 128>
struct LoudsCC {
  static_assert(spill_threshold <= 128);
  static constexpr int spill_threshold_ = spill_threshold;

  struct BitVector {
    struct Block {
      uint32_t rank1_;       // rank1 before block
      uint32_t rank00_;      // rank00 before block; used for value navigation
      /** select index for computing select0(rank1(pos))
       * msb: indicates if the select index is spilled or not
       * not spilled - bits 0-23: maximum block index with rank0 < this->rank1_; bits 24-30: distance to the upper bound
       * spilled - bits 0-30: spill index (only half the bits are 0, so no overflow)
       */
      uint32_t select0_;
      uint8_t subranks_[4];  // rank1 in block before each uint64_t element
      uint64_t bits_[4];     // 256 actual bits

      auto build_index() -> uint32_t {
        uint32_t subrank = 0;
        for (int i = 0; i < 4; i++) {
          subranks_[i] = subrank;
          subrank += __builtin_popcountll(bits_[i]);
        }
        return subrank;
      }

      __HOT __ALWAYS_INLINE auto rank1(uint32_t size) const -> uint32_t {
        return subranks_[size/64] + __builtin_popcountll(bits_[size/64] & MASK(size%64));
      }

      __HOT __ALWAYS_INLINE auto rank00(uint32_t size, bool prev) const -> uint32_t {
        uint32_t ret = 0;
        for (int i = 0; i < size / 64; i++) {  // maybe make branchless?
          ret += rank00ll(bits_[i], prev);
          prev = (bits_[i] >> 63);
        }
        ret += rank00ll(bits_[size/64] | ~MASK(size%64), prev);
        return ret;
      }

      __HOT __ALWAYS_INLINE auto select0(uint32_t rank) const -> uint32_t {
        if (rank <= 64*2 - subranks_[2]) {
          if (rank <= 64*1 - subranks_[1]) {
            return 0*64 + select0ll(bits_[0], rank);
          } else {
            return 1*64 + select0ll(bits_[1], rank - (64*1 - subranks_[1]));
          }
        } else {
          if (rank <= 64*3 - subranks_[3]) {
            return 2*64 + select0ll(bits_[2], rank - (64*2 - subranks_[2]));
          } else {
            return 3*64 + select0ll(bits_[3], rank - (64*3 - subranks_[3]));
          }
        }
      }
    };  // 48 bytes

  #ifdef __BENCH_BV__
    static size_t get_time_;
    static size_t rank_time_;
    static size_t select_time_;
    static size_t num_selects_;
    static size_t probed_blocks_;
    static size_t max_probe_dist_;

    static void print_microbenchmark() {
      printf("get time: %lf ms; rank time: %lf ms; select time: %lf ms\n", (double)get_time_/1000000,
             (double)rank_time_/1000000, (double)select_time_/1000000);
      printf("%ld select probed %ld blocks, avg probe dist = %lf, max probe dist = %ld\n", num_selects_, probed_blocks_, (double)probed_blocks_/num_selects_, max_probe_dist_);
    }

    static void clear_microbenchmark() {
      max_probe_dist_ = probed_blocks_ = num_selects_ = get_time_ = rank_time_ = select_time_ = 0;
    }
  #else
    static void print_microbenchmark() {}
    static void clear_microbenchmark() {}
  #endif

    Block *blocks_{nullptr};

    // in blocks where 0 bits are sparse, spills_ contains precomputed select0 results for EVERY possible input
    uint32_t *spills_{nullptr};

    uint32_t capacity_{0};

    uint32_t size_{0};

    uint32_t rank1_{0};

    uint32_t rank00_{0};

    uint32_t spill_size_{0};

    BitVector() = default;

    BitVector(size_t size) {
      reserve(size);
    }

    ~BitVector() {
      free(blocks_);
      free(spills_);
    }

    auto size_in_bytes() const -> size_t {
      return sizeof(Block) * (capacity_/256 + 1) + sizeof(uint32_t) * spill_size_ + sizeof(BitVector);
    }

    auto size_in_bits() const -> size_t {
      return size_in_bytes() * 8;
    }

    auto size() const -> size_t {
      return size_;
    }

    __HOT __ALWAYS_INLINE auto rank1() const -> size_t {
      return rank1_;
    }

    __HOT __ALWAYS_INLINE auto rank00() const -> size_t {
      return rank00_;
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

    void clear() {
      free(blocks_);
      free(spills_);
      blocks_ = nullptr;
      spills_ = nullptr;
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

    void init_select() {
      int num_blocks = (size_ + 255) / 256;

      // build select index
      int i = 0, j = 0;
      while (i < num_blocks) {
        while ((j+1)*256 - blocks_[j+1].rank1_ < blocks_[i].rank1_) {
          j++;
        }
        // j is the maximum block index satisfying blocks_[j].rank0_ < blocks_[i].rank1_
        blocks_[i].select0_ = j;
        i++;
      }
      blocks_[capacity_/256].select0_ = capacity_/256;  // sentinel
    }

    void init_spills() {
      int num_blocks = capacity_ / 256;
      int num_spills = 0;
      for (int i = 0; i < num_blocks; i++) {
        uint32_t dist = blocks_[i+1].select0_ - blocks_[i].select0_;
        if (dist >= spill_threshold_) {
          blocks_[i].select0_ |= BIT(31);
          // also precompute blocks_[i].rank1_ just in case block starts with 0 bits
          num_spills += blocks_[i+1].rank1_ - blocks_[i].rank1_ + 1;
        } else {
          blocks_[i].select0_ |= (dist << 24); // # bits fits in 32 bits, so # blocks fits in 24 bits
        }
      }

      spills_ = reinterpret_cast<uint32_t *>(realloc(spills_, num_spills * sizeof(uint32_t)));
      spill_size_ = num_spills;
      int spilled = 0;
      for (int i = 0; i < num_blocks; i++) {
        if (!GET_BIT(blocks_[i].select0_, 31)) {
          continue;
        }
        uint32_t block_idx = blocks_[i].select0_ & MASK(31);
        blocks_[i].select0_ = BIT(31) | spilled;

        uint32_t remainder = blocks_[i].rank1_ - (block_idx*256 - blocks_[block_idx].rank1_);
        uint32_t pos = block_idx*256 + blocks_[block_idx].select0(remainder);
        for (uint32_t j = 0; j <= blocks_[i+1].rank1_ - blocks_[i].rank1_; j++) {
          spills_[spilled++] = pos;
          pos = next0(pos + 1);
          assert(spilled <= spill_size_);
        }
      }
      assert(spilled == spill_size_);
    }

    void build() {
      if (capacity_ - size_ >= 256) {
        capacity_ = (size_ + 255) / 256 * 256;
        blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_/256 + 1)));
      }

      if (blocks_ == nullptr) {
        return;
      }

      // clear trailing bits
      uint32_t remainder = size_ % 256;
      blocks_[size_/256].bits_[remainder/64] &= MASK(remainder%64);
      for (uint32_t i = (remainder + 63) / 64; i < 4; i++) {
        blocks_[size_/256].bits_[i] = 0;
      }
      for (uint32_t i = 0; i < 4; i++) {
        blocks_[capacity_/256].subranks_[i] = 0;
      }

      rank1_ = rank00_ = 0;
      bool prev = true;
      for (uint32_t i = 0; i < capacity_ / 256; i++) {
        uint32_t block_rank1 = blocks_[i].build_index();
        blocks_[i].rank1_ = rank1_;
        rank1_ += block_rank1;

        blocks_[i].rank00_ = rank00_;
        rank00_ += rank00ll(blocks_[i].bits_[0], prev) + rank00ll(blocks_[i].bits_[1], blocks_[i].bits_[0] >> 63) +
                   rank00ll(blocks_[i].bits_[2], blocks_[i].bits_[1] >> 63) +
                   rank00ll(blocks_[i].bits_[3], blocks_[i].bits_[2] >> 63);
        prev = blocks_[i].bits_[3] >> 63;
      }
      if (size_ < capacity_) {
        rank00_ -= capacity_ - size_ - get(size_ - 1);
      }
      blocks_[capacity_/256].rank1_ = rank1_;
      blocks_[capacity_/256].rank00_ = rank00_;

      init_select();
      init_spills();
    }

    // get the `pos`-th bit
    __HOT __ALWAYS_INLINE auto get(size_t pos) const -> bool {
      assert(pos < size_);

      BENCH_BV( auto start_time = std::chrono::high_resolution_clock::now(); )
      bool ret = GET_BIT(blocks_[pos/256].bits_[(pos%256)/64], pos%64);
      BENCH_BV(
        auto end_time = std::chrono::high_resolution_clock::now();
        get_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // return the number of 1 bits in the first `size` bits
    __HOT __ALWAYS_INLINE auto rank1(size_t size) const -> size_t {
      assert(size <= size_);

      BENCH_BV( auto start_time = std::chrono::high_resolution_clock::now(); )
      const auto &block = blocks_[size / 256];
      size_t ret = block.rank1_ + block.rank1(size % 256);
      BENCH_BV(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // return the number of 0 bits in the first `size` bits
    __HOT __ALWAYS_INLINE auto rank0(size_t size) const -> size_t {
      return size - rank1(size);
    }

    // return the number of 00 patterns in the first `size` bits
    __HOT __ALWAYS_INLINE auto rank00(size_t size) const -> size_t {
      assert(size <= size_);

      BENCH_BV( auto start_time = std::chrono::high_resolution_clock::now(); )
      const auto &block = blocks_[size / 256];
      bool prev = (size >= 256 && (blocks_[size/256 - 1].bits_[3] >> 63));
      size_t ret = block.rank00_ + block.rank00(size % 256, prev);
      BENCH_BV(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )
      return ret;
    }

    // return the position of the closest 0 bit after position `pos` (inclusive)
    // if not found, return `size_`
    __HOT __ALWAYS_INLINE auto next0(size_t pos) const -> size_t {
      BENCH_BV( auto start = std::chrono::high_resolution_clock::now(); )
      uint32_t idx = pos / 64;
      uint64_t elt = ~blocks_[idx/4].bits_[idx%4] & ~MASK(pos%64);
      while (elt == 0) {
        idx++;
        elt = ~blocks_[idx/4].bits_[idx%4];
      }
      size_t ret = idx*64 + __builtin_ctzll(elt);
      BENCH_BV(
        auto end = std::chrono::high_resolution_clock::now();
        get_time_ += (end - start).count();
      )
      return ret;
    }

    // return {rank1(pos+1), select0(rank1(pos+1))}
    __HOT __ALWAYS_INLINE auto r1s0(size_t pos) const -> std::pair<size_t, size_t> {
      assert(pos < size_);

      Block &block = blocks_[(pos+1)/256];
      BENCH_BV( auto start_time = std::chrono::high_resolution_clock::now(); )
      size_t rank = block.rank1_ + block.rank1((pos+1)%256);
      BENCH_BV(
        auto end_time = std::chrono::high_resolution_clock::now();
        rank_time_ += (end_time - start_time).count();
      )

      BENCH_BV( num_selects_++; )

      BENCH_BV( start_time = std::chrono::high_resolution_clock::now(); )
      if (GET_BIT(block.select0_, 31)) {
        uint32_t spill_idx = block.select0_ & MASK(31);
        BENCH_BV(
          end_time = std::chrono::high_resolution_clock::now();
          select_time_ += (start_time - end_time).count();
          probed_blocks_++;
        )
        return {rank, spills_[spill_idx + (rank - block.rank1_)]};
      }

      BENCH_BV( size_t probe_dist = 1; )
      int left = block.select0_ & MASK(24), right = left + (block.select0_ >> 24) + 1;
      assert(left*256 - blocks_[left].rank1_ < rank);
      assert(right*256 - blocks_[right].rank1_ >= rank);

      while (right - left > 8) {
        int mid = (left + right + 1) / 2;
        if (mid*256 - blocks_[mid].rank1_ < rank) {
          left = mid;
        } else {
          right = mid - 1;
        }
        BENCH_BV( probe_dist++; )
      }
      while ((left+1)*256 - blocks_[left+1].rank1_ < rank) {
        left++;
        BENCH_BV( probe_dist++; )
      }

      size_t remainder = rank - (left*256 - blocks_[left].rank1_);
      size_t ret = left*256 + blocks_[left].select0(remainder);
      BENCH_BV(
        end_time = std::chrono::high_resolution_clock::now();
        select_time_ += (end_time - start_time).count();
        probed_blocks_ += probe_dist;
        if (max_probe_dist_ < probe_dist) {
          // printf("pos %ld: probe dist %d\n", pos, probe_dist);
          max_probe_dist_ = probe_dist;
        }
      )
      return {rank, ret};
    }

    // used for debugging only
    auto select0(size_t rank) const -> size_t {
      assert(rank <= rank1_);

      int left = 0;
      int right = size_ / 256;

      while (left + 8 < right) {  // binary search when the gap is large
        int mid = (left + right + 1) / 2;
        if (mid*256 - blocks_[mid].rank1_ < rank) {
          left = mid;
        } else {
          right = mid - 1;
        }
      }
      int block_idx = left;  // switch to linear search
      while ((block_idx+1)*256 - blocks_[block_idx+1].rank1_ < rank) {
        block_idx++;
      }

      assert((rank == 0 || block_idx*256 - blocks_[block_idx].rank1_ < rank) &&
             (block_idx+1)*256 - blocks_[block_idx+1].rank1_ >= rank);

      auto &block = blocks_[block_idx];
      size_t remainder = rank - (block_idx*256 - block.rank1_);
      return block_idx*256 + block.select0(remainder);
    }
  };

#ifdef __BENCH_LOUDS__
  static size_t select_time_;
#endif

  BitVector bv_;

  LoudsCC() = default;

  LoudsCC(size_t size) : bv_(size*2 + 1) {
    bv_.append1();
    bv_.append0();
  }

  ~LoudsCC() = default;

#ifdef __BENCH_LOUDS__
  static void print_microbenchmark() {
    printf("select time: %lf ms\n", (double)select_time_/1000000);
    BitVector::print_microbenchmark();
  }

  static void clear_microbenchmark() {
    select_time_ = 0;
    BitVector::clear_microbenchmark();
  }
#else
  static void print_microbenchmark() {
    BitVector::print_microbenchmark();
  }

  static void clear_microbenchmark() {
    BitVector::clear_microbenchmark();
  }
#endif

  // index of the root
  static constexpr size_t root_idx = 2;

  void add_node(size_t num_child) {
    for (int i = 0; i < num_child; i++) {
      bv_.append1();
    }
    bv_.append0();
  }

  void build_rank_select_ds() {
    bv_.build();
  }

  auto size_in_bytes() const -> size_t {
    return bv_.size_in_bytes();
  }

  inline size_t size_in_bits() const {
    return size_in_bytes() * 8;
  }

  // move to the k-th child of the node started at bit `pos`
  // return the start of child node
  auto move_to_child(size_t pos, size_t k) const -> std::pair<size_t, size_t> {
    BENCH_LOUDS( auto start = std::chrono::high_resolution_clock::now(); )
    auto [rank, child_pos] = bv_.r1s0(pos + k);
    BENCH_LOUDS(
      auto end = std::chrono::high_resolution_clock::now();
      select_time_ += (end - start).count();
    )
    return {rank, child_pos + 1};
  }

  auto n_th_child_rank(size_t pos, size_t k) const -> size_t {
    return bv_.rank1(pos + k + 1);
  }

  auto num_child(size_t pos) const -> size_t {
    assert(pos == 0 || bv_.get(pos - 1) == 0);
    size_t next0 = bv_.next0(pos);
    return next0 - pos;
  }

  auto is_leaf(size_t pos) const -> bool {
    assert(pos == 0 || bv_.get(pos - 1) == 0);
    assert(pos < bv_.size());
    return bv_.get(pos) == 0;
  }

  auto leaf_rank(size_t pos) const -> size_t {
    assert(pos < bv_.size());
    return bv_.rank00(pos);
  }

  // from position in the bv to internal node index
  auto internal_rank(size_t pos, size_t node_rank) const -> size_t {
    return node_rank - leaf_rank(pos);
  }
};

#ifdef __BENCH_BV__
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::get_time_ = 0;
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::rank_time_ = 0;
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::select_time_ = 0;
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::num_selects_ = 0;
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::probed_blocks_ = 0;
template<int spill_threshold> size_t LoudsCC<spill_threshold>::BitVector::max_probe_dist_ = 0;
#endif

#ifdef __BENCH_LOUDS__
template<int spill_threshold> size_t LoudsCC<spill_threshold>::select_time_ = 0;
#endif

#undef __BENCH_LOUDS__
#undef __BENCH_BV__
#undef BENCH_BV
#undef BENCH_LOUDS
