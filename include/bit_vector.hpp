#pragma once

#include "utils.hpp"


// rank-only bit vector; an external select index can be built to support select queries
struct BitVector {
  struct Block {
    uint32_t rank_{0};
    uint8_t subrank_[4]{0};
    uint64_t bits_[4]{0};

    auto build_index() -> uint32_t {
      uint32_t rank = 0;
      for (int i = 0; i < 4; i++) {
        subrank_[i] = rank;
        rank += __builtin_popcountll(bits_[i]);
      }
      return rank;
    }

    auto get(uint32_t pos) const -> bool {
      assert(pos < 256);
      return GET_BIT(bits_[pos/64], pos % 64);
    }

    auto rank1(uint32_t size) const -> uint32_t {
      return subrank_[size/64] + __builtin_popcountll(bits_[size/64] & MASK(size%64));
    }

    auto rank0(uint32_t size) const -> uint32_t {
      return size - rank1(size);
    }

    auto select1(uint32_t rank) const -> uint32_t {
      assert(rank <= rank1());
      if (rank <= subrank_[2]) {
        if (rank <= subrank_[1]) {
          return 64*0 + selectll(bits_[0], rank);
        } else {
          return 64*1 + selectll(bits_[1], rank - subrank_[1]);
        }
      } else {
        if (rank <= subrank_[3]) {
          return 64*2 + selectll(bits_[2], rank - subrank_[2]);
        } else {
          return 64*3 + selectll(bits_[3], rank - subrank_[3]);
        }
      }
    }

    auto rank1() const -> uint32_t {
      return subrank_[3] + __builtin_popcountll(bits_[3]);
    }
  };  // 40 bytes

  Block *blocks_{nullptr};

  uint32_t capacity_{0};

  uint32_t size_{0};

  uint32_t rank_{0};

  BitVector() = default;

  ~BitVector() {
    clear();
  }

  auto size() const -> uint32_t {
    return size_;
  }

  auto rank1() const -> uint32_t {
    return rank_;
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

  void load_bits(const uint64_t *bits, size_t start, uint32_t size) {
    if (size_ + size > capacity_) {
      capacity_ = std::max<uint32_t>(((size_ + size) * 2 + 255) / 256 * 256, 256 * 8);
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

  auto size_in_bytes() const -> size_t {
    return sizeof(BitVector) + sizeof(Block) * capacity_/256;
  }

  auto size_in_bits() const -> size_t {
    return size_in_bytes() * 8;
  }

  void shrink_to_fit() {
    if (capacity_ - size_ >= 256) {
      capacity_ = (size_ + 255) / 256 * 256;
      blocks_ = reinterpret_cast<Block *>(realloc(blocks_, sizeof(Block) * (capacity_ / 256 + 1)));
    }
  }

  void build() {
    if (blocks_ == nullptr) {
      return;
    }

    shrink_to_fit();

    // clear trailing bits
    uint32_t remainder = size_ % 256;
    for (uint32_t i = (remainder + 63) / 64; i < 4; i++) {
      blocks_[size_ / 256].bits_[i] = 0;
    }
    for (uint32_t i = 0; i < 4; i++) {
      blocks_[capacity_ / 256].subrank_[i] = 0;
      blocks_[capacity_ / 256].bits_[i] = 0;
    }

    // build rank index
    rank_ = 0;
    for (uint32_t i = 0; i < capacity_ / 256; i++) {
      blocks_[i].rank_ = rank_;
      rank_ += blocks_[i].build_index();
    }
    blocks_[capacity_ / 256].rank_ = rank_;
  }

  // get the `pos`-th bit of bv<bvnum>
  auto get(uint32_t pos) const -> bool {
    assert(pos < size_);
    return blocks_[pos/256].get(pos % 256);
  }

  auto rank1(uint32_t size) const -> uint32_t {
    assert(size <= size_);
    const auto &block = blocks_[size/256];
    return block.rank_ + block.rank1(size % 256);
  }

  auto rank0(uint32_t size) const -> uint32_t {
    assert(size <= size_);
    return size - rank1(size);
  }
};