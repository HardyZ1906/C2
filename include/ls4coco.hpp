#pragma once

#include "utils.hpp"
#include "fst_builder.hpp"
#include "static_vector.hpp"

#include <limits>
#include <vector>


// #define __DEBUG_LS4COCO__


// louds-sparse trie used for building CoCo-trie
// similar to LoudsSparseCC, except that 1) cutoff is always 0, and 2) supports parent navigation
template<typename Key>
class LS4CoCo {
 public:
  using key_type = Key;
  using label_vec = StaticVector<uint8_t>;
  using builder_type = FstBuilder<key_type>;

  static constexpr uint8_t terminator_ = 0;

 private:
  struct BitVector {
    struct Block {  // packed has-child and louds bitvectors
      uint32_t rank0_{0};    // cumulative block ranks of bv<0> (i.e. has-child)
      uint32_t rank1_{0};    // cumulative block ranks of bv<1> (i.e. louds)
      uint32_t select0_{0};  // index of the last block such that blocks_[select0_].rank1_ + 1 < rank0_; used for parent navigation
      uint32_t select1_{0};  // index of the last block such that blocks_[select1_].rank0_ - 1 < rank1_; used for child navigation
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
      auto rank0(uint32_t size) const -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);
        return size - rank1<bvnum>(size);
      }

      template<int bvnum>
      auto select1(uint32_t rank) const -> uint32_t {
        static_assert(bvnum == 0 || bvnum == 1);
        if constexpr (bvnum == 0) {
          assert(rank <= rank1<0>());
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
          assert(rank <= rank1<1>());
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
    };  // 88 bytes

    Block *blocks_{nullptr};

    uint32_t capacity_{0};

    uint32_t size_{0};

    uint32_t rank_[2]{0};

    BitVector() = default;

    ~BitVector() {
      free(blocks_);
    }

    auto size() const -> uint32_t {
      return size_;
    }

    template<int bvnum>
    auto rank1() const -> uint32_t {
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

    void load_bits(const uint64_t *bits0, const uint64_t *bits1, size_t start, uint32_t size) {
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

      // build rank index
      rank_[0] = rank_[1] = 0;
      for (uint32_t i = 0; i < capacity_ / 256; i++) {
        blocks_[i].rank0_ = rank_[0];
        blocks_[i].rank1_ = rank_[1];
        rank_[0] += blocks_[i].template build_index<0>();
        rank_[1] += blocks_[i].template build_index<1>();
      }
      blocks_[capacity_ / 256].rank0_ = rank_[0];
      blocks_[capacity_ / 256].rank1_ = rank_[1];

      // build select index
      init_select();
    }

    void init_select() {
      int num_blocks = capacity_ / 256;

      // build index for bv<0>.select1(bv<1>.rank1(pos) - 1)
      uint32_t i = 0, j = 0;
      while (i <= num_blocks) {
        while (j < num_blocks && blocks_[j+1].rank0_ + 1 < blocks_[i].rank1_) {
          j++;
        }
        blocks_[i].select0_ = j;
        i++;
      }

      // build index for bv<1>.select1(bv<0>.rank1(pos) + 1)
      i = j = 0;
      while (i <= num_blocks) {
        while (j < num_blocks && blocks_[j+1].rank1_ < blocks_[i].rank0_ + 1) {
          j++;
        }
        blocks_[i].select1_ = j;
        i++;
      }
    }

    // get the `pos`-th bit of bv<bvnum>
    template<int bvnum>
    auto get(uint32_t pos) const -> bool {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(pos < size_);
      return blocks_[pos/256].template get<bvnum>(pos % 256);
    }

    // compute the rank of the first `size` bits of bv<bvnum>
    template<int bvnum>
    auto rank1(uint32_t size) const -> uint32_t {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(size <= size_);

      const auto &block = blocks_[size/256];
      if constexpr (bvnum == 0) {
        return block.rank0_ + block.template rank1<0>(size % 256);
      } else {
        return block.rank1_ + block.template rank1<1>(size % 256);
      }
    }

    template<int bvnum>
    auto rank0(uint32_t size) const -> uint32_t {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(size <= size_);
      return size - rank1(size);
    }

    // select the `rank`-th 1 bit from bv<bvnum>; used in build phase only
    template<int bvnum>
    auto select1(uint32_t rank) const -> uint32_t {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(rank <= rank_[bvnum]);

      uint32_t left = 0, right = capacity_/256;
      if constexpr (bvnum == 0) {
        while (left + 8 < right) {
          uint32_t mid = (left + right + 1) / 2;
          if (blocks_[mid].rank0_ < rank) {
            left = mid;
          } else {
            right = mid - 1;
          }
        }
        while (blocks_[left+1].rank0_ < rank) {
          left++;
        }
        return left*256 + blocks_[left].template select1<0>(rank - blocks_[left].rank0_);
      } else {
        while (left + 8 < right) {
          uint32_t mid = (left + right + 1) / 2;
          if (blocks_[mid].rank1_ < rank) {
            left = mid;
          } else {
            right = mid - 1;
          }
        }
        while (blocks_[left+1].rank1_ < rank) {
          left++;
        }
        return left*256 + blocks_[left].template select1<1>(rank - blocks_[left].rank1_);
      }
    }

    // return the position of the closest 1 bit before position `pos` (inclusive)
    // if not found, return 0
    template<int bvnum>
    auto prev1(uint32_t pos) const -> uint32_t {
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
    auto next1(uint32_t pos) const -> uint32_t {
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

    // bvnum == 0: returns bv<0>.select1(bv<1>.rank1(pos + 1) - 1); used for child navigation
    // bvnum == 1: returns bv<1>.select1(bv<0>.rank1(pos + 1) + 1); used for parent navigation
    template<int bvnum>
    auto rs(uint32_t pos) const -> uint32_t {
      static_assert(bvnum == 0 || bvnum == 1);
      assert(pos < size_);

      Block &block = blocks_[(pos+1)/256];
      if constexpr (bvnum == 0) {
        uint32_t rank = block.rank1_ + block.rank1<1>((pos+1)%256) - 1;
        assert(rank != static_cast<uint32_t>(-1));
        uint32_t left = block.select0_, right = blocks_[(pos+1)/256 + 1].select0_ + 1;
        assert(rank == 0 || blocks_[left].rank0_ < rank);
        assert(blocks_[right].rank0_ >= rank);
        while (left + 8 < right) {
          uint32_t mid = (left + right + 1) / 2;
          if (blocks_[mid].rank0_ < rank) {
            left = mid;
          } else {
            right = mid - 1;
          }
        }
        while (blocks_[left+1].rank0_ < rank) {
          left++;
        }
        return left*256 + blocks_[left].template select1<0>(rank - blocks_[left].rank0_);
      } else {
        uint32_t rank = block.rank0_ + block.rank1<0>((pos+1)%256) + 1;
        uint32_t left = block.select1_, right = blocks_[(pos+1)/256 + 1].select1_ + 1;
        assert(blocks_[left].rank1_ < rank);
        assert(blocks_[right].rank1_ >= rank);
        while (left + 8 < right) {
          uint32_t mid = (left + right + 1) / 2;
          if (blocks_[mid].rank1_ < rank) {
            left = mid;
          } else {
            right = mid - 1;
          }
        }
        while (blocks_[left+1].rank1_ < rank) {
          left++;
        }
        return left*256 + blocks_[left].template select1<1>(rank - blocks_[left].rank1_);
      }
    }

    auto size_in_bytes() const -> size_t {
      return sizeof(BitVector) + sizeof(Block) * capacity_/256;
    }

    auto size_in_bits() const -> size_t {
      return size_in_bytes() * 8;
    }
  };
  using bitvec = BitVector;
 public:
  // helper class for walking down the trie and traversing macro-node keys; used by the optimizer
  struct walker {
    using trie_t = LS4CoCo;

    key_type key_;
    const trie_t *trie_;
    uint32_t pos_;
    uint32_t level_;  // number of walked levels

    walker(const trie_t *trie, uint32_t pos) : trie_(trie), pos_(pos), level_(1) {
      uint8_t label = trie_->get_label(pos_);
      if (label != terminator_) {
        key_.push_back(label);
      }
    }

    walker(const walker &other) : key_(other.key_), trie_(other.trie_), pos_(other.pos_), level_(other.level_) {}

    // is the key legitimately terminated?
    auto valid() const -> bool {
      return !trie_->has_child(pos_);
    }

    // is the current key a prefix key?
    auto prefix_key() const -> bool {
      return trie_->get_label(pos_) == terminator_;
    }

    auto key() const -> const key_type & {
      return key_;
    }

    // move to the leftmost label in node
    void move_to_front() {
      uint32_t front = trie_->node_start(pos_);
      if (trie_->get_label(front) == terminator_) {
        pos_ = front;
        key_.pop_back();
      } else {
        pos_ = front;
        key_.back() = trie_->get_label(pos_);
      }
    }

    // move to the rightmost label in node
    void move_to_back() {
      uint32_t back = trie_->node_end(pos_) - 1;
      if (trie_->get_label(pos_) == terminator_) {
        pos_ = back;
        key_.push_back(trie_->get_label(pos_));
      } else {
        pos_ = back;
        key_.back() = trie_->get_label(pos_);
      }
    }

    // does NOT regress if current level is already greater than `max_level`
    void get_min_key(uint32_t max_level = std::numeric_limits<uint32_t>::max()) {
      while (level_ < max_level && trie_->has_child(pos_)) {
        pos_ = trie_->child_pos(pos_);  // keep taking the leftmost branch
        uint8_t label = trie_->get_label(pos_);
        if (label != terminator_) {
          key_.push_back(label);
        }
        level_++;
      }
    }

    // does NOT regress if current level is already greater than `max_level`
    void get_max_key(uint32_t max_level = std::numeric_limits<uint32_t>::max()) {
      while (level_ < max_level && trie_->has_child(pos_)) {
        pos_ = trie_->child_pos(pos_);
        pos_ = trie_->node_end(pos_) - 1;  // keep taking the rightmost branch
        uint8_t label = trie_->get_label(pos_);
        if (label != terminator_) {
          key_.push_back(label);
        }
        level_++;
      }
    }

    // make sure to call `get_min_key(max_level)` before calling this
    // return true on success and false if there is no more key
    auto next(uint32_t max_level = std::numeric_limits<uint32_t>::max()) -> bool {
      while (level_ > 0) {
        if (trie_->get_label(pos_) == terminator_) {  // terminator is never the last label
          pos_++;
          key_.push_back(trie_->get_label(pos_));
          get_min_key(max_level);
          return true;
        } else if (pos_ + 1 < trie_->bv_.size() && !trie_->louds(pos_ + 1)) {  // not the last label in node
          pos_++;
          key_.back() = trie_->get_label(pos_);
          get_min_key(max_level);
          return true;
        }
        // last label in node; regress to parent
        key_.pop_back();
        pos_ = trie_->parent_pos(pos_);
        level_--;
      }
      return false;
    }

    // move to the leftmost next-level node in subtrie
    // return true if found and false otherwise
    // calling any of the `move_down` functions after false is returned is undefined behavior
    auto move_down_one_level_left() -> bool {
      uint32_t next_level = level_ + 1;

      get_min_key(next_level);
      while (level_ < next_level) {
        assert(!trie_->has_child(pos_));  // key terminates before `next_level`
        while (true) {
          if (trie_->get_label(pos_) != terminator_) {
            key_.pop_back();
          }
          uint32_t end = trie_->node_end(pos_);
          uint32_t next = trie_->bv_.next1<0>(pos_ + 1);  // has_child.next1(pos_ + 1)
          if (next < end) {  // trace next branch
            pos_ = next;
            key_.push_back(trie_->get_label(pos_));
            break;
          }
          // regress to parent
          pos_ = trie_->parent_pos(pos_);
          level_--;
          if (level_ == 0) {
            return false;
          }
        }
        get_min_key(next_level);
      }
      return true;
    }

    // move to the rightmost next-level node in subtrie
    // return true if found and false otherwise
    // calling any of the `move_down` functions after false is returned is undefined behavior
    auto move_down_one_level_right() -> bool {
      uint32_t next_level = level_ + 1;

      get_max_key(next_level);
      while (level_ < next_level) {
        assert(!trie_->has_child(pos_));  // key terminates before `next_level`
        while (true) {
          if (trie_->get_label(pos_) != terminator_) {
            key_.pop_back();
          }
          uint32_t start = trie_->node_start(pos_);
          uint32_t prev = trie_->bv_.prev1<0>(pos_ - 1);  // has_child.prev1(pos_)
          if (prev >= start) {  // trace previous branch
            pos_ = prev;
            key_.push_back(trie_->get_label(pos_));
            break;
          }
          // regress to parent
          pos_ = trie_->parent_pos(pos_);
          level_--;
          if (level_ == 0) {
            return false;
          }
        }
        get_max_key(next_level);
      }
      return true;
    }
  };
 public:
  LS4CoCo() = default;

  // keys must be sorted and unique
  template<typename Iterator>
  void build(Iterator begin, Iterator end) {
    builder_type builder;
    builder.build(begin, end);
    build(builder);
  #ifdef __DEBUG_LS4COCO__
    auto left_walker = walker(this, 0);
    left_walker.get_min_key();
    uint32_t cnt = 0;
    while (begin != end) {
      // printf("%d:%s:%s\n", cnt, left_walker.key().c_str(), begin->c_str());
      assert(left_walker.key() == *begin);
      ++begin;
      ++cnt;
      left_walker.next();
    }
  #endif
  }

  void build(builder_type &builder) {
    assert(builder.s_labels_.size() > 0);

    clear();

    depth_ = builder.s_labels_.size();

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
    bv_.build();
  }

  void clear() {
    labels_.clear();
    bv_.clear();
    depth_ = 0;
  }

  auto node_start(uint32_t pos) const -> uint32_t {
    return bv_.prev1<1>(pos);
  }

  auto node_end(uint32_t pos) const -> uint32_t {
    return bv_.next1<1>(pos + 1);
  }

  auto node_degree(uint32_t pos) const -> uint32_t {
    return node_end(pos) - node_start(pos);
  }

  auto node_id(uint32_t pos) const -> uint32_t {
    assert(louds(pos));
    return bv_.rank1<1>(pos);
  }

  auto value_pos(uint32_t pos) const -> uint32_t {
    assert(!has_child(pos));
    return bv_.rank0<0>(pos);
  }

  auto child_pos(uint32_t pos) const -> uint32_t {
    return bv_.rs<1>(pos);
  }

  auto parent_pos(uint32_t pos) const -> uint32_t {
    return bv_.rs<0>(pos);
  }

  auto has_child(uint32_t pos) const -> bool {
    return bv_.get<0>(pos);
  }

  auto louds(uint32_t pos) const -> bool {
    return bv_.get<1>(pos);
  }

  auto get_label(uint32_t pos) const -> uint8_t {
    return labels_.at(pos);
  }

  auto num_nodes() const -> uint32_t {
    return bv_.rank1<1>();
  }

  auto depth() const -> uint32_t {
    return depth_;
  }

  // return the cutoffs between levels
  auto get_level_boundaries() const -> std::vector<uint32_t> {
    std::vector<uint32_t> ret(depth_ + 1);
    uint32_t num_children = 0;
    ret[0] = 0;
    for (uint32_t i = 1; i <= depth_; i++) {
      uint32_t pos = node_end(bv_.select1<1>(num_children + 1));  // louds.select1(num_children)
      ret[i] = pos;
      num_children = bv_.rank1<0>(pos);  // has_child.rank1(pos)
    }
    assert(ret[depth_] == bv_.size_);  // sentinel
    return ret;
  }

  auto size_in_bytes() const -> size_t {
    return labels_.size_in_bytes() + bv_.size_in_bytes() + sizeof(uint32_t);
  }

  auto size_in_bits() const -> size_t {
    return size_in_bytes() * 8;
  }

 private:
  // 8 bits per label; indicates valid branches at each node, in level order
  // labels belonging to the same node are sorted in ascending order
  label_vec labels_;

  // 2 bits per label; packed has-child and louds
  bitvec bv_;

  uint32_t depth_{0};

  friend class walker;
  template <typename K> friend class CoCoOptimizer;
  template <typename K, bool r> friend class CoCoCC;
};
