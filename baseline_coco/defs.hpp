#pragma once

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <sys/time.h>

#include <immintrin.h>


#define BIT(n)  (1ull << (n))
#define MASK(n) (BIT(n) - 1ull)
#define GET_BIT(bits, pos) ((bits) & BIT(pos))
#define SET_BIT(bits, pos) ((bits) |= BIT(pos))
#define CLEAR_BIT(bits, pos) ((bits) &= ~BIT(pos))

#define EXPECT(expr1, expr2, op) \
do { \
  if (!((expr1) op (expr2))) { \
    printf("assertion failed: %ld, %ld\n", static_cast<uint64_t>(expr1), static_cast<uint64_t>(expr2)); \
    fflush(stdout); \
    assert((expr1) op (expr2)); \
  } \
} while (0)


// gcc attributues
#define __ALWAYS_INLINE __attribute__((always_inline))
#define __NOINLINE __attribute__((noinline))
#define __HOT __attribute__((hot))


auto usec() -> uint64_t {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

auto log2_ceil(uint64_t x) -> int {
  assert(x > 0);
  if (x & (x-1) == 0) {
    return __builtin_ctzll(x);
  }
  return 64 - __builtin_clzll(x);
}

// 64-bit select 1; refer to https://graphics.stanford.edu/~seander/bithacks.html
auto selectll(uint64_t bits, int rank) -> int {
  // int pos;
  // uint64_t c2, c4, c8, c16;  // popcounts of each 2-bit, 4-bit, 8-bit and 16-bit chunks
  // uint32_t mid;  // popcount of half of the current chunk; used for binary search

  // c2 = bits - ((bits >> 1) & ~0UL/3);
  // c4 = (c2 & ~0UL/5) + ((c2 >> 2) & ~0UL/5);
  // c8 = (c4 + (c4 >> 4)) & ~0UL/0x11;
  // c16 = (c8 + (c8 >> 8)) & ~0UL/0x101;
                                          
  // // binary search, starting with chunk size 64
  // pos = 0;
  // mid = ((c16 >> 16) + c16) & 0xff;
  // if (mid < rank) {
  //   pos += 32;
  //   rank -= mid;
  // }

  // mid = (c16 >> pos) & 0xff;
  // if (mid < rank) {
  //   pos += 16;
  //   rank -= mid;
  // }

  // mid = (c8 >> pos) & 0xf;
  // if (mid < rank) {
  //   pos += 8;
  //   rank -= mid;
  // }

  // mid = (c4 >> pos) & 0x7;
  // if (mid < rank) {
  //   pos += 4;
  //   rank -= mid;
  // }

  // mid = (c2 >> pos) & 0x3;
  // if (mid < rank) {
  //   pos += 2;
  //   rank -= mid;
  // }

  // mid = (bits >> pos) & 0x1;
  // if (mid < rank) {
  //   pos++;
  // }

  // return pos;
  if (rank == 0) {
    return 0;
  }
  return sux::select64(bits, rank - 1);
}

// 64-bit select 0; refer to https://graphics.stanford.edu/~seander/bithacks.html
auto select0ll(uint64_t bits, int rank) -> int {
  return selectll(~bits, rank);
}

auto rank00ll(uint64_t bits, bool prev) -> int {
  auto ret = (!prev & ~bits) + __builtin_popcountll(~bits & (~bits >> 1));
  return ret;
}

auto rank(const uint64_t *bits, size_t begin, size_t end) -> size_t {
  uint64_t block = bits[begin / 64] & ~MASK(begin % 64);
  size_t ret = 0;
  int first_block_idx = begin / 64, last_block_idx = end / 64;
  for (int i = first_block_idx; i < last_block_idx; i++) {
    ret += __builtin_popcountll(block);
    block = bits[i + 1];
  }
  return ret + __builtin_popcountll(block & MASK(end % 64));
};


/**
 * SIMD-related variables & functions
 * naming convention: <what>_<element width>_<vector width>
 * there is NO alignment requirements
 */

const int16_t inc_16_128[8][8] = {
  {1,1,1,1,1,1,1,1},{0,1,1,1,1,1,1,1},
  {0,0,1,1,1,1,1,1},{0,0,0,1,1,1,1,1},
  {0,0,0,0,1,1,1,1},{0,0,0,0,0,1,1,1},
  {0,0,0,0,0,0,1,1},{0,0,0,0,0,0,0,1}
};

const int32_t inc_32_256[8][8] = {
  {1,1,1,1,1,1,1,1},{0,1,1,1,1,1,1,1},
  {0,0,1,1,1,1,1,1},{0,0,0,1,1,1,1,1},
  {0,0,0,0,1,1,1,1},{0,0,0,0,0,1,1,1},
  {0,0,0,0,0,0,1,1},{0,0,0,0,0,0,0,1}
};

// do an element-wise ge comparison, using SIMD instructions
inline auto cmpge_16_128(const int16_t *src, int16_t val) -> int {
  __m128i val_vec = _mm_set1_epi16(val - 1);
  __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  __m128i res_vec = _mm_cmpgt_epi16(src_vec, val_vec);
  auto mask = _mm_movemask_epi8(res_vec);
  return mask;
}

// do an element-wise ge comparison, using SIMD instructions
inline auto cmpge_32_256(const int32_t *src, int32_t val) -> int {
  __m256i val_vec = _mm256_set1_epi32(val - 1);
  __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  __m256i res_vec = _mm256_cmpgt_epi32(src_vec, val_vec);
  auto mask = _mm256_movemask_epi8(res_vec);
  return mask;
}

// add vector `src` to vector `dst`, assuming no overflows
inline void add_16_128(int16_t *dst, const int16_t *src) {
  __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst));
  __m128i res_vec = _mm_add_epi16(src_vec, dst_vec);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), res_vec);
}

// subtract vector `src` from vector `dst`, assuming no underflows
inline void sub_16_128(int16_t *dst, const int16_t *src) {
  __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst));
  __m128i res_vec = _mm_sub_epi16(dst_vec, src_vec);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), res_vec);
}

// add vector `src` to vector `dst`, assuming no overflows
inline void add_32_256(int32_t *dst, const int32_t *src) {
  __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
  __m256i res_vec = _mm256_add_epi32(src_vec, dst_vec);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), res_vec);
}

// subtract vector `src` from vector `dst`, assuming no underflows
inline void sub_32_256(int32_t *dst, const int32_t *src) {
  __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
  __m256i res_vec = _mm256_sub_epi32(dst_vec, src_vec);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), res_vec);
}

/**
 * bit copying
 */
// copy `len` bits starting from the `src_start`-th bit of `src` to the `dst_start`-th bit of `dst`
// returns the number of bits copied (which is always `len`)
auto copy_bits(uint64_t *dst, size_t dst_start, const uint64_t *src, size_t src_start, size_t len) -> size_t {
  int offset1 = dst_start % 64, offset2 = src_start % 64;
  int remainder1 = (dst_start + len) % 64, remainder2 = (src_start + len) % 64;
  int first_block1 = dst_start / 64, last_block1 = (dst_start + len) / 64;
  int first_block2 = src_start / 64, last_block2 = (src_start + len) / 64;

  if (offset1 == offset2) {
    uint64_t low_bits = (dst[first_block1] & MASK(offset1));
    for (int i = 0; i < last_block1 - first_block1; i++) {
      dst[first_block1 + i] = (low_bits | (src[first_block2 + i] & ~MASK(offset2)));
      low_bits = src[first_block2 + i + 1] & MASK(offset2);  // potential read out of bound
    }
    if (remainder1 > 0) {  // prevent write out of bound
      dst[last_block1] = (low_bits | (src[last_block2] & ~MASK(offset2))) & MASK(remainder1);
    }
  } else if (offset1 < offset2) {
    int rshift = offset2 - offset1;

    uint64_t low_bits = (dst[first_block1] & MASK(offset1)) | ((src[first_block2] >> offset2) << offset1);
    for (int i = 0; i < last_block1 - first_block1; i++) {
      dst[first_block1 + i] = (low_bits | (src[first_block2 + i + 1] << (64 - rshift)));
      low_bits = (src[first_block2 + i + 1] >> rshift);  // potential read out of bound
    }
    if (last_block2 - first_block2 == last_block1 - first_block1) {
      assert(remainder2 > 0);
      dst[last_block1] = (low_bits & MASK(remainder1));
    } else {  // one more source block
      assert(remainder1 > 0);
      assert(last_block2 - first_block2 == last_block1 - first_block1 + 1);
      if (remainder2 > 0) {  // prevent write out of bound
        dst[last_block1] = ((low_bits | (src[last_block2] << (64 - rshift))) & MASK(remainder1));
      } else {
        dst[last_block1] = (low_bits & MASK(remainder1));
      }
    }
  } else {  // offset1 > offset2
    int lshift = offset1 - offset2;

    uint64_t low_bits = (dst[first_block1] & MASK(offset1));
    for (int i = 0; i < last_block2 - first_block2; i++) {
      dst[first_block1 + i] = (low_bits | (((src[first_block2 + i] >> offset2) & MASK(64 - offset1)) << offset1));
      // potential read out of bound
      low_bits = ((src[first_block2 + i] >> (64 - lshift)) | ((src[first_block2 + i + 1] & MASK(offset2)) << lshift));  
    }
    if (last_block2 - first_block2 == last_block1 - first_block1) {
      assert(remainder1 > 0);
      dst[last_block1] = (low_bits | (((src[last_block2] >> offset2) & MASK(64 - offset1)) << offset1)) & MASK(remainder1);
    } else {  // one more destination block
      assert(remainder2 > 0);
      assert(last_block2 - first_block2 == last_block1 - first_block1 - 1);
      dst[last_block1 - 1] = (low_bits | (((src[last_block2] >> offset2) & MASK(64 - offset1)) << offset1));
      if (remainder1 > 0) {  // prevent write out of bound
        dst[last_block1] = (src[last_block2] >> (64 - lshift)) & MASK(remainder1);
      }
    }
  }

  return len;
}

// similar to `copy_bits`, except that bits are traversed and copied in reverse order
auto copy_bits_backward(uint64_t *dst, size_t dst_end, const uint64_t *src, size_t src_end, size_t len) -> size_t {
  assert(dst_end >= len);
  assert(src_end >= len);

  int offset1 = dst_end % 64, offset2 = src_end % 64;
  int remainder1 = (dst_end - len) % 64, remainder2 = (src_end - len) % 64;
  int last_block1 = dst_end / 64, first_block1 = (dst_end - len) / 64;
  int last_block2 = src_end / 64, first_block2 = (src_end - len) / 64;

  if (offset1 == offset2) {
    if (offset1 == 0) {  // prevent array out of bound
      uint64_t high_bits = src[last_block2 - 1];
      for (int i = 1; i < last_block1 - first_block1; i++) {  // TODO
        dst[last_block1 - i] = high_bits;
        high_bits = src[last_block2 - i - 1];
      }
      dst[first_block1] = high_bits & ~MASK(remainder1);
    } else {
      uint64_t high_bits = dst[last_block2] & ~MASK(offset1);
      for (int i = 0; i < last_block1 - first_block1; i++) {
        dst[last_block1 - i] = (high_bits | (src[last_block2 - i] & MASK(offset2)));
        high_bits = src[last_block2 - i - 1] & ~MASK(offset1);
      }
      dst[first_block1] = (high_bits | (src[first_block2] & MASK(offset2))) & ~MASK(remainder1);
    }
  } else if (offset1 < offset2) {
    int rshift = offset2 - offset1;

    if (offset1 == 0) {  // prevent array out of bound
      uint64_t high_bits = (src[last_block2] << (64 - rshift));
      for (int i = 1; i < last_block2 - first_block2; i++) {  // TODO
        dst[last_block1 - i] = (high_bits | (src[last_block2 - i] >> rshift));
        high_bits = ((src[last_block2 - i] << (64 - rshift)) | (src[last_block2 - i - 1] >> rshift));
      }
      if (last_block2 - first_block2 == last_block1 - first_block1) {
        dst[first_block1] = (high_bits | ((src[first_block2] & ~MASK(remainder2)) >> rshift));
      } else {  // one more destination block
        assert(last_block2 - first_block2 == last_block1 - first_block1 - 1);
        dst[first_block1 + 1] = (high_bits | (src[first_block2] >> rshift));
        dst[first_block1] = (src[first_block2] & ~MASK(remainder2)) << (64 - rshift);
      }
    } else {
      uint64_t high_bits = dst[last_block1] & ~MASK(offset1);
      for (int i = 0; i < last_block2 - first_block2; i++) {
        dst[last_block1 - i] = (high_bits | ((src[last_block2 - i] & MASK(offset2)) >> rshift));
        high_bits = ((src[last_block2 - i] << (64 - rshift)) | (src[last_block2 - i - 1] >> offset2) << offset1);
      }
      if (last_block2 - first_block2 == last_block1 - first_block1) {
        dst[first_block1] = (high_bits | ((src[first_block2] & MASK(offset2) & ~MASK(remainder2)) >> rshift));
      } else {  // one more destination block
        assert(last_block2 - first_block2 == last_block1 - first_block1 - 1);
        dst[first_block1 + 1] = (high_bits | ((src[first_block2] & MASK(offset2)) >> rshift));
        dst[first_block1] = (src[first_block2] & ~MASK(remainder2)) << (64 - rshift);
      }
    }
  } else {  // offset1 > offset2
    int lshift = offset1 - offset2;

    if (offset2 == 0) {  // prevent array out of bound
      uint64_t high_bits = dst[last_block1] & ~MASK(offset1);
      for (int i = 0; i < last_block1 - first_block1; i++) {
        dst[last_block1 - i] = (high_bits | (src[last_block2 - i - 1] >> (64 - lshift)));
        high_bits = src[last_block2 - i - 1] << lshift;
      }
      if (last_block2 - first_block2 == last_block1 - first_block1) {
        dst[first_block1] = high_bits & ~MASK(remainder1);
      } else {  // one more source block
        assert(last_block2 - first_block2 == last_block1 - first_block1 + 1);
        dst[first_block1] = (high_bits | (src[first_block2] >> (64 - lshift))) & ~MASK(remainder1);
      }
    } else {
      uint64_t high_bits = dst[last_block1] & ~MASK(offset1);
      for (int i = 0; i < last_block1 - first_block1; i++) {
        uint64_t low_bits = ((src[last_block2 - i] & MASK(offset2)) << lshift) | (src[last_block2 - i - 1] >> (64 - lshift));
        dst[last_block1 - i] = (high_bits | low_bits);
        high_bits = (src[last_block2 - i - 1] & ~MASK(offset2)) << lshift;
      }
      if (last_block2 - first_block2 == last_block1 - first_block1) {
        uint64_t low_bits = ((src[first_block2] & MASK(offset2)) << lshift);
        dst[first_block1] = (high_bits | low_bits) & ~MASK(remainder1);
      } else {  // one more source block
        assert(last_block2 - first_block2 == last_block1 - first_block1 + 1);
        uint64_t low_bits = ((src[first_block2 + 1] & MASK(offset2)) << lshift) | (src[first_block2] >> (64 - lshift));
        dst[first_block1] = (high_bits | low_bits) & ~MASK(remainder1);
      }
    }
  }

  return len;
}

// #define __DEBUG__

#ifdef __DEBUG__

void print_bits(const uint64_t *bits, size_t start, size_t len) {
  printf("bits: ");
  for (size_t i = 0; i < len; i++) {
    bool bit = GET_BIT(bits[(start + i) / 64], (start + i) % 64);
    printf("%d", bit);
    if (i % 8 == 7) {
      printf(",");
    }
  }
  printf("\n");
  fflush(stdout);
}

void print_hex(const uint64_t *bits, size_t len) {
  for (size_t i = 0; i < len; i++) {
    printf("%016lx,", bits[i]);
  }
  printf("\n");
  fflush(stdout);
}

#include <random>

class CopyBitsTest {
 public:
  static void test_all() {
    test_copy_equal();
    test_copy_rshift();
    test_copy_lshift();

    test_copy_backward_equal();
    test_copy_backward_rshift();
    test_copy_backward_lshift();
  }

  static void test_copy_equal() {  // src/dst offests are equal
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY EQUAL]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset = dist(gen) % 64;
      size_t len = dist(gen) % (max_num_blocks * 64 - offset + 1);
      // printf("offset: %ld, len: %ld\n", offset, len);
      copy_bits(dst, offset, src, offset, len);
      EXPECT(compare_bits(dst, offset, src, offset, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY EQUAL SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      src[0] = dist(gen);
      size_t offset = dist(gen) % 64;
      size_t len = dist(gen) % 64;
      // printf("offset: %ld, len: %ld\n", offset, len);
      copy_bits(dst, offset, src, offset, len);
      EXPECT(compare_bits(dst, offset, src, offset, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static void test_copy_rshift() {  // src offset less than dst offset
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY RSHIFT]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 >= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % (max_num_blocks * 64 - offset2 + 1);
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits(dst, offset1, src, offset2, len);
      EXPECT(compare_bits(dst, offset1, src, offset2, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY RSHIFT SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 >= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % 64;
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits(dst, offset1, src, offset2, len);
      EXPECT(compare_bits(dst, offset1, src, offset2, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static void test_copy_lshift() {  // src offset greater than dst offset
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY LSHIFT]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 <= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % (max_num_blocks * 64 - offset1 + 1);
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits(dst, offset1, src, offset2, len);
      if (!compare_bits(dst, offset1, src, offset2, len)) {
        print_bits(dst, offset1, len);
        print_bits(src, offset2, len);
      }
      EXPECT(compare_bits(dst, offset1, src, offset2, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY LSHIFT SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 <= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % 64;
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits(dst, offset1, src, offset2, len);
      EXPECT(compare_bits(dst, offset1, src, offset2, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static void test_copy_backward_equal() {  // src/dst offests are equal
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY BACKWARD EQUAL]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset = dist(gen) % 64;
      size_t len = dist(gen) % ((max_num_blocks - 1) * 64 + offset + 1);
      // printf("offset: %ld, len: %ld\n", offset, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset, src, (max_num_blocks - 1) * 64 + offset, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset - len, src, (max_num_blocks - 1) * 64 + offset - len, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY BACKWARD EQUAL SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      src[0] = dist(gen);
      size_t offset = dist(gen) % 64;
      size_t len = dist(gen) % 64;
      // printf("offset: %ld, len: %ld\n", offset, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset, src, (max_num_blocks - 1) * 64 + offset, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset - len, src, (max_num_blocks - 1) * 64 + offset - len, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static void test_copy_backward_rshift() {  // src offset less than dst offset
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY BACKWARD RSHIFT]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 >= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % ((max_num_blocks - 1) * 64 + offset1 + 1);
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset1, src, (max_num_blocks - 1) * 64 + offset2, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset1 - len, src, (max_num_blocks - 1) * 64 + offset2 - len, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY BACKWARD RSHIFT SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 >= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % 64;
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset1, src, (max_num_blocks - 1) * 64 + offset2, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset1 - len, src, (max_num_blocks - 1) * 64 + offset2 - len, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static void test_copy_backward_lshift() {  // src offset greater than dst offset
    const int num_passes = 100;
    const int max_num_blocks = 100;
    
    uint64_t src[max_num_blocks];
    uint64_t dst[max_num_blocks];

    std::random_device rd;

    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    printf("[TEST COPY BACKWARD LSHIFT]\n");
    for (int i = 0; i < num_passes; i++) {
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 <= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % ((max_num_blocks - 1) * 64 + offset2 + 1);
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset1, src, (max_num_blocks - 1) * 64 + offset2, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset1 - len, src, (max_num_blocks - 1) * 64 + offset2 - len, len), true, ==);
    }
    printf("[PASSED]\n");

    printf("[TEST COPY BACKWARD LSHIFT SMALL]\n");
    for (int i = 0; i < num_passes; i++) {  // special case: copy less than 64 bits
      printf("[PASS %d]\n", i);

      for (int i = 0; i < max_num_blocks; i++) {
        src[i] = dist(gen);
      }
      size_t offset1 = dist(gen) % 64, offset2 = dist(gen) % 64;
      while (offset1 <= offset2) {
        offset1 = dist(gen) % 64;
        offset2 = dist(gen) % 64;
      }
      size_t len = dist(gen) % 64;
      // printf("offset1: %ld, offset2: %ld, len: %ld\n", offset1, offset2, len);
      copy_bits_backward(dst, (max_num_blocks - 1) * 64 + offset1, src, (max_num_blocks - 1) * 64 + offset2, len);
      EXPECT(compare_bits(dst, (max_num_blocks - 1) * 64 + offset1 - len, src, (max_num_blocks - 1) * 64 + offset2 - len, len), true, ==);
    }
    printf("[PASSED]\n");
  }

  static auto compare_bits(const uint64_t *bits1, size_t start1, const uint64_t *bits2, size_t start2, size_t len) -> bool {
    for (size_t i = 0; i < len; i++) {
      bool bit1 = GET_BIT(bits1[(start1 + i) / 64], (start1 + i) % 64);
      bool bit2 = GET_BIT(bits2[(start2 + i) / 64], (start2 + i) % 64);
      if (bit1 != bit2) {
        printf("bit %ld not equal: %d, %d\n", i, bit1, bit2);
        return false;
      }
    }
    return true;
  }
};
#endif