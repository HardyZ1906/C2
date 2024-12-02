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


#include <sdsl/bit_vectors.hpp>
#include <sdsl/util.hpp>
#include <sux/bits/SimpleSelectZero.hpp>
#include <sux/bits/SimpleSelectZeroHalf.hpp>

#pragma once

// #define __BENCH_SUX__
#ifdef __BENCH_SUX__
# include <chrono>
# define BENCH_SUX(foo) foo
#else
# define BENCH_SUX(foo)
#endif


template<typename rank_support1 = sdsl::rank_support_v<1>,
        typename rank_support00 = sdsl::rank_support_v<0, 2>,
        typename select0_type = sux::bits::SimpleSelectZero<>>
struct louds_sux {
    sdsl::bit_vector bv;
    // pointer used while building
    size_t build_p = 0;
    rank_support1 rs1;
    rank_support00 rs00;
    select0_type ss0;

#ifdef __BENCH_SUX__
    static size_t get_time_;
    static size_t rank_time_;
    static size_t select_time_;

    static void print_microbenchmark() {
        printf("get time: %lf ms, rank time: %lf ms, select time: %lf ms\n", (double)get_time_/1000000,
               (double)rank_time_/1000000, (double)select_time_/1000000);
    }

    static void clear_microbenchmark() {
        get_time_ = rank_time_ = select_time_ = 0;
    }
#else
    static void print_microbenchmark() {}
    static void clear_microbenchmark() {}
#endif

    louds_sux() = default;

    louds_sux(size_t n) {
        // n is the number of nodes
        bv = sdsl::bit_vector(1 + 2 * n, 0);
        bv[0] = 1;
        bv[1] = 0;
        build_p = 2;
    }

    ~louds_sux() {}

    louds_sux(sdsl::bit_vector &_bv) {
        size_t _bv_size = _bv.size();
        bv.swap(_bv);
        assert(bv.size() == _bv_size);
        rs1 = rank_support1(&bv);
        rs00 = rank_support00(&bv);
        if constexpr (std::is_same_v<select0_type, sux::bits::SimpleSelectZero<>>)
            ss0 = select0_type(bv.data(), bv.size(), 2);
        else if constexpr (std::is_same_v<select0_type, sux::bits::SimpleSelectZeroHalf<>>)
            ss0 = select0_type(bv.data(), bv.size());
        else
            ss0 = select0_type(&bv);
    }

    // index of the root
    static const size_t root_idx = 2;

    void add_node(size_t num_child) {
        for (auto i = 0; i < num_child; ++i) {
            bv[build_p++] = 1;
        }
        bv[build_p++] = 0;
    }

    void build_rank_select_ds() {
        // init rank select support
        rs1 = rank_support1(&bv);
        rs00 = rank_support00(&bv);
        if constexpr (std::is_same_v<select0_type, sux::bits::SimpleSelectZero<>>)
            ss0 = select0_type(bv.data(), bv.size(), 2);
        else if constexpr (std::is_same_v<select0_type, sux::bits::SimpleSelectZeroHalf<>>)
            ss0 = select0_type(bv.data(), bv.size());
        else
            ss0 = select0_type(&bv);
        assert(bv.size() == this->build_p);
    }

    // from position in the bv to node index (0-based)
    size_t node_rank(size_t v) const {
        // it is just rank_0(v-1)
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )
        size_t ret = (v - 1) - rs1.rank(v - 1);
        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            rank_time_ += (end - start).count();
        )
        return ret;
    }

    // from index of nodes to position in the bv
    size_t node_select(size_t i) {
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )
        size_t ret;
        if constexpr (std::is_same_v<select0_type, sdsl::select_support_mcl<0>>)
            ret = ss0.select(i) + 1;
        else
            ret = ss0.selectZero(i - 1) + 1;
        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            select_time_ += (end - start).count();
        )
        return ret;
    }

    size_t size_in_bytes() const {
        size_t to_return = 0;
        if constexpr (std::is_same_v<select0_type, sdsl::select_support_mcl<0>>)
            to_return += sdsl::size_in_bytes(ss0);
        else
            to_return += ss0.bitCount() / CHAR_BIT;
        to_return += sdsl::size_in_bytes(rs1);
        to_return += sdsl::size_in_bytes(rs00);
        to_return += sdsl::size_in_bytes(bv);
        to_return += sizeof(louds_sux);
        return to_return;
    }


    inline size_t size_in_bits() const {
        size_t to_return = size_in_bytes();
        return to_return * CHAR_BIT;
    }

    // takes position in the bv returns number of children
    size_t num_child(size_t v) const {
        size_t next0 = nextZero(v);
        assert(next0 >= v);
        return next0 - v;
    }

    // takes position in the bv and an integer > 0 and returns the index of the nth child
    size_t n_th_child_rank(size_t v, size_t n) {
        assert(v < bv.size());
        assert((n - 1) < num_child(v));
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )
        size_t ret = rs1.rank(v - 1 + n);
        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            rank_time_ += (end - start).count();
        )
        return ret;
    }

    inline bool is_leaf(size_t v) {
        assert(bv[v - 1] == 0);
        assert(v < bv.size());
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )
        bool ret = (bv[v] == 0);
        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            get_time_ += (end - start).count();
        )
        return ret;
    }

    // from position in the bv to internal node index
    size_t internal_rank(size_t v, size_t node_rank) {
        assert(v < bv.size());
        assert(node_rank >= rs00.rank(v));
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )
        size_t ret = node_rank - rs00.rank(v);
        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            rank_time_ += (end - start).count();
        )
        return ret;
    }

    // from sux::bits::SimpleSelectZeroHalf::selectZero(const uint64_t rank, uint64_t *const next)
    size_t nextZero(const uint64_t bv_idx) const {
        BENCH_SUX( auto start = std::chrono::high_resolution_clock::now(); )

        uint64_t curr = (bv_idx - 1) / 64;

        uint64_t window = ~bv.data()[curr] & -1ULL << (bv_idx - 1);
        window &= window - 1;

        while (window == 0) window = ~bv.data()[++curr];
        size_t ret = curr * 64 + __builtin_ctzll(window);

        BENCH_SUX(
            auto end = std::chrono::high_resolution_clock::now();
            get_time_ += (end - start).count();
        )
        return ret;
    }

};

#ifdef __BENCH_SUX__

#define LOUDS_SUX_TMPL_ARGS template<typename rank_support1, typename rank_support00, typename select0_type>
#define LOUDS_SUX_TMPL louds_sux<rank_support1, rank_support00, select0_type>

LOUDS_SUX_TMPL_ARGS size_t LOUDS_SUX_TMPL::get_time_ = 0;
LOUDS_SUX_TMPL_ARGS size_t LOUDS_SUX_TMPL::rank_time_ = 0;
LOUDS_SUX_TMPL_ARGS size_t LOUDS_SUX_TMPL::select_time_ = 0;

#undef LOUDS_SUX_TMPL_ARGS
#undef LOUDS_SUX_TMPL

#endif

#undef __BENCH_SUX__
#undef BENCH_SUX