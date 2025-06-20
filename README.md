# $C^2$

## Intro

Re-implementation of three state-of-the-art succinct tries (FST, CoCo-trie and Marisa) with cache-conscious bitvector redesign and adaptive unary path compression.

## Building the Project

In the project directory, run:
```
git submodule update --init --recursive
bash lib/adapted_code/move.sh
bash baseline_marisa/build_marisa.sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Running the Project

The binary executable `benchmark` in the `build` directory evaluates the chosen tries by running pre-defined benchmarks:
```
./benchmark [trie] [space_relaxation] [max_recursion] [tail_mask]
```

`trie`: 0 - $C^2$-FST, 1 - $C^2$-CoCo, 2 - $C^2$-Marisa, 3 - FST, 4 - CoCo (with 5% space relaxation), 5 - Marisa, 6 - original PDT (with Re-pair string pool), 7 - ART, 8 - C-ART, 9 - $C^2$-CoCo (using LOUDS encoding), 10 - CoCo' (obtained by replacing $C^2$-CoCo's topology with CoCo's), 11 - topology performance comparison between $C^2$-CoCo and CoCo', 12 - topology performance comparison between $C^2$-Marisa and Marisa. Default value is 0. Since CoCo is likely to fail to build on large datasets, we use CoCo' as the baseline on such datasets instead.

`space_relaxation`: space relaxation parameter for $C^2$-CoCo and CoCo'. Ignored by other data structures (including CoCo-trie, which implements this parameter as a template argument). A larger value results in worse space efficiency but may improve query performance. Default value is 0.

`max_recursion`: the maximum level of recursion for $C^2$ tries as well as Marisa. A larger value may improve space efficiency at the cost of degraded query performance. Ignored by other data structures. Default value is 0.

`tail_mask`: the choice of the tail container. `tail_mask` >= 4 || `tail_mask` == 0: FSST; 2 <= `tail_mask` < 4: Re-pair; `tail_mask` == 1: Sorted. Default choice is FSST.