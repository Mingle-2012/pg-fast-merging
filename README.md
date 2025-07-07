# Fast Graph-based Indexes Merging for Approximate Nearest Neighbor Search

## Contents:

- [Introduction](#introduction)
- [Usage](#usage)
  - [Environment](#environment)
  - [Running the code](#run-the-code)
  - [Extensibility](#extensibility)
- [Performance](#performance)
  - [Datasets](#datasets)
- [License](#license)

## Introduction

This repository contains the source code for the paper "Fast Graph-based Indexes Merging for Approximate Nearest Neighbor Search". In the paper, we propose a framework for merging multiple indexes into a single index that can be used for effective ANNS.

## Usage

### Environment

- GCC 11.4.0
- CMake 3.29
- OpenMP 4.5
- Ubuntu 22.04
- C++17

### Run the code

Change the dataset path and the metric in C++ files in the `tests` directory if you want to use your own dataset.


Currently, `l2` and `cosine` metrics are supported. The dataset should be in `fvecs` (also `ivecs`, which is used by the Sift1M dataset) format, or `fbin` (also `ibin` that is used by DiskANN) format.


If you don't have the dataset, you can download the SIFT1M dataset with the following command:

```bash
python3 dataset.py
```

Then, compile the code with the following commands:

```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```

We provide three test files:

- `test_index_construction.cpp` - test the construction of HNSW and Vamana indexes.
- `test_merge_from_raw.cpp` - test the merging of the mainstream graph-based indexes. (2 sub-graphs)
- `test_merge_two_pg_indexes.cpp` - test the merging of given graphs. (2 sub-graphs)

Here are the examples of running the test files:

```bash
cd tests
$ ./test_index_construction  <base path> <metric> <query path> <groundtruth path> <topk> <algorithm> <algorithm parameters>
$ ./test_merge_from_raw <base path> <metric> <query path> <groundtruth path> <M>
$ ./test_merge_two_pg_indexes <graph 1> <base path 1> <graph 2> <bath path 2> <M> <metric> <output>
```

> The graph structure is formatted as follows:
> 
> [number of vertices]
> 
> [number of neighbors of vertex 1] [neighbor 1] [neighbor 2]...
> 
> [number of neighbors of vertex 2] [neighbor 1] [neighbor 2]...
> 
> ...

### Parameters

Merging parameters:

- `M` - maximum number of neighbors for each vertex.

Search parameters:

- `K` - the number of nearest neighbors to search for.
- `L` - the search pool size.

### Extensibility

Other pruning algorithms can be added to the `src/fgim.cpp` or `src/mgraph.cpp` files.

## Performance

### Datasets

| Dataset         | #Points   | #Dimensions | #Queries |
|-----------------|-----------|-------------|----------|
| SIFT1M          | 1,000,000 | 128         | 10,000   |
| GIST1M          | 1,000,000 | 960         | 1,000    |
| DEEP1M          | 1,000,000 | 96          | 10,000   |
| MSong           | 994,185   | 420         | 1,000    |
| GloVe           | 1,183,514 | 100         | 10,000   |
| Crawl           | 1,989,995 | 300         | 10,000   |
| Internet Search | 9,991,307 | 768         | 172      |

- For the SIFT1M and GIST1M datasets, you can download them from the [TEXMEX](http://corpus-texmex.irisa.fr/) website.
- Thanks to [Yusuke Matsui](https://github.com/matsui528), you can download the DEEP1M dataset from his [repository](https://github.com/matsui528/deep1b_gt).
- For the MSong dataset, you can download it from the [CUHK-GQR](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html)
- Thanks to the authors of [SSG](https://github.com/ZJULearning/SSG), you can download the Crawl, GloVe datasets from their repository.

## License

All source code is made available under a MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.
