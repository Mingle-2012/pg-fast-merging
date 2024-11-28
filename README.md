# Fast Graph-based Indexes Merging for Approximate Nearest Neighbor Search

## Contents:

- [Introduction](#introduction)
- [Usage](#usage)
  - [Environment](#environment)
  - [Running the code](#running-the-code)
  - [Extensibility](#extensibility)
- [Performance](#performance)
  - [Datasets](#datasets)
  - [Compared methods](#compared-methods)
  - [Results](#results)
- [License](#license)

## Introduction

This repository contains the source code for the paper "Fast Graph-based Indexes Merging for Approximate Nearest Neighbor Search". In the paper, we propose a framework for merging multiple indexes into a single index that can be used for ANNS.

## Usage

### Environment

- GCC 11.4.0
- CMake 3.29
- OpenMP 4.5

### Running the code

Compile On Ubuntu 22.04:

```bash
$ sudo apt-get install g++ cmake libomp-dev
$ git clone https://github.com/Mingle-2012/pg-fast-merging.git
$ cd Merge
$ mkdir build && cd build
$ cmake ..
$ make -j
```

### Extensibility

## Performance

### Datasets

### Compared methods

### Results

## License

All source code is made available under a MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.
