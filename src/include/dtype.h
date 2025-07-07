//
// Created by XiaoWu on 2024/11/23.
//

/**
 * This implementation is based on the following references:
 * See https://github.com/facebookresearch/faiss and
 * https://github.com/JieFengWang/mini_rnn for more details.
 */

#ifndef MYANNS_DTYPE_H
#define MYANNS_DTYPE_H

#include <malloc.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "logger.h"

#ifdef __GNUC__
#ifdef __AVX__
#define ALIGNMENT 32
#else
#ifdef __SSE2__
#define ALIGNMENT 16
#else
#define ALIGNMENT 4
#endif
#endif
#endif

#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

namespace graph {
template <typename T>
class Matrix {
    unsigned col{};
    unsigned row{};
    size_t stride{};

    std::shared_ptr<char> data;

    size_t offset_bytes{};

    void
    reset(const unsigned r, const unsigned c) {
        row = r;
        col = c;
        stride = (sizeof(T) * c + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        auto new_data_ptr = static_cast<char*>(memalign(ALIGNMENT, row * stride));
        if (!new_data_ptr) {
            throw std::bad_alloc();
        }
        if (data) {
            data.reset();
        }
        data = std::shared_ptr<char>(new_data_ptr, [](char* p) { free(p); });
    }

    void
    load_vecs_data(std::ifstream& is, const unsigned int skip, const unsigned int gap) {
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        size -= skip;
        is.seekg(0, std::ios::beg);
        unsigned dim;
        is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
        unsigned line = sizeof(T) * dim + gap;
        unsigned N = size / line;
        logger << "Vector size: " << N << std::endl;
        logger << "Vector dimension: " << dim << std::endl;
        reset(N, dim);
        zero();
        is.seekg(skip, std::ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            is.seekg(gap, std::ios::cur);
            is.read(data.get() + stride * i, sizeof(T) * dim);
        }
    }

    void
    load_bin_data(std::ifstream& is) {
        unsigned size, dim;
        is.read(reinterpret_cast<char*>(&size), sizeof(unsigned int));
        is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
        logger << "Vector size: " << size << std::endl;
        logger << "Vector dimension: " << dim << std::endl;
        reset(size, dim);
        zero();
        for (unsigned i = 0; i < size; ++i) {
            is.read(data.get() + stride * i, sizeof(T) * dim);
        }
    }

    void
    load_hdf5_data(std::ifstream& is) {
        throw std::runtime_error("HDF5 loading is not implemented.");
    }

public:
    Matrix() : data(nullptr) {
    }

    Matrix(const unsigned r, const unsigned c) {
        data = nullptr;
        reset(r, c);
    }

    Matrix(const Matrix& m) {
        col = m.col, row = m.row, stride = m.stride, data = m.data;
    }

    Matrix(const Matrix& original_matrix, unsigned start_row, unsigned num_rows)
        : col(original_matrix.col),
          row(num_rows),
          stride(original_matrix.stride),
          data(original_matrix.data) {
        offset_bytes = original_matrix.offset_bytes + start_row * original_matrix.stride;
        if (start_row + num_rows > original_matrix.row) {
            throw std::out_of_range("Sub-matrix dimensions out of bounds of original matrix.");
        }
    }

    explicit Matrix(std::vector<std::shared_ptr<Matrix> >& matrices) {
        auto& base = matrices[0];
        col = base->col;
        stride = base->stride;
        row = 0;
        for (const auto& matrix : matrices) {
            row += matrix->row;
        }
        data = std::shared_ptr<char>(static_cast<char*>(memalign(ALIGNMENT, row * stride)),
                                     [](char* p) { free(p); });
        if (data == nullptr) {
            throw std::runtime_error("Cannot allocate memory for matrix.");
        }
        size_t offset = 0;
        for (const auto& matrix : matrices) {
            memcpy(data.get() + offset, matrix->data.get(), matrix->row * stride);
            offset += matrix->row * stride;
        }
    }

    ~Matrix() {
        if (data)
            data.reset();
    }

    [[nodiscard]] bool
    empty() const {
        return data == nullptr;
    }

    [[nodiscard]] unsigned
    size() const {
        return row;
    }

    [[nodiscard]] unsigned
    dim() const {
        return col;
    }

    [[nodiscard]] size_t
    step() const {
        return stride;
    }

    [[nodiscard]] size_t
    offset() const {
        return offset_bytes;
    }

    void
    resize(unsigned r, unsigned c) {
        reset(r, c);
    }

    bool
    belong(const Matrix& m) const {
        return data.get() == m.data.get();
    }

    T* operator[](unsigned i) {
        return reinterpret_cast<T*>(data.get() + stride * i + offset_bytes);
    }

    T const* operator[](unsigned i) const {
        return reinterpret_cast<T const*>(data.get() + stride * i + offset_bytes);
    }

    T&
    operator()(unsigned i, unsigned j) {
        return reinterpret_cast<T*>(data.get() + stride * i + offset_bytes)[j];
    }

    Matrix&
    operator=(const Matrix& m) {
        if (this == &m) {
            return *this;
        }
        if (row * col != m.row * m.col) {
            if (data) {
                data.reset();
            }
            data = std::shared_ptr<char>(static_cast<char*>(memalign(ALIGNMENT, m.row * m.stride)),
                                         [](char* p) { free(p); });
        }
        memcpy(data.get(), m.data.get(), m.row * m.stride);
        row = m.row;
        col = m.col;
        stride = m.stride;
        return *this;
    }

    void
    zero() const {
        memset(data.get(), 0, row * stride);
    }

    void
    load(const std::string& path) {
        logger << "Loading data from " << path << std::endl;
        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is) {
            throw std::runtime_error("Cannot open file " + path);
        }

        if (path.find(".fbin") != std::string::npos || path.find(".ibin") != std::string::npos) {
            load_bin_data(is);
        } else if (path.find(".fvecs") != std::string::npos ||
                   path.find(".ivecs") != std::string::npos) {
            load_vecs_data(is, 0, 4);
        } else if (path.find(".hdf5") != std::string::npos) {
            load_hdf5_data(is);
        } else {
            throw std::runtime_error("Unsupported file format: " + path);
        }
    }

    /**
     * Append a matrix to the current matrix.
     * @param matrix
     */
    void
    append(const Matrix& matrix) {
        size_t new_rows = row + matrix.row;
        size_t new_columns = col;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        auto new_data = std::make_shared<char>(
            static_cast<char*>(memalign(32, new_rows * new_stride)), [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row * stride);
        memcpy(new_data.get() + row * stride, matrix.data.get(), matrix.row * matrix.step());
        data.reset();
        data = new_data;
        row = new_rows;
        col = new_columns;
        stride = new_stride;
    }

    /**
     * Append a list of matrices to the current matrix.
     * @param matrices std::vector<Matrix>
     */
    void
    append(const std::vector<Matrix>& matrices) {
        size_t new_rows = row;
        for (const auto& matrix : matrices) {
            new_rows += matrix.row;
        }
        auto new_data = std::make_shared<char>(static_cast<char*>(memalign(32, new_rows * stride)),
                                               [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row * stride);
        size_t offset = row * stride;
        for (const auto& matrix : matrices) {
            memcpy(new_data.get() + offset, matrix.data.get(), matrix.row * matrix.stride);
            offset += matrix.row * matrix.stride;
        }
        data.reset();
        data = new_data;
        row = new_rows;
    }

    /**
     * Append a list of matrices to the current matrix.
     * @param matrices std::vector<std::shared_ptr<Matrix>>
     */
    void
    append(const std::vector<std::shared_ptr<Matrix> >& matrices) {
        size_t new_rows = row;
        for (const auto& matrix : matrices) {
            new_rows += matrix->row;
        }
        auto new_data = std::shared_ptr<char>(static_cast<char*>(memalign(32, new_rows * stride)),
                                              [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row * stride);
        size_t offset = row * stride;
        for (const auto& matrix : matrices) {
            memcpy(new_data.get() + offset, matrix->data.get(), matrix->row * matrix->stride);
            offset += matrix->row * matrix->step();
        }
        data.reset();
        data = new_data;
        row = new_rows;
    }

    /**
     * Split the matrix into num parts. Note that the original matrix will be resized but not freed.
     * @param num The number of parts to split.
     * @return
     */
    std::vector<Matrix>
    split(const size_t num) {
        size_t new_rows = row / num;
        size_t remaining = row % num;
        size_t new_columns = col;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        std::vector<Matrix> matrices;
        for (size_t i = 1; i < num - 1; ++i) {
            Matrix part(new_rows, new_columns);
            memcpy(part.data.get(), data.get() + i * new_rows * new_stride, new_rows * new_stride);
            matrices.push_back(std::move(part));
        }

        Matrix last_part(new_rows + remaining, new_columns);
        memcpy(last_part.data.get(),
               data.get() + (num - 1) * new_rows * new_stride,
               (new_rows + remaining) * new_stride);
        matrices.push_back(std::move(last_part));

        row = new_rows;
        return matrices;
    }

    void
    halve(Matrix& other) {
        size_t total = row;
        size_t new_rows = row / 2;
        row = new_rows;
        auto tmp = std::shared_ptr<char>(static_cast<char*>(memalign(32, new_rows * stride)),
                                         [](char* p) { free(p); });
        if (!tmp) {
            throw std::bad_alloc();
        }
        memcpy(tmp.get(), data.get(), new_rows * stride);
        auto new_data =
            std::shared_ptr<char>(static_cast<char*>(memalign(32, (total - new_rows) * stride)),
                                  [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get() + new_rows * stride, (total - new_rows) * stride);
        data.reset();
        data = tmp;
        other.data = new_data;
        other.row = total - new_rows;
        other.col = col;
        other.stride = stride;
    }
};

template <typename T>
using MatrixPtr = std::shared_ptr<Matrix<T> >;

template <typename T>
void
mergeMatrix(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& merged) {
    size_t r1 = m1.size();
    size_t r2 = m2.size();
    size_t c = m1.dim();
    if (c != m2.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }

    if (&m1 == &merged) {
        Matrix<T> temp(m1);
        merged.resize(r1 + r2, c);
        for (size_t i = 0; i < r1; ++i) {
            std::copy(temp[i], temp[i] + c, merged[i]);
        }
    } else {
        merged.resize(r1 + r2, c);
        for (size_t i = 0; i < r1; ++i) {
            std::copy(m1[i], m1[i] + c, merged[i]);
        }
    }
    for (size_t i = 0; i < r2; ++i) {
        std::copy(m2[i], m2[i] + c, merged[i + r1]);
    }
}

template <typename T>
class MatrixProxy {
    unsigned rows{0};
    unsigned cols{0};
    size_t stride{0};
    uint8_t const* data{nullptr};

public:
    explicit MatrixProxy(Matrix<T> const& m) {
        reset(m);
    }

    void
    reset(Matrix<T> const& m) {
        rows = m.size();
        cols = m.dim();
        stride = m.step();
        data = reinterpret_cast<uint8_t const*>(m[0]);
    }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
    /// Construct from FLANN matrix.
    MatrixProxy(flann::Matrix<float> const& m_)
        : rows(m_.rows), cols(m_.cols), stride(m_.stride), data(m_.data) {
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
    }
#endif
#ifdef CV_MAJOR_VERSION
    /// Construct from OpenCV matrix.
    MatrixProxy(cv::Mat const& m_) : rows(m_.rows), cols(m_.cols), stride(m_.step), data(m_.data) {
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
    }
#endif
#ifdef NPY_NDARRAYOBJECT_H
    /// Construct from NumPy matrix.
    MatrixProxy(PyArrayObject* obj) {
        if (!obj || (obj->nd != 2))
            throw invalid_argument("bad array shape");
        rows = obj->dimensions[0];
        cols = obj->dimensions[1];
        stride = obj->strides[0];
        data = reinterpret_cast<uint8_t const*>(obj->data);
        if (obj->descr->elsize != sizeof(float))
            throw invalid_argument("bad data type size");
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
        if (!(stride >= cols * sizeof(float)))
            throw invalid_argument("bad stride");
    }
#endif
#endif

    [[nodiscard]] unsigned
    size() const {
        return rows;
    }

    [[nodiscard]] unsigned
    dim() const {
        return cols;
    }

    T const* operator[](const unsigned i) const {
#ifdef USE_SSE
        _mm_prefetch(data + stride * i, _MM_HINT_T0);
#endif
        return reinterpret_cast<T const*>(data + stride * i);
    }

    T* operator[](unsigned i) {
#ifdef USE_SSE
        _mm_prefetch(data + stride * i, _MM_HINT_T0);
#endif
        return const_cast<float*>(reinterpret_cast<T const*>(data + stride * i));
    }
};

template <typename T>
class IndexOracle {
public:
    [[nodiscard]] virtual unsigned
    size() const = 0;

    [[nodiscard]] virtual unsigned
    dim() const = 0;

    virtual void
    reset(const Matrix<T>& m) = 0;

    virtual T
    operator()(unsigned i, unsigned j) const = 0;

    virtual T
    operator()(unsigned, const T*) const = 0;

    virtual T
    operator()(const T*, const T*) const = 0;

    virtual T* operator[](unsigned i) const = 0;

    virtual ~IndexOracle() = default;
};

using OraclePtr = std::shared_ptr<IndexOracle<float> >;

template <typename T, typename DIST_TYPE>
class MatrixOracle : public IndexOracle<T> {
public:
    MatrixProxy<T> proxy;

    explicit MatrixOracle(const Matrix<T>& m) : proxy(m) {
    }

    void
    reset(const Matrix<T>& m) override {
        proxy.reset(m);
    }

    [[nodiscard]] unsigned
    size() const override {
        return proxy.size();
    }

    [[nodiscard]] unsigned
    dim() const override {
        return proxy.dim();
    }

    T
    operator()(unsigned i, unsigned j) const override {
        return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
    }

    T
    operator()(unsigned i, const T* vec) const override {
        return DIST_TYPE::apply((T*)proxy[i], vec, proxy.dim());
    }

    T
    operator()(const T* vec1, const T* vec2) const override {
        return DIST_TYPE::apply(vec1, vec2, proxy.dim());
    }

    T* operator[](unsigned i) const override {
        return const_cast<T*>(proxy[i]);
    }

    static std::shared_ptr<IndexOracle<T> >
    getInstance(const Matrix<T>& m) {
        return std::make_shared<MatrixOracle<T, DIST_TYPE> >(m);
    }
};
}  // namespace graph

#endif  // MYANNS_DTYPE_H
