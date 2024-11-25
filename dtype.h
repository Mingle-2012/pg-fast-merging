//
// Created by XiaoWu on 2024/11/23.
//

/**
 * This implementation is based on the following references:
 * See https://github.com/facebookresearch/faiss and https://github.com/JieFengWang/mini_rnn for more details.
 */

#ifndef MERGE_DTYPE_H
#define MERGE_DTYPE_H

#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <vector>
#include "metric.h"

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

namespace merge {
    template<typename T, unsigned A = ALIGNMENT>
    class Matrix {
        unsigned col{};
        unsigned row{};
        size_t stride{};
        char *data;

        void reset(unsigned r,
                   unsigned c) {
            row = r;
            col = c;
            stride = (sizeof(T) * c + A - 1) / A * A;
            if (data)
                free(data);
            data = (char *) memalign(A, row * stride);
        }

    public:
        Matrix() : col(0), row(0), stride(0), data(nullptr) {}

        Matrix(unsigned r,
               unsigned c) : data(nullptr) {
            reset(r, c);
        }

        Matrix(const Matrix &m) : col(m.col), row(m.row), stride(m.stride), data(0) {
            data = (char *) memalign(A, row * stride);
            memcpy(data, m.data, row * stride);
        }

        ~Matrix() {
            if (data)
                free(data);
        }

        unsigned size() const {
            return row;
        }

        unsigned dim() const {
            return col;
        }

        size_t step() const {
            return stride;
        }

        void resize(unsigned r,
                    unsigned c) {
            reset(r, c);
        }

        T *operator[](unsigned i) {
            return reinterpret_cast<T *>(&data[stride * i]);
        }

        T const *operator[](unsigned i) const {
            return reinterpret_cast<T const *>(&data[stride * i]);
        }

        T &operator()(unsigned i,
                      unsigned j) {
            return reinterpret_cast<T *>(&data[stride * i])[j];
        }

        Matrix &operator=(const Matrix &m) {
            if (this == &m) {
                return *this;
            }
            if (row * col != m.row * m.col) {
                delete[] data;
                data = (char *) memalign(A, m.row * m.stride);
            }
            memcpy(data, m.data, m.row * m.stride);
            row = m.row;
            col = m.col;
            stride = m.stride;
            return *this;
        }

        void zero() {
            memset(data, 0, row * stride);
        }

        void load(const std::string &path,
                  unsigned skip = 0,
                  unsigned gap = 4) {
            std::ifstream is(path.c_str(), std::ios::binary);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
            is.seekg(0, std::ios::beg);
            unsigned dim;
            is.read((char *) &dim, sizeof(unsigned int));
            std::cout << "Read Dimension: " << dim << std::endl;
            unsigned line = sizeof(T) * dim + gap;
            unsigned N = size / line;
            reset(N, dim);
            zero();
            is.seekg(skip, std::ios::beg);
            for (unsigned i = 0; i < N; ++i) {
                is.seekg(gap, std::ios::cur);
                is.read(&data[stride * i], sizeof(T) * dim);
            }
        }

        void append(const Matrix<T> &matrix) {
            size_t new_rows = row + matrix.getRows();
            size_t new_columns = col;
            size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
            char *new_data = (char *) memalign(32, new_rows * new_stride);
            memcpy(new_data, data, row * stride);
            memcpy(new_data + row * stride, matrix.data, matrix.getRows() * matrix.step());
            free(data);
            data = new_data;
            row = new_rows;
            col = new_columns;
            stride = new_stride;
        }

        void split(std::vector<Matrix<T> *> &matrices) {
            size_t new_rows = row / matrices.size();
            size_t new_columns = col;
            size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
            for (size_t i = 0; i < matrices.size(); ++i) {
                matrices[i] = new Matrix<T>(new_rows, new_columns);
                memcpy(matrices[i]->data, data + i * new_rows * new_stride, new_rows * new_stride);
            }
        }

        void halve(Matrix<T> &other) {
            size_t total = row;
            size_t new_rows = row / 2;
            row = new_rows;

            char *tmp = (char *) memalign(32, new_rows * stride);
            memcpy(tmp, data, new_rows * stride);
            char *new_data = (char *) memalign(32, (total - new_rows) * stride);
            memcpy(new_data, data + new_rows * stride, (total - new_rows) * stride);
            free(data);
            data = tmp;
            other.data = new_data;
            other.row = total - new_rows;
            other.col = col;
            other.stride = stride;
        }
    };

    template<class DATA_TYPE>
    void merge(const Matrix<DATA_TYPE> &m1,
               const Matrix<DATA_TYPE> &m2,
               Matrix<DATA_TYPE> &merged) {
        size_t r1 = m1.size();
        size_t r2 = m2.size();
        size_t c = m1.dim();
        if (c != m2.dim()) {
            throw std::runtime_error("Dimension mismatch");
        }

        if (&m1 == &merged) {
            Matrix<DATA_TYPE> temp(m1);
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

    template<typename DATA_TYPE, unsigned A = ALIGNMENT>
    class MatrixProxy {
        unsigned rows;
        unsigned cols;
        size_t stride;
        uint8_t const *data;

    public:
        explicit MatrixProxy(Matrix<DATA_TYPE> const &m)
                : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0])) {
        }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy(flann::Matrix<DATA_TYPE> const &m_)
            : rows(m_.rows), cols(m_.cols), stride(m_.stride), data(m_.data)
        {
            if (stride % A)
                throw invalid_argument("bad alignment");
        }
#endif
#ifdef CV_MAJOR_VERSION
        /// Construct from OpenCV matrix.
        MatrixProxy(cv::Mat const &m_)
            : rows(m_.rows), cols(m_.cols), stride(m_.step), data(m_.data)
        {
            if (stride % A)
                throw invalid_argument("bad alignment");
        }
#endif
#ifdef NPY_NDARRAYOBJECT_H
        /// Construct from NumPy matrix.
        MatrixProxy(PyArrayObject *obj)
        {
            if (!obj || (obj->nd != 2))
                throw invalid_argument("bad array shape");
            rows = obj->dimensions[0];
            cols = obj->dimensions[1];
            stride = obj->strides[0];
            data = reinterpret_cast<uint8_t const *>(obj->data);
            if (obj->descr->elsize != sizeof(DATA_TYPE))
                throw invalid_argument("bad data type size");
            if (stride % A)
                throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE)))
                throw invalid_argument("bad stride");
        }
#endif
#endif

        unsigned size() const {
            return rows;
        }

        unsigned dim() const {
            return cols;
        }

        DATA_TYPE const *operator[](unsigned i) const {
            return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
        }

        DATA_TYPE *operator[](unsigned i) {
            return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
        }
    };

    template<typename DATA_TYPE>
    class IndexOracle {
    public:
        virtual unsigned size() const = 0;

        virtual unsigned dim() const = 0;

        virtual float operator()(unsigned i,
                                 unsigned j) const = 0;

        virtual float operator()(unsigned,
                                 const DATA_TYPE *) const = 0;

        virtual DATA_TYPE *operator[](unsigned i) const = 0;
    };

    template<typename DATA_TYPE, typename DIST_TYPE>
    class MatrixOracle : public IndexOracle<DATA_TYPE> {
    public:
        MatrixProxy<DATA_TYPE> proxy;

        template<typename MATRIX_TYPE>
        explicit MatrixOracle(MATRIX_TYPE const &m) : proxy(m) {
        }

        virtual unsigned size() const {
            return proxy.size();
        }

        virtual unsigned dim() const {
            return proxy.dim();
        }

        virtual float operator()(unsigned i,
                                 unsigned j) const {
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
        }

        virtual float operator()(unsigned i,
                                 const DATA_TYPE *vec) const {
            return DIST_TYPE::apply((DATA_TYPE *) proxy[i], vec, proxy.dim());
        }

        virtual DATA_TYPE *operator[](const unsigned i) const {
            return const_cast<DATA_TYPE *>(proxy[i]);
        }
    };
}


#endif //MERGE_DTYPE_H
