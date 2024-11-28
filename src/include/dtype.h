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
    class Matrix {
        unsigned col{};
        unsigned row{};
        size_t stride{};
        char *data;

        void reset(unsigned r,
                   unsigned c);

    public:
        Matrix() : col(0), row(0), stride(0), data(nullptr) {}

        Matrix(unsigned r,
               unsigned c);

        Matrix(const Matrix &m);

        ~Matrix();

        unsigned size() const;

        unsigned dim() const;

        size_t step() const;

        void resize(unsigned r,
                    unsigned c);

        float *operator[](unsigned i);

        float const *operator[](unsigned i) const;

        float &operator()(unsigned i,
                      unsigned j);

        Matrix &operator=(const Matrix &m);

        void zero();

        void load(const std::string &path,
                  unsigned int skip = 0,
                  unsigned int gap = 4);

        void append(const Matrix &matrix);

        void split(std::vector<Matrix *> &matrices);

        void halve(Matrix &other);
    };

    void mergeMatrix(const Matrix &m1,
                            const Matrix &m2,
                            Matrix &merged);
    
    class MatrixProxy {
        unsigned rows;
        unsigned cols;
        size_t stride;
        uint8_t const *data;

    public:
        explicit MatrixProxy(Matrix const &m);

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy(flann::Matrix<float> const &m_)
            : rows(m_.rows), cols(m_.cols), stride(m_.stride), data(m_.data)
        {
            if (stride % ALIGNMENT)
                throw invalid_argument("bad alignment");
        }
#endif
#ifdef CV_MAJOR_VERSION
        /// Construct from OpenCV matrix.
        MatrixProxy(cv::Mat const &m_)
            : rows(m_.rows), cols(m_.cols), stride(m_.step), data(m_.data)
        {
            if (stride % ALIGNMENT)
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
            if (obj->descr->elsize != sizeof(float))
                throw invalid_argument("bad data type size");
            if (stride % ALIGNMENT)
                throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(float)))
                throw invalid_argument("bad stride");
        }
#endif
#endif

        unsigned size() const;

        unsigned dim() const;

        float const *operator[](unsigned i) const;

        float *operator[](unsigned i) ;
    };

    class IndexOracle {
    public:
        virtual unsigned size() const = 0;

        virtual unsigned dim() const = 0;

        virtual float operator()(unsigned i,
                                 unsigned j) const = 0;

        virtual float operator()(unsigned,
                                 const float *) const = 0;

        virtual float *operator[](unsigned i) const = 0;
    };

    template<typename DIST_TYPE>
    class MatrixOracle : public IndexOracle {
    public:
        MatrixProxy proxy;

        explicit MatrixOracle(const Matrix &m) : proxy(m) {}

        virtual unsigned size() const{
            return proxy.size();
        }

        virtual unsigned dim() const{
            return proxy.dim();
        }

        virtual float operator()(unsigned i,
                                 unsigned j) const{
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
        }

        virtual float operator()(unsigned i,
                                 const float *vec) const{
            return DIST_TYPE::apply((float *) proxy[i], vec, proxy.dim());
        }


        virtual float *operator[](unsigned i) const{
            return const_cast<float *>(proxy[i]);
        }

    };
}

#endif //MERGE_DTYPE_H
