#include "dtype.h"
#include "logger.h"

using namespace merge;

void Matrix::reset(unsigned int r,
                         unsigned int c) {
    row = r;
    col = c;
    stride = (sizeof(float) * c + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    if (data)
        free(data);
    data = (char *) memalign(ALIGNMENT, row * stride);
}

Matrix::Matrix(unsigned int r,
                     unsigned int c) {
    data = nullptr;
    reset(r, c);
}

Matrix::Matrix(const Matrix &m) {
    col = m.col, row = m.row, stride = m.stride, data = nullptr;
    data = (char *) memalign(ALIGNMENT, row * stride);
    memcpy(data, m.data, row * stride);
}

Matrix::~Matrix() {
    if (data)
        free(data);
}

unsigned Matrix::size() const {
    return row;
}

unsigned Matrix::dim() const {
    return col;
}

size_t Matrix::step() const {
    return stride;
}

void Matrix::resize(unsigned int r,
                          unsigned int c) {
    reset(r, c);
}

float *Matrix::operator[](unsigned int i) {
    return reinterpret_cast<float *>(&data[stride * i]);
}


float const *Matrix::operator[](unsigned int i) const {
    return reinterpret_cast<float const *>(&data[stride * i]);
}


float &Matrix::operator()(unsigned int i,
                            unsigned int j) {
    return reinterpret_cast<float *>(&data[stride * i])[j];
}


Matrix &Matrix::operator=(const Matrix &m) {
    if (this == &m) {
        return *this;
    }
    if (row * col != m.row * m.col) {
        delete[] data;
        data = (char *) memalign(ALIGNMENT, m.row * m.stride);
    }
    memcpy(data, m.data, m.row * m.stride);
    row = m.row;
    col = m.col;
    stride = m.stride;
    return *this;
}


void Matrix::zero() {
    memset(data, 0, row * stride);
}


void Matrix::load(const std::string &path,
                        unsigned int skip,
                        unsigned int gap) {
    logger << "Loading data from " << path << std::endl;
    std::ifstream is(path.c_str(), std::ios::binary);
    if (!is) {
        throw std::runtime_error("Cannot open file " + path);
    }
    is.seekg(0, std::ios::end);
    size_t size = is.tellg();
    size -= skip;
    is.seekg(0, std::ios::beg);
    unsigned dim;
    is.read((char *) &dim, sizeof(unsigned int));
    logger << "Vector dimension: " << dim << std::endl;
    unsigned line = sizeof(float) * dim + gap;
    unsigned N = size / line;
    reset(N, dim);
    zero();
    is.seekg(skip, std::ios::beg);
    for (unsigned i = 0; i < N; ++i) {
        is.seekg(gap, std::ios::cur);
        is.read(&data[stride * i], sizeof(float) * dim);
    }
}


void Matrix::append(const Matrix &matrix) {
    size_t new_rows = row + matrix.row;
    size_t new_columns = col;
    size_t new_stride = (sizeof(float) * new_columns + 31) / 32 * 32;
    char *new_data = (char *) memalign(32, new_rows * new_stride);
    memcpy(new_data, data, row * stride);
    memcpy(new_data + row * stride, matrix.data, matrix.row * matrix.step());
    free(data);
    data = new_data;
    row = new_rows;
    col = new_columns;
    stride = new_stride;
}


void Matrix::split(std::vector<Matrix *> &matrices){
    size_t new_rows = row / matrices.size();
    size_t new_columns = col;
    size_t new_stride = (sizeof(float) * new_columns + 31) / 32 * 32;
    for (size_t i = 0; i < matrices.size(); ++i) {
        matrices[i] = new Matrix(new_rows, new_columns);
        memcpy(matrices[i]->data, data + i * new_rows * new_stride, new_rows * new_stride);
    }
}


void Matrix::halve(Matrix &other) {
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

void merge::mergeMatrix(const Matrix &m1,
           const Matrix &m2,
           Matrix &merged) {
    size_t r1 = m1.size();
    size_t r2 = m2.size();
    size_t c = m1.dim();
    if (c != m2.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }

    if (&m1 == &merged) {
        Matrix temp(m1);
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

MatrixProxy::MatrixProxy(const Matrix &m) {
    rows = m.size();
    cols = m.dim();
    stride = m.step();
    data = reinterpret_cast<uint8_t const *>(m[0]);
}

unsigned MatrixProxy::size() const {
    return rows;
}


unsigned MatrixProxy::dim() const {
    return cols;
}


float const *MatrixProxy::operator[](unsigned int i) const {
    return reinterpret_cast<float const *>(data + stride * i);
}


float *MatrixProxy::operator[](unsigned int i) {
    return const_cast<float *>(reinterpret_cast<float const *>(data + stride * i));
}
