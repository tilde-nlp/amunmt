#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <cblas.h>

#include "types.h"
#include "base_matrix.h"
#include "phoenix_functions.h"
#include "gemmlowp/public/gemmlowp.h"
#include "quant/quantize.h"


namespace mblas {

using namespace boost::phoenix::placeholders;

template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;

    TMatrix()
    : rows_(0), cols_(0), min_(0), max_(0)
    {}

    TMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows_ * cols_)
    {}

    TMatrix(size_t rows, size_t cols, value_type val)
    : rows_(rows), cols_(cols), data_(rows_ * cols_, val)
    {}

    TMatrix(TMatrix&& m)
    : rows_(m.rows_), cols_(m.cols_), min_(m.min_), max_(m.max_), data_(std::move(m.data_)) {}

    TMatrix(const TMatrix& m) = delete;

    value_type operator()(size_t i, size_t j) const {
      return data_[i * cols_ + j];
    }

    void Set(size_t i, size_t j, data_t value)  {
      data_[i * cols_ + j] = value;
    }

    size_t Rows() const {
      return rows_;
    }

    size_t Cols() const {
      return cols_;
    }
    
    float Min() const {
      return min_;
    }
  
    float Max() const {
      return max_;
    }

    void Resize(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_);
    }

    void Resize(size_t rows, size_t cols, value_type val) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_, val);
    }

    void Reserve(size_t rows, size_t cols) {
      data_.reserve(rows * cols);
    }

    void Reshape(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
    }

    void SetRange(float min, float max) {
      min_ = min;
      max_ = max;
    }

    void Purge() {
      Clear();
      VecType temp;
      data_.swap(temp);
    }

    void Clear() {
      data_.clear();
      rows_ = 0;
      cols_ = 0;
    }

    VecType& GetVec() {
      return data_;
    }

    const VecType& GetVec() const {
      return data_;
    }

    value_type* data() {
      return data_.data();
    }

    const value_type* data() const {
      return data_.data();
    }

    iterator begin() {
      return data_.begin();
    }

    iterator end() {
      return data_.end();
    }

    const_iterator begin() const{
      return data_.begin();
    }

    const_iterator end() const {
      return data_.end();
    }

    size_t size() const {
      return data_.size();
    }

  private:
    size_t rows_;
    size_t cols_;
    float min_;
    float max_;
    VecType data_;
};

typedef std::vector<data_t> FVec;
typedef TMatrix<FVec> Matrix;

typedef std::vector<data32_t> FVec32;
typedef TMatrix<FVec32> Matrix32;

template <class M>
void debug1(const M& m, size_t pos = 0, size_t l = 5) {
  std::cerr << "rows=" << m.Rows() << " cols=" << m.Cols() << " min=" << m.Min() << " max=" << m.Max() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    for(size_t j = pos; j < m.Cols() && j < pos + l; ++j) {
      std::cerr << QuantizedToFloat(m.GetVec()[i * m.Cols() + j], m.Min(), m.Max()) << " ";
      //std::cerr << (int)m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
    if(i == 4)
      break;
  }
}

template <class M>
M& Swap(M& Out, M& In) {
  size_t iRows = In.Rows();
  size_t iCols = In.Cols();
  size_t oRows = Out.Rows();
  size_t oCols = Out.Cols();

  float iMin = In.Min();
  float iMax = In.Max();
  float oMin = Out.Min();
  float oMax = Out.Max();

  Out.Reshape(iRows, iCols);
  Out.SetRange(iMin, iMax);
  
  In.Reshape(oRows, oCols);
  In.SetRange(oMin, oMax);

  In.GetVec().swap(Out.GetVec());
  return Out;
}

template <class M>
M& Transpose(M& Out, const M& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();

  Out.Resize(n, m);
  Out.SetRange(In.Min(), In.Max());

  const typename M::value_type* d_in = In.data();
  typename M::value_type* d_out = Out.data();
  
  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      d_out[j * m + i] = d_in[i * n  + j];
    
  return Out;
}

template <class M>
M& Transpose(M& Out) {
  M Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}


Matrix& Mean(Matrix& Out, const Matrix& In);

Matrix& Copy(Matrix& Out, const Matrix& In);

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0, const size_t c = 0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0, const size_t c = 0);

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef std::vector<RowPair> DeviceRowPairs;

Matrix& Concat(Matrix& Out, const Matrix& In);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

void Prod(gemmlowp::GemmContext& context,
          Matrix& C, const Matrix& A, const Matrix& B,
          bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out);
Matrix& SoftmaxLog(Matrix& Out);

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows1 = Out.Rows();
  size_t rows2 = In.Rows();

  size_t rows = rows1 * rows2;
  size_t cols  = Out.Cols();

  Matrix Temp(rows, cols, 1.0);

  data_t* d_out = Temp.data();
  const data_t* d_in1 = Out.data();
  const data_t* d_in2 = In.data();

  for(int j = 0; j < rows; ++j) {
    data_t* rowOut = d_out + j * cols;
    const data_t* rowIn1 = d_in1 + (j % rows1) * cols;
    const data_t* rowIn2 = d_in2 + (j / rows1) * cols;
    
    for(int i = 0; i < cols; ++i)
      rowOut[i] = functor(rowIn1[i], rowIn2[i]);
  }
  
  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
  // @TODO: Make this efficient with special kernel!
  Matrix InTemp;
  Transpose(InTemp, In);

  Transpose(Out);
  Broadcast(functor, Out, InTemp);
  Transpose(Out);
  return Out;
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  data_t* d_out = Out.data();
  const data_t* d_in = In.data();

  for(int j = 0; j < cols; ++j) {    
    for(int i = 0; i < rows; ++i) {
      data_t* rowOut = d_out + i * cols + j;
      const data_t* rowIn  = d_in + i;
      *rowOut = functor(*rowOut, *rowIn);      
    }
  }
  return Out;
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  data_t* d_out = Out.data();
  const data_t* d_in = In.data();

  for(int j = 0; j < rows; ++j) {
    data_t* rowOut = d_out + j * cols;
    for(int i = 0; i < cols; ++i)
      rowOut[i] = functor(rowOut[i], d_in[i]);
  }
  
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor, Matrix& Out) {
  data_t* d_out = Out.data();
  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i]);
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In) {
  data_t* d_out = Out.data();
  const data_t* d_in = In.data();

  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i], d_in[i]);

  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2) {
  
  data_t* d_out = Out.data();
  const data_t* d_in1 = In1.data();
  const data_t* d_in2 = In2.data();
  
  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i], d_in1[i], d_in2[i]);

  return Out;
}

}
