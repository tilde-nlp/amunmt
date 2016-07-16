#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#define EIGEN_DONT_PARALLELIZE
#include <eigen3/Eigen/Dense>

#include "cblas.h"
#include "phoenix_functions.h"

namespace mblas {

typedef Eigen::Matrix<float,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> Matrix;

typedef Eigen::Matrix<float,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> RMatrix;

typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> RVector;

typedef Eigen::Map<Matrix> MatrixMap;

//Matrix operator*(const Matrix& m1, const Matrix& m2);
                      
template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 5) {
  std::cerr << m.rows() << " " << m.cols() << std::endl;
  for(size_t i = 0; i < m.rows(); ++i) {
    for(size_t j = pos; j < m.cols() && j < pos + l; ++j) {
      std::cerr << m(i, j) << " ";
    }
    std::cerr << std::endl;
    if(i == 4)
      break;
  }
}

template <class M>
void Debug2(const M& m) {
  std::cerr << m.rows() << " " << m.cols() << std::endl;
  for(size_t i = 0; i < m.rows(); ++i) {
    for(size_t j = 0; j < m.cols(); ++j) {
      std::cerr << m(i, j) << " ";
    }
    std::cerr << std::endl;
  }
}

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef std::vector<RowPair> DeviceRowPairs;

template <class M>
M& Assemble(M& Out, const M& In,
            const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.resize(rowPairs.size(), In.cols());
  
  for(int j = 0; j < rowPairs.size(); ++j) {
    size_t dstId = rowPairs[j].first;
    size_t srcId = rowPairs[j].second;
    Out.row(dstId) = In.row(srcId);
  }
  
  return Out;
}

template <class M>
Matrix Softmax(const M& m) {
  Matrix nums = m.unaryExpr(&expapprox);
  Matrix denoms = nums.rowwise().sum();
  for(size_t i = 0; i < m.rows(); ++i)
    nums.row(i) = nums.row(i) / denoms(i);
  return std::move(nums);
}

template <class M>
Matrix SoftmaxCol(const M& m) {
  Matrix nums = m.unaryExpr(&expapprox);
  Matrix denoms = nums.colwise().sum();
  for(size_t i = 0; i < m.cols(); ++i)
    nums.col(i) = nums.col(i) / denoms(i);
  return std::move(nums);
}

template <class M1, class M2>
Matrix Add3D(const M1& m1, const M2& m2) {
  size_t rows1 = m1.rows();
  size_t rows2 = m2.rows();

  size_t rows = rows1 * rows2;
  size_t cols = m1.cols();
  
  Matrix out(rows, cols);
  
  for(size_t i = 0; i < rows; ++i) {
    size_t r1 = i % rows1;
    size_t r2 = i / rows1;
    out.row(i) = m1.row(r1) + m2.row(r2);
  }
  return std::move(out);
}

}
