#include "matrix.h"
#include "simd_math_prims.h"
#include "quant/qgemm.h"

namespace mblas {

Matrix& Mean(Matrix& Out, const Matrix& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();

  Out.Resize(1, n, 0.f);
  Matrix Ones(1, m, 1.f);

  float alpha = 1.0 / m;
  float beta  = 0.0;
  cblas_sgemv(CblasColMajor, CblasNoTrans, n, m, alpha, (float*)In.data(), n,
              (float*)Ones.data(), 1, beta, (float*)Out.data(), 1);
  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.Rows(), In.Cols());
  Out.SetRange(In.Min(), In.Max());
  std::copy(In.begin(), In.end(), Out.begin());
  return Out;
}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c) {
  size_t start = r * Out.Cols() + c;
  std::copy(In.begin(), In.end(), Out.begin() + start);
  Out.SetRange(In.Min(), In.Max());
  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.Cols() - c;
  Out.Resize(1, length);
  size_t start = r * In.Cols() + c;
  size_t end   = start + length;
  std::copy(In.begin() + start, In.begin() + end, Out.begin());
  Out.SetRange(In.Min(), In.Max());
  return Out;
}

void gCopyRows(data_t* out, const data_t* in, size_t cols,
               const RowPair* devPairs, size_t numPairs) {
  for(int j = 0; j < numPairs; ++j) {
      size_t dstId = devPairs[j].first;
      size_t srcId = devPairs[j].second;

      data_t* rowOut = out + dstId * cols;
      const data_t* rowIn = in + srcId * cols;

      for(int i = 0; i < cols; ++i)
          rowOut[i] = rowIn[i];
  }
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs) {
  data_t* d_out = Out.data();
  const data_t* d_in = In.data();

  gCopyRows(d_out, d_in, In.Cols(), devPairs, numPairs);
  return Out;
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs) {
  CopyRows(Out, In, pairs.data(), pairs.size());
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.size();
  
  float min_new = std::min(Out.Min(), In.Min());
  float max_new = std::max(Out.Max(), In.Max());
  
  RequantizeManyInNewRange(Out.data(), Out.size(), Out.Min(), Out.Max(),
                           min_new, max_new, Out.data());
  
  Out.Resize(Out.Rows() + In.Rows(), Out.Cols());
  
  RequantizeManyInNewRange(In.data(), In.size(), In.Min(), In.Max(),
                           min_new, max_new, Out.data() + oldSize);
  
  Out.SetRange(min_new, max_new);
  return Out;
}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.Resize(rowPairs.size(), In.Cols());
  CopyRows(Out, In, rowPairs);
  Out.SetRange(In.Min(), In.Max());
  return Out;
}

void gSlice(data_t* out, const data_t* in,
            size_t n, size_t dim,
            size_t rows, size_t cols) {
  for(int j = 0; j < rows; j++) {
    data_t* rowOut = out + j * dim;
    const data_t* rowIn = in + j * cols + n * dim;

    for(int i = 0; i < dim; ++i)
        rowOut[i] = rowIn[i];
  }
}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.Resize(In.Rows(), dim);

  data_t* d_out = Out.data();
  const data_t* d_in = In.data();

  gSlice(d_out, d_in, n, dim, In.Rows(), In.Cols());
  return Out;
}

void Prod(gemmlowp::GemmContext& context,
          Matrix& C, const Matrix& A, const Matrix& B,
          bool transA, bool transB) {
  bool transC = false;
  if(B.Cols() > A.Rows()) {
    transA = !transA;
    transB = !transB;
    transC = !transC;
    QGemm(context, B, transB, A, transA, C, transC);
  }
  else {
    QGemm(context, A, transA, B, transB, C, transC);
  }
}

void gSoftMax(data_t* d, size_t rows, size_t cols) {
  data_t sum[rows];
  for(int j = 0; j < rows; ++j) {
    sum[j] = 0;
    data_t* out = d + j * cols;
    for(int i = 0; i < cols; ++i) {
      out[i] = expapprox(out[i]);
      sum[j]+= out[i];
    }
    for(int i = 0; i < cols; ++i) {
      out[i] /= sum[j];
    }
  }
}

Matrix& Softmax(Matrix& Out) {
  gSoftMax(Out.data(), Out.Rows(), Out.Cols());
  return Out;
}

void gSoftMaxLog(data_t* d, size_t rows, size_t cols) {
  data_t sum[rows];
  for(int j = 0; j < rows; ++j) {
    sum[j] = 0;
    data_t* out = d + j * cols;
    for(int i = 0; i < cols; ++i) {
      out[i] = expapprox(out[i]);
      sum[j]+= out[i];
    }
    for(int i = 0; i < cols; ++i) {
      out[i] = logapprox(out[i] / sum[j]);
    }
  }
}

Matrix& SoftmaxLog(Matrix& Out) {
  gSoftMaxLog(Out.data(), Out.Rows(), Out.Cols());
}


}
