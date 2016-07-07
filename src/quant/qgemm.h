#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <limits>

#define GEMMLOWP_SSE4_64

#include "types.h"
#include "mblas/matrix.h"
#include "gemmlowp/public/gemmlowp.h"
#include "quant/operators.h"
#include "quant/quantize.h"

// We have to be able to detect and handle overflows in int32_t, so this function
// uses doubles and int64_t's to make sure we have enough room.

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int32_t c_highest = static_cast<int32_t>(std::numeric_limits<T3>::max());
  const int32_t c_lowest = static_cast<int32_t>(std::numeric_limits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class Context, class M>
void QGemm(Context& context, const M& A, bool transA, const M& B, bool transB, M& C, bool transC) {
  
  const float min_a = A.Min();
  const float max_a = A.Max();
  const float min_b = B.Min();
  const float max_b = B.Max();
  
  const data32_t offset_a = FloatToQuantizedUnclamped<data_t>(0.0f, min_a, max_a);
  const data32_t offset_b = FloatToQuantizedUnclamped<data_t>(0.0f, min_b, max_b);
  
  const std::tuple<> empty_pipeline = {};
  
  mblas::Matrix32 C32;
  if(transC) {
    C.Resize(A.Cols(), B.Rows());
    C32.Resize(A.Cols(), B.Rows());
  } else {
    C.Resize(A.Rows(), B.Cols());
    C32.Resize(A.Rows(), B.Cols());
  }
  
  if(transA) {
    gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::ColMajor> a(A.data(), A.Cols(), A.Rows(), A.Cols());
    if(transB) {
      gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::ColMajor> b(B.data(), B.Cols(), B.Rows(), B.Cols());
      if(transC) {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::ColMajor> c(C32.data(), A.Cols(), B.Rows(), B.Cols());
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
      else {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::RowMajor> c(C32.data(), A.Rows(), B.Cols(), B.Cols());      
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
    }
    else {
      gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::RowMajor> b(B.data(), B.Rows(), B.Cols(), B.Cols());
      if(transC) {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::ColMajor> c(C32.data(), A.Cols(), B.Rows(), B.Cols());        
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
      else {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::RowMajor> c(C32.data(), A.Rows(), B.Cols(), B.Cols());      
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
    }
  }
  else {
    gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::RowMajor> a(A.data(), A.Rows(), A.Cols(), A.Cols());
    if(transB) {
      gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::ColMajor> b(B.data(), B.Cols(), B.Rows(), B.Cols());
      if(transC) {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::ColMajor> c(C32.data(), A.Cols(), B.Rows(), B.Cols());        
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
      else {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::RowMajor> c(C32.data(), A.Rows(), B.Cols(), B.Cols());      
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
    }
    else {
      gemmlowp::MatrixMap<const data_t, gemmlowp::MapOrder::RowMajor> b(B.data(), B.Rows(), B.Cols(), B.Cols());
      if(transC) {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::ColMajor> c(C32.data(), A.Cols(), B.Rows(), B.Cols());        
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
      else {
        gemmlowp::MatrixMap<data32_t, gemmlowp::MapOrder::RowMajor> c(C32.data(), A.Rows(), B.Cols(), B.Cols());      
        gemmlowp::GemmWithOutputPipeline<data_t, data32_t, gemmlowp::DefaultL8R8BitDepthParams>(
          &context, a, b, &c, -offset_a, -offset_b, empty_pipeline);
      }
    }
  }
  float min_c_value;
  float max_c_value;  
  QuantizationRangeForMultiplication<data_t, data_t, data32_t>(min_a, max_a, min_b, max_b,                                                               &min_c_value, &max_c_value);
  C32.SetRange(min_c_value, max_c_value);
  QuantizeDownAndShrinkRange(C, C32);
}
