#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <limits>

#define GEMMLOWP_SSE4_64

#include "gemmlowp/public/gemmlowp.h"

// We have to be able to detect and handle overflows in int32_t, so this function
// uses doubles and int64_t's to make sure we have enough room.
template <class T>
inline int64_t FloatToQuantizedUnclamped(float input, float range_min, float range_max) {
  if (range_min == range_max) {
    return 0;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64_t quantized =
      (round(input * range_scale) - round(range_min * range_scale));
  const int64_t lowest_quantized =
      static_cast<double>(std::numeric_limits<T>::lowest());
  quantized += lowest_quantized;
  return quantized;
}

// This converts the float into the final quantized type, clamping/saturating
// any over or underflows.
template <class T>
inline T FloatToQuantized(float input, float range_min, float range_max) {
  int64_t quantized = FloatToQuantizedUnclamped<T>(input, range_min, range_max);
  const int64_t lowest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::lowest());
  const int64_t highest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::max());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32_t>(quantized));
}

template <class T>
float QuantizedToFloat(T input, float range_min, float range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64_t lowest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::lowest());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double result = range_min + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T>
float FloatForOneQuantizedLevel(float range_min, float range_max) {
  const int32_t highest = static_cast<int32_t>(std::numeric_limits<T>::max());
  const int32_t lowest = static_cast<int32_t>(std::numeric_limits<T>::lowest());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

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

template <bool TransposeA, bool TransposeB, bool TransposeC>
void GemmlowpMultiply(const uint8_t* a_data, const uint8_t* b_data,
                      int32_t* c_data, int m, int n, int k, int offset_a,
                      int offset_b, int lda, int ldb, int ldc) {
  const uint8_t* a_data_as_uint8 = a_data;
  const uint8_t* b_data_as_uint8 = b_data;
  int32_t* c_data_as_int32_t = c_data;
  static const gemmlowp::MapOrder ResultOrder =
      !TransposeC ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder LhsOrder =
      !TransposeA ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder RhsOrder =
      !TransposeB ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  gemmlowp::MatrixMap<const std::uint8_t, LhsOrder> lhs(a_data_as_uint8, m, k,
                                                        lda);
  gemmlowp::MatrixMap<const std::uint8_t, RhsOrder> rhs(b_data_as_uint8, k, n,
                                                        ldb);
  gemmlowp::MatrixMap<std::int32_t, ResultOrder> result(c_data_as_int32_t, m, n,
                                                        ldc);
  const std::tuple<> empty_pipeline = {};
  gemmlowp::GemmContext context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &context, lhs, rhs, &result, -offset_a, -offset_b, empty_pipeline);
}

template <class M>
void QGemm(M& C, const M& A, const M& B) {
  const float min_a = *std::min_element(A.begin(), A.end());
  const float max_a = *std::max_element(A.begin(), A.end());
  const float min_b = *std::min_element(B.begin(), B.end());
  const float max_b = *std::max_element(B.begin(), B.end());

  const int32_t offset_a = FloatToQuantizedUnclamped<uint8_t>(0.0f, min_a, max_a);
  const int32_t offset_b = FloatToQuantizedUnclamped<uint8_t>(0.0f, min_b, max_b);
  const int32_t offset_c = 0;
  const int32_t mult_c = 1;
  const int32_t shift_c = 0;

  C.Resize(A.Rows(), B.Cols());
  
  const size_t m = A.Rows();
  const size_t n = B.Cols();
  const size_t k = A.Cols();
  const size_t lda = A.Cols();
  const size_t ldb = B.Cols();
  const size_t ldc = C.Cols();
  
  uint8_t* a_data = new uint8_t[m * k];
  for(int i = 0; i < m * k; ++i)
    a_data[i] = FloatToQuantized<uint8_t>(A.data()[i], min_a, max_a);
  
  uint8_t* b_data = new uint8_t[k*n];
  for(int i = 0; i < k * n; ++i)
    b_data[i] = FloatToQuantized<uint8_t>(B.data()[i], min_b, max_b);
  
  int32_t* c_data = new int32_t[m*n];
  GemmlowpMultiply<false, false, false>(a_data, b_data, c_data, m, n, k,
                                        offset_a, offset_b, lda, ldb, ldc);
  
  delete[] a_data;
  delete[] b_data;

  float min_c_value;
  float max_c_value;
  QuantizationRangeForMultiplication<uint8_t, uint8_t, int32_t>(min_a, max_a, min_b, max_b,
                                                                 &min_c_value, &max_c_value);
  
  for(int i = 0; i < m * n; ++i)
    C.data()[i] = QuantizedToFloat(c_data[i], min_c_value, max_c_value);
    
  delete[] c_data;
}

