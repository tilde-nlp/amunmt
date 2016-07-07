#pragma once

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
inline void FloatToQuantizedMany(T* output, const float* input, size_t num,
                                 float range_min, float range_max) {
  if (range_min == range_max) {
    for(size_t i = 0; i < num; ++i)
      output[i] = 0;
    return;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  const int64_t lowest_quantized =
      static_cast<double>(std::numeric_limits<T>::lowest());
  const int64_t highest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::max());
  const int64_t rounding_error = round(range_min * range_scale);
  
  for(size_t i = 0; i < num; ++i) {
    int64_t quantized = round(input[i] * range_scale) - rounding_error + lowest_quantized;
    quantized = std::max(quantized, lowest_quantized);
    quantized = std::min(quantized, highest_quantized);
    output[i] = static_cast<T>(quantized);
  }
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

template <class T1, class T2>
inline void RequantizeManyInNewRange(T1* input, size_t count,
                                     float min_input, float max_input,
                                     float min_output, float max_output,
                                     T2* output) {
  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(input[index], min_input, max_input);
    output[index] = FloatToQuantized<T2>(input_float, min_output, max_output);
  }
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRange<data32_t, data_t>(
    data32_t* input, size_t count, float min_input, float max_input,
    float min_output, float max_output, data_t* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const uint64_t recip_output_range_fp =
      static_cast<uint64_t>(recip_output_range * (1 << fp_shift));
  const uint64_t range_scale_fp =
      static_cast<uint64_t>(255.0 * (1 << fp_shift) * input_range / output_range);
  const uint64_t input_offset_fp =
      (min_input * recip_output_range_fp) + (range_scale_fp >> 1);
  const uint64_t output_offset_fp =
      output_range == 0.0 ? 0.0 : round((min_output * 255.0) / output_range);
  const uint64_t rounding_delta = 1 << (fp_shift - 1);

  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (size_t index = 0; index < count; ++index) {
    const uint64_t input_value = static_cast<uint64_t>(input[index]);
    const uint64_t fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const uint64_t round_intermediate =
        ((fp_value >= 0) ? (fp_value + rounding_delta)
                         : (fp_value - rounding_delta)) >>
        fp_shift;
    uint64_t quantized_int64 = (round_intermediate - output_offset_fp);
    quantized_int64 = std::max(quantized_int64, 0UL);
    quantized_int64 = std::min(quantized_int64, 255UL);
    output[index] = static_cast<data_t>(static_cast<data32_t>(quantized_int64));
  }
}

template <class Matrix>
float MinTotal(Matrix& m) {
  return m.Min();
}

template <class Matrix, class ...Args>
float MinTotal(Matrix& m, Args&... args) {
  return std::min(m.Min(), MinTotal(args...));
}

template <class Matrix>
float MaxTotal(Matrix& m) {
  return m.Max();
}

template <class Matrix, class ...Args>
float MaxTotal(Matrix& m, Args&... args) {
  return std::max(m.Max(), MaxTotal(args...));
}

template <class M>
void RequantizeTotal(float min, float max, M& m) {
  if(m.Min() != min || m.Max() != max) {
    RequantizeManyInNewRange(m.data(), m.size(),
                             m.Min(), m.Max(),
                             min, max, m.data());
    m.SetRange(min, max);
  }
}

template <class M, class ...Args>
void RequantizeTotal(float min, float max, M& m, Args&... args) {
  RequantizeTotal(min, max, m);
  RequantizeTotal(min, max, args...);
}

template <class ...Args>
void TotalSpace(Args&... args) {
  float min = MinTotal(args...);
  float max = MaxTotal(args...);
  RequantizeTotal(min, max, args...);
}

template <class MInput, class MOutput>
void QuantizeDownAndShrinkRange(MOutput& output, const MInput& input) {
  const float input_min_float = input.Min();
  const float input_max_float = input.Max();

  typedef typename MInput::value_type T1;
  typedef typename MOutput::value_type T2;
  
  const int32_t input_lowest_quantized =
      static_cast<int32_t>(std::numeric_limits<T1>::min());
  const int32_t input_highest_quantized =
      static_cast<int32_t>(std::numeric_limits<T1>::max());
  T1 actual_min_quantized = input_highest_quantized;
  T1 actual_max_quantized = input_lowest_quantized;
  for (int i = 0; i < input.size(); ++i) {
    const T1 value = input.data()[i];
    actual_min_quantized = std::min(actual_min_quantized, value);
    actual_max_quantized = std::max(actual_max_quantized, value);
  }

  // We want to make sure that the minimum is no larger than zero, so that the
  // convolution operation can run efficiently.
  const float actual_min_float =
        std::min(0.0f, QuantizedToFloat(actual_min_quantized, input_min_float,
                                        input_max_float));
  const float actual_max_float = QuantizedToFloat(
        actual_max_quantized, input_min_float, input_max_float);

  RequantizeManyInNewRange(input.data(), input.size(),
                           input_min_float, input_max_float, actual_min_float,
                           actual_max_float, output.data());
  output.SetRange(actual_min_float, actual_max_float);
}