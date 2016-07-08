#pragma once

template <typename T>
inline int32_t qsumUnclamped(T zero, T q1, T q2) {
  return q1 + q2 - zero;
}

template <typename T, typename ...Args>
inline int32_t qsumUnclamped(T zero, T q, Args ...args) {
  return q + qsumUnclamped(zero, args...) - zero;
}

template <typename T, typename ...Args>
inline T qsum(T zero, Args ...args) {
  int32_t qmax = (1 << sizeof(T) * 8) - 1;
  int32_t qmin = 0;
  return static_cast<T>(std::max(qmin, std::min(qmax, qsumUnclamped(zero, args...))));
}

template <typename T>
inline int32_t qsubUnclamped(T zero, T q1, T q2) {
  return q1 - q2 + zero;
}

template <typename T>
inline T qsub(T zero, T q1, T q2) {
  int32_t qmax = (1 << sizeof(T) * 8) - 1;
  int32_t qmin = 0;
  return static_cast<T>(std::max(qmin, std::min(qmax, qsubUnclamped(zero, q1, q2))));
}

template <typename T>
inline int32_t qmulUnclamped(T zero, T one, T q1, T q2) {
  int32_t q1zero = q1 - zero;
  int32_t q2zero = q2 - zero;
  int32_t div = one - zero;
  return (q1zero * q2zero) / div + zero;
}

template <typename T>
inline T qmul(T zero, T one, T q1, T q2) {
  int32_t qmax = (1 << sizeof(T) * 8) - 1;
  int32_t qmin = 0;
  return static_cast<T>(std::max(qmin, std::min(qmax, qmulUnclamped(zero, one, q1, q2))));
}
