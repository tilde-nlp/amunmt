#pragma once

#include <cmath>
#include <utility>

namespace mblas {
    
    template <typename T>
    class Scalar {
      public:
        typedef T type;
        
        Scalar(const T& scalar) : scalar_(std::move(scalar)) {}
        
        const T& operator()() const {
          return std::move(scalar_);
        }
                
      private:
        const T scalar_;
    };
    
    template <class E>
    const Scalar<E> s(const E e) {
      return Scalar<E>(e);
    }

    template <class Expression, class Op>
    class UnaryOp {
      private:
        const Expression exp_;
        const Op op_{};

      public:
        typedef decltype( op_(exp_()) ) type;
        
        UnaryOp(const Expression& exp)
        : exp_(std::move(exp)) {}
        
        auto operator()() const -> decltype( op_(exp_()) ) { 
            return std::move(op_(exp_()));
        }
    };
    
    template <class Expression1, class Expression2, class Op>
    class BinaryOp {
      private:
        const Expression1 exp1_;
        const Expression2 exp2_;
        const Op op_{};

      public:
        typedef decltype( op_(exp1_(), exp2_()) ) type;
        
        BinaryOp(const Expression1& exp1, const Expression2 &exp2)
        : exp1_(std::move(exp1)), exp2_(std::move(exp2)) {}
        
        auto operator()() const -> decltype( op_(exp1_(), exp2_()) ) {
            return std::move(op_(exp1_(), exp2_()));
        }        
    };
    
    template <typename T1, typename T2>
    struct PlusOp {
      auto operator()(const T1& a, const T2& b) const -> decltype ( a + b ) {
        return a + b;
      }
    };
        
    template <class E1, class E2>
    using Plus = BinaryOp<E1, E2, PlusOp<typename E1::type, typename E2::type>>;
    
    template <class E1, class E2>
    const Plus<E1, E2> operator+(const E1& e1, const E2& e2) {
        return Plus<E1, E2>(e1, e2);   
    }
    
    template <typename T1, typename T2>
    struct MultOp {
      auto operator()(const T1& a, const T2& b) const -> decltype ( a * b ) {
        return a * b;
      }
    };
    
    template <class E1, class E2>
    using Mult = BinaryOp<E1, E2, MultOp<typename E1::type, typename E2::type>>;
    
    template <class E1, class E2>
    const Mult<E1, E2> operator*(const E1& e1, const E2& e2) {
        return Mult<E1, E2>(e1, e2);   
    }

    template <typename T1, typename T2>
    struct DivOp {
      auto operator()(const T1& a, const T2& b) const -> decltype ( a / b ) {
        return a / b;
      }
    };
    
    template <class E1, class E2>
    using Div = BinaryOp<E1, E2, DivOp<typename E1::type, typename E2::type>>;
    
    template <class E1, class E2>
    const Div<E1, E2> operator/(const E1& e1, const E2& e2) {
        return Div<E1, E2>(e1, e2);   
    }

    template <typename T1, typename T2>
    struct SubOp {
      auto operator()(const T1& a, const T2& b) const -> decltype ( a - b ) {
        return a - b;
      }
    };
    
    template <class E1, class E2>
    using Sub = BinaryOp<E1, E2, SubOp<typename E1::type, typename E2::type>>;
    
    template <class E1, class E2>
    const Sub<E1, E2> operator-(const E1& e1, const E2& e2) {
        return Sub<E1, E2>(e1, e2);   
    }
    
    template <typename T1, typename T2>
    struct PowOp {
      auto operator()(const T1& a, const T2& b) const -> decltype ( pow(a, b) ) {
        return pow(a, b);
      }
    };
    
    template <class E1, class E2>
    using Pow = BinaryOp<E1, E2, PowOp<typename E1::type, typename E2::type>>;
    
    template <class E1, class E2>
    const Pow<E1, E2> pow(const E1& e1, const E2& e2) {
        return Pow<E1, E2>(e1, e2);   
    }
    
    //template <class E1, class E2, T>
    //auto GradOp(const Plus<E1, E2>& plus, const T& x) const -> decltype( PlusOp(GradOp(plus.exp1, x), GradOp(plus.exp2, x)) ) {
    //  return PlusOp(GradOp(plus.exp1, x), GradOp(plus.exp2, x));
    //}    
    //
    //template <T>
    //Scalar<E> GradOp(const T& t, const T& x) const {
    //  return t == x ? s(1) : s(0);
    //}
    //
    //template <class E, T>
    //Scalar<E> GradOp(const Scalar<E>& scalar, const T& x) const {
    //  return s(0);
    //}
    //
    
    template <class InIt, class OutIt>
    void Copy(InIt it, InIt end, OutIt out) {
        
    }
    
    template <class MatrixType>
    MatrixType& Transpose(MatrixType& matrix) {
        return matrix;
    }
}