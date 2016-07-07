#pragma once

#include "decoder/god.h"
#include "mblas/matrix.h"
#include "quant/qgemm.h"

void gElementwiseOps(data_t* out,
                    const data_t* state,
                    const data_t* ruh,
                    const data_t* t,
                    const data_t* b,
                    const data_t* bx1,
                    const data_t* bx2,
                    float min, float max,
                    size_t rows, size_t cols);

template <class Weights>
class GRU {
  public:
    GRU(const Weights& model)
    : w_(model) {
      context_.set_max_num_threads(God::Get<size_t>("threads-openblas"));
      
      using namespace mblas;
      Transpose(WWx_, w_.W_);
      Matrix WxT;
      Transpose(WxT, w_.Wx_);
      Concat(WWx_, WxT);
      Transpose(WWx_);
      
      Transpose(UUx_, w_.U_);
      Matrix UxT;
      Transpose(UxT, w_.Ux_);
      Concat(UUx_, UxT);
      Transpose(UUx_); 
    }

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
      using namespace mblas;
      
      Prod(context_, RUH_, Context, WWx_);
      Prod(context_, Temp_, State, UUx_);
      
      mblas::Matrix B;
      mblas::Copy(B, w_.B_);
      mblas::Matrix Bx1;
      mblas::Copy(Bx1, w_.Bx1_);
      mblas::Matrix Bx2;
      mblas::Copy(Bx2, w_.Bx2_);
      
      TotalSpace(RUH_, Temp_, B, Bx1, Bx2);
      
      ElementwiseOps(NextState, State, RUH_, Temp_, B, Bx1, Bx2);
    }
                
    void ElementwiseOps(mblas::Matrix& NextState,
                        const mblas::Matrix& State,
                        mblas::Matrix& RUH,
                        mblas::Matrix& Temp,
                        mblas::Matrix& B,
                        mblas::Matrix& Bx1,
                        mblas::Matrix& Bx2) const {
      const size_t rows = State.Rows();
      const size_t cols = State.Cols();
      NextState.Resize(rows, cols);
      
      gElementwiseOps(NextState.data(), State.data(),
                      RUH.data(), Temp.data(),
                      B.data(), Bx1.data(), Bx2.data(),
                      RUH.Min(), RUH.Max(),
                      rows, cols);
    }
    
    size_t GetStateLength() const {
      return w_.U_.Rows();
    }

    
  private:
    // Model matrices
    const Weights& w_;
        
    // reused to avoid allocation    
    mutable mblas::Matrix WWx_;
    mutable mblas::Matrix UUx_;
    
    mutable mblas::Matrix RUH_;
    mutable mblas::Matrix Temp_;
    
    mutable gemmlowp::GemmContext context_;
};