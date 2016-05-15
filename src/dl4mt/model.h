#pragma once

#include <map>
#include <string>

#include "npz_converter.h"

template <class Backend>
struct Weights {
  typedef typename Backend::Matrix Matrix;
  
  //////////////////////////////////////////////////////////////////////////////
  
  struct EncEmbeddings {
    EncEmbeddings(const NpzConverter<Backend>& model)
    : E_(model["Wemb"])
    {}
    
    const Matrix E_;
  };
  
  struct EncForwardGRU {
    EncForwardGRU(const NpzConverter<Backend>& model) 
    : W_(model["encoder_W"]),  
      B_(model("encoder_b", true)),
      U_(model["encoder_U"]),
      Wx_(model["encoder_Wx"]),
      Bx1_(model("encoder_bx", true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model["encoder_Ux"])
    { }
    
    const Matrix W_;
    const Matrix B_;
    const Matrix U_;
    const Matrix Wx_;
    const Matrix Bx1_;
    const Matrix Bx2_;
    const Matrix Ux_;
  };
  
  struct EncBackwardGRU {
    EncBackwardGRU(const NpzConverter<Backend>& model) 
    : W_(model["encoder_r_W"]),  
      B_(model("encoder_r_b", true)),
      U_(model["encoder_r_U"]),
      Wx_(model["encoder_r_Wx"]),
      Bx1_(model("encoder_r_bx", true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model["encoder_r_Ux"])
    {}
    
    const Matrix W_;
    const Matrix B_;
    const Matrix U_;
    const Matrix Wx_;
    const Matrix Bx1_;
    const Matrix Bx2_;
    const Matrix Ux_;
  };
  
  //////////////////////////////////////////////////////////////////////////////
  
  struct DecEmbeddings {
    DecEmbeddings(const NpzConverter<Backend>& model)
    : E_(model["Wemb_dec"])
    {}
    
    const Matrix E_;
  };

  struct DecInit {
    DecInit(const NpzConverter<Backend>& model)
    : Wi_(model["ff_state_W"]),
      Bi_(model("ff_state_b", true))
    {}
    
    const Matrix Wi_;
    const Matrix Bi_;
  };
  
  struct DecGRU1 {
    DecGRU1(const NpzConverter<Backend>& model)
    : W_(model["decoder_W"]),
      B_(model("decoder_b", true)),
      U_(model["decoder_U"]),      
      Wx_(model["decoder_Wx"]),
      Bx1_(model("decoder_bx", true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model["decoder_Ux"])
    {}
    
    const Matrix W_;
    const Matrix B_;
    const Matrix U_;
    const Matrix Wx_;
    const Matrix Bx1_;
    const Matrix Bx2_;
    const Matrix Ux_;
  };
  
  struct DecGRU2 {
    DecGRU2(const NpzConverter<Backend>& model)
    : W_(model["decoder_Wc"]),
      B_(model("decoder_b_nl", true)),
      U_(model["decoder_U_nl"]),      
      Wx_(model["decoder_Wcx"]),
      Bx2_(model("decoder_bx_nl", true)),
      Bx1_(Bx2_.Rows(), Bx2_.Cols(), 0.0),
      Ux_(model["decoder_Ux_nl"])
    {}
          
    const Matrix W_;
    const Matrix B_;
    const Matrix U_;
    const Matrix Wx_;
    const Matrix Bx2_;
    const Matrix Bx1_;
    const Matrix Ux_;
  };
  
  struct DecAlignment {
    DecAlignment(const NpzConverter<Backend>& model)
    : V_(model("decoder_U_att", true)),
      W_(model["decoder_W_comb_att"]),
      B_(model("decoder_b_att", true)),
      U_(model["decoder_Wc_att"]),
      C_(model["decoder_c_tt"]) // scalar?
    {}
          
    const Matrix V_;
    const Matrix W_;
    const Matrix B_;
    const Matrix U_;
    const Matrix C_;
  };
  
  struct DecSoftmax {
    DecSoftmax(const NpzConverter<Backend>& model)
    : W1_(model["ff_logit_lstm_W"]),
      B1_(model("ff_logit_lstm_b", true)),
      W2_(model["ff_logit_prev_W"]),
      B2_(model("ff_logit_prev_b", true)),
      W3_(model["ff_logit_ctx_W"]),
      B3_(model("ff_logit_ctx_b", true)),
      W4_(model["ff_logit_W"]),
      B4_(model("ff_logit_b", true))
    {}
          
    const Matrix W1_;
    const Matrix B1_;
    const Matrix W2_;
    const Matrix B2_;
    const Matrix W3_;
    const Matrix B3_;
    const Matrix W4_;
    const Matrix B4_;
  };
  
  Weights(const std::string& npzFile, size_t device = 0)
  : Weights(NpzConverter<Backend>(npzFile), device)
  {}
  
  Weights(const NpzConverter<Backend>& model, size_t device = 0)
  : encEmbeddings_(model),
    encForwardGRU_(model),
    encBackwardGRU_(model),
    decEmbeddings_(model),
    decInit_(model),
    decGru1_(model),
    decGru2_(model),
    decAlignment_(model),
    decSoftmax_(model),
    device_(device)
    {}
  
  size_t GetDevice() {
    return device_;
  }
  
  const EncEmbeddings encEmbeddings_;
  const DecEmbeddings decEmbeddings_;
  const EncForwardGRU encForwardGRU_;
  const EncBackwardGRU encBackwardGRU_;
  const DecInit decInit_;
  const DecGRU1 decGru1_;
  const DecGRU2 decGru2_;
  const DecAlignment decAlignment_;
  const DecSoftmax decSoftmax_;
  
  const size_t device_;
};
