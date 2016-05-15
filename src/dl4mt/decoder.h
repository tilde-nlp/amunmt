#pragma once

#include "dl4mt/model.h"
#include "dl4mt/gru.h"

template <class Backend>
class Decoder {
  private:
    typedef Backend::Matrix Matrix;
    
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}
            
        void Lookup(Matrix& Rows, const std::vector<size_t>& ids) {
          Assemble(Rows, w_.E_, ids);
        }
        
        size_t GetCols() {
          return w_.E_.Cols();    
        }
        
        size_t GetRows() const {
          return w_.E_.Rows();    
        }
        
      private:
        const Weights& w_;
    };
    
    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}          
        
        void InitializeState(Matrix& State,
                             const Matrix& SourceContext,
                             const size_t batchSize = 1) {
          using namespace mblas;
          
          Mean(Temp1_, SourceContext);
          Temp2_.Clear();
          Temp2_.Resize(batchSize, SourceContext.Cols(), 0.0);
          BroadcastVec(_1 + _2, Temp2_, Temp1_);
          Prod(State, Temp2_, w_.Wi_);
          BroadcastVec(Tanh(_1 + _2), State, w_.Bi_);
        }
        
        void GetNextState(Matrix& NextState,
                          const Matrix& State,
                          const Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }
        
      private:
        const Weights1& w_;
        const GRU<Weights2> gru_;
        
        Matrix Temp1_;
        Matrix Temp2_;
    };
    
    template <class Weights>
    class RNNFinal {
      public:
        RNNFinal(const Weights& model)
        : gru_(model) {}          
        
        void GetNextState(Matrix& NextState,
                          const Matrix& State,
                          const Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }
        
      private:
        const GRU<Weights> gru_;
    };
        
    template <class Weights>
    class Alignment {
      public:
        Alignment(const Weights& model)
        : w_(model)
        {
          //for(int i = 0; i < 2; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);            
          //}
        }
          
        void GetAlignedSourceContext(Matrix& AlignedSourceContext,
                                     const Matrix& HiddenState,
                                     const Matrix& SourceContext) {
          using namespace mblas;  
          
          Prod(/*h_[0],*/ Temp1_, SourceContext, w_.U_);
          Prod(/*h_[1],*/ Temp2_, HiddenState, w_.W_);
          BroadcastVec(_1 + _2, Temp2_, w_.B_/*, s_[1]*/);
          
          //cudaDeviceSynchronize();
          
          Broadcast(Tanh(_1 + _2), Temp1_, Temp2_);
          
          Prod(A_, w_.V_, Temp1_, false, true);
          
          size_t rows1 = SourceContext.Rows();
          size_t rows2 = HiddenState.Rows();     
          A_.Reshape(rows2, rows1); // due to broadcasting above
          Element(_1 + w_.C_(0,0), A_);
          
          Backend::Softmax(A_);
          Prod(AlignedSourceContext, A_, SourceContext);
        }
        
        void GetAttention(Matrix& Attention) {
          Backend::Copy(Attention, A_);
        }
      
      private:
        const Weights& w_;
        
        cublasHandle_t h_[2];
        cudaStream_t s_[2];
        
        Matrix Temp1_;
        Matrix Temp2_;
        Matrix A_;
        
        Matrix Ones_;
        Matrix Sums_;
    };
    
    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          //for(int i = 0; i < 3; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);            
          //}
        }
          
        void GetProbs(Matrix& Probs,
                  const Matrix& State,
                  const Matrix& Embedding,
                  const Matrix& AlignedSourceContext) {
          using namespace mblas;
          
          Prod(/*h_[0],*/ T1_, State, w_.W1_);
          Prod(/*h_[1],*/ T2_, Embedding, w_.W2_);
          Prod(/*h_[2],*/ T3_, AlignedSourceContext, w_.W3_);
          
          BroadcastVec(_1 + _2, T1_, w_.B1_ /*,s_[0]*/);
          BroadcastVec(_1 + _2, T2_, w_.B2_ /*,s_[1]*/);
          BroadcastVec(_1 + _2, T3_, w_.B3_ /*,s_[2]*/);
      
          //cudaDeviceSynchronize();
      
          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);
          
          Prod(Probs, T1_, w_.W4_);
          BroadcastVec(_1 + _2, Probs, w_.B4_);
          Backend::Softmax(Probs);
          Element(Log(_1), Probs);
        }
    
        void Filter(const std::vector<size_t>& ids) {
        }
       
      private:        
        const Weights& w_;
        
        cublasHandle_t h_[3];
        cudaStream_t s_[3];
        
        bool filtered_;
        Matrix FilteredWo_;
        Matrix FilteredWoB_;
        
        Matrix T1_;
        Matrix T2_;
        Matrix T3_;
    };
    
  public:
    Decoder(const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
      alignment_(model.decAlignment_),
      softmax_(model.decSoftmax_)
    {}
    
    void MakeStep(Matrix& NextState,
                  Matrix& Probs,
                  const Matrix& State,
                  const Matrix& Embeddings,
                  const Matrix& SourceContext) {
      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      GetProbs(Probs, NextState, Embeddings, AlignedSourceContext_);
    }
    
    void EmptyState(Matrix& State,
                    const Matrix& SourceContext,
                    size_t batchSize = 1) {
      rnn1_.InitializeState(State, SourceContext, batchSize);
    }
    
    void EmptyEmbedding(Matrix& Embedding,
                        size_t batchSize = 1) {
      Embedding.Clear();
      Embedding.Resize(batchSize, embeddings_.GetCols(), 0);
    }
    
    void Lookup(Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }
    
    void Filter(const std::vector<size_t>& ids) {
      // @TODO
    }
      
    void GetAttention(Matrix& Attention) {
      alignment_.GetAttention(Attention);
    }
    
    size_t GetVocabSize() const {
      return embeddings_.GetRows();
    }
    
  private:
    
    void GetHiddenState(Matrix& HiddenState,
                        const Matrix& PrevState,
                        const Matrix& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }
    
    void GetAlignedSourceContext(Matrix& AlignedSourceContext,
                                 const Matrix& HiddenState,
                                 const Matrix& SourceContext) {
      alignment_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext);
    }
    
    void GetNextState(Matrix& State,
                      const Matrix& HiddenState,
                      const Matrix& AlignedSourceContext) {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }
    
    
    void GetProbs(Matrix& Probs,
                  const Matrix& State,
                  const Matrix& Embedding,
                  const Matrix& AlignedSourceContext) {
      softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
    }
    
  private:
    Matrix HiddenState_;
    Matrix AlignedSourceContext_;  
    
    Embeddings<Weights<Backend>::DecEmbeddings> embeddings_;
    RNNHidden<Weights<Backend>::DecInit, Weights::DecGRU1> rnn1_;
    RNNFinal<Weights<Backend>::DecGRU2> rnn2_;
    Alignment<Weights<Backend>::DecAlignment> alignment_;
    Softmax<Weights<Backend>::DecSoftmax> softmax_;
};
