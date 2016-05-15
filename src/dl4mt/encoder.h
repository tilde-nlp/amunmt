#pragma once

#include "dl4mt/model.h"
#include "dl4mt/gru.h"
 
template <Backend>
class Encoder {
  private:
    typedef Backend::Matrix Matrix;
    
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}
          
        void Lookup(Matrix& Row, size_t i) {
          using namespace mblas;
          if(i < w_.E_.Rows())
            CopyRow(Row, w_.E_, i);
          else
            CopyRow(Row, w_.E_, 1); // UNK
        }
      
      private:
        const Weights& w_;
    };
    
    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}
        
        void InitializeState(size_t batchSize = 1) {
          State_.Clear();
          State_.Resize(batchSize, gru_.GetStateLength(), 0.0);
        }
        
        void GetNextState(Matrix& NextState,
                          const Matrix& State,
                          const Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }
        
        template <class It>
        void GetContext(It it, It end, 
                        Matrix& Context, bool invert) {
          InitializeState();
          
          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(State_, State_, *it++);
            if(invert)
              PasteRow(Context, State_, n - i - 1, gru_.GetStateLength());
            else
              PasteRow(Context, State_, i, 0);
            ++i;
          }
        }
        
        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }
        
      private:
        // Model matrices
        const GRU<Weights> gru_;
        
        Matrix State_;
    };
    
  public:
    Encoder(const Weights& model)
    : embeddings_(model.encEmbeddings_),
      forwardRnn_(model.encForwardGRU_),
      backwardRnn_(model.encBackwardGRU_)
    {}
    
    void GetContext(const std::vector<size_t>& words,
                    Matrix& Context) {
      std::vector<Matrix> embeddedWords;
      
      Context.Resize(words.size(), forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());
      for(auto& w : words) {
        embeddedWords.emplace_back();
        embeddings_.Lookup(embeddedWords.back(), w);
      }
      
      forwardRnn_.GetContext(embeddedWords.cbegin(),
                             embeddedWords.cend(),
                             Context, false);
      backwardRnn_.GetContext(embeddedWords.crbegin(),
                              embeddedWords.crend(),
                              Context, true);
    }
    
  private:
    Embeddings<Weights<Backend>::EncEmbeddings> embeddings_;
    RNN<Weights<Backend>::EncForwardGRU> forwardRnn_;
    RNN<Weights<Backend>::EncBackwardGRU> backwardRnn_;
};
