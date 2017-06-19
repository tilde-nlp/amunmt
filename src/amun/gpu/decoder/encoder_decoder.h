#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"
#include "common/base_best_hyps.h"
#include "common/threadpool.h"
#include "gpu/types-gpu.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/handles.h"


namespace amunmt {
namespace GPU {

class EncoderDecoderState;
class Encoder;
class Decoder;
class Weights;

////////////////////////////////////////////
class EncoderDecoder : public Scorer {
  private:
    typedef EncoderDecoderState EDState;

  public:
    EncoderDecoder(const God &god,
    			   const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual ~EncoderDecoder();

    virtual void Decode(const State& in, State& out, const std::vector<uint>& beamSizes);

    virtual State* NewState() const;

    virtual void BeginSentenceState(State& state, size_t batchSize=1);

    virtual void SetSource(const Sentences& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);

    mblas::Matrix& GetAttention();
    virtual BaseMatrix& GetProbs();

    size_t GetVocabSize() const;

    void Filter(const std::vector<size_t>& filterIds);

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;
    DeviceVector<uint> indices_;
    DeviceVector<uint> batchMapping_;
      // set in Encoder::GetContext() to length (maxSentenceLength * batchSize). 1 if it's a word, 0 otherwise

    std::unique_ptr<mblas::Matrix> SourceContext_;

    EncoderDecoder(const EncoderDecoder&) = delete;
};

////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const EncoderDecoderLoader&) = delete;
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);
    virtual ~EncoderDecoderLoader();
    
    virtual void Load(const God &god);

    virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
    virtual BestHypsBasePtr GetBestHyps(const God &god) const;

  private:
    std::vector<std::unique_ptr<Weights>> weights_; // MUST be indexed by gpu id. eg. weights_[2] is for gpu2
};

}
}

