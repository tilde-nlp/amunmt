#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

namespace amunmt {

Sentence::Sentence(const God &god, size_t vLineNum, const std::string& line)
  : lineNum_(vLineNum)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  if (tabs.size() == 0) {
    tabs.push_back("");
  }

  size_t maxLength = god.Get<size_t>("max-length");
  size_t i = 0;
  for (auto& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");

    if (maxLength && lineTokens.size() > maxLength) {
      lineTokens.resize(maxLength);
    }

    std::vector<std::vector<std::string>> lineFactors;
    for (const std::string& token : lineTokens) {
      std::vector<std::string> wordFactors;
      Split(token, wordFactors, "|");
      for (std::string fact : wordFactors) {
        std::cerr << " " << fact;
      }
      std::cerr << std::endl;
      lineFactors.push_back(wordFactors);
    }

    std::cerr << "------------" << std::endl;
    auto processed = god.Preprocess(i, lineFactors);
    for (auto word : processed) {
      for (std::string fact : word) {
        std::cerr << " " << fact;
      }
      std::cerr << std::endl;
    }
    // TODO: refactor the rest of the code to use the structured
    // vector of vector of factor representation everywhere.
    // Currently we merge it into a single (periodic) vector
    // of factors, i.e., w11 f11 f12 f13 w21 f21 f22 f23 ...
    // for compatability reasons
    // std::vector<std::string> merged;
    words_.push_back(std::vector<Word>());
    for (const std::vector<std::string>& wordFactors : processed) {
      size_t vocabIdx = 0;
      for (const std::string& factor : wordFactors) {
        words_.back().push_back(god.GetSourceVocab(i, vocabIdx++)[factor]);
      }
      // merged.insert(merged.end(), wordFactors.begin(), wordFactors.end());
      // words_.push_back(god.GetSourceVocab(i++)(merged));
    }
    std::cerr << "------------" << std::endl;
    for (auto fact : words_.back()) {
      std::cerr << " " << fact;
      std::cerr << std::endl;
    }
    i++;
  }
}

Sentence::Sentence(const God &god, size_t lineNum, const std::vector<std::string>& words)
  : lineNum_(lineNum) {
    auto processed = god.Preprocess(0, words);
    words_.push_back(god.GetSourceVocab(0)(processed));
}

Sentence::Sentence(God&, size_t lineNum, const std::vector<size_t>& words)
  : lineNum_(lineNum) {
    words_.push_back(words);
}


size_t Sentence::GetLineNum() const {
  return lineNum_;
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::size(size_t index) const {
  return words_[index].size();
}


}

