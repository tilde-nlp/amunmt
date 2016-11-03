#pragma once

#include <vector>
#include <unordered_set>
#include <regex>
#include <algorithm>

#include "common/god.h"
#include "common/history.h"
#include "common/utils.h"
#include "common/vocab.h"
#include "common/soft_alignment.h"
#include "common/utils.h"

template <class OStream>
void Printer(const History& history, size_t lineNo, OStream& out) {
  if(God::Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = God::GetScorerNames();
    const NBestList &nbl = history.NBest(God::Get<size_t>("beam-size"));
    if(God::Get<bool>("wipo")) {
      out << "OUT: " << nbl.size() << std::endl;
    }
    for(size_t i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      if(God::Get<bool>("wipo"))
        out << "OUT: ";
      out << lineNo << " ||| " << Join(God::Postprocess(God::GetTargetVocab()(words))) << " |||";
      for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
        out << " " << scorerNames[j] << "= " << hypo->GetCostBreakdown()[j];
      }
      if(God::Get<bool>("normalize")) {
        out << " ||| " << hypo->GetCost() / words.size() << std::endl;
      }
      else {
        out << " ||| " << hypo->GetCost() << std::endl;
      }
    }
  }
  else
  {
    std::vector<std::string> sourceWordList = history.sentence.words;
    std::string source = Join(sourceWordList);
    LOG(progress) << "Source: " << source;

    std::vector<std::string> targetWordList = God::GetTargetVocab()(history.Top().first);
    std::string translation = Join(targetWordList);
    LOG(progress) << "Translation: " << translation;

    auto last = history.Top().second;
    std::vector<SoftAlignment> aligns;
    while (last->GetPrevHyp().get() != nullptr) {
      aligns.push_back(*(last->GetAlignment(0)));
      last = last->GetPrevHyp();
    }
    std::reverse(aligns.begin(), aligns.end());

    auto unknownWords = history.sentence.unknownWords;
    auto unknownSourceWordIndexes = history.sentence.unknownWordIndexes;

    const std::string placeholder = "βIDβ";
    std::set<size_t> unknownTargetWordIndexes;
    for(size_t targetWordIndex; targetWordIndex < targetWordList.size(); targetWordIndex++) {
      if(targetWordList[targetWordIndex].compare(placeholder) == 0) {
        unknownTargetWordIndexes.insert(targetWordIndex);
      } 
    }

    while (unknownTargetWordIndexes.size() > 0 && unknownSourceWordIndexes.size() > 0) {
      float max = 0;
      size_t maxSourceWordIndex, maxTargetWordIndex;
      size_t unknownWordCounter = 0;
      for(auto sourceWordIndex : unknownSourceWordIndexes) {
        for(auto targetWordIndex : unknownTargetWordIndexes) {
          if(aligns[targetWordIndex][sourceWordIndex] > max) {
            max = aligns[targetWordIndex][sourceWordIndex];
            maxSourceWordIndex = sourceWordIndex;
            maxTargetWordIndex = targetWordIndex;
          }
        }
      }
      std::cerr << "Unknown word placeholder, source index: " << maxSourceWordIndex << ", target index: " << maxTargetWordIndex <<  ", text: " << unknownWords[maxSourceWordIndex] << std::endl;
      unknownSourceWordIndexes.erase(maxSourceWordIndex);
      unknownTargetWordIndexes.erase(maxTargetWordIndex);
      targetWordList[maxTargetWordIndex] = unknownWords[maxSourceWordIndex];
    }

    for(auto sourceWordIndex : unknownSourceWordIndexes) {
      std::cerr << "Could not find placeholder in translation, source index: " << sourceWordIndex << ", text: " << unknownWords[sourceWordIndex] << std::endl;      
    }

    std::string postprocessedTranslation = Join(God::Postprocess(targetWordList));
    out << postprocessedTranslation;

    if (God::Get<bool>("return-alignment")) {

      std::stringstream ss;

      if (God::Has("bpe")) {
        const std::string seperator = "@@";
        std::unordered_set<size_t> splitSourceWords;
        for(size_t counter=0; counter< sourceWordList.size(); counter++) {
          if(EndsWith(sourceWordList[counter], seperator)){
            splitSourceWords.insert(counter);
          }
        }
        std::unordered_set<size_t> splitTargetWords;
        for(size_t counter=0; counter< targetWordList.size(); counter++) {
          if(EndsWith(targetWordList[counter], seperator)){
            splitTargetWords.insert(counter);
          }
        }
        size_t targetWordCount = targetWordList.size() + 1 - splitTargetWords.size();
        size_t sourceWordCount = sourceWordList.size() + 1 - splitSourceWords.size();
	float compressedAlignment[targetWordCount][sourceWordCount];
	memset(compressedAlignment, 0,  sizeof compressedAlignment);

        size_t targetCounter = 0;
        bool lastTargetJoinable = false;

        for(size_t x = 0; x< aligns.size(); x++) {
          size_t sourceCounter = 0;
          for (size_t y = 0; y< aligns[x].size(); y++) {
            compressedAlignment[targetCounter][sourceCounter] += aligns[x][y];
	    if(splitSourceWords.find(y) == splitSourceWords.end()){
              if(lastTargetJoinable){
                compressedAlignment[targetCounter][sourceCounter] /= 2;
              }
	      sourceCounter++;
            }
          }
          if(splitTargetWords.find(x) == splitTargetWords.end()){
            targetCounter++;
            lastTargetJoinable = false;
          } else {
            lastTargetJoinable = true;
          }
        }

        for(size_t x = 0; x < targetWordCount; x++){
            ss << "(";
          for(size_t y = 0; y < sourceWordCount; y++){
            ss << compressedAlignment[x][y] << " ";
          }
            ss << ") | ";
        }

      } else {
    
        for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
          ss << "(";
          for (auto sIt = it->begin(); sIt != it->end(); ++sIt) {
            ss << *sIt << " ";
          }
          ss << ") | ";
        }
      }
      out << " ||| " << ss.str();
    }
    out << std::endl;
  }
}
