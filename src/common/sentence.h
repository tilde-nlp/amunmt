#pragma once
#include <vector>
#include <string>
#include "types.h"
#include <set>
#include <map>

class Sentence {
  public:
    Sentence();
    Sentence(size_t lineNo, const std::string& line);
    
    const Words& GetWords(size_t index = 0) const;
    
    size_t GetLine() const;

    std::string GetText() const;

    std::set<size_t> unknownWordIndexes;
    std::map<size_t, std::string> unknownWords;
    std::vector<std::string> words;
    
  private:
    std::vector<Words> words_;
    size_t lineNo_;
    std::string line_;
};

