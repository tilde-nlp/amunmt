#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

Sentence::Sentence(){
}

Sentence::Sentence(size_t lineNo, const std::string& line)
: lineNo_(lineNo), line_(line)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  size_t i = 0;
  for(auto&& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");
    auto processed = God::Preprocess(i, lineTokens);
    auto vocab = God::GetSourceVocab(0);

    for(size_t wordCounter = 0; wordCounter < processed.size(); wordCounter++){
	for(size_t wordPartCounter = 0; wordPartCounter < processed[wordCounter].size(); wordPartCounter++) {
	    // if any part of the word is UNK
	    // use non-translatable token instead
	    // this token should be configurable
	    // it depends on what placeholder names are trained into translation model
	    if(vocab[processed[wordCounter][wordPartCounter]] == 1) { //UNK
		    unknownWordIndexes.insert(words.size());
		    unknownWords[words.size()] = lineTokens[wordCounter];
		    processed[wordCounter].clear();
		    processed[wordCounter].push_back("βIDβ");
		    break;
	    }
	}
	for(size_t wordPartCounter = 0; wordPartCounter < processed[wordCounter].size(); wordPartCounter++) {
	    words.push_back(processed[wordCounter][wordPartCounter]);
	}
    }

    words_.push_back(God::GetSourceVocab(i++)(words));
  }
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::GetLine() const {
  return lineNo_;
}

std::string Sentence::GetText() const {
  return line_;
}



