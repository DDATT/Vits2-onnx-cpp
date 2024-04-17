#ifndef PHOEMIZE_H_
#define PHOEMIZE_H_

#include <map>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <locale>
#include <codecvt>


typedef char32_t Phoneme;
typedef std::map<Phoneme, std::vector<Phoneme>> PhonemeMap;

struct eSpeakPhonemeConfig {
  std::string voice = "en-us";

  Phoneme period = U'.';      // CLAUSE_PERIOD
  Phoneme comma = U',';       // CLAUSE_COMMA
  Phoneme question = U'?';    // CLAUSE_QUESTION
  Phoneme exclamation = U'!'; // CLAUSE_EXCLAMATION
  Phoneme colon = U':';       // CLAUSE_COLON
  Phoneme semicolon = U';';   // CLAUSE_SEMICOLON
  Phoneme space = U' ';

  // Remove language switch flags like "(en)"
  bool keepLanguageFlags = false;

  std::shared_ptr<PhonemeMap> phonemeMap;
};

// Phonemizes text using espeak-ng.
// Returns phonemes for each sentence as a separate std::vector.
//
// Assumes espeak_Initialize has already been called.
std::string phonemize_eSpeak(std::string text, eSpeakPhonemeConfig &config);
std::vector<int64_t> text_to_sequence(const std::string& text, eSpeakPhonemeConfig &config);
#endif // PHONEMIZE_H_