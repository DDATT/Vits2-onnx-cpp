#include "phonemize.h"


// convert UTF-8 string to wstring
static std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}


PhonemizerEngine::PhonemizerEngine() {
}
void PhonemizerEngine::Init(std::string voice) {
    std::string espeak_data = "espeak-ng/share/espeak-ng-data/";
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0,
        espeak_data.c_str(), 0);
    if (result < 0) {
        throw std::runtime_error("Failed to initialize eSpeak");
    }
    int setvoice = espeak_SetVoiceByName(voice.c_str());
    if (setvoice != 0) {
        throw std::runtime_error("Failed to set eSpeak-ng voice");

    }
}
PhonemizerEngine::~PhonemizerEngine() {

}

std::vector<int64_t> PhonemizerEngine::text_to_sequence(const std::string& text) {
    std::vector<int64_t> sequence;
    std::string clean_text = phonemize_eSpeak(text);
    std::wstring clean_text_wstring = utf8_to_wstring(clean_text);

    for (char16_t symbol : clean_text_wstring) {
        if (this->_symbol_to_id.count(symbol) > 0) {
            int symbol_id = _symbol_to_id[symbol];
            sequence.push_back(symbol_id);
        } 
    }
    return sequence;
}

std::string PhonemizerEngine::phonemize_eSpeak(std::string text) {
    std::string textCopy(text);

    std::vector<char32_t>* sentencePhonemes = nullptr;
    const char* inputTextPointer = textCopy.c_str();
    int terminator = 0;
    std::string res = "";
    while (inputTextPointer != NULL) {

        std::string clausePhonemes(espeak_TextToPhonemes(
            (const void**)&inputTextPointer,
            /*textmode*/ espeakCHARS_AUTO,
            /*phonememode = IPA*/ 0x02));

        res = res + clausePhonemes;
        res = res + ", ";

    } // while inputTextPointer != NULL
    res.pop_back();
    res.pop_back();
    res += ".";
    return res;
} /* phonemize_eSpeak */