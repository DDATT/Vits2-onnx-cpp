#include "phonemize.h"
#include "espeak-ng/speak_lib.h"
#include <algorithm>
// language -> phoneme -> [phoneme, ...]

std::unordered_map<char16_t, int> _symbol_to_id = {
    {u'_', 0}, {u';', 1}, {u':', 2}, {u',', 3}, {u'.', 4}, {u'!', 5}, {u'?', 6}, {u'¡', 7}, {u'¿', 8}, {u'—', 9}, {u'…', 10},
    {u'"', 11}, {u'«', 12}, {u'»', 13}, {u'“', 14}, {u'”', 15}, {u' ', 16}, {u'A', 17}, {u'B', 18}, {u'C', 19}, {u'D', 20},
    {u'E', 21}, {u'F', 22}, {u'G', 23}, {u'H', 24}, {u'I', 25}, {u'J', 26}, {u'K', 27}, {u'L', 28}, {u'M', 29}, {u'N', 30},
    {u'O', 31}, {u'P', 32}, {u'Q', 33}, {u'R', 34}, {u'S', 35}, {u'T', 36}, {u'U', 37}, {u'V', 38}, {u'W', 39}, {u'X', 40},
    {u'Y', 41}, {u'Z', 42}, {u'a', 43}, {u'b', 44}, {u'c', 45}, {u'd', 46}, {u'e', 47}, {u'f', 48}, {u'g', 49}, {u'h', 50}, 
    {u'i', 51}, {u'j', 52}, {u'k', 53}, {u'l', 54}, {u'm', 55}, {u'n', 56}, {u'o', 57}, {u'p', 58}, {u'q', 59}, {u'r', 60}, 
    {u's', 61}, {u't', 62}, {u'u', 63}, {u'v', 64}, {u'w', 65}, {u'x', 66}, {u'y', 67}, {u'z', 68}, {u'ɑ', 69}, {u'ɐ', 70}, 
    {u'ɒ', 71}, {u'æ', 72}, {u'ɓ', 73}, {u'ʙ', 74}, {u'β', 75}, {u'ɔ', 76}, {u'ɕ', 77}, {u'ç', 78}, {u'ɗ', 79}, {u'ɖ', 80}, 
    {u'ð', 81}, {u'ʤ', 82}, {u'ə', 83}, {u'ɘ', 84}, {u'ɚ', 85}, {u'ɛ', 86}, {u'ɜ', 87}, {u'ɝ', 88}, {u'ɞ', 89}, {u'ɟ', 90}, 
    {u'ʄ', 91}, {u'ɡ', 92}, {u'ɠ', 93}, {u'ɢ', 94}, {u'ʛ', 95}, {u'ɦ', 96}, {u'ɧ', 97}, {u'ħ', 98}, {u'ɥ', 99}, {u'ʜ', 100},
    {u'ɨ', 101}, {u'ɪ', 102}, {u'ʝ', 103}, {u'ɭ', 104}, {u'ɬ', 105}, {u'ɫ', 106}, {u'ɮ', 107}, {u'ʟ', 108}, {u'ɱ', 109}, 
    {u'ɯ', 110}, {u'ɰ', 111}, {u'ŋ', 112}, {u'ɳ', 113}, {u'ɲ', 114}, {u'ɴ', 115}, {u'ø', 116}, {u'ɵ', 117}, {u'ɸ', 118}, 
    {u'θ', 119}, {u'œ', 120}, {u'ɶ', 121}, {u'ʘ', 122}, {u'ɹ', 123}, {u'ɺ', 124}, {u'ɾ', 125}, {u'ɻ', 126}, {u'ʀ', 127}, 
    {u'ʁ', 128}, {u'ɽ', 129}, {u'ʂ', 130}, {u'ʃ', 131}, {u'ʈ', 132}, {u'ʧ', 133}, {u'ʉ', 134}, {u'ʊ', 135}, {u'ʋ', 136}, 
    {u'ⱱ', 137}, {u'ʌ', 138}, {u'ɣ', 139}, {u'ɤ', 140}, {u'ʍ', 141}, {u'χ', 142}, {u'ʎ', 143}, {u'ʏ', 144}, {u'ʑ', 145}, 
    {u'ʐ', 146}, {u'ʒ', 147}, {u'ʔ', 148}, {u'ʡ', 149}, {u'ʕ', 150}, {u'ʢ', 151}, {u'ǀ', 152}, {u'ǁ', 153}, {u'ǂ', 154}, 
    {u'ǃ', 155}, {u'ˈ', 156}, {u'ˌ', 157}, {u'ː', 158}, {u'ˑ', 159}, {u'ʼ', 160}, {u'ʴ', 161}, {u'ʰ', 162}, {u'ʱ', 163}, 
    {u'ʲ', 164}, {u'ʷ', 165}, {u'ˠ', 166}, {u'ˤ', 167}, {u'˞', 168}, {u'↓', 169}, {u'↑', 170}, {u'→', 171}, {u'↗', 172},
    {u'↘', 173}, { u'̩', 175}, {u'ᵻ', 177}
}; 
    
std::unordered_map<int, char16_t> _id_to_symbol = {
    {0, u'_'}, {1, u';'}, {2, u':'}, {3, u','}, {4, u'.'}, {5, u'!'}, {6, u'?'}, {7, u'¡'}, {8, u'¿'}, {9, u'—'}, {10, u'…'},
    {11, u'"'}, {12, u'«'}, {13, u'»'}, {14, u'“'}, {15, u'”'}, {16, u' '}, {17, u'A'}, {18, u'B'}, {19, u'C'}, {20, u'D'},
    {21, u'E'}, {22, u'F'}, {23, u'G'}, {24, u'H'}, {25, u'I'}, {26, u'J'}, {27, u'K'}, {28, u'L'}, {29, u'M'}, {30, u'N'},
    {31, u'O'}, {32, u'P'}, {33, u'Q'}, {34, u'R'}, {35, u'S'}, {36, u'T'}, {37, u'U'}, {38, u'V'}, {39, u'W'}, {40, u'X'},
    {41, u'Y'}, {42, u'Z'}, {43, u'a'}, {44, u'b'}, {45, u'c'}, {46, u'd'}, {47, u'e'}, {48, u'f'}, {49, u'g'}, {50, u'h'}, 
    {51, u'i'}, {52, u'j'}, {53, u'k'}, {54, u'l'}, {55, u'm'}, {56, u'n'}, {57, u'o'}, {58, u'p'}, {59, u'q'}, {60, u'r'}, 
    {61, u's'}, {62, u't'}, {63, u'u'}, {64, u'v'}, {65, u'w'}, {66, u'x'}, {67, u'y'}, {68, u'z'}, {69, u'ɑ'}, {70, u'ɐ'}, 
    {71, u'ɒ'}, {72, u'æ'}, {73, u'ɓ'}, {74, u'ʙ'}, {75, u'β'}, {76, u'ɔ'}, {77, u'ɕ'}, {78, u'ç'}, {79, u'ɗ'}, {80, u'ɖ'}, 
    {81, u'ð'}, {82, u'ʤ'}, {83, u'ə'}, {84, u'ɘ'}, {85, u'ɚ'}, {86, u'ɛ'}, {87, u'ɜ'}, {88, u'ɝ'}, {89, u'ɞ'}, {90, u'ɟ'}, 
    {91, u'ʄ'}, {92, u'ɡ'}, {93, u'ɠ'}, {94, u'ɢ'}, {95, u'ʛ'}, {96, u'ɦ'}, {97, u'ɧ'}, {98, u'ħ'}, {99, u'ɥ'}, {100, u'ʜ'}, 
    {101, u'ɨ'}, {102, u'ɪ'}, {103, u'ʝ'}, {104, u'ɭ'}, {105, u'ɬ'}, {106, u'ɫ'}, {107, u'ɮ'}, {108, u'ʟ'}, {109, u'ɱ'}, {110, u'ɯ'}, 
    {111, u'ɰ'}, {112, u'ŋ'}, {113, u'ɳ'}, {114, u'ɲ'}, {115, u'ɴ'}, {116, u'ø'}, {117, u'ɵ'}, {118, u'ɸ'}, {119, u'θ'}, {120, u'œ'}, 
    {121, u'ɶ'}, {122, u'ʘ'}, {123, u'ɹ'}, {124, u'ɺ'}, {125, u'ɾ'}, {126, u'ɻ'}, {127, u'ʀ'}, {128, u'ʁ'}, {129, u'ɽ'}, {130, u'ʂ'}, 
    {131, u'ʃ'}, {132, u'ʈ'}, {133, u'ʧ'}, {134, u'ʉ'}, {135, u'ʊ'}, {136, u'ʋ'}, {137, u'ⱱ'}, {138, u'ʌ'}, {139, u'ɣ'}, {140, u'ɤ'}, 
    {141, u'ʍ'}, {142, u'χ'}, {143, u'ʎ'}, {144, u'ʏ'}, {145, u'ʑ'}, {146, u'ʐ'}, {147, u'ʒ'}, {148, u'ʔ'}, {149, u'ʡ'}, {150, u'ʕ'}, 
    {151, u'ʢ'}, {152, u'ǀ'}, {153, u'ǁ'}, {154, u'ǂ'}, {155, u'ǃ'}, {156, u'ˈ'}, {157, u'ˌ'}, {158, u'ː'}, {159, u'ˑ'}, {160, u'ʼ'}, 
    {161, u'ʴ'}, {162, u'ʰ'}, {163, u'ʱ'}, {164, u'ʲ'}, {165, u'ʷ'}, {166, u'ˠ'}, {167, u'ˤ'}, {168, u'˞'}, {169, u'↓'}, {170, u'↑'}, 
    {171, u'→'}, {172, u'↗'}, {173, u'↘'}, {175, u'̩'}, {177, u'ᵻ'}
};

// convert UTF-8 string to wstring
std::wstring utf8_to_wstring (const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}
std::vector<char16_t> char_vector_from_string(const std::string& str) {
    return std::vector<char16_t>(str.begin(), str.end()); 
}
std::vector<int64_t> text_to_sequence(const std::string& text, eSpeakPhonemeConfig &config) {
    std::vector<int64_t> sequence; 
    std::string clean_text = phonemize_eSpeak(text, config); 
    std::wstring clean_text_wstring= utf8_to_wstring(clean_text);

    // Sử dụng thư viện locale để định dạng chuỗi
    for (char16_t symbol: clean_text_wstring) {
        if (_symbol_to_id.count(symbol) > 0) { // Kiểm tra symbol có trong map không
            int symbol_id = _symbol_to_id[symbol]; 
            sequence.push_back(symbol_id);
        }  // else không có gì thêm vào nếu symbol không hợp lệ
    }
    return sequence;
}

std::string phonemize_eSpeak(std::string text, eSpeakPhonemeConfig &config) {
    auto voice = config.voice;
    int result = espeak_SetVoiceByName(voice.c_str());
    if (result != 0) {
        throw std::runtime_error("Failed to set eSpeak-ng voice");

    }

    // Modified by eSpeak
    std::string textCopy(text);

    std::vector<Phoneme> *sentencePhonemes = nullptr;
    const char *inputTextPointer = textCopy.c_str();
    int terminator = 0;
        std::string res = "";
    while (inputTextPointer != NULL) {
        // Modified espeak-ng API to get access to clause terminator

        std::string clausePhonemes(espeak_TextToPhonemes(
            (const void **)&inputTextPointer,
            /*textmode*/ espeakCHARS_AUTO,
            /*phonememode = IPA*/ 0x02));

        res = res + clausePhonemes;
        res = res + ", ";

    } // while inputTextPointer != NULL
    res.pop_back();
    res.pop_back();
    res+=".";
    std::cout<<res<<std::endl;
    return res;
} /* phonemize_eSpeak */
