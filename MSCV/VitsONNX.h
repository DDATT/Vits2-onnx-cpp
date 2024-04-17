#pragma once
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "phonemize.h"

#include <any>
#include <map>
#include <optional>
#include <algorithm>


struct SynthesisResult {
	double inferSeconds = 0;
	double audioSeconds = 0;
	double realTimeFactor = 0;
};

class VitsONNX
{
public:
	VitsONNX() = delete;
	VitsONNX(const std::string& onnx_model_path);
	virtual ~VitsONNX();

	std::map<std::string, int> getSynthesisConfig();
	std::vector < int16_t > inference(std::string text_input);
	
private:
	void PrintModelInfo(Ort::Session& session);
	std::vector<std::u32string> Phonemes;
	std::vector<int32_t> PhonemeIDs;
private:
	Ort::Env m_env;
	Ort::Session m_session;

	std::optional<int64_t> speakerId;
	float noiseScale = 0.667f;
	float lengthScale = 1.0f;
	float noiseW = 0.8f;

	// Audio settings
	int sampleRate = 22050;
	int sampleWidth = 2; // 16-bit
	int channels = 1;    // mono

	const float MAX_WAV_VALUE = 32767.0f;

	PhonemizerEngine phonemizer;
};

