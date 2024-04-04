#include <iostream>
#include <chrono>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <fstream>

#include "wavfile.hpp"
#include <onnxruntime_cxx_api.h>
#include "espeak-ng/speak_lib.h"
#include "phonemize.h"

const std::string instanceName{"vits"};

typedef int64_t SpeakerId;

const float MAX_WAV_VALUE = 32767.0f;

struct ModelSession {
    Ort::Session onnx;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions options;
    Ort::Env env;

    ModelSession() : onnx(nullptr){};
};

struct SynthesisResult {
  double inferSeconds;
  double audioSeconds;
  double realTimeFactor;
};


struct SynthesisConfig {
  // VITS inference settings
  float noiseScale = 0.667f;
  float lengthScale = 1.0f;
  float noiseW = 0.8f;

  // Audio settings
  int sampleRate = 22050;
  int sampleWidth = 2; // 16-bit
  int channels = 1;    // mono

  // Speaker id from 0 to numSpeakers - 1
  std::optional<SpeakerId> speakerId;

  // Extra silence
  float sentenceSilenceSeconds = 0.2f;
  std::optional<std::map<char32_t, float>> phonemeSilenceSeconds;
};




void loadModel(std::string modelPath, ModelSession &session, bool useCuda) {
    session.env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                            instanceName.c_str());
    session.env.DisableTelemetryEvents();

    if (useCuda) {
        // Use CUDA provider
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        session.options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Slows down performance by ~2x
    // session.options.SetIntraOpNumThreads(1);

    // Roughly doubles load time for no visible inference benefit
    // session.options.SetGraphOptimizationLevel(
    //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session.options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Slows down performance very slightly
    // session.options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    session.options.DisableCpuMemArena();
    session.options.DisableMemPattern();
    session.options.DisableProfiling();


    #ifdef _WIN32
    auto modelPathW = std::wstring(modelPath.begin(), modelPath.end());
    auto modelPathStr = modelPathW.c_str();
    #else
    auto modelPathStr = modelPath.c_str();
    #endif

    session.onnx = Ort::Session(session.env, modelPathStr, session.options);
}

void Synthesize(std::vector<int64_t> &phonemeIds,
                SynthesisConfig &synthesisConfig, ModelSession &session,
                std::vector<int16_t> &audioBuffer, SynthesisResult &result){

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
                        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    std::vector<int64_t> phonemeIdLengths{(int64_t)phonemeIds.size()};
    std::vector<float> scales{synthesisConfig.noiseScale,
                                synthesisConfig.lengthScale,
                                synthesisConfig.noiseW};

    std::vector<Ort::Value> inputTensors;
    std::vector<int64_t> phonemeIdsShape{1, (int64_t)phonemeIds.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIds.data(), phonemeIds.size(), phonemeIdsShape.data(),
        phonemeIdsShape.size()));

    std::vector<int64_t> phomemeIdLengthsShape{(int64_t)phonemeIdLengths.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
        phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size()));

    std::vector<int64_t> scalesShape{(int64_t)scales.size()};
    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(),
                                        scalesShape.data(), scalesShape.size()));
    // Add speaker id.
    // NOTE: These must be kept outside the "if" below to avoid being deallocated.
    std::vector<int64_t> speakerId{(int64_t)synthesisConfig.speakerId.value_or(0)};
    std::vector<int64_t> speakerIdShape{(int64_t)speakerId.size()};

    if (synthesisConfig.speakerId) {
        inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, speakerId.data(), speakerId.size(), speakerIdShape.data(),
            speakerIdShape.size()));
    }
    // From export_onnx.py
    std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales",
                                                "sid"};
    std::array<const char *, 1> outputNames = {"output"};

    // Infer
    auto startTime = std::chrono::steady_clock::now();
    auto outputTensors = session.onnx.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
        inputTensors.size(), outputNames.data(), outputNames.size());
    auto endTime = std::chrono::steady_clock::now();
    auto inferDuration = std::chrono::duration<double>(endTime - startTime);
    std::cout<<"Infertime: " << inferDuration.count() <<std::endl;
    if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor())) {
        throw std::runtime_error("Invalid output tensors");
    }

    const float *audio = outputTensors.front().GetTensorData<float>();
    auto audioShape =
        outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioCount = audioShape[audioShape.size() - 1];

    result.audioSeconds = (double)audioCount / (double)synthesisConfig.sampleRate;
    result.realTimeFactor = 0.0;
    if (result.audioSeconds > 0) {
        result.realTimeFactor = result.inferSeconds / result.audioSeconds;
    }

    // Get max audio value for scaling
    float maxAudioValue = 0.01f;
    for (int64_t i = 0; i < audioCount; i++) {
        float audioValue = abs(audio[i]);
        if (audioValue > maxAudioValue) {
        maxAudioValue = audioValue;
        }
    }

    // We know the size up front
    audioBuffer.reserve(audioCount);

    // Scale audio to fill range and convert to int16
    float audioScale = (MAX_WAV_VALUE / std::max(0.01f, maxAudioValue));
    for (int64_t i = 0; i < audioCount; i++) {
        int16_t intAudioValue = static_cast<int16_t>(
            std::clamp(audio[i] * audioScale,
                    static_cast<float>(std::numeric_limits<int16_t>::min()),
                    static_cast<float>(std::numeric_limits<int16_t>::max())));

        audioBuffer.push_back(intAudioValue);
    }

    // Clean up
    for (std::size_t i = 0; i < outputTensors.size(); i++) {
        Ort::detail::OrtRelease(outputTensors[i].release());
    }

    for (std::size_t i = 0; i < inputTensors.size(); i++) {
        Ort::detail::OrtRelease(inputTensors[i].release());
    }
}

int main(){
    // std::cout<<"Hello world!"<<std::endl;
    std::string model_path = "vits2_model.onnx";
    std::string text = "";
    ModelSession session;
    eSpeakPhonemeConfig eSpeakConfig;
    SynthesisConfig syncfig;
    SynthesisResult res;
    std::vector<int16_t> audio;
    std::vector<std::vector<Phoneme>> phonemes;
    std::string espeak_data = "espeak-ng/share/espeak-ng-data/";
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0,
                        espeak_data.c_str(), 0);
    if (result < 0) {
      throw std::runtime_error("Failed to initialize eSpeak");
    }

    std::ofstream audioFile("test.wav", std::ios::binary);
    loadModel(model_path, session, false);
    // std::string test = phonemize_eSpeak(text, eSpeakConfig);
    std::vector<int64_t> Phonemeid = text_to_sequence(text, eSpeakConfig);
    
    Synthesize(Phonemeid, syncfig, session, audio, res);
    
    writeWavHeader(syncfig.sampleRate, syncfig.sampleWidth, syncfig.channels, (int32_t)audio.size(),audioFile);
    audioFile.write((const char *)audio.data(), sizeof(int16_t) * audio.size());
    return 0;
}
