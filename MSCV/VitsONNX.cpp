#include <thread>

#include "VitsONNX.h"





static std::basic_string<ORTCHAR_T> string_to_wstring(const std::string& str)
{
	std::wstring wide_string_arg2 = std::wstring(str.begin(), str.end());
	std::basic_string<ORTCHAR_T> modelFilepath = std::basic_string<ORTCHAR_T>(wide_string_arg2);
	return modelFilepath;
}

VitsONNX::VitsONNX(const std::string& onnx_model_path):m_session(nullptr), m_env(nullptr) 
{
	m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Vits_onnxruntime_cpu");
	std::wstring onnx_model_path_wstr = string_to_wstring(onnx_model_path);
	int cpu_processor_num = std::thread::hardware_concurrency();
	cpu_processor_num /= 2;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(cpu_processor_num);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetLogSeverityLevel(4);

	if (!OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0))
		std::cerr << "";
	m_session = Ort::Session(m_env, onnx_model_path_wstr.c_str(), session_options);
	//PrintModelInfo(m_session);
	this->phonemizer.Init("en-us");
}


VitsONNX::~VitsONNX() {
}

std::map<std::string, int> VitsONNX::getSynthesisConfig() {
	std::map<std::string, int> SynthesisConfig = {
		{"sampleRate", 22050}, {"sampleWidth", 2}, {"channels", 1}
	};
	return SynthesisConfig;
}

void VitsONNX::PrintModelInfo(Ort::Session& session)
{
	// print the number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();
	std::cout << "Number of input node is:" << num_input_nodes << std::endl;
	std::cout << "Number of output node is:" << num_output_nodes << std::endl;

	// print node name
	Ort::AllocatorWithDefaultOptions allocator;
	std::cout << std::endl;
	for (auto i = 0; i < num_input_nodes; i++)
		std::cout << "The input op-name " << i << " is:" << session.GetInputNameAllocated(i, allocator) << std::endl;
	for (auto i = 0; i < num_output_nodes; i++)
		std::cout << "The output op-name " << i << " is:" << session.GetOutputNameAllocated(i, allocator) << std::endl;


	// print input and output dims
	for (auto i = 0; i < num_input_nodes; i++)
	{
		std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "input " << i << " dim is: ";
		for (auto j = 0; j < input_dims.size(); j++)
			std::cout << input_dims[j] << " ";
	}
	for (auto i = 0; i < num_output_nodes; i++)
	{
		std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "output " << i << " dim is: ";
		for (auto j = 0; j < output_dims.size(); j++)
			std::cout << output_dims[j] << " ";
	}
}


std::vector < int16_t > VitsONNX::inference(std::string text_input) {
	std::vector < int16_t > audioBuffer;
	std::vector<int64_t> phonemeIds = this->phonemizer.text_to_sequence(text_input);
	/*for (int i = 0; i < phonemeIds.size(); i++) {
		std::cout << phonemeIds[i] << " ";
	}
	std::cout<<std::endl;*/
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

	std::vector<int64_t> phonemeIdLengths{ (int64_t)phonemeIds.size() };

	std::vector<float> scales{ this->noiseScale,
									this->lengthScale,
										this->noiseW };
	
	std::vector<Ort::Value> inputTensors;

	std::vector<int64_t> phonemeIdsShape{ 1, (int64_t)phonemeIds.size() };
	inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, phonemeIds.data(), phonemeIds.size(), 
															phonemeIdsShape.data(),	phonemeIdsShape.size()));

	std::vector<int64_t> phomemeIdLengthsShape{ (int64_t)phonemeIdLengths.size() };
	inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
															phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size()));

	std::vector<int64_t> scalesShape{ (int64_t)scales.size() };
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(),
															scalesShape.data(), scalesShape.size()));

	std::vector<int64_t> speakerId{(int64_t)this->speakerId.value_or(0) };
	std::vector<int64_t> speakerIdShape{ (int64_t)speakerId.size() };


	if (this->speakerId) {
		inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, speakerId.data(), speakerId.size(),
																speakerIdShape.data(), speakerIdShape.size()));
	}

	// From export_onnx.py
	std::array<const char*, 4> inputNames = { "input", "input_lengths", "scales",
											  "sid" };
	std::array<const char*, 1> outputNames = { "output" };

	// Infer
	auto startTime = std::chrono::steady_clock::now();

	auto outputTensors = m_session.Run(
		Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(),
		inputTensors.size(), outputNames.data(), outputNames.size());
	auto endTime = std::chrono::steady_clock::now();
	

	if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor())) {
		throw std::runtime_error("Invalid output tensors");
	}
	auto inferDuration = std::chrono::duration<double>(endTime - startTime);
	std::cout << "infer time: " << inferDuration.count() << std::endl;

	const float* audio = outputTensors.front().GetTensorData<float>();
	auto audioShape = outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();
	int64_t audioCount = audioShape[audioShape.size() - 1];

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

	return audioBuffer;
}