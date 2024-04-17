#include <iostream>
#include <fstream>
#include "VitsONNX.h"
#include "wavfile.h"
#include <conio.h> 

int main() {
	const std::string& model_path = "vits2_model.onnx";
	std::ofstream audioFile("test.wav", std::ios::binary);

	while (true){
	std::string text = "";
	std::cout << "Text input: ";
	std::getline(std::cin, text);
	std::cout << std::endl;

	if (text == "quit") {
		break;
	}

	VitsONNX vitsmodel(model_path);
	std::map<std::string, int> synthesisConfig = vitsmodel.getSynthesisConfig();

	std::vector < int16_t > audio = vitsmodel.inference(text);
	//std::cout << synthesisConfig["sampleWidth"];
	writeWavHeader(synthesisConfig["sampleRate"], synthesisConfig["sampleWidth"], synthesisConfig["channels"], (int32_t)audio.size(), audioFile);
	audioFile.write((const char*)audio.data(), sizeof(int16_t) * audio.size());
	}
	return 0;
}