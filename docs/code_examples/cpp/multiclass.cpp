#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>

using json = nlohmann::json;

std::vector<int32_t> preprocess_text(const std::string& text, const std::string& tokenizer_file) {
    std::vector<int32_t> vector(30, 0);

    std::ifstream tf(tokenizer_file);
    json tokenizer; tf >> tokenizer;

    std::string text_lower = text;
    std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
    std::vector<std::string> words;
    size_t start = 0, end;
    while ((end = text_lower.find(' ', start)) != std::string::npos) {
        if (end > start) words.push_back(text_lower.substr(start, end - start));
        start = end + 1;
    }
    if (start < text_lower.length()) words.push_back(text_lower.substr(start));

    for (size_t i = 0; i < std::min(words.size(), size_t(30)); i++) {
        auto it = tokenizer.find(words[i]);
        if (it != tokenizer.end()) {
            vector[i] = it->get();
        } else {
            auto oov = tokenizer.find("");
            vector[i] = oov != tokenizer.end() ? oov->get() : 1;
        }
    }
    return vector;
}

int main() {
    std::string text = "I hate you";
    auto vector = preprocess_text(text, "hate_speech(English)/vocab.json");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "hate_speech(English)/model.onnx", session_options);

    std::vector<int64_t> input_shape = {1, 30};
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, vector.data(), vector.size(),
                                                     input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"sequential"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                                    output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::ifstream lf("hate_speech(English)/scaler.json");
    json label_map; lf >> label_map;

    auto max_it = std::max_element(output_data, output_data + output_size);
    int predicted_idx = std::distance(output_data, max_it);
    std::string label = label_map[std::to_string(predicted_idx)];
    float score = *max_it;

    std::cout << "C++ ONNX output: " << label << " (Score: " << std::fixed << std::setprecision(4)
              << score << ")" << std::endl;
    return 0;
}
