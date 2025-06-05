#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <string>

using json = nlohmann::json;

std::vector<float> preprocess_text(const std::string& text, const std::string& vocab_file, const std::string& scaler_file) {
    std::vector<float> vector(5000, 0.0f);

    // Load vocab.json
    std::ifstream vf(vocab_file);
    json tfidf_data; vf >> tfidf_data;
    auto vocab = tfidf_data["vocab"];
    std::vector<float> idf = tfidf_data["idf"];

    // Load scaler.json
    std::ifstream sf(scaler_file);
    json scaler_data; sf >> scaler_data;
    std::vector<float> mean = scaler_data["mean"];
    std::vector<float> scale = scaler_data["scale"];

    // TF-IDF
    std::string text_lower = text;
    std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
    std::map<std::string, int> word_counts;
    size_t start = 0, end;
    while ((end = text_lower.find(' ', start)) != std::string::npos) {
        if (end > start) word_counts[text_lower.substr(start, end - start)]++;
        start = end + 1;
    }
    if (start < text_lower.length()) word_counts[text_lower.substr(start)]++;
    for (const auto& [word, count] : word_counts) {
        if (vocab.contains(word)) {
            vector[vocab[word]] = count * idf[vocab[word]];
        }
    }

    // Scale
    for (size_t i = 0; i < 5000; i++) {
        vector[i] = (vector[i] - mean[i]) / scale[i];
    }
    return vector;
}

int main() {
    std::string text = "This is a positive test string";
    auto vector = preprocess_text(text, "model_vocab.json", "model_scaler.json");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "model.onnx", session_options);

    std::vector<int64_t> input_shape = {1, 5000};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, vector.data(), vector.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {"float_input"};
    std::vector<const char*> output_names = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int64_t* output_shape;
    size_t shape_count;
    output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(output_shape, shape_count);

    // Load label mapping
    std::ifstream sf("model_scaler.json");
    json scaler_data; sf >> scaler_data;
    std::cout << "C++ Multilabel ONNX output:\n";
    for (size_t i = 0; i < output_shape[1]; i++) {
        std::string idx = std::to_string(i);
        std::string label = scaler_data.contains(idx) ? scaler_data[idx].get<std::string>() : idx;
        std::cout << label << ": " << output_data[i] << "\n";
    }

    return 0;
}
