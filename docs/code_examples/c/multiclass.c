#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "onnxruntime-osx-universal2-1.22.0/include/onnxruntime_c_api.h"
#include <cjson/cJSON.h>

const OrtApi* g_ort = NULL;

int32_t* preprocess_text(const char* text, const char* tokenizer_file) {
    int32_t* vector = calloc(30, sizeof(int32_t));

    FILE* f = fopen(tokenizer_file, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_str = malloc(len + 1);
    fread(json_str, 1, len, f);
    json_str[len] = 0;
    fclose(f);

    cJSON* tokenizer = cJSON_Parse(json_str);
    if (!tokenizer) {
        free(json_str);
        return NULL;
    }

    char* text_copy = strdup(text);
    for (char* p = text_copy; *p; p++) *p = tolower(*p);

    char* word = strtok(text_copy, " \t\n");
    int idx = 0;
    while (word && idx < 30) {
        cJSON* token = cJSON_GetObjectItem(tokenizer, word);
        vector[idx++] = token ? token->valueint : (cJSON_GetObjectItem(tokenizer, "<OOV>") ? cJSON_GetObjectItem(tokenizer, "<OOV>")->valueint : 1);
        word = strtok(NULL, " \t\n");
    }

    free(text_copy);
    free(json_str);
    cJSON_Delete(tokenizer);
    return vector;
}

int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return 1;

    const char* text = "That's really thoughtful feedback — thank you.";
    int32_t* vector = preprocess_text(text, "hate_speech(English)/vocab.json");
    if (!vector) return 1;

    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status) return 1;

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status) return 1;

    OrtSession* session;
    status = g_ort->CreateSession(env, "hate_speech(English)/model.onnx", session_options, &session);
    if (status) return 1;

    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) return 1;

    int64_t input_shape[] = {1, 30};
    OrtValue* input_tensor;
    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, vector, 30 * sizeof(int32_t), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_tensor);
    if (status) return 1;

    const char* input_names[] = {"input"};
    const char* output_names[] = {"sequential"};
    OrtValue* output_tensor = NULL;
    status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    if (status) return 1;

    float* output_data;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) return 1;

    FILE* f = fopen("hate_speech(English)/scaler.json", "r");
    if (!f) return 1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_str = malloc(len + 1);
    fread(json_str, 1, len, f);
    json_str[len] = 0;
    fclose(f);

    cJSON* label_map = cJSON_Parse(json_str);
    if (!label_map) {
        free(json_str);
        return 1;
    }

    int predicted_idx = 0;
    float max_prob = output_data[0];
    int num_classes = cJSON_GetArraySize(label_map);
    for (int i = 1; i < num_classes; i++) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            predicted_idx = i;
        }
    }

    char idx_str[16];
    snprintf(idx_str, sizeof(idx_str), "%d", predicted_idx);
    cJSON* label = cJSON_GetObjectItem(label_map, idx_str);
    if (!label) return 1;

    printf("C ONNX output: %s (Score: %.4f)\n", label->valuestring, max_prob);

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    free(vector);
    free(json_str);
    cJSON_Delete(label_map);

    return 0;
}
