#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "onnxruntime-osx-universal2-1.22.0/include/onnxruntime_c_api.h"
#include <cjson/cJSON.h>

const OrtApi* g_ort = NULL;

float* preprocess_text(const char* text, const char* vocab_file, const char* scaler_file) {
    float* vector = calloc(5000, sizeof(float));
    if (!vector) return NULL;

    FILE* f = fopen(vocab_file, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_str = malloc(len + 1);
    fread(json_str, 1, len, f);
    json_str[len] = 0;
    fclose(f);

    cJSON* tfidf_data = cJSON_Parse(json_str);
    if (!tfidf_data) {
        free(json_str);
        return NULL;
    }

    cJSON* vocab = cJSON_GetObjectItem(tfidf_data, "vocab");
    cJSON* idf = cJSON_GetObjectItem(tfidf_data, "idf");
    if (!vocab || !idf) {
        free(json_str);
        cJSON_Delete(tfidf_data);
        return NULL;
    }

    f = fopen(scaler_file, "r");
    if (!f) {
        free(json_str);
        cJSON_Delete(tfidf_data);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* scaler_str = malloc(len + 1);
    fread(scaler_str, 1, len, f);
    scaler_str[len] = 0;
    fclose(f);

    cJSON* scaler_data = cJSON_Parse(scaler_str);
    if (!scaler_data) {
        free(json_str);
        free(scaler_str);
        cJSON_Delete(tfidf_data);
        return NULL;
    }

    cJSON* mean = cJSON_GetObjectItem(scaler_data, "mean");
    cJSON* scale = cJSON_GetObjectItem(scaler_data, "scale");
    if (!mean || !scale) {
        free(json_str);
        free(scaler_str);
        cJSON_Delete(tfidf_data);
        cJSON_Delete(scaler_data);
        return NULL;
    }

    char* text_copy = strdup(text);
    for (char* p = text_copy; *p; p++) *p = tolower(*p);

    char* word = strtok(text_copy, " \t\n");
    while (word) {
        cJSON* idx = cJSON_GetObjectItem(vocab, word);
        if (idx) {
            int i = idx->valueint;
            if (i < 5000) {
                vector[i] += cJSON_GetArrayItem(idf, i)->valuedouble;
            }
        }
        word = strtok(NULL, " \t\n");
    }

    for (int i = 0; i < 5000; i++) {
        vector[i] = (vector[i] - cJSON_GetArrayItem(mean, i)->valuedouble) /
                    cJSON_GetArrayItem(scale, i)->valuedouble;
    }

    free(text_copy);
    free(json_str);
    free(scaler_str);
    cJSON_Delete(tfidf_data);
    cJSON_Delete(scaler_data);
    return vector;
}

int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return 1;

    const char* text = "Earn $5000 a week from home — no experience required!";
    float* vector = preprocess_text(text, "vocab.json", "scaler.json");
    if (!vector) return 1;

    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status) return 1;

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status) return 1;

    OrtSession* session;
    status = g_ort->CreateSession(env, "model.onnx", session_options, &session);
    if (status) return 1;

    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) return 1;

    int64_t input_shape[] = {1, 5000};
    OrtValue* input_tensor;
    status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, vector, 5000 * sizeof(float),
                                                 input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                 &input_tensor);
    if (status) return 1;

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;
    status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
                       output_names, 1, &output_tensor);
    if (status) return 1;

    float* output_data;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) return 1;

    printf("C ONNX output: %s (Score: %.4f)\n",
           output_data[0] > 0.5 ? "Spam" : "Not Spam",
           output_data[0]);

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    free(vector);
    return 0;
  }
