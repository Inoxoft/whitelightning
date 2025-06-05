#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <onnxruntime_c_api.h>
#include <cJSON.h>

#define INPUT_DIM 5000

const OrtApi* g_ort = NULL;

float* preprocess_text(const char* text, const char* vocab_file, const char* scaler_file, int* num_labels) {
    float* vector = (float*)calloc(INPUT_DIM, sizeof(float));
    if (!vector) return NULL;

    // Load vocab.json
    FILE* vf = fopen(vocab_file, "r");
    fseek(vf, 0, SEEK_END);
    long vsize = ftell(vf);
    fseek(vf, 0, SEEK_SET);
    char* vjson = (char*)malloc(vsize + 1);
    fread(vjson, 1, vsize, vf);
    vjson[vsize] = '\0';
    fclose(vf);
    cJSON* tfidf_data = cJSON_Parse(vjson);
    cJSON* vocab = cJSON_GetObjectItem(tfidf_data, "vocab");
    cJSON* idf = cJSON_GetObjectItem(tfidf_data, "idf");

    // Load scaler.json
    FILE* sf = fopen(scaler_file, "r");
    fseek(sf, 0, SEEK_END);
    long ssize = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    char* sjson = (char*)malloc(ssize + 1);
    fread(sjson, 1, ssize, sf);
    sjson[ssize] = '\0';
    fclose(sf);
    cJSON* scaler_data = cJSON_Parse(sjson);
    cJSON* mean = cJSON_GetObjectItem(scaler_data, "mean");
    cJSON* scale = cJSON_GetObjectItem(scaler_data, "scale");

    // Count labels
    *num_labels = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, scaler_data) {
        if (strcmp(item->string, "mean") && strcmp(item->string, "scale")) (*num_labels)++;
    }

    // TF-IDF
    char* text_lower = strdup(text);
    for (char* p = text_lower; *p; p++) *p = tolower(*p);
    char* word_counts[1000];
    int counts[1000] = {0};
    int word_count = 0;
    char* token = strtok(text_lower, " ");
    while (token) {
        int found = 0;
        for (int i = 0; i < word_count; i++) {
            if (strcmp(word_counts[i], token) == 0) {
                counts[i]++;
                found = 1;
                break;
            }
        }
        if (!found && word_count < 1000) {
            word_counts[word_count] = strdup(token);
            counts[word_count++] = 1;
        }
        token = strtok(NULL, " ");
    }
    for (int i = 0; i < word_count; i++) {
        cJSON* vocab_item = cJSON_GetObjectItem(vocab, word_counts[i]);
        if (vocab_item) {
            int idx = vocab_item->valueint;
            vector[idx] = counts[i] * (float)cJSON_GetArrayItem(idf, idx)->valuedouble;
        }
        free(word_counts[i]);
    }
    free(text_lower);

    // Scale
    for (int i = 0; i < INPUT_DIM; i++) {
        vector[i] = (vector[i] - (float)cJSON_GetArrayItem(mean, i)->valuedouble) /
                    (float)cJSON_GetArrayItem(scale, i)->valuedouble;
    }

    cJSON_Delete(tfidf_data);
    cJSON_Delete(scaler_data);
    free(vjson);
    free(sjson);
    return vector;
}

int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    OrtSession* session;
    g_ort->CreateSession(env, L"model.onnx", session_options, &session);

    const char* text = "This is a positive test string";
    int num_labels;
    float* vector = preprocess_text(text, "model_vocab.json", "model_scaler.json", &num_labels);

    int64_t input_shape[] = {1, INPUT_DIM};
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info);
    OrtValue* input_tensor;
    g_ort->CreateTensorWithDataAsOrtValue(memory_info, vector, INPUT_DIM * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;
    g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);

    float* output_data;
    g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);

    // Load label mapping
    FILE* sf = fopen("model_scaler.json", "r");
    fseek(sf, 0, SEEK_END);
    long ssize = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    char* sjson = (char*)malloc(ssize + 1);
    fread(sjson, 1, ssize, sf);
    sjson[ssize] = '\0';
    fclose(sf);
    cJSON* scaler_data = cJSON_Parse(sjson);
    printf("C Multilabel ONNX output:\n");
    for (int i = 0; i < num_labels; i++) {
        char idx[10];
        snprintf(idx, sizeof(idx), "%d", i);
        cJSON* label = cJSON_GetObjectItem(scaler_data, idx);
        printf("%s: %.4f\n", label ? label->valuestring : idx, output_data[i]);
    }

    cJSON_Delete(scaler_data);
    free(sjson);
    free(vector);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    return 0;
}
