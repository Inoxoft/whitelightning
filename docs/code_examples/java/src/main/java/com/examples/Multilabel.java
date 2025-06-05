import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class MultilabelInference {
    public static float[] preprocessText(String text, String vocabFile, String scalerFile) throws Exception {
        float[] vector = new float[5000];

        // Load vocab.json
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> tfidfData = mapper.readValue(new File(vocabFile), Map.class);
        Map<String, Integer> vocab = (Map<String, Integer>) tfidfData.get("vocab");
        float[] idf = mapper.convertValue(tfidfData.get("idf"), float[].class);

        // Load scaler.json
        Map<String, Object> scalerData = mapper.readValue(new File(scalerFile), Map.class);
        float[] mean = mapper.convertValue(scalerData.get("mean"), float[].class);
        float[] scale = mapper.convertValue(scalerData.get("scale"), float[].class);

        // TF-IDF
        String textLower = text.toLowerCase();
        Map<String, Integer> wordCounts = new HashMap<>();
        for (String word : textLower.split("\\s+")) {
            wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        }
        for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
            Integer idx = vocab.get(entry.getKey());
            if (idx != null) {
                vector[idx] = entry.getValue() * idf[idx];
            }
        }

        // Scale
        for (int i = 0; i < 5000; i++) {
            vector[i] = (vector[i] - mean[i]) / scale[i];
        }
        return vector;
    }

    public static void main(String[] args) throws Exception {
        String text = "This is a positive test string";
        float[] vector = preprocessText(text, "model_vocab.json", "model_scaler.json");

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("model.onnx", sessionOptions);

        long[] inputShape = {1, 5000};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, vector, inputShape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("float_input", inputTensor);
        OrtSession.Result results = session.run(inputs);

        float[] outputData = ((float[][]) results.get(0).getValue())[0];

        // Load label mapping
        ObjectMapper mapper = new ObjectMapper();
        Map<String, String> scalerData = mapper.readValue(new File("model_scaler.json"), Map.class);
        System.out.println("Java Multilabel ONNX output:");
        for (int i = 0; i < outputData.length; i++) {
            String idx = String.valueOf(i);
            String label = scalerData.getOrDefault(idx, idx);
            System.out.printf("%s: %.4f%n", label, outputData[i]);
        }

        results.close();
        inputTensor.close();
        session.close();
        sessionOptions.close();
        env.close();
    }
}
