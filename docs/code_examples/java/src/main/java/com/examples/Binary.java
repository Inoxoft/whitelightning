import ai.onnxruntime.*;
import org.json.JSONObject;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class BinaryClassifier {
    private Map<String, Integer> vocab;
    private List<Float> idf;
    private List<Float> mean;
    private List<Float> scale;
    private OrtSession session;

    public BinaryClassifier(String modelPath, String vocabPath, String scalerPath) throws Exception {
        // Load vocabulary and IDF weights
        String vocabJson = new String(Files.readAllBytes(Paths.get(vocabPath)));
        JSONObject vocabData = new JSONObject(vocabJson);
        this.vocab = new HashMap<>();
        JSONObject vocabObj = vocabData.getJSONObject("vocab");
        for (String key : vocabObj.keySet()) {
            this.vocab.put(key, vocabObj.getInt(key));
        }
        this.idf = new ArrayList<>();
        vocabData.getJSONArray("idf").forEach(item -> this.idf.add(((Number) item).floatValue()));

        // Load scaling parameters
        String scalerJson = new String(Files.readAllBytes(Paths.get(scalerPath)));
        JSONObject scalerData = new JSONObject(scalerJson);
        this.mean = new ArrayList<>();
        this.scale = new ArrayList<>();
        scalerData.getJSONArray("mean").forEach(item -> this.mean.add(((Number) item).floatValue()));
        scalerData.getJSONArray("scale").forEach(item -> this.scale.add(((Number) item).floatValue()));

        // Initialize ONNX Runtime session
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    private float[] preprocessText(String text) {
        float[] vector = new float[5000];
        Map<String, Integer> wordCounts = new HashMap<>();

        // Count word frequencies
        for (String word : text.toLowerCase().split("\\s+")) {
            wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        }

        // Compute TF-IDF
        for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
            Integer idx = vocab.get(entry.getKey());
            if (idx != null) {
                vector[idx] = entry.getValue() * idf.get(idx);
            }
        }

        // Scale features
        for (int i = 0; i < 5000; i++) {
            vector[i] = (vector[i] - mean.get(i)) / scale.get(i);
        }

        return vector;
    }

    public float predict(String text) throws OrtException {
        float[] inputData = preprocessText(text);
        float[][] inputArray = new float[1][5000];
        inputArray[0] = inputData;

        OnnxTensor inputTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), inputArray);
        String inputName = session.getInputNames().iterator().next();
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

        float[][] outputArray = (float[][]) result.get(0).getValue();
        return outputArray[0][0];
    }

    public static void main(String[] args) {
        try {
            BinaryClassifier classifier = new BinaryClassifier(
                "src/main/resources/model.onnx",
                "src/main/resources/model_vocab.json",
                "src/main/resources/model_scaler.json"
            );

            String text = "This is a positive test string";
            float probability = classifier.predict(text);
            System.out.printf("Java ONNX output: Probability = %.4f%n", probability);
            System.out.println("Classification: " + (probability > 0.5 ? "Positive" : "Negative"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
