import ai.onnxruntime.*;
import org.json.JSONObject;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class ONNXModelRunner {
    public static void main(String[] args) {
        try {
            LabelVocabLoader loader = new LabelVocabLoader("resources/labelMap.json", "resources/vocab.json");
            Map<Integer, String> labelMap = loader.getLabelMap();
            Map<String, Integer> vocab = loader.getVocab();

            String modelPath = "resources/model.onnx";
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions());

            String inputText = "The government announced new policies to boost the economy";

            Tokenizer tokenizer = new Tokenizer(vocab);
            int maxLen = 30;
            int[] tokenizedInput = tokenizer.tokenize(inputText);
            int[] paddedInput = new int[maxLen];
            for (int i = 0; i < maxLen; i++) {
                if (i < tokenizedInput.length) {
                    paddedInput[i] = tokenizedInput[i];
                } else {
                    paddedInput[i] = 0;
                }
            }

            int[][] inputData = new int[1][maxLen];
            inputData[0] = paddedInput;

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);

            String inputName = session.getInputNames().iterator().next();
            OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

            float[][] outputArray = (float[][]) result.get(0).getValue();
            int maxIndex = 0;
            float maxScore = outputArray[0][0];
            for (int i = 1; i < outputArray[0].length; i++) {
                if (outputArray[0][i] > maxScore) {
                    maxScore = outputArray[0][i];
                    maxIndex = i;
                }
            }

            System.out.println("Java ONNX output: " + labelMap.get(maxIndex) +
                             " (Score: " + String.format("%.4f", maxScore) + ")");

            session.close();
            env.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static class Tokenizer {
        private Map<String, Integer> vocab;

        public Tokenizer(Map<String, Integer> vocab) {
            this.vocab = vocab;
        }

        public int[] tokenize(String text) {
            String[] words = text.toLowerCase().split("\\s+");
            int[] tokenized = new int[words.length];
            for (int i = 0; i < words.length; i++) {
                Integer token = vocab.getOrDefault(words[i], vocab.get(""));
                tokenized[i] = token;
            }
            return tokenized;
        }
    }

    static class LabelVocabLoader {
        private Map<Integer, String> labelMap;
        private Map<String, Integer> vocab;

        public LabelVocabLoader(String labelMapPath, String vocabPath) throws Exception {
            String labelMapJson = new String(Files.readAllBytes(Paths.get(labelMapPath)));
            JSONObject labelMapObject = new JSONObject(labelMapJson);
            this.labelMap = new HashMap<>();
            for (String key : labelMapObject.keySet()) {
                this.labelMap.put(Integer.parseInt(key), labelMapObject.getString(key));
            }

            String vocabJson = new String(Files.readAllBytes(Paths.get(vocabPath)));
            JSONObject vocabObject = new JSONObject(vocabJson);
            this.vocab = new HashMap<>();
            for (String key : vocabObject.keySet()) {
                this.vocab.put(key, vocabObject.getInt(key));
            }
        }

        public Map<Integer, String> getLabelMap() {
            return labelMap;
        }

        public Map<String, Integer> getVocab() {
            return vocab;
        }
    }
}
