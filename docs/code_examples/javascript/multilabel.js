const ort = require('onnxruntime-node');
const fs = require('fs');

function preprocessText(text, vocabFile, scalerFile) {
    const vector = new Float32Array(5000).fill(0);

    // Load vocab.json
    const tfidfData = JSON.parse(fs.readFileSync(vocabFile, 'utf8'));
    const vocab = tfidfData.vocab;
    const idf = tfidfData.idf;

    // Load scaler.json
    const scalerData = JSON.parse(fs.readFileSync(scalerFile, 'utf8'));
    const mean = scalerData.mean;
    const scale = scalerData.scale;

    // TF-IDF
    const textLower = text.toLowerCase();
    const wordCounts = {};
    textLower.split(/\s+/).forEach(word => {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
    });
    for (const [word, count] of Object.entries(wordCounts)) {
        if (word in vocab) {
            vector[vocab[word]] = count * idf[vocab[word]];
        }
    }

    // Scale
    for (let i = 0; i < 5000; i++) {
        vector[i] = (vector[i] - mean[i]) / scale[i];
    }
    return vector;
}

async function main() {
    const text = "This is a positive test string";
    const vector = preprocessText(text, "model_vocab.json", "model_scaler.json");

    const session = await ort.InferenceSession.create("model.onnx");
    const inputTensor = new ort.Tensor("float32", vector, [1, 5000]);
    const feeds = { float_input: inputTensor };
    const results = await session.run(feeds);

    const outputData = results.output.data;
    const scalerData = JSON.parse(fs.readFileSync("model_scaler.json", 'utf8'));
    console.log("JavaScript Multilabel ONNX output:");
    for (let i = 0; i < outputData.length; i++) {
        const label = scalerData[i.toString()] || `label_${i}`;
        console.log(`${label}: ${outputData[i].toFixed(4)}`);
    }
}

main().catch(console.error);
