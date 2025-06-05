async function preprocessText(text, vocabUrl, scalerUrl) {
const tfidfResp = await fetch(vocabUrl);
const tfidfData = await tfidfResp.json();
const vocab = tfidfData.vocab;
const idf = tfidfData.idf;

const scalerResp = await fetch(scalerUrl);
const scalerData = await scalerResp.json();
const mean = scalerData.mean;
const scale = scalerData.scale;

// TF-IDF
const vector = new Float32Array(5000).fill(0);
const words = text.toLowerCase().split(/\s+/);
const wordCounts = {};
words.forEach(word => wordCounts[word] = (wordCounts[word] || 0);
for (const word in wordCounts) {
    if (vocab[word] !== undefined) {
        vector[vocab[word]] = wordCounts[word] * idf[vocab[word]];
    }
}

# Scale
for (let i = 0; i < 5000; i++) {
    vector[i] = (vector[i] - mean[i]) / scale[i];
}
return vector;
}

async function runModel(text) {
const session = await ort.InferenceSession.create("model.onnx");
const vector = await preprocessText(text, "model_vocab.json", "model_scaler.json");
const tensor = new ort.Tensor("float32", vector, [1, 5000]);
const feeds = { input: tensor };
const output = await session.run(feeds);
const probability = output[Object.keys(output)[0]].data[0];
console.log(`JS ONNX output: Probability = ${probability.toFixed(4)}`);
}

runModel("This is a positive test string");
