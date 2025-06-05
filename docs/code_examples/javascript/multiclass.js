async function preprocessText(text, tokenizerUrl) {
const tokenizerResp = await fetch(tokenizerUrl);
const tokenizer = await tokenizerResp.json();

const oovToken = '<OOV>';
const words = text.toLowerCase().split(/\s+/);
const sequence = words.map(word => tokenizer[word] || tokenizer[oovToken] || 1).slice(0, 30);
const padded = new Int32Array(30).fill(0);
sequence.forEach((val, idx) => padded[idx] = val);
return padded;
}

async function runModel(text) {
const session = await ort.InferenceSession.create('model.onnx');
const vector = await preprocessText(text, 'vocab.json');
const tensor = new ort.Tensor('int32', vector, [1, 30]);
const feeds = { input: tensor };
const output = await session.run(feeds);

const labelResp = await fetch('scaler.json');
const labelMap = await labelResp.json();

const probabilities = output[Object.keys(output)[0]].data;
const predictedIdx = probabilities.reduce((maxIdx, val, idx) => val > probabilities[maxIdx] ? idx : maxIdx, 0);
const label = labelMap[predictedIdx];
const score = probabilities[predictedIdx];
console.log(`JS ONNX output: ${label} (Score: ${score.toFixed(4)})`);
}

runModel('The government announced new policies to boost the economy');
