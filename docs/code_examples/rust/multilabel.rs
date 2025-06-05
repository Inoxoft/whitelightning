use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use ort::{Environment, Session, Tensor, RunOptions};

fn preprocess_text(text: &str, vocab_file: &str, scaler_file: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; 5000];

    // Load vocab.json
    let mut vf = File::open(vocab_file).unwrap();
    let mut vjson = String::new();
    vf.read_to_string(&mut vjson).unwrap();
    let tfidf_data: Value = serde_json::from_str(&vjson).unwrap();
    let vocab = tfidf_data["vocab"].as_object().unwrap();
    let idf: Vec<f32> = tfidf_data["idf"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    // Load scaler.json
    let mut sf = File::open(scaler_file).unwrap();
    let mut sjson = String::new();
    sf.read_to_string(&mut sjson).unwrap();
    let scaler_data: Value = serde_json::from_str(&sjson).unwrap();
    let mean: Vec<f32> = scaler_data["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let scale: Vec<f32> = scaler_data["scale"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    // TF-IDF
    let text_lower = text.to_lowercase();
    let mut word_counts = HashMap::new();
    for word in text_lower.split_whitespace() {
        *word_counts.entry(word.to_string()).or_insert(0) += 1;
    }
    for (word, count) in word_counts {
        if let Some(idx) = vocab.get(&word).and_then(|v| v.as_u64()) {
            vector[idx as usize] = count as f32 * idf[idx as usize];
        }
    }

    // Scale
    for i in 0..5000 {
        vector[i] = (vector[i] - mean[i]) / scale[i];
    }
    vector
}

fn main() -> ort::Result<()> {
    let text = "This is a positive test string";
    let vector = preprocess_text(&text, "model_vocab.json", "model_scaler.json");

    let env = Environment::builder().with_name("test").build()?;
    let session = Session::builder()?.commit_from_file("model.onnx")?;

    let input_tensor = Tensor::from_array(vector.as_slice(), &[1, 5000])?;
    let outputs = session.run(RunOptions::default(), &[("float_input", input_tensor)], &["output"])?;

    let output_data = outputs["output"].as_tensor::<f32>().unwrap().as_slice().unwrap();

    // Load label mapping
    let mut sf = File::open("model_scaler.json").unwrap();
    let mut sjson = String::new();
    sf.read_to_string(&mut sjson).unwrap();
    let scaler_data: Value = serde_json::from_str(&sjson).unwrap();
    println!("Rust Multilabel ONNX output:");
    for i in 0..output_data.len() {
        let idx = i.to_string();
        let label = scaler_data.get(&idx).map_or(&idx, |v| v.as_str().unwrap_or(&idx));
        println!("{}: {:.4}", label, output_data[i]);
    }

    Ok(())
}
