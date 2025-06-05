use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json::Value as JsonValue;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

struct Tokenizer {
    vocab: HashMap<String, i32>,
}

impl Tokenizer {
    fn new(vocab_file: &str) -> Result<Self> {
        let mut file = File::open(vocab_file)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let vocab: HashMap<String, i32> = serde_json::from_str(&contents)?;
        Ok(Tokenizer { vocab })
    }

    fn tokenize(&self, text: &str) -> Vec<i32> {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        let mut tokens = Vec::with_capacity(30);

        for word in words.iter().take(30) {
            let token = self.vocab.get(*word)
                .or_else(|| self.vocab.get(""))
                .copied()
                .unwrap_or(1);
            tokens.push(token);
        }

        while tokens.len() < 30 {
            tokens.push(0);
        }

        tokens
    }
}

fn load_label_map(file_path: &str) -> Result<HashMap<String, String>> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let label_map: HashMap<String, String> = serde_json::from_str(&contents)?;
    Ok(label_map)
}

fn main() -> Result<()> {
    // Initialize tokenizer and load label map
    let tokenizer = Tokenizer::new("vocab.json")?;
    let label_map = load_label_map("scaler.json")?;

    // Create ONNX Runtime environment and session
    let environment = Environment::builder()
        .with_name("onnx-rust-demo")
        .build()?;

    let session = SessionBuilder::new(&environment)?
        .with_model_from_file("model.onnx")?;

    // Prepare input text
    let text = "The government announced new policies to boost the economy";
    let tokens = tokenizer.tokenize(text);

    // Create input tensor
    let input_tensor = Value::from_array(([1, 30], tokens))?;

    // Run inference
    let outputs = session.run(vec![input_tensor])?;
    let output: &Value = outputs[0].downcast_ref().unwrap();
    let probabilities = output.as_slice::<f32>()?;

    // Find predicted class
    let (predicted_idx, &score) = probabilities.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let label = label_map.get(&predicted_idx.to_string())
        .ok_or_else(|| anyhow::anyhow!("Label not found"))?;

    println!("Rust ONNX output: {} (Score: {:.4})", label, score);
    Ok(())
}
