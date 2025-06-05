use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use ndarray::Array2;

struct BinaryClassifier {
    vocab: HashMap<String, usize>,
    idf: Vec<f32>,
    mean: Vec<f32>,
    scale: Vec<f32>,
    session: Session,
}

impl BinaryClassifier {
    fn new(model_path: &str, vocab_path: &str, scaler_path: &str) -> Result<Self> {
        let vocab_file = File::open(vocab_path)?;
        let vocab_reader = BufReader::new(vocab_file);
        let vocab_data: JsonValue = serde_json::from_reader(vocab_reader)?;

        let mut vocab = HashMap::new();
        let vocab_obj = vocab_data["vocab"].as_object().unwrap();
        for (key, value) in vocab_obj {
            vocab.insert(key.clone(), value.as_u64().unwrap() as usize);
        }

        let idf: Vec<f32> = vocab_data["idf"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let scaler_file = File::open(scaler_path)?;
        let scaler_reader = BufReader::new(scaler_file);
        let scaler_data: JsonValue = serde_json::from_reader(scaler_reader)?;

        let mean: Vec<f32> = scaler_data["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let scale: Vec<f32> = scaler_data["scale"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let environment = Arc::new(Environment::builder()
            .with_name("binary_classifier")
            .build()?);
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file(model_path)?;

        Ok(BinaryClassifier {
            vocab,
            idf,
            mean,
            scale,
            session,
        })
    }

    fn preprocess_text(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; 5000];
        let mut word_counts: HashMap<&str, usize> = HashMap::new();

        let text_lower = text.to_lowercase();
        for word in text_lower.split_whitespace() {
            *word_counts.entry(word).or_insert(0) += 1;
        }

        for (word, count) in word_counts {
            if let Some(&idx) = self.vocab.get(word) {
                vector[idx] = count as f32 * self.idf[idx];
            }
        }

        for i in 0..5000 {
            vector[i] = (vector[i] - self.mean[i]) / self.scale[i];
        }

        vector
    }

    fn predict(&self, text: &str) -> Result<f32> {
        let input_data = self.preprocess_text(text);
        let input_array = Array2::from_shape_vec((1, 5000), input_data)?;
        let input_dyn = input_array.into_dyn();
        let input_cow = ndarray::CowArray::from(input_dyn.view());
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;

        let outputs = self.session.run(vec![input_tensor])?;
        let output_view = outputs[0].try_extract::<f32>()?;
        let output_data = output_view.view();

        Ok(output_data[[0, 0]])
    }
}

fn main() -> Result<()> {
    let classifier = BinaryClassifier::new(
        "spam_classifier/model.onnx",
        "spam_classifier/vocab.json",
        "spam_classifier/scaler.json",
    )?;

    let text = "Act now! Get 70% off on all products. Visit our site today!";
    let probability = classifier.predict(text)?;

    println!("Rust ONNX output: Probability = {:.4}", probability);
    println!("Classification: {}",
        if probability > 0.5 { "Positive" } else { "Negative" }
    );

    Ok(())
}
