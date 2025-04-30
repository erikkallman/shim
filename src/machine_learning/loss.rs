//! Loss functions for machine learning models

use ndarray::Array1;

/// Trait for loss functions
pub trait LossFunction {
    /// Type of input for loss calculation
    type Input;

    /// Calculate the loss between predictions and targets
    fn calculate_loss(&self, predictions: &Self::Input, targets: &Self::Input) -> f64;

    /// Calculate gradients of the loss with respect to predictions
    fn calculate_gradients(&self, predictions: &Self::Input, targets: &Self::Input) -> Self::Input;
}

/// Mean Squared Error loss
#[derive(Debug, Clone, Copy)]
pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    type Input = Array1<f64>;

    fn calculate_loss(&self, predictions: &Self::Input, targets: &Self::Input) -> f64 {
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        squared_diff.sum() / predictions.len() as f64
    }

    fn calculate_gradients(&self, predictions: &Self::Input, targets: &Self::Input) -> Self::Input {
        let diff = predictions - targets;
        2.0 * diff / predictions.len() as f64
    }
}

/// Cross-Entropy loss for binary classification
#[derive(Debug, Clone, Copy)]
pub struct BinaryCrossEntropy;

impl LossFunction for BinaryCrossEntropy {
    type Input = Array1<f64>;

    fn calculate_loss(&self, predictions: &Self::Input, targets: &Self::Input) -> f64 {
        let n = predictions.len() as f64;
        let mut loss = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            // Clip predictions to avoid numerical issues
            let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
            loss -= t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln();
        }

        loss / n
    }

    fn calculate_gradients(&self, predictions: &Self::Input, targets: &Self::Input) -> Self::Input {
        let n = predictions.len() as f64;
        let mut grads = Array1::zeros(predictions.len());

        for i in 0..predictions.len() {
            // Clip predictions to avoid numerical issues
            let p_clipped = predictions[i].max(1e-15).min(1.0 - 1e-15);
            grads[i] = -(targets[i] / p_clipped - (1.0 - targets[i]) / (1.0 - p_clipped)) / n;
        }

        grads
    }
}

/// Cross-Entropy loss for multi-class classification
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropy;

impl LossFunction for CrossEntropy {
    type Input = Array1<f64>;

    fn calculate_loss(&self, predictions: &Self::Input, targets: &Self::Input) -> f64 {
        let n = predictions.len() as f64;
        let mut loss = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            if *t > 0.0 {
                // Clip predictions to avoid numerical issues
                let p_clipped = p.max(1e-15);
                loss -= t * p_clipped.ln();
            }
        }

        loss / n
    }

    fn calculate_gradients(&self, predictions: &Self::Input, targets: &Self::Input) -> Self::Input {
        let n = predictions.len() as f64;
        let mut grads = Array1::zeros(predictions.len());

        for i in 0..predictions.len() {
            if targets[i] > 0.0 {
                // Clip predictions to avoid numerical issues
                let p_clipped = predictions[i].max(1e-15);
                grads[i] = -targets[i] / p_clipped / n;
            }
        }

        grads
    }
}

/// Helper function to compute softmax probabilities
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|xi| (xi - max_val).exp());
    let sum_exp = exp_x.sum();
    exp_x / sum_exp
}

/// Categorical cross-entropy with softmax activation
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxCrossEntropy;

impl LossFunction for SoftmaxCrossEntropy {
    type Input = Array1<f64>;

    fn calculate_loss(&self, logits: &Self::Input, targets: &Self::Input) -> f64 {
        let probs = softmax(logits);
        let n = probs.len() as f64;
        let mut loss = 0.0;

        for (p, t) in probs.iter().zip(targets.iter()) {
            if *t > 0.0 {
                // Clip probabilities to avoid numerical issues
                let p_clipped = p.max(1e-15);
                loss -= t * p_clipped.ln();
            }
        }

        loss / n
    }

    fn calculate_gradients(&self, logits: &Self::Input, targets: &Self::Input) -> Self::Input {
        let probs = softmax(logits);
        let n = probs.len() as f64;

        // Gradient of softmax cross-entropy is (p - y)/n
        (probs - targets) / n
    }
}
