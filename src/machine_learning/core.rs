//! Core traits and types for machine learning models

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::error::Error;
use ndarray::{Array1, Array2};

use crate::machine_learning::optimizer::Optimizer;
use crate::machine_learning::loss::LossFunction;

/// Errors that can occur in machine learning models
#[derive(Debug, Clone)]
pub enum ModelError {
    /// Dimensionality mismatch in input or output data
    DimensionMismatch(String),

    /// Error during forward computation
    ForwardError(String),

    /// Error during backward computation or gradient calculation
    BackwardError(String),

    /// Error during parameter update
    UpdateError(String),

    /// Input data is outside the valid range
    InputRangeError(String),

    /// Generic model error
    Other(String),
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ModelError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            ModelError::ForwardError(msg) => write!(f, "Forward computation error: {}", msg),
            ModelError::BackwardError(msg) => write!(f, "Backward computation error: {}", msg),
            ModelError::UpdateError(msg) => write!(f, "Parameter update error: {}", msg),
            ModelError::InputRangeError(msg) => write!(f, "Input range error: {}", msg),
            ModelError::Other(msg) => write!(f, "Model error: {}", msg),
        }
    }
}

impl Error for ModelError {}

/// Base trait for all machine learning models
pub trait Model {
    /// Type of input data
    type Input;

    /// Type of output predictions
    type Output;

    /// Type of errors this model can produce
    type Error: Error + From<ModelError>;

    /// Returns the number of trainable parameters in the model
    fn parameter_count(&self) -> usize;

    /// Gets the current model parameters
    fn get_parameters(&self) -> Vec<f64>;

    /// Sets the model parameters
    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error>;

    /// Returns the input and output dimensions
    fn dimensions(&self) -> (usize, usize);

    /// Clones the model into a boxed trait object (useful for collections)
    fn clone_boxed(&self) -> Box<dyn Model<Input=Self::Input, Output=Self::Output, Error=Self::Error>>
    where
        Self: Sized + Clone + 'static
    {
        Box::new(self.clone())
    }
}

/// Trait for models that can make predictions
pub trait PredictiveModel: Model {
    /// Make a prediction for a single input
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Make predictions for a batch of inputs
    fn predict_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    /// Clones the predictive model into a boxed trait object
    fn clone_box_predictive(&self) -> Box<dyn PredictiveModel<Input=Self::Input, Output=Self::Output, Error=Self::Error>>
    where
        Self: Sized + Clone + 'static
    {
        Box::new(self.clone())
    }
}

/// Trait for models that can be trained
pub trait TrainableModel: PredictiveModel {
    /// Type of loss function this model can use
    type LossFunction: LossFunction<Input=Self::Output>;

    /// Train the model on a dataset
    fn train<O: Optimizer>(
        &mut self,
        inputs: &[Self::Input],
        targets: &[Self::Output],
        optimizer: &O,
        loss_fn: &Self::LossFunction
    ) -> Result<f64, Self::Error>;

    /// Calculate gradients for a single input-target pair
    fn calculate_gradients(
        &self,
        input: &Self::Input,
        target: &Self::Output,
        loss_fn: &Self::LossFunction
    ) -> Result<Vec<f64>, Self::Error>;
}

/// Trait for models that have specific architectures
pub trait ArchitecturalModel: Model {
    /// Returns the model's architecture as a string
    fn architecture_description(&self) -> String;

    /// Returns the model's layer information
    fn layer_information(&self) -> Vec<(String, usize)>;
}

/// Basic feedforward neural network model
#[derive(Clone)]
pub struct FeedForwardNN {
    input_dim: usize,
    output_dim: usize,
    #[allow(dead_code)]
    hidden_layers: Vec<usize>,
    parameters: Vec<f64>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl FeedForwardNN {
    /// Creates a new feedforward neural network
    pub fn new(input_dim: usize, hidden_layers: Vec<usize>, output_dim: usize) -> Self {
        // Construct layer dimensions including input and output
        let layer_dims: Vec<usize> = std::iter::once(input_dim)
            .chain(hidden_layers.iter().cloned())
            .chain(std::iter::once(output_dim))
            .collect();

        // Initialize weights and biases
        let mut weights = Vec::with_capacity(layer_dims.len() - 1);
        let mut biases = Vec::with_capacity(layer_dims.len() - 1);
        let mut parameters = Vec::new();

        for i in 0..layer_dims.len() - 1 {
            let rows = layer_dims[i + 1];
            let cols = layer_dims[i];

            // Xavier initialization for weights
            let weight_scale = 1.0 / (cols as f64).sqrt();
            let mut weight_matrix = Array2::zeros((rows, cols));

            for r in 0..rows {
                for c in 0..cols {
                    let w = (2.0 * rand::random::<f64>() - 1.0) * weight_scale;
                    weight_matrix[[r, c]] = w;
                    parameters.push(w);
                }
            }

            weights.push(weight_matrix);

            // Initialize biases to zero
            let mut bias_vector = Array1::zeros(rows);
            for r in 0..rows {
                bias_vector[r] = 0.0;
                parameters.push(0.0);
            }

            biases.push(bias_vector);
        }

        FeedForwardNN {
            input_dim,
            output_dim,
            hidden_layers: hidden_layers.clone(),
            parameters,
            weights,
            biases,
        }
    }

    /// Applies the activation function to a value
    fn activate(&self, x: f64) -> f64 {
        // ReLU activation
        x.max(0.0)
    }

    /// Applies the activation function derivative to a value
    #[allow(dead_code)]
    fn activate_derivative(&self, x: f64) -> f64 {
        // ReLU derivative
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    /// Forward pass through the network
    fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>, ModelError> {
        if input.len() != self.input_dim {
            return Err(ModelError::DimensionMismatch(
                format!("Expected input dim {}, got {}", self.input_dim, input.len())
            ));
        }

        // Propagate through each layer
        let mut current = input.clone();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            // Linear transformation: Wx + b
            let mut next = weight.dot(&current) + bias;

            // Apply activation to all except the output layer
            if &weight != self.weights.last().unwrap() {
                next.mapv_inplace(|x| self.activate(x));
            }

            current = next;
        }

        Ok(current)
    }
}

impl Model for FeedForwardNN {
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        if parameters.len() != self.parameters.len() {
            return Err(ModelError::DimensionMismatch(
                format!("Expected {} parameters, got {}", self.parameters.len(), parameters.len())
            ));
        }

        self.parameters = parameters.to_vec();

        // Update weights and biases
        let mut param_idx = 0;

        for i in 0..self.weights.len() {
            let rows = self.weights[i].shape()[0];
            let cols = self.weights[i].shape()[1];

            // Update weights
            for r in 0..rows {
                for c in 0..cols {
                    self.weights[i][[r, c]] = parameters[param_idx];
                    param_idx += 1;
                }
            }

            // Update biases
            for r in 0..rows {
                self.biases[i][r] = parameters[param_idx];
                param_idx += 1;
            }
        }

        Ok(())
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }
}

impl PredictiveModel for FeedForwardNN {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        self.forward(input)
    }
}

// Linear model implementation (for comparison with quantum models)
#[derive(Clone)]
pub struct LinearModel {
    input_dim: usize,
    output_dim: usize,
    weight: Array2<f64>,
    bias: Array1<f64>,
    parameters: Vec<f64>,
}

impl LinearModel {
    /// Creates a new linear model
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with small random values
        let mut weight = Array2::zeros((output_dim, input_dim));
        let bias = Array1::zeros(output_dim);
        let mut parameters = Vec::with_capacity(output_dim * (input_dim + 1));

        // Initialize weights and collect parameters
        for i in 0..output_dim {
            for j in 0..input_dim {
                let w = (rand::random::<f64>() - 0.5) * 0.01;
                weight[[i, j]] = w;
                parameters.push(w);
            }
        }

        // Add biases to parameters
        for _i in 0..output_dim {
            parameters.push(0.0); // Initialize biases to zero
        }

        LinearModel {
            input_dim,
            output_dim,
            weight,
            bias,
            parameters,
        }
    }
}

impl Model for LinearModel {
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        if parameters.len() != self.parameters.len() {
            return Err(ModelError::DimensionMismatch(
                format!("Expected {} parameters, got {}", self.parameters.len(), parameters.len())
            ));
        }

        self.parameters = parameters.to_vec();

        // Update weights
        let mut param_idx = 0;
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                self.weight[[i, j]] = parameters[param_idx];
                param_idx += 1;
            }
        }

        // Update biases
        for i in 0..self.output_dim {
            self.bias[i] = parameters[param_idx];
            param_idx += 1;
        }

        Ok(())
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }
}

impl PredictiveModel for LinearModel {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        if input.len() != self.input_dim {
            return Err(ModelError::DimensionMismatch(
                format!("Expected input dim {}, got {}", self.input_dim, input.len())
            ));
        }

        // Linear transformation: Wx + b
        let output = self.weight.dot(input) + &self.bias;
        Ok(output)
    }
}
