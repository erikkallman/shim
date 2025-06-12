//! Variational quantum models for machine learning

use ndarray::Array1;
use rand::Rng;
use std::f64::consts::PI;

use crate::machine_learning::core::{Model, PredictiveModel};
use crate::machine_learning::quantum::model::{QuantumModel, QuantumModelError, EncodingStrategy, DecodingStrategy};
use crate::machine_learning::quantum::circuit_model::ParametrizedCircuitModel;
use crate::quantum::circuit::QuantumCircuit;
use crate::quantum::state::{QuantumState, StateVector};
/// Trait for variational quantum models
pub trait VariationalQuantumModel: QuantumModel {
    /// Adds a variational layer to the model
    fn add_variational_layer(&mut self) -> Result<(), Self::Error>;

    /// Returns the number of variational layers
    fn layer_count(&self) -> usize;
}

/// Quantum Neural Network
#[derive(Clone)]
pub struct QuantumNeuralNetwork {
    /// The underlying parametrized circuit model
    circuit_model: ParametrizedCircuitModel,
    /// Number of variational layers
    layer_count: usize,
}

impl QuantumNeuralNetwork {
    /// Creates a new quantum neural network
    pub fn new(
        qubit_count: usize,
        input_dim: usize,
        output_dim: usize,
        layer_count: usize
    ) -> Self {
        // Estimate the number of parameters needed
        let params_per_layer = qubit_count * 3; // Rx, Ry, Rz for each qubit
        let total_params = params_per_layer * layer_count;

        // Initialize with random parameters
        let parameters = (0..total_params)
            .map(|_| rand::thread_rng().gen_range(0.0..2.0 * PI))
            .collect::<Vec<_>>();

        // Create the variational circuit model
        let mut circuit_model = ParametrizedCircuitModel::new(
            qubit_count,
            parameters,
            input_dim,
            output_dim
        );

        // Set appropriate encoding/decoding strategies
        circuit_model.encoding_strategy = EncodingStrategy::AngleEncoding;

        let measurement_qubits = if output_dim <= qubit_count {
            (0..output_dim).collect()
        } else {
            (0..qubit_count).collect()
        };

        circuit_model.decoding_strategy = DecodingStrategy::MeasurementBased(measurement_qubits);

        // Create the model
        let mut qnn = QuantumNeuralNetwork {
            circuit_model,
            layer_count: 0,
        };

        // Add variational layers
        for _ in 0..layer_count {
            // Ignore errors during initialization
            let _ = qnn.add_variational_layer();
        }

        qnn
    }

    /// Sets the encoding strategy
    pub fn with_encoding_strategy(mut self, strategy: EncodingStrategy) -> Self {
        self.circuit_model.encoding_strategy = strategy;
        self
    }

    /// Sets the decoding strategy
    pub fn with_decoding_strategy(mut self, strategy: DecodingStrategy) -> Self {
        self.circuit_model.decoding_strategy = strategy;
        self
    }
}

impl Model for QuantumNeuralNetwork {
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    type Error = QuantumModelError;

    fn parameter_count(&self) -> usize {
        self.circuit_model.parameter_count()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.circuit_model.get_parameters()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        self.circuit_model.set_parameters(parameters)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.circuit_model.dimensions()
    }
}

impl PredictiveModel for QuantumNeuralNetwork {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        self.circuit_model.predict(input)
    }
}

impl QuantumModel for QuantumNeuralNetwork {
    fn circuit(&self) -> &QuantumCircuit {
        self.circuit_model.circuit()
    }

    fn circuit_mut(&mut self) -> &mut QuantumCircuit {
        self.circuit_model.circuit_mut()
    }

    fn qubit_count(&self) -> usize {
        self.circuit_model.qubit_count()
    }

    fn encode_input(&self, input: &Self::Input) -> Result<StateVector, Self::Error> {
        self.circuit_model.encode_input(input)
    }

    fn decode_output<S: QuantumState>(&self, state: &S) -> Result<Self::Output, Self::Error> {
        self.circuit_model.decode_output(state)
    }

    fn run_circuit(&self, input_state: &StateVector) -> Result<StateVector, QuantumModelError> {
        self.circuit_model.run_circuit(input_state)
    }
}

impl VariationalQuantumModel for QuantumNeuralNetwork {
    fn add_variational_layer(&mut self) -> Result<(), Self::Error> {
        self.circuit_model.add_variational_layer()?;
        self.layer_count += 1;
        Ok(())
    }

    fn layer_count(&self) -> usize {
        self.layer_count
    }
}

/// Variational Quantum Classifier
#[derive(Clone)]
pub struct VariationalQuantumClassifier {
    /// The underlying quantum neural network
    qnn: QuantumNeuralNetwork,
    /// Number of classes
    #[allow(dead_code)]
    class_count: usize,
}

impl VariationalQuantumClassifier {
    /// Creates a new variational quantum classifier
    pub fn new(
        qubit_count: usize,
        input_dim: usize,
        class_count: usize,
        layer_count: usize
    ) -> Self {
        // We need at least log2(class_count) qubits for measurement
        let required_measurement_qubits = (class_count as f64).log2().ceil() as usize;

        if required_measurement_qubits > qubit_count {
            panic!("Need at least {} qubits to represent {} classes",
                   required_measurement_qubits, class_count);
        }

        // Create a quantum neural network
        let mut qnn = QuantumNeuralNetwork::new(
            qubit_count,
            input_dim,
            class_count,  // output dim matches class count for probabilities
            layer_count
        );

        // Set appropriate decoding strategy for classification
        // Measure the qubits needed to distinguish classes
        let measurement_qubits = (0..required_measurement_qubits).collect();
        qnn.circuit_model.decoding_strategy = DecodingStrategy::MeasurementBased(measurement_qubits);

        VariationalQuantumClassifier {
            qnn,
            class_count,
        }
    }

    /// Get class probabilities
    pub fn get_probabilities(&self, input: &Array1<f64>) -> Result<Array1<f64>, QuantumModelError> {
        self.qnn.predict(input)
    }

    /// Get the predicted class (as index)
    pub fn predict_class(&self, input: &Array1<f64>) -> Result<usize, QuantumModelError> {
        let probs = self.get_probabilities(input)?;

        // Return the class with highest probability
        let mut max_prob = probs[0];
        let mut max_class = 0;

        for (class, &prob) in probs.iter().enumerate().skip(1) {
            if prob > max_prob {
                max_prob = prob;
                max_class = class;
            }
        }

        Ok(max_class)
    }
}

impl Model for VariationalQuantumClassifier {
    type Input = Array1<f64>;
    type Output = Array1<f64>;  // Class probabilities
    type Error = QuantumModelError;

    fn parameter_count(&self) -> usize {
        self.qnn.parameter_count()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.qnn.get_parameters()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        self.qnn.set_parameters(parameters)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.qnn.dimensions()
    }
}

impl PredictiveModel for VariationalQuantumClassifier {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        self.qnn.predict(input)
    }
}

impl QuantumModel for VariationalQuantumClassifier {
    fn circuit(&self) -> &QuantumCircuit {
        self.qnn.circuit()
    }

    fn circuit_mut(&mut self) -> &mut QuantumCircuit {
        self.qnn.circuit_mut()
    }

    fn qubit_count(&self) -> usize {
        self.qnn.qubit_count()
    }

    fn encode_input(&self, input: &Self::Input) -> Result<StateVector, Self::Error> {
        self.qnn.encode_input(input)
    }

    fn decode_output<S: QuantumState>(&self, state: &S) -> Result<Self::Output, Self::Error> {
        self.qnn.decode_output(state)
    }

    fn run_circuit(&self, input_state: &StateVector) -> Result<StateVector, QuantumModelError> {
        self.qnn.run_circuit(input_state)
    }

}

impl VariationalQuantumModel for VariationalQuantumClassifier {
    fn add_variational_layer(&mut self) -> Result<(), Self::Error> {
        self.qnn.add_variational_layer()
    }

    fn layer_count(&self) -> usize {
        self.qnn.layer_count()
    }
}
