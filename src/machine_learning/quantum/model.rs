//! Core quantum model abstractions

use std::error::Error;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::quantum::circuit::QuantumCircuit;
use crate::machine_learning::categorical::functors::DataToCircuitFunctor;
use crate::machine_learning::categorical::{TrainingTransformation};
use crate::quantum::state::{QuantumState, StateVector};
use crate::machine_learning::core::{Model, ModelError};
use crate::machine_learning::prelude::Optimizer;
use crate::machine_learning::prelude::LossFunction;
use crate::machine_learning::categorical::functors::CircuitToPredictionFunctor;

/// Errors specific to quantum models
#[derive(Debug, Clone)]
pub enum QuantumModelError {
    /// Error in quantum circuit construction or execution
    CircuitError(String),

    /// Error in encoding classical data to quantum states
    EncodingError(String),

    /// Error in decoding quantum states to classical outputs
    DecodingError(String),

    /// Error from the underlying machine learning model
    ModelError(ModelError),

    /// Generic quantum model error
    Other(String),
}

impl std::fmt::Display for QuantumModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumModelError::CircuitError(msg) => write!(f, "Circuit error: {}", msg),
            QuantumModelError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
            QuantumModelError::DecodingError(msg) => write!(f, "Decoding error: {}", msg),
            QuantumModelError::ModelError(err) => write!(f, "Model error: {}", err),
            QuantumModelError::Other(msg) => write!(f, "Quantum model error: {}", msg),
        }
    }
}

impl Error for QuantumModelError {}

impl From<ModelError> for QuantumModelError {
    fn from(error: ModelError) -> Self {
        QuantumModelError::ModelError(error)
    }
}

/// Strategies for encoding classical data into quantum states
pub enum EncodingStrategy {
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    Custom(Box<dyn Fn(&Array1<f64>, usize) -> Result<StateVector, QuantumModelError> + Send + Sync>),
}

impl Clone for EncodingStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::AmplitudeEncoding => Self::AmplitudeEncoding,
            Self::AngleEncoding => Self::AngleEncoding,
            Self::BasisEncoding => Self::BasisEncoding,
            Self::Custom(_) => {
                // We can't actually clone the closure, so we need a fallback strategy
                // Option 1: Return a default strategy instead
                Self::AngleEncoding  // Use angle encoding as a default

                // Option 2: Panic (not recommended)
                // panic!("Cannot clone Custom encoding strategy")
            }
        }
    }
}

// Manually implement Debug
impl std::fmt::Debug for EncodingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AmplitudeEncoding => write!(f, "AmplitudeEncoding"),
            Self::AngleEncoding => write!(f, "AngleEncoding"),
            Self::BasisEncoding => write!(f, "BasisEncoding"),
            Self::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}


pub enum DecodingStrategy {
    MeasurementBased(Vec<usize>),
    ExpectationBased(Vec<Array2<Complex64>>),
    Custom(Box<dyn Fn(&StateVector) -> Result<Array1<f64>, QuantumModelError> + Send + Sync>),
}

impl Clone for DecodingStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::MeasurementBased(qubits) => Self::MeasurementBased(qubits.clone()),
            Self::ExpectationBased(observables) => Self::ExpectationBased(observables.clone()),
            Self::Custom(_) => {
                // Return a default strategy
                Self::MeasurementBased(vec![0])  // Default to measuring qubit 0
            }
        }
    }
}

impl std::fmt::Debug for DecodingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeasurementBased(qubits) => write!(f, "MeasurementBased({:?})", qubits),
            Self::ExpectationBased(_) => write!(f, "ExpectationBased(...)"),
            Self::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

/// Core trait for quantum machine learning models
pub trait QuantumModel: Model<Error = QuantumModelError> {
    /// Gets a reference to the quantum circuit
    fn circuit(&self) -> &QuantumCircuit;

    /// Gets a mutable reference to the quantum circuit
    fn circuit_mut(&mut self) -> &mut QuantumCircuit;

    /// Gets the number of qubits in the circuit
    fn qubit_count(&self) -> usize;

    /// Encodes classical input data into a quantum state
    fn encode_input(&self, input: &Self::Input) -> Result<StateVector, QuantumModelError>;

    /// Decodes a quantum state into classical output
    fn decode_output<S: QuantumState>(&self, state: &S) -> Result<Self::Output, QuantumModelError>;

    /// Runs the quantum circuit on an input state
    fn run_circuit(&self, input_state: &StateVector) -> Result<StateVector, QuantumModelError> {
        self.circuit().apply(input_state)
            .map_err(|e| QuantumModelError::CircuitError(e.to_string()))
    }
}

// Extension trait for quantum models that use categorical structures
pub trait CategoricalQuantumModel: QuantumModel {

    /// Returns the functor that maps from the data category to the quantum circuit category
    fn encoding_functor(&self) -> DataToCircuitFunctor;

    /// Returns the functor that maps from the quantum circuit category to the prediction category
    fn decoding_functor(&self) -> CircuitToPredictionFunctor;

    /// Returns a natural transformation representing the training process
    fn training_transformation<O: Optimizer + Clone, L: LossFunction + Clone>(&self, optimizer: &O, loss: &L)
        -> TrainingTransformation<Self, O, L>
    where
        Self: Sized + Clone;
}


/// Helper trait for quantum gradient computation
///
/// This is separate from QuantumModel to maintain object safety
pub trait QuantumGradients: QuantumModel {
    /// Computes gradients using the parameter shift rule
    fn compute_gradients_parameter_shift<F>(
        &self,
        input: &Self::Input,
        target: &Self::Output,
        loss_fn: F,
        parameter_indices: &[usize],
        shift: f64,
    ) -> Result<Vec<f64>, QuantumModelError>
    where
        F: Fn(&Self::Output, &Self::Output) -> f64;
}

/// Default implementation of gradient computation
pub struct ParameterShiftGradients<M: QuantumModel + Clone>(pub M);

impl<M: QuantumModel + Clone> ParameterShiftGradients<M> {
    /// Computes gradients for a quantum model using parameter shift rule
    pub fn compute_gradients<F>(
        &self,
        input: &M::Input,
        target: &M::Output,
        loss_fn: F,
        parameter_indices: &[usize],
        shift: f64,
    ) -> Result<Vec<f64>, QuantumModelError>
    where
        F: Fn(&M::Output, &M::Output) -> f64
    {
        let model = &self.0;
        let n_params = parameter_indices.len();
        let mut gradients = vec![0.0; n_params];

        // Encode input
        let input_state = model.encode_input(input)?;

        // Get current parameters
        let current_params = model.get_parameters();

        // For each parameter, compute gradient using parameter shift rule
        for (i, &param_idx) in parameter_indices.iter().enumerate() {
            if param_idx >= current_params.len() {
                return Err(QuantumModelError::Other(
                    format!("Parameter index {} out of bounds", param_idx)
                ));
            }

            // Forward evaluation with positive shift
            let mut params_plus = current_params.clone();
            params_plus[param_idx] += shift;

            // Create a clone of the model with shifted parameters
            let mut model_plus = model.clone();
            model_plus.set_parameters(&params_plus)?;

            // Run the circuit with positive shift
            let state_plus = model_plus.run_circuit(&input_state)?;
            let output_plus = model_plus.decode_output(&state_plus)?;
            let loss_plus = loss_fn(&output_plus, target);

            // Forward evaluation with negative shift
            let mut params_minus = current_params.clone();
            params_minus[param_idx] -= shift;

            // Update the clone with new parameters
            let mut model_minus = model.clone();
            model_minus.set_parameters(&params_minus)?;

            // Run the circuit with negative shift
            let state_minus = model_minus.run_circuit(&input_state)?;
            let output_minus = model_minus.decode_output(&state_minus)?;
            let loss_minus = loss_fn(&output_minus, target);

            // Compute gradient using finite difference
            gradients[i] = (loss_plus - loss_minus) / (2.0 * shift);
        }

        Ok(gradients)
    }
}

/// Helper function to tensor product two quantum models
// pub fn tensor_quantum_models<M1, M2>(
//     model1: &M1,
//     model2: &M2
// ) -> Result<Box<dyn QuantumModel<Input=Array1<f64>, Output=Array1<f64>>>, QuantumModelError>
pub fn tensor_quantum_models<M1, M2>(
    _model1: &M1,
    _model2: &M2
) -> Result<Box<dyn std::error::Error>, Box<dyn std::error::Error>>
where
    M1: QuantumModel<Input=Array1<f64>, Output=Array1<f64>> + Clone + 'static,
    M2: QuantumModel<Input=Array1<f64>, Output=Array1<f64>> + Clone + 'static,
{
    // placeholder for now. implement this based on QuantumCircuit
    Err("Tensor product of two quantum models not implemented".into())
}
