//! Quantum kernel methods for machine learning

use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::machine_learning::core::{Model, PredictiveModel};
use crate::machine_learning::quantum::model::{QuantumModel, QuantumModelError, EncodingStrategy};
use crate::quantum::circuit::{QuantumCircuit, CircuitBuilder};
use crate::quantum::state::{QuantumState, StateVector};

/// Quantum kernel for machine learning
#[derive(Clone)]
pub struct QuantumKernel {
    /// The quantum circuit used for the feature map
    circuit: QuantumCircuit,
    /// Number of qubits
    qubit_count: usize,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Function to map classical data to quantum states
    feature_map: Arc<dyn Fn(&Array1<f64>) -> Result<StateVector, QuantumModelError> + Send + Sync>,
}

impl QuantumKernel {
    /// Creates a new quantum kernel
    pub fn new(qubit_count: usize, input_dim: usize) -> Self {
        let circuit = QuantumCircuit::new(qubit_count);

        // Create a default feature map using ZZ feature mapping
        let feature_map = Arc::new(move |input: &Array1<f64>| -> Result<StateVector, QuantumModelError> {
            if input.len() != input_dim {
                return Err(QuantumModelError::EncodingError(
                    format!("Expected input dimension {}, got {}", input_dim, input.len())
                ));
            }

            // Build a ZZ feature map circuit
            let mut builder = CircuitBuilder::new(qubit_count);

            // First layer: Apply Hadamard to all qubits
            for q in 0..qubit_count {
                builder.h(q).map_err(|e|
                    QuantumModelError::EncodingError(e.to_string())
                )?;
            }

            // Second layer: Apply rotations based on data
            for (i, &value) in input.iter().enumerate() {
                if i < qubit_count {
                    builder.rz(i, value).map_err(|e|
                        QuantumModelError::EncodingError(e.to_string())
                    )?;
                }
            }

            // Third layer: Apply entangling ZZ rotations
            for q1 in 0..qubit_count {
                for q2 in (q1+1)..qubit_count {
                    if q1 < input_dim && q2 < input_dim {
                        let angle = PI * input[q1] * input[q2];

                        // Apply CNOT - RZ - CNOT for effective ZZ rotation
                        builder.cnot(q1, q2).map_err(|e|
                            QuantumModelError::EncodingError(e.to_string())
                        )?;

                        builder.rz(q2, angle).map_err(|e|
                            QuantumModelError::EncodingError(e.to_string())
                        )?;

                        builder.cnot(q1, q2).map_err(|e|
                            QuantumModelError::EncodingError(e.to_string())
                        )?;
                    }
                }
            }

            // Create and run the circuit
            let circuit = builder.build();
            let initial_state = StateVector::zero_state(qubit_count);
            circuit.apply(&initial_state)
                .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
        });

        QuantumKernel {
            circuit,
            qubit_count,
            input_dim,
            output_dim: 1, // Kernels produce a scalar output
            feature_map,
        }
    }

    /// Creates a quantum kernel with a custom feature map
    pub fn with_feature_map(
        mut self,
        feature_map: Arc<dyn Fn(&Array1<f64>) -> Result<StateVector, QuantumModelError> + Send + Sync>
    ) -> Self {
        self.feature_map = feature_map;
        self
    }

    /// Creates a quantum kernel with a predefined feature map type
    pub fn with_encoding_strategy(mut self, strategy: EncodingStrategy) -> Self {
        match strategy {
            EncodingStrategy::AmplitudeEncoding => {
                let qubit_count = self.qubit_count;
                self.feature_map = Arc::new(move |input: &Array1<f64>| -> Result<StateVector, QuantumModelError> {
                    // Normalize the input vector
                    let norm = input.dot(input).sqrt();
                    if norm < 1e-10 {
                        return Err(QuantumModelError::EncodingError(
                            "Input vector has zero norm".to_string()
                        ));
                    }

                    let normalized = input / norm;

                    // Convert to complex amplitudes
                    let mut amplitudes = Vec::with_capacity(1 << qubit_count);
                    for i in 0..normalized.len() {
                        if i < (1 << qubit_count) {
                            amplitudes.push(num_complex::Complex64::new(normalized[i], 0.0));
                        } else {
                            break;
                        }
                    }

                    // Pad with zeros if needed
                    while amplitudes.len() < (1 << qubit_count) {
                        amplitudes.push(num_complex::Complex64::new(0.0, 0.0));
                    }

                    StateVector::new(qubit_count, amplitudes.into())
                        .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
                });
            },
            EncodingStrategy::AngleEncoding => {
                let qubit_count = self.qubit_count;
                self.feature_map = Arc::new(move |input: &Array1<f64>| -> Result<StateVector, QuantumModelError> {
                    let mut builder = CircuitBuilder::new(qubit_count);

                    // Apply rotation gates based on input values
                    for (i, &value) in input.iter().enumerate() {
                        if i < qubit_count {
                            builder.ry(i, value).map_err(|e|
                                QuantumModelError::EncodingError(e.to_string())
                            )?;
                        }
                    }

                    // Add entanglement (as commonly used in variational circuits)
                    for q in 0..qubit_count-1 {
                        builder.cnot(q, q+1).map_err(|e|
                            QuantumModelError::EncodingError(e.to_string())
                        )?;
                    }

                    // Second rotation layer
                    for (i, &value) in input.iter().enumerate() {
                        if i < qubit_count {
                            builder.rz(i, value).map_err(|e|
                                QuantumModelError::EncodingError(e.to_string())
                            )?;
                        }
                    }

                    // Create and run the circuit
                    let circuit = builder.build();
                    let initial_state = StateVector::zero_state(qubit_count);
                    circuit.apply(&initial_state)
                        .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
                });
            },
            EncodingStrategy::BasisEncoding => {
                let qubit_count = self.qubit_count;
                self.feature_map = Arc::new(move |input: &Array1<f64>| -> Result<StateVector, QuantumModelError> {
                    // Convert input to binary representation
                    let mut index = 0;
                    for (i, &value) in input.iter().enumerate() {
                        if value > 0.5 && i < qubit_count {
                            index |= 1 << i;
                        }
                    }

                    StateVector::computational_basis(qubit_count, index)
                        .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
                });
            },
            EncodingStrategy::Custom(encoder) => {
                // We need to make sure the encoder has the same signature as our feature_map
                // The error suggests encoder might have a different signature
                if let Some(converted_encoder) = Self::convert_encoder(encoder.into()) {
                    self.feature_map = converted_encoder;
                }
            },
        }

        self
    }

    /// Helper method to convert an encoder that might have a different signature
    fn convert_encoder(
        encoder: Arc<dyn Fn(&Array1<f64>, usize) -> Result<StateVector, QuantumModelError> + Send + Sync>
    ) -> Option<Arc<dyn Fn(&Array1<f64>) -> Result<StateVector, QuantumModelError> + Send + Sync>> {
        // placeholder for the logic to adapt encoders with different signatures
        // Example approach: create a wrapper that adapts the two-argument encoder to a one-argument encoder
        let qubit_count = 0; //obtain from somewhere.
        let adapted_encoder = Arc::new(move |input: &Array1<f64>| -> Result<StateVector, QuantumModelError> {
            encoder(input, qubit_count)
        });

        Some(adapted_encoder)
    }

    /// Compute kernel value between two data points
    pub fn compute_kernel(
        &self,
        x1: &Array1<f64>,
        x2: &Array1<f64>
    ) -> Result<f64, QuantumModelError> {
        // Map the inputs to quantum states
        let state1 = (self.feature_map)(x1)?;
        let state2 = (self.feature_map)(x2)?;

        // Compute the inner product
        let inner_product = state1.inner_product(&state2);

        // Return the squared absolute value (fidelity)
        Ok(inner_product.norm_sqr())
    }

    /// Compute kernel matrix for a dataset
    pub fn compute_kernel_matrix(
        &self,
        data: &[Array1<f64>]
    ) -> Result<Array2<f64>, QuantumModelError> {
        let n = data.len();
        let mut kernel_matrix = Array2::zeros((n, n));

        // Precompute quantum states for all data points
        let mut quantum_states = Vec::with_capacity(n);
        for x in data {
            quantum_states.push((self.feature_map)(x)?);
        }

        // Compute kernel matrix
        for i in 0..n {
            for j in i..n {
                let inner_product = quantum_states[i].inner_product(&quantum_states[j]);
                let kernel_value = inner_product.norm_sqr();

                kernel_matrix[[i, j]] = kernel_value;
                if i != j {
                    kernel_matrix[[j, i]] = kernel_value; // Symmetric
                }
            }
        }

        Ok(kernel_matrix)
    }

    /// Train a kernel-based classifier using provided labeled data
    pub fn train_classifier(
        &self,
        data: &[Array1<f64>],
        labels: &[f64],
        regularization: f64
    ) -> Result<KernelModel, QuantumModelError> {
        if data.len() != labels.len() {
            return Err(QuantumModelError::EncodingError(
                format!("Data size ({}) doesn't match labels size ({})", data.len(), labels.len())
            ));
        }

        // Compute the kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(data)?;

        // Add regularization to diagonal
        let mut kernel_reg = kernel_matrix.clone();
        for i in 0..kernel_reg.shape()[0] {
            kernel_reg[[i, i]] += regularization;
        }

        // Solve the system K*alpha = y to find alpha
        // Using a very simple solver for the example (not efficient)
        // In a real implementation, use a proper linear algebra library

        // Convert labels to Array1
        let y = Array1::from_vec(labels.to_vec());

        // Start with random alphas
        let mut alpha = Array1::zeros(data.len());

        // Simple gradient descent to solve the system
        let lr = 0.01;
        let iterations = 1000;

        for _ in 0..iterations {
            let residual = kernel_reg.dot(&alpha) - &y;
            alpha = alpha - lr * kernel_reg.dot(&residual);
        }

        // Create the model
        let model = KernelModel {
            kernel: self.clone(),
            support_vectors: data.to_vec(),
            alpha: alpha.to_vec(),
        };

        Ok(model)
    }
}

impl Model for QuantumKernel {
    type Input = Array1<f64>;
    type Output = f64;
    type Error = QuantumModelError;

    fn parameter_count(&self) -> usize {
        0 // Quantum kernels typically don't have trainable parameters
    }

    fn get_parameters(&self) -> Vec<f64> {
        Vec::new()
    }

    fn set_parameters(&mut self, _parameters: &[f64]) -> Result<(), Self::Error> {
        Ok(()) // No parameters to set
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }
}

impl QuantumModel for QuantumKernel {
    fn circuit(&self) -> &QuantumCircuit {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut QuantumCircuit {
        &mut self.circuit
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn encode_input(&self, input: &Self::Input) -> Result<StateVector, Self::Error> {
        (self.feature_map)(input)
    }


    fn decode_output<S: QuantumState>(&self, _state: &S) -> Result<Self::Output, Self::Error> {
        // Kernels don't directly decode to output, they compute kernel values
        Err(QuantumModelError::DecodingError(
            "Quantum kernels don't support direct state decoding".to_string()
        ))
    }
}

/// Kernel-based machine learning model
#[derive(Clone)]
pub struct KernelModel {
    /// The underlying quantum kernel
    kernel: QuantumKernel,
    /// Support vectors (training data points)
    support_vectors: Vec<Array1<f64>>,
    /// Coefficients for each support vector
    alpha: Vec<f64>,
}

impl KernelModel {
    /// Creates a new kernel model
    pub fn new(
        kernel: QuantumKernel,
        support_vectors: Vec<Array1<f64>>,
        alpha: Vec<f64>
    ) -> Self {
        KernelModel {
            kernel,
            support_vectors,
            alpha,
        }
    }
}

impl Model for KernelModel {
    type Input = Array1<f64>;
    type Output = f64;
    type Error = QuantumModelError;

    fn parameter_count(&self) -> usize {
        self.alpha.len()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.alpha.clone()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        if parameters.len() != self.alpha.len() {
            return Err(QuantumModelError::EncodingError(
                format!("Expected {} parameters, got {}", self.alpha.len(), parameters.len())
            ));
        }

        self.alpha = parameters.to_vec();
        Ok(())
    }

    fn dimensions(&self) -> (usize, usize) {
        let (input_dim, _) = self.kernel.dimensions();
        (input_dim, 1)
    }
}

impl PredictiveModel for KernelModel {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let mut output = 0.0;

        // Compute the kernel value for each support vector
        for (sv, &alpha) in self.support_vectors.iter().zip(self.alpha.iter()) {
            let kernel_value = self.kernel.compute_kernel(input, sv)?;
            output += alpha * kernel_value;
        }

        Ok(output)
    }
}
