//! Parametrized quantum circuit models

use ndarray::Array1;
use num_complex::Complex64;

use crate::quantum::circuit::{QuantumCircuit, CircuitBuilder};
use crate::quantum::gate::{QuantumGate, StandardGate, ParametrizedGate};
use crate::quantum::state::{QuantumState, StateVector};
use crate::machine_learning::core::{Model, PredictiveModel};
use crate::machine_learning::quantum::model::{
    QuantumModel, CategoricalQuantumModel, QuantumModelError,
    EncodingStrategy, DecodingStrategy
};
use crate::machine_learning::optimizer::Optimizer;
use crate::machine_learning::loss::LossFunction;
use crate::machine_learning::categorical::functors::DataToCircuitFunctor;
use crate::machine_learning::categorical::{TrainingTransformation};
use crate::machine_learning::categorical::functors::CircuitToPredictionFunctor;

/// Enum representing types of parametrized gates
#[derive(Debug, Clone, PartialEq)]
pub enum ParamGateType {
    /// Rotation around X axis
    Rx,
    /// Rotation around Y axis
    Ry,
    /// Rotation around Z axis
    Rz,
    /// Phase shift gate
    Phase,
    /// Controlled rotation around X axis
    CRx,
    /// Controlled rotation around Y axis
    CRy,
    /// Controlled rotation around Z axis
    CRz,
    /// Controlled phase shift
    CPhase,
    /// Custom gate (identified by name)
    Custom(String),
}

/// Structure mapping parameters to gates in the circuit
#[derive(Debug, Clone)]
pub struct ParameterMapping {
    /// Index of the gate in the circuit
    pub gate_index: usize,
    /// Index of the parameter in the parameters vector
    pub param_index: usize,
    /// Type of the gate
    pub gate_type: ParamGateType,
}

/// Parametrized quantum circuit model
#[derive(Clone, Debug)]
pub struct ParametrizedCircuitModel {
    /// The quantum circuit
    circuit: QuantumCircuit,
    /// Trainable parameters
    parameters: Vec<f64>,
    /// Mapping between parameters and gates
    parameter_mapping: Vec<ParameterMapping>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Strategy for encoding classical data to quantum states
    pub encoding_strategy: EncodingStrategy,
    /// Strategy for decoding quantum states to classical outputs
    pub decoding_strategy: DecodingStrategy,
}

impl ParametrizedCircuitModel {
    /// Creates a new parametrized circuit model
    pub fn new(
        qubit_count: usize,
        parameters: Vec<f64>,
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        ParametrizedCircuitModel {
            circuit: QuantumCircuit::new(qubit_count),
            parameters,
            parameter_mapping: Vec::new(),
            input_dim,
            output_dim,
            encoding_strategy: EncodingStrategy::AngleEncoding,
            decoding_strategy: DecodingStrategy::MeasurementBased(vec![0]),
        }
    }

    /// Adds a parametrized gate to the circuit with a reference to a parameter
    pub fn add_parametrized_gate(
        &mut self,
        gate_type: ParamGateType,
        target_qubit: usize,
        param_index: usize,
    ) -> Result<(), QuantumModelError> {
        if param_index >= self.parameters.len() {
            return Err(QuantumModelError::CircuitError(
                format!("Parameter index {} out of bounds", param_index)
            ));
        }

        let param_value = self.parameters[param_index];
        let gate: Box<dyn QuantumGate> = match gate_type {
            ParamGateType::Rx => Box::new(ParametrizedGate::Rx(param_value)),
            ParamGateType::Ry => Box::new(ParametrizedGate::Ry(param_value)),
            ParamGateType::Rz => Box::new(ParametrizedGate::Rz(param_value)),
            ParamGateType::Phase => Box::new(ParametrizedGate::Phase(param_value)),
            ParamGateType::Custom(_) => return Err(QuantumModelError::CircuitError(
                "Custom gate types not implemented".to_string()
            )),
            _ => return Err(QuantumModelError::CircuitError(
                format!("Gate type {:?} requires multiple qubits", gate_type)
            )),
        };

        let gate_index = self.circuit.gates.len();
        self.circuit.add_gate(gate, &[target_qubit])
            .map_err(|e| QuantumModelError::CircuitError(e.to_string()))?;

        self.parameter_mapping.push(ParameterMapping {
            gate_index,
            param_index,
            gate_type,
        });

        Ok(())
    }

    /// Adds a controlled parametrized gate
    pub fn add_controlled_parametrized_gate(
        &mut self,
        gate_type: ParamGateType,
        control_qubit: usize,
        target_qubit: usize,
        param_index: usize,
    ) -> Result<(), QuantumModelError> {
        if param_index >= self.parameters.len() {
            return Err(QuantumModelError::CircuitError(
                format!("Parameter index {} out of bounds", param_index)
            ));
        }

        let param_value = self.parameters[param_index];
        let gate: Box<dyn QuantumGate> = match gate_type {
            ParamGateType::CRx => Box::new(ParametrizedGate::CRx(param_value)),
            ParamGateType::CRy => Box::new(ParametrizedGate::CRy(param_value)),
            ParamGateType::CRz => Box::new(ParametrizedGate::CRz(param_value)),
            ParamGateType::CPhase => Box::new(ParametrizedGate::CPhase(param_value)),
            ParamGateType::Custom(_) => return Err(QuantumModelError::CircuitError(
                "Custom gate types not implemented".to_string()
            )),
            _ => return Err(QuantumModelError::CircuitError(
                format!("Gate type {:?} is not a controlled gate", gate_type)
            )),
        };

        let gate_index = self.circuit.gates.len();
        self.circuit.add_gate(gate, &[control_qubit, target_qubit])
            .map_err(|e| QuantumModelError::CircuitError(e.to_string()))?;

        self.parameter_mapping.push(ParameterMapping {
            gate_index,
            param_index,
            gate_type,
        });

        Ok(())
    }

    /// Adds a variational layer (pattern of parametrized gates)
    pub fn add_variational_layer(&mut self) -> Result<(), QuantumModelError> {
        let qubit_count = self.circuit.qubit_count;
        let param_offset = self.parameter_mapping.len();

        // Check if we have enough parameters
        if param_offset + 3 * qubit_count > self.parameters.len() {
            return Err(QuantumModelError::CircuitError(
                "Not enough parameters for variational layer".to_string()
            ));
        }

        // Add rotation gates on each qubit (Rx, Ry, Rz)
        for qubit in 0..qubit_count {
            self.add_parametrized_gate(
                ParamGateType::Rx,
                qubit,
                param_offset + qubit * 3
            )?;

            self.add_parametrized_gate(
                ParamGateType::Ry,
                qubit,
                param_offset + qubit * 3 + 1
            )?;

            self.add_parametrized_gate(
                ParamGateType::Rz,
                qubit,
                param_offset + qubit * 3 + 2
            )?;
        }

        // Add entangling gates between adjacent qubits
        for qubit in 0..qubit_count - 1 {
            self.circuit.add_gate(Box::new(StandardGate::CNOT), &[qubit, qubit + 1])
                .map_err(|e| QuantumModelError::CircuitError(e.to_string()))?;
        }

        Ok(())
    }

    /// Update the circuit gates with current parameters
    fn update_circuit_parameters(&mut self) -> Result<(), QuantumModelError> {
        for mapping in &self.parameter_mapping {
            let param_value = self.parameters[mapping.param_index];
            let (gate, _qubits) = &mut self.circuit.gates[mapping.gate_index];

            // Create a new gate with the updated parameter
            let new_gate: Box<dyn QuantumGate> = match mapping.gate_type {
                ParamGateType::Rx => Box::new(ParametrizedGate::Rx(param_value)),
                ParamGateType::Ry => Box::new(ParametrizedGate::Ry(param_value)),
                ParamGateType::Rz => Box::new(ParametrizedGate::Rz(param_value)),
                ParamGateType::Phase => Box::new(ParametrizedGate::Phase(param_value)),
                ParamGateType::CRx => Box::new(ParametrizedGate::CRx(param_value)),
                ParamGateType::CRy => Box::new(ParametrizedGate::CRy(param_value)),
                ParamGateType::CRz => Box::new(ParametrizedGate::CRz(param_value)),
                ParamGateType::CPhase => Box::new(ParametrizedGate::CPhase(param_value)),
                ParamGateType::Custom(_) => return Err(QuantumModelError::CircuitError(
                    "Custom gate types not implemented".to_string()
                )),
            };

            // Replace the gate
            *gate = new_gate;
        }

        Ok(())
    }
}

impl Model for ParametrizedCircuitModel {
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    type Error = QuantumModelError;

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn get_parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), Self::Error> {
        if parameters.len() != self.parameters.len() {
            return Err(QuantumModelError::from(
                crate::machine_learning::core::ModelError::DimensionMismatch(
                    format!("Expected {} parameters, got {}", self.parameters.len(), parameters.len())
                )
            ));
        }

        self.parameters = parameters.to_vec();
        self.update_circuit_parameters()?;

        Ok(())
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }
}

impl QuantumModel for ParametrizedCircuitModel {
    fn circuit(&self) -> &QuantumCircuit {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut QuantumCircuit {
        &mut self.circuit
    }

    fn qubit_count(&self) -> usize {
        self.circuit.qubit_count
    }

    fn encode_input(&self, input: &Self::Input) -> Result<StateVector, QuantumModelError> {
        if input.len() != self.input_dim {
            return Err(QuantumModelError::EncodingError(
                format!("Expected input dimension {}, got {}", self.input_dim, input.len())
            ));
        }

        match &self.encoding_strategy {
            EncodingStrategy::AmplitudeEncoding => {
                // Normalize the input vector to create a valid quantum state
                let norm = input.dot(input).sqrt();
                if norm < 1e-10 {
                    return Err(QuantumModelError::EncodingError(
                        "Input vector has zero norm".to_string()
                    ));
                }

                let normalized = input / norm;

                // Convert to complex amplitudes
                let mut amplitudes = Vec::with_capacity(1 << self.qubit_count());
                for i in 0..normalized.len() {
                    if i < (1 << self.qubit_count()) {
                        amplitudes.push(Complex64::new(normalized[i], 0.0));
                    } else {
                        break;
                    }
                }

                // Pad with zeros if needed
                while amplitudes.len() < (1 << self.qubit_count()) {
                    amplitudes.push(Complex64::new(0.0, 0.0));
                }

                StateVector::new(self.qubit_count(), amplitudes.into())
                    .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
            },

            EncodingStrategy::AngleEncoding => {
                // Start with |0⟩ state
                let mut builder = CircuitBuilder::new(self.qubit_count());

                // Apply rotation gates based on input values
                for (i, &value) in input.iter().enumerate() {
                    if i < self.qubit_count() {
                        // Apply Ry and Rz rotations
                        builder.ry(i, value)
                            .map_err(|e| QuantumModelError::EncodingError(e.to_string()))?;

                        if i + 1 < input.len() {
                            builder.rz(i, input[i + 1])
                                .map_err(|e| QuantumModelError::EncodingError(e.to_string()))?;
                        }
                    }
                }

                // Build and run the encoding circuit
                let encoding_circuit = builder.build();
                let initial_state = StateVector::zero_state(self.qubit_count());

                encoding_circuit.apply(&initial_state)
                    .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
            },

            EncodingStrategy::BasisEncoding => {
                // Convert input to binary representation
                let mut index = 0;
                for (i, &value) in input.iter().enumerate() {
                    if value > 0.5 && i < self.qubit_count() {
                        index |= 1 << i;
                    }
                }

                StateVector::computational_basis(self.qubit_count(), index)
                    .map_err(|e| QuantumModelError::EncodingError(e.to_string()))
            },

            EncodingStrategy::Custom(encoder) => {
                encoder(input, self.qubit_count())
            },
        }
    }

    fn decode_output<S: QuantumState>(&self, state: &S) -> Result<Self::Output, QuantumModelError> {
        // First check if the state is a StateVector

        let state_vec = state
            .as_any()
            .downcast_ref::<StateVector>()
            .ok_or_else(|| QuantumModelError::DecodingError(
                "Only StateVector representation is supported for output decoding".to_string()
            ))?;

        // Now proceed with the same logic as before
        match &self.decoding_strategy {
            DecodingStrategy::MeasurementBased(qubits) => {
                // Get measurement probabilities for the specified qubits
                let amplitudes = state_vec.amplitudes();
                let mut result = Array1::zeros(self.output_dim);
                // Simple strategy: map probability of |1⟩ on each measured qubit to output
                for (i, &qubit) in qubits.iter().enumerate() {
                    if i < self.output_dim {
                        // Calculate probability of measuring |1⟩
                        let mut prob_1 = 0.0;
                        for (j, amp) in amplitudes.iter().enumerate() {
                            if (j >> qubit) & 1 == 1 {
                                prob_1 += amp.norm_sqr();
                            }
                        }
                        result[i] = prob_1;
                    }
                }
                Ok(result)
            },
            DecodingStrategy::ExpectationBased(observables) => {
                let mut result = Array1::zeros(self.output_dim);
                // Calculate expectation value for each observable
                for (i, observable) in observables.iter().enumerate() {
                    if i < self.output_dim {
                        let amplitude_vec = Array1::from_iter(state_vec.amplitudes().iter().cloned());
                        let expectation = amplitude_vec.dot(&observable.dot(&amplitude_vec.clone())).re;
                        result[i] = expectation;
                    }
                }
                Ok(result)
            },
            DecodingStrategy::Custom(decoder) => {
                decoder(state_vec)
            },
        }
    }
}

impl PredictiveModel for ParametrizedCircuitModel {
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        // Encode the input as a quantum state
        let input_state = self.encode_input(input)?;

        // Run the quantum circuit
        let output_state = self.run_circuit(&input_state)?;

        // Decode the output state to classical output
        self.decode_output(&output_state)
    }
}

impl CategoricalQuantumModel for ParametrizedCircuitModel {

    fn encoding_functor(&self) -> DataToCircuitFunctor {
        DataToCircuitFunctor {
            qubit_count: self.qubit_count(),
            encoding_strategy: match &self.encoding_strategy {
                EncodingStrategy::AmplitudeEncoding => EncodingStrategy::AmplitudeEncoding,
                EncodingStrategy::AngleEncoding => EncodingStrategy::AngleEncoding,
                EncodingStrategy::BasisEncoding => EncodingStrategy::BasisEncoding,
                EncodingStrategy::Custom(_) => EncodingStrategy::AngleEncoding, // Default to angle encoding for custom strategies
            },
        }
    }

    fn decoding_functor(&self) -> CircuitToPredictionFunctor {
        CircuitToPredictionFunctor {
            qubit_count: self.qubit_count(),
            decoding_strategy: match &self.decoding_strategy {
                DecodingStrategy::MeasurementBased(qubits) => DecodingStrategy::MeasurementBased(qubits.clone()),
                DecodingStrategy::ExpectationBased(obs) => DecodingStrategy::ExpectationBased(obs.clone()),
                DecodingStrategy::Custom(_) => DecodingStrategy::MeasurementBased(vec![0]), // Default for custom strategies
            },
        }
    }

    fn training_transformation<O: Optimizer + Clone, L: LossFunction + Clone>(
        &self,
        optimizer: &O,
        loss: &L
    ) -> TrainingTransformation<Self, O, L>
    where
        Self: Sized + Clone,
    {
        TrainingTransformation::new(self.clone(), optimizer.clone(), loss.clone())
    }
}
