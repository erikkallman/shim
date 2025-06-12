//! Statevector simulator with categorical structure
//!
//! This module implements a statevector simulator for quantum circuits
//! leveraging the categorical structure of quantum operations.
use std::collections::HashMap;
use std::fmt;
use num_complex::Complex64;
use ndarray::{Array1, Array2};

use crate::quantum::circuit::QuantumCircuit;
use crate::quantum::QuantumCircuitCategory;
use crate::quantum::state::{StateVector, QuantumState, QuantumStateCategory};
use crate::quantum::gate::{QuantumGate, QuantumGateCategory};
use crate::category::prelude::*;
use crate::quantum::CircuitOptimizer;
use crate::quantum::circuit_to_gate;
use crate::quantum::optimizer::OptimizationEndofunctor;
use crate::quantum::circuit::CircuitToGateFunctor;
/// A measurement outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Measurement yielded 0
    Zero,
    /// Measurement yielded 1
    One,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Zero => write!(f, "0"),
            Outcome::One => write!(f, "1"),
        }
    }
}

/// An outcome of measuring multiple qubits
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementOutcome {
    /// The outcomes for each measured qubit
    pub outcomes: Vec<Outcome>,
    /// The probability of this outcome
    pub probability: f64,
}

impl fmt::Display for MeasurementOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for outcome in &self.outcomes {
            write!(f, "{}", outcome)?;
        }
        write!(f, " (p={})", self.probability)
    }
}

/// A statevector simulator for quantum circuits
/// using categorical structure
#[derive(Clone)]
pub struct StatevectorSimulator {
    /// The current state of the simulator
    state: StateVector,
    /// Random number generator for measurements
    rng: rand::rngs::ThreadRng,
    /// Category structures for operations
    gate_category: QuantumGateCategory,
    state_category: QuantumStateCategory,
}

impl StatevectorSimulator {
    /// Create a new statevector simulator with the specified number of qubits
    pub fn new(qubit_count: usize) -> Self {
        StatevectorSimulator {
            state: StateVector::zero_state(qubit_count),
            rng: rand::thread_rng(),
            gate_category: QuantumGateCategory,
            state_category: QuantumStateCategory,
        }
    }

    /// Create a simulator from an existing state vector
    pub fn from_state(state: StateVector) -> Self {
        StatevectorSimulator {
            state,
            rng: rand::thread_rng(),
            gate_category: QuantumGateCategory,
            state_category: QuantumStateCategory,
        }
    }

    /// Get the current state vector
    pub fn state(&self) -> &StateVector {
        &self.state
    }

    /// Set the state vector
    pub fn set_state(&mut self, state: StateVector) {
        self.state = state;
    }

    /// Reset the simulator to the |0...0⟩ state
    pub fn reset(&mut self) {
        self.state = StateVector::zero_state(self.state.qubit_count());
    }

    /// Get the number of qubits in the simulator
    pub fn qubit_count(&self) -> usize {
        self.state.qubit_count()
    }

    /// Apply a quantum gate to the specified qubits using categorical operations
    pub fn apply_gate(&mut self, gate: &dyn QuantumGate, qubits: &[usize]) -> Result<(), String> {
        // Apply the gate directly (which now uses categorical operations internally)
        let new_state = gate.apply_to_qubits(&self.state, qubits)?;
        self.state = new_state;
        Ok(())
    }
    /// Apply a quantum circuit using categorical composition
    pub fn run_circuit(&mut self, circuit: &QuantumCircuit) -> Result<(), String> {
        if circuit.qubit_count > self.qubit_count() {
            return Err(format!(
                "Circuit has {} qubits, but simulator has only {} qubits",
                circuit.qubit_count,
                self.qubit_count()
            ));
        }

        // Get the circuit as a single gate using categorical composition
        let circuit_gate = circuit.as_single_gate()?;

        // Apply the composed gate to the state
        let qubits: Vec<usize> = (0..circuit.qubit_count).collect();
        self.state = circuit_gate.apply_to_qubits(&self.state, &qubits)?;

        Ok(())
    }

    /// Measure a single qubit without collapsing the state
    pub fn measure_qubit_probability(&self, qubit: usize) -> Result<HashMap<Outcome, f64>, String> {
        if qubit >= self.qubit_count() {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let mut probabilities = HashMap::new();
        let dim = 1 << self.qubit_count();
        let mut prob_zero = 0.0;
        let mut prob_one = 0.0;

        // Calculate probabilities for outcome 0 and 1
        // This involves computing Z-projections which are morphisms in the quantum state category
        for i in 0..dim {
            let bit = (i >> (self.qubit_count() - 1 - qubit)) & 1;
            let prob = self.state.probability(i);

            if bit == 0 {
                prob_zero += prob;
            } else {
                prob_one += prob;
            }
        }

        probabilities.insert(Outcome::Zero, prob_zero);
        probabilities.insert(Outcome::One, prob_one);

        Ok(probabilities)
    }

    /// Calculate the expectation value of an observable on a single qubit
    /// using categorical structure to represent the observable
    pub fn expectation_value_single_qubit(&self, observable: &Array2<Complex64>, qubit: usize)
                                          -> Result<f64, String> {

        if qubit >= self.qubit_count() {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        // Ensure the observable is a 2x2 matrix (single qubit)
        if observable.shape() != [2, 2] {
            return Err(format!(
                "Single-qubit observable must be 2x2, got {}x{}",
                observable.shape()[0], observable.shape()[1]
            ));
        }

        // For the Z operator specifically, we can optimize by using measurement probabilities
        if self.is_z_operator(observable) {
            let probabilities = self.measure_qubit_probability(qubit)?;
            let prob_zero = probabilities.get(&Outcome::Zero).unwrap_or(&0.0);
            let prob_one = probabilities.get(&Outcome::One).unwrap_or(&0.0);
            return Ok(prob_zero - prob_one);
        }

        // For other observables, expand to full system size and use the general method
        // This is conceptually a tensor product in the monoidal category structure
        let full_observable = self.expand_single_qubit_observable(observable, qubit)?;
        self.expectation_value(&full_observable)
    }

    // Helper method to check if an observable is the Z operator
    fn is_z_operator(&self, observable: &Array2<Complex64>) -> bool {
        let z00 = Complex64::new(1.0, 0.0);
        let z11 = Complex64::new(-1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        (observable[[0, 0]] - z00).norm_sqr() < 1e-10 &&
        (observable[[0, 1]] - zero).norm_sqr() < 1e-10 &&
        (observable[[1, 0]] - zero).norm_sqr() < 1e-10 &&
        (observable[[1, 1]] - z11).norm_sqr() < 1e-10
    }

    // Expand an observable to act on a specified qubit, implementing
    // monoidal category tensor structure
    fn expand_single_qubit_observable(&self,
                                      observable: &Array2<Complex64>,
                                      qubit: usize
    ) -> Result<Array2<Complex64>, String> {
        if qubit >= self.qubit_count() {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        // Total qubits in system
        let total_qubits = self.qubit_count();
        let dim = 1 << total_qubits;
        let mut result = Array2::zeros((dim, dim));

        // Use the monoidal structure of the state category to construct the full operator
        // This implements I ⊗ ... ⊗ O ⊗ ... ⊗ I where O is at the specified qubit
        for i in 0..dim {
            for j in 0..dim {
                // Check if all bits except the qubit of interest match
                let mut matches = true;
                for q in 0..total_qubits {
                    if q != qubit {
                        let shift = total_qubits - 1 - q;
                        let bit_i = (i >> shift) & 1;
                        let bit_j = (j >> shift) & 1;
                        if bit_i != bit_j {
                            matches = false;
                            break;
                        }
                    }
                }

                if matches {
                    // Extract the bits for the target qubit
                    let shift = total_qubits - 1 - qubit;
                    let bit_i = (i >> shift) & 1;
                    let bit_j = (j >> shift) & 1;

                    // Apply the observable to the target qubit
                    result[[i, j]] = observable[[bit_i, bit_j]];
                }
            }
        }

        Ok(result)
    }

    /// Measure multiple qubits without collapsing the state
    pub fn measure_qubits_probability(&self, qubits: &[usize]) -> Result<HashMap<Vec<Outcome>, f64>, String> {
        // Validate qubit indices
        for &q in qubits {
            if q >= self.qubit_count() {
                return Err(format!("Qubit index {} out of range", q));
            }
        }

        let mut probabilities = HashMap::new();
        let dim = 1 << self.qubit_count();

        // Go through all computational basis states
        for i in 0..dim {
            let prob = self.state.probability(i);
            if prob > 1e-10 {  // Only consider non-zero probability amplitudes
                let mut outcomes = Vec::with_capacity(qubits.len());

                // Extract the bit values for each qubit
                for &q in qubits {
                    let shift = self.qubit_count() - 1 - q;  // Big-endian bit position
                    let bit = (i >> shift) & 1;
                    let outcome = if bit == 0 { Outcome::Zero } else { Outcome::One };
                    outcomes.push(outcome);
                }

                // Add to probability map
                *probabilities.entry(outcomes).or_insert(0.0) += prob;
            }
        }

        Ok(probabilities)
    }

    /// Measure a single qubit and collapse the state
    /// This operation uses the categorical structure to implement projection
    pub fn measure_qubit(&mut self, qubit: usize) -> Result<Outcome, String> {
        use rand::Rng;

        let probabilities = self.measure_qubit_probability(qubit)?;
        let prob_zero = probabilities[&Outcome::Zero];

        // Generate random value to determine measurement outcome
        let random_val = self.rng.gen::<f64>();
        let outcome = if random_val < prob_zero {
            Outcome::Zero
        } else {
            Outcome::One
        };

        // Collapse the state using categorical projection
        self.collapse_to_outcome(qubit, outcome)?;

        Ok(outcome)
    }

    /// Measure multiple qubits and collapse the state
    pub fn measure_qubits(&mut self, qubits: &[usize]) -> Result<Vec<Outcome>, String> {
        use rand::Rng;

        let probabilities = self.measure_qubits_probability(qubits)?;

        // Convert to a cumulative distribution for sampling
        let mut cumulative_probs = Vec::new();
        let mut total_prob = 0.0;

        for (outcomes, prob) in &probabilities {
            total_prob += prob;
            cumulative_probs.push((outcomes.clone(), total_prob));
        }

        // Generate random value to determine measurement outcome
        let random_val = self.rng.gen::<f64>() * total_prob;

        // Find the outcome based on the random value
        let mut measured_outcomes = Vec::new();
        for (outcomes, cum_prob) in cumulative_probs {
            if random_val <= cum_prob {
                measured_outcomes = outcomes;
                break;
            }
        }

        // Collapse the state using categorical projection
        self.collapse_to_outcomes(qubits, &measured_outcomes)?;

        Ok(measured_outcomes)
    }

    /// Collapse the state to a specific outcome for a qubit
    /// This implements a categorical projection morphism
    fn collapse_to_outcome(&mut self, qubit: usize, outcome: Outcome) -> Result<(), String> {
        if qubit >= self.qubit_count() {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        // Create a projection operator in the state category
        // This is a morphism in the QuantumStateCategory
        let dim = 1 << self.qubit_count();
        let mut new_amplitudes = Array1::zeros(dim);
        let mut norm_factor = 0.0;

        // Project onto the subspace where the qubit has the measured value
        for i in 0..dim {
            let shift = self.qubit_count() - 1 - qubit;  // Big-endian bit position
            let bit = (i >> shift) & 1;
            let bit_matches = (bit == 0 && outcome == Outcome::Zero) ||
                             (bit == 1 && outcome == Outcome::One);

            if bit_matches {
                new_amplitudes[i] = self.state.amplitudes()[i];
                norm_factor += new_amplitudes[i].norm_sqr();
            }
        }

        // Normalize the new state
        if norm_factor < 1e-10 {
            return Err(format!("Zero probability for outcome {:?} on qubit {}", outcome, qubit));
        }

        let norm_factor = 1.0 / norm_factor.sqrt();
        for i in 0..dim {
            new_amplitudes[i] *= Complex64::new(norm_factor, 0.0);
        }

        // Update the state - this is the result of applying the projection morphism
        self.state = StateVector::new(self.qubit_count(), new_amplitudes)?;

        Ok(())
    }

    /// Collapse the state to specific outcomes for multiple qubits
    /// This implements a composite categorical projection
    fn collapse_to_outcomes(&mut self, qubits: &[usize], outcomes: &[Outcome]) -> Result<(), String> {
        if qubits.len() != outcomes.len() {
            return Err(format!(
                "Number of qubits ({}) doesn't match number of outcomes ({})",
                qubits.len(), outcomes.len()
            ));
        }

        for &q in qubits {
            if q >= self.qubit_count() {
                return Err(format!("Qubit index {} out of range", q));
            }
        }

        // Create a projection operator that is a tensor product of individual projections
        // This uses the monoidal structure of the state category
        let dim = 1 << self.qubit_count();
        let mut new_amplitudes = Array1::zeros(dim);
        let mut norm_factor = 0.0;

        // Project onto the subspace where all qubits have their measured values
        for i in 0..dim {
            let mut matches = true;

            for (j, &q) in qubits.iter().enumerate() {
                let shift = self.qubit_count() - 1 - q;  // Big-endian bit position
                let bit = (i >> shift) & 1;
                let expected_bit = match outcomes[j] {
                    Outcome::Zero => 0,
                    Outcome::One => 1,
                };

                if bit != expected_bit {
                    matches = false;
                    break;
                }
            }

            if matches {
                new_amplitudes[i] = self.state.amplitudes()[i];
                norm_factor += new_amplitudes[i].norm_sqr();
            }
        }

        // Normalize the new state
        if norm_factor < 1e-10 {
            return Err(format!(
                "Zero probability for specified outcomes on qubits {:?}",
                qubits
            ));
        }

        let norm_factor = 1.0 / norm_factor.sqrt();
        for i in 0..dim {
            new_amplitudes[i] *= Complex64::new(norm_factor, 0.0);
        }

        // Update the state with the result of the composite projection
        self.state = StateVector::new(self.qubit_count(), new_amplitudes)?;

        Ok(())
    }

    /// Calculate the expectation value of a Hermitian observable
    /// This represents an inner product in the state category
    pub fn expectation_value(&self, observable: &Array2<Complex64>) -> Result<f64, String> {
        // Ensure the observable is the right size
        let dim = 1 << self.qubit_count();
        if observable.shape() != [dim, dim] {
            return Err(format!(
                "Observable dimension mismatch: expected {}x{}, got {}x{}",
                dim, dim, observable.shape()[0], observable.shape()[1]
            ));
        }

        // Calculate ⟨ψ|O|ψ⟩ - this is an inner product in the state category
        let state_vec = self.state.amplitudes();
        let o_psi = observable.dot(state_vec);

        let mut expectation = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            expectation += state_vec[i].conj() * o_psi[i];
        }

        // The expectation value should be real for a Hermitian observable
        if expectation.im.abs() > 1e-10 {
            return Err(format!(
                "Non-real expectation value: {}. Observable might not be Hermitian.",
                expectation
            ));
        }

        Ok(expectation.re)
    }

    /// Sample measurement outcomes multiple times
    pub fn sample_measurements(&mut self, qubits: &[usize], shots: usize) -> Result<HashMap<Vec<Outcome>, usize>, String> {
        let mut results = HashMap::new();
        let original_state = self.state.clone();

        for _ in 0..shots {
            // Reset to the original state for each measurement
            self.state = original_state.clone();

            // Measure and record the outcome
            let outcome = self.measure_qubits(qubits)?;
            *results.entry(outcome).or_insert(0) += 1;
        }

        // Restore the original state
        self.state = original_state;

        Ok(results)
    }

    /// Calculate all possible measurement outcomes and their probabilities
    pub fn get_measurement_outcomes(&self, qubits: &[usize]) -> Result<Vec<MeasurementOutcome>, String> {
        let probabilities = self.measure_qubits_probability(qubits)?;

        let mut outcomes = Vec::new();
        for (outcome_vec, probability) in probabilities {
            outcomes.push(MeasurementOutcome {
                outcomes: outcome_vec,
                probability,
            });
        }

        // Sort by probability (highest first)
        outcomes.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        Ok(outcomes)
    }

    /// Implement categorical tensor product of simulators
    pub fn tensor(&self, other: &StatevectorSimulator) -> Result<StatevectorSimulator, String> {
        // Tensor product of states using the monoidal structure
        let combined_state = self.state.tensor(&other.state);

        // Create a new simulator with the combined state
        Ok(StatevectorSimulator {
            state: combined_state,
            rng: rand::thread_rng(),
            gate_category: self.gate_category.clone(),
            state_category: self.state_category.clone(),
        })
    }

    /// Execute a unitary operation defined by a categorical morphism
    pub fn apply_categorical_operation(&mut self, morphism: &Array2<Complex64>) -> Result<(), String> {
        // The morphism should be a unitary operator in the state category
        let dim = 1 << self.qubit_count();
        if morphism.shape() != [dim, dim] {
            return Err(format!(
                "Morphism dimension mismatch: expected {}x{}, got {}x{}",
                dim, dim, morphism.shape()[0], morphism.shape()[1]
            ));
        }

        // Apply the morphism to the state
        self.state = self.state.apply_matrix(morphism)?;

        Ok(())
    }

    /// Run circuit with optimization using proper categorical composition
    pub fn run_circuit_with_optimization(&mut self, circuit: &QuantumCircuit) -> Result<(), String> {
        // Check compatibility
        if circuit.qubit_count > self.qubit_count() {
            return Err(format!(
                "Circuit has {} qubits, but simulator has only {} qubits",
                circuit.qubit_count,
                self.qubit_count()
            ));
        }

        // 1. Circuit to gate: functor from CircuitCategory to GateCategory
        let circuit_to_gate_functor = CircuitToGateFunctor;
        let gate = circuit_to_gate_functor.map_morphism(
            &QuantumCircuitCategory,
            &self.gate_category,
            circuit
        );

        // 2. Optimization: endofunctor on GateCategory
        let optimization_endofunctor = OptimizationEndofunctor::new(
            CircuitOptimizer::default()
        );
        let optimized_gate = optimization_endofunctor.map_morphism(
            &self.gate_category,
            &self.gate_category,
            &gate
        );

        // 3. Apply gate: morphism from GateCategory to StateCategory
        let qubits: Vec<usize> = (0..circuit.qubit_count).collect();
        self.apply_gate(&*optimized_gate, &qubits)
    }

    /// Run multiple optimized circuits in sequence, treating them as a single composition
    pub fn run_circuit_sequence(&mut self, circuits: &[QuantumCircuit]) -> Result<(), String> {
        if circuits.is_empty() {
            return Ok(());
        }

        // Check compatibility of all circuits
        let qubit_count = circuits[0].qubit_count;
        for (i, circuit) in circuits.iter().enumerate().skip(1) {
            if circuit.qubit_count != qubit_count {
                return Err(format!(
                    "Circuit at index {} has different qubit count: {} != {}",
                    i, circuit.qubit_count, qubit_count
                ));
            }
        }

        // Convert all circuits to gates and compose using categorical composition
        let gates: Result<Vec<Box<dyn QuantumGate>>, String> = circuits
            .iter()
            .map(|c| Ok(circuit_to_gate(c)))
            .collect();

        let gates = gates?;

        // Use categorical composition to create a single gate
        let mut composed_gate = gates[0].clone_box();
        for gate in gates.iter().skip(1) {
            composed_gate = self.gate_category.compose(&composed_gate, gate)
                .ok_or_else(|| "Failed to compose gates".to_string())?;
        }

        // Optimize the composed circuit
        let optimizer = CircuitOptimizer::default();
        let optimized_gate = optimizer.optimize_categorical(&composed_gate);

        // Apply the optimized gate
        let qubits: Vec<usize> = (0..qubit_count).collect();
        self.apply_gate(&*optimized_gate, &qubits)
    }
}
