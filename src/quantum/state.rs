// src/quantum/state.rs
//! Quantum state representations
//!
//! This module defines quantum states and their categorical structure.

use std::fmt::{self, Debug, Display};
use num_complex::Complex64;
use ndarray::{Array1, Array2};
use crate::quantum::QuantumCircuit;
use crate::category::prelude::*;
use std::any::Any;
use crate::quantum::DensityMatrix;

/// Trait for quantum states in different representations
pub trait QuantumState: Clone + Debug where Self: 'static {
    /// Returns the number of qubits in this quantum state
    fn qubit_count(&self) -> usize;

    /// Returns the dimension of the Hilbert space (2^n for n qubits)
    fn dimension(&self) -> usize {
        1 << self.qubit_count()
    }

    /// Check if the state is valid (e.g., normalized)
    fn is_valid(&self) -> bool;

    /// Tensor product with another quantum state
    fn tensor(&self, other: &Self) -> Self;

    /// Partial trace over specified qubits
    fn partial_trace(&self, qubits: &[usize]) -> Option<Self>;

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// State vector representation of a quantum state
#[derive(Clone, Debug)]
pub struct StateVector {
    /// Number of qubits
    pub qubit_count: usize,

    /// The state vector as an array of complex amplitudes
    amplitudes: Array1<Complex64>,
}

impl StateVector {
    /// Create a new state vector with the given amplitudes
    pub fn new(qubit_count: usize, amplitudes: Array1<Complex64>) -> Result<Self, String> {
        let expected_dim = 1 << qubit_count;

        if amplitudes.len() != expected_dim {
            return Err(format!(
                "State vector dimension mismatch: expected {}, got {}",
                expected_dim, amplitudes.len()
            ));
        }

        let state = StateVector {
            qubit_count,
            amplitudes,
        };

        if !state.is_valid() {
            return Err("State vector is not normalized".to_string());
        }

        Ok(state)
    }

    /// Set the amplitudes data for this state vector
    pub fn set_data(&mut self, data: Vec<Complex64>) -> Result<(), String> {
        if data.len() != self.dimension() {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension(), data.len()
            ));
        }

        // Convert Vec to Array1
        self.amplitudes = Array1::from(data);

        // Check if the state is normalized
        if !self.is_valid() {
            return Err("State vector is not normalized".to_string());
        }

        Ok(())
    }
    /// Create a new state vector in the computational basis state |index⟩
    pub fn computational_basis(qubit_count: usize, index: usize) -> Result<Self, String> {
        let dim = 1 << qubit_count;

        if index >= dim {
            return Err(format!(
                "Index {} is out of range for {}-qubit state",
                index, qubit_count
            ));
        }

        let mut amplitudes = Array1::zeros(dim);
        amplitudes[index] = Complex64::new(1.0, 0.0);

        Ok(StateVector {
            qubit_count,
            amplitudes,
        })
    }

    /// Create the zero state |00...0⟩
    pub fn zero_state(qubit_count: usize) -> Self {
        Self::computational_basis(qubit_count, 0).unwrap()
    }

    /// Inner product with another state vector
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        if self.qubit_count != other.qubit_count {
            panic!("Dimension mismatch in inner product");
        }

        // Compute ⟨self|other⟩
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension() {
            result += self.amplitudes[i].conj() * other.amplitudes[i];
        }

        result
    }

    /// Calculate the probability of measuring the given bit string
    pub fn probability(&self, bit_string: usize) -> f64 {
        if bit_string >= self.dimension() {
            return 0.0;
        }

        let amplitude = self.amplitudes[bit_string];
        amplitude.norm_sqr()
    }

    /// Get a reference to the amplitudes
    pub fn amplitudes(&self) -> &Array1<Complex64> {
        &self.amplitudes
    }

    /// Apply a gate matrix to this state vector
    pub fn apply_matrix(&self, matrix: &Array2<Complex64>) -> Result<Self, String> {
        let dim = self.dimension();

        if matrix.shape() != [dim, dim] {
            return Err(format!(
                "Matrix dimension mismatch: expected {}x{}, got {}x{}",
                dim, dim, matrix.shape()[0], matrix.shape()[1]
            ));
        }

        let new_amplitudes = matrix.dot(&self.amplitudes);

        Ok(StateVector {
            qubit_count: self.qubit_count,
            amplitudes: new_amplitudes,
        })
    }

    pub fn partial_trace(&self, qubits: &[usize]) -> Option<Self> {
        // Convert to density matrix
        let dm = DensityMatrix::from_state_vector(self);

        // Perform partial trace on the density matrix
        let traced_dm = dm.partial_trace(qubits)?;

        // Try to convert back to a state vector (succeeds only for pure states)
        traced_dm.to_state_vector()
    }
}

impl QuantumState for StateVector {

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn is_valid(&self) -> bool {
        // Check if the state vector is normalized
        let norm_sqr: f64 = self.amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum();

        (norm_sqr - 1.0).abs() < 1e-10
    }

    fn tensor(&self, other: &Self) -> Self {
        let self_dim = self.dimension();
        let other_dim = other.dimension();
        let new_dim = self_dim * other_dim;
        let new_qubit_count = self.qubit_count + other.qubit_count;

        let mut new_amplitudes = Array1::zeros(new_dim);

        for i in 0..self_dim {
            for j in 0..other_dim {
                let idx = i * other_dim + j;
                new_amplitudes[idx] = self.amplitudes[i] * other.amplitudes[j];
            }
        }

        StateVector {
            qubit_count: new_qubit_count,
            amplitudes: new_amplitudes,
        }
    }

    fn partial_trace(&self, qubits: &[usize]) -> Option<Self> {
        self.partial_trace(qubits)
    }
}

impl Display for StateVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}-qubit state:", self.qubit_count)?;

        let threshold = 1e-10;
        let mut has_entries = false;

        for i in 0..self.dimension() {
            let amp = self.amplitudes[i];
            if amp.norm_sqr() > threshold {
                has_entries = true;

                // Convert i to binary representation for the ket label
                let bit_string = format!("{:0width$b}", i, width = self.qubit_count);

                write!(f, "  ({:.6}{:+.6}i) |{}⟩", amp.re, amp.im, bit_string)?;

                // Add probability
                let prob = amp.norm_sqr();
                if prob > threshold {
                    write!(f, " [{:.1}%]", prob * 100.0)?;
                }

                writeln!(f)?;
            }
        }

        if !has_entries {
            writeln!(f, "  (zero state)")?;
        }

        Ok(())
    }
}

/// A single qubit state, which can be represented more efficiently
#[derive(Clone, Debug)]
pub struct Qubit {
    /// The qubit state in the form α|0⟩ + β|1⟩
    alpha: Complex64,
    beta: Complex64,
}

impl Qubit {
    /// Create a new qubit state
    pub fn new(alpha: Complex64, beta: Complex64) -> Result<Self, String> {
        let qubit = Qubit { alpha, beta };

        if !qubit.is_valid() {
            return Err("Qubit state is not normalized".to_string());
        }

        Ok(qubit)
    }

    /// Create the |0⟩ state
    pub fn zero() -> Self {
        Qubit {
            alpha: Complex64::new(1.0, 0.0),
            beta: Complex64::new(0.0, 0.0),
        }
    }

    /// Create the |1⟩ state
    pub fn one() -> Self {
        Qubit {
            alpha: Complex64::new(0.0, 0.0),
            beta: Complex64::new(1.0, 0.0),
        }
    }

    /// Create the |+⟩ state
    pub fn plus() -> Self {
        Qubit {
            alpha: Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            beta: Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        }
    }

    /// Create the |-⟩ state
    pub fn minus() -> Self {
        Qubit {
            alpha: Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            beta: Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        }
    }

    /// Convert to a StateVector
    pub fn to_state_vector(&self) -> StateVector {
        let mut amplitudes = Array1::zeros(2);
        amplitudes[0] = self.alpha;
        amplitudes[1] = self.beta;

        StateVector {
            qubit_count: 1,
            amplitudes,
        }
    }
}

impl QuantumState for Qubit {

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        1
    }

    fn is_valid(&self) -> bool {
        let norm_sqr = self.alpha.norm_sqr() + self.beta.norm_sqr();
        (norm_sqr - 1.0).abs() < 1e-10
    }

    fn tensor(&self, other: &Self) -> Self {
        // When tensoring two qubits, we get a 2-qubit state
        // which can't be represented as a Qubit, so we need to convert
        let self_sv = self.to_state_vector();
        let other_sv = other.to_state_vector();

        let _result_sv = self_sv.tensor(&other_sv);

        // For proper implementation, we should return a StateVector or a multi-qubit type
        // but for demonstration we'll just return |0⟩
        Self::zero()
    }

    fn partial_trace(&self, _qubits: &[usize]) -> Option<Self> {
        // Partial trace of a single qubit makes no sense
        None
    }
}

/// A category of quantum states
#[derive(Debug, Clone)]
pub struct QuantumStateCategory;

impl QuantumStateCategory {
    /// Convert a quantum circuit to a morphism in the quantum state category
    pub fn circuit_to_morphism(&self, circuit: &QuantumCircuit) -> Array2<Complex64> {
        // If the circuit is empty, return the identity matrix
        if circuit.gates.is_empty() {
            return self.identity(&circuit.qubit_count);
        }

        // Convert the circuit to a single quantum gate
        match circuit.as_single_gate() {
            Ok(gate) => gate.matrix(),
            Err(_) => {
                // In case of error, return identity
                // A more robust implementation might propagate the error
                self.identity(&circuit.qubit_count)
            }
        }
    }
}

/// Define objects and morphisms in the category of quantum states
impl Category for QuantumStateCategory {
    type Object = usize; // Number of qubits
    type Morphism = Array2<Complex64>; // Linear operator as a complex matrix

    fn domain(&self, _f: &Self::Morphism) -> Self::Object {
        // For simplicity, we don't track the domain and codomain in the morphism itself
        // In a more sophisticated implementation, morphisms would include this information
        0 // Placeholder
    }

    fn codomain(&self, _f: &Self::Morphism) -> Self::Object {
        // Same as above
        0 // Placeholder
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        // Identity matrix of dimension 2^obj
        let dim = 1 << obj;
        Array2::from_diag(&Array1::from_elem(dim, Complex64::new(1.0, 0.0)))
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        // Matrix multiplication g · f
        // Note the order: g comes after f in the composition
        if f.shape()[0] != g.shape()[1] {
            return None;
        }

        Some(g.dot(f))
    }
}

/// Implement MonoidalCategory for QuantumStateCategory
impl MonoidalCategory for QuantumStateCategory {
    fn unit(&self) -> Self::Object {
        0 // 0-qubit system (scalar)
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b // Tensor product adds qubit counts
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        // Kronecker product of matrices
        let f_rows = f.shape()[0];
        let f_cols = f.shape()[1];
        let g_rows = g.shape()[0];
        let g_cols = g.shape()[1];

        let mut result = Array2::zeros((f_rows * g_rows, f_cols * g_cols));

        for i in 0..f_rows {
            for j in 0..f_cols {
                for k in 0..g_rows {
                    for l in 0..g_cols {
                        result[[i * g_rows + k, j * g_cols + l]] = f[[i, j]] * g[[k, l]];
                    }
                }
            }
        }

        result
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Left unitor: I ⊗ A → A
        // For quantum states, this is trivial since tensoring with 0 qubits doesn't change anything
        self.identity(a)
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Right unitor: A ⊗ I → A
        // Similarly trivial
        self.identity(a)
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        // Associator: (A ⊗ B) ⊗ C → A ⊗ (B ⊗ C)
        // For quantum states, this is trivial since tensor product is strictly associative
        let total_qubits = a + b + c;
        self.identity(&total_qubits)
    }
}

/// Implement SymmetricMonoidalCategory for QuantumStateCategory
impl SymmetricMonoidalCategory for QuantumStateCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Braiding: A ⊗ B → B ⊗ A
        // This is the swap operator that exchanges subsystems

        let dim_a = 1 << a;
        let dim_b = 1 << b;
        let total_dim = dim_a * dim_b;

        let mut result = Array2::zeros((total_dim, total_dim));

        for i in 0..dim_a {
            for j in 0..dim_b {
                // Map |i⟩⊗|j⟩ to |j⟩⊗|i⟩
                let src_idx = i * dim_b + j;
                let dst_idx = j * dim_a + i;
                result[[dst_idx, src_idx]] = Complex64::new(1.0, 0.0);
            }
        }

        result
    }
}

/// A functor from classical data to quantum states
pub struct StatePreparationFunctor;

/// We'll need to define a simple category for classical data
#[derive(Debug)]
pub struct ClassicalDataCategory;

impl Category for ClassicalDataCategory {
    type Object = usize; // Dimension of the data space
    type Morphism = Array2<f64>; // Classical data transformation

    fn domain(&self, _f: &Self::Morphism) -> Self::Object {
        // Placeholder
        0
    }

    fn codomain(&self, _f: &Self::Morphism) -> Self::Object {
        // Placeholder
        0
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        let dim = *obj;
        Array2::eye(dim)
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        if f.shape()[0] != g.shape()[1] {
            return None;
        }

        Some(g.dot(f))
    }
}

impl Functor<ClassicalDataCategory, QuantumStateCategory> for StatePreparationFunctor {
    fn map_object(&self, _c: &ClassicalDataCategory, _d: &QuantumStateCategory, obj: &usize) -> usize {
        // Map classical dimension to qubit count
        // We need log2(dimension) qubits to represent the classical space
        (*obj as f64).log2().ceil() as usize
    }

    fn map_morphism(
        &self,
        _c: &ClassicalDataCategory,
        _d: &QuantumStateCategory,
        f: &Array2<f64>
    ) -> Array2<Complex64> {
        // Map classical transformation to quantum operator
        // This is a simplification; a real implementation would be more sophisticated
        let shape = f.shape();
        let mut result = Array2::zeros((shape[0], shape[1]));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result[[i, j]] = Complex64::new(f[[i, j]], 0.0);
            }
        }

        result
    }
}
