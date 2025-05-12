// src/quantum/gate.rs
//! Quantum gates implementation
//!
//! This module defines quantum gates as morphisms in the category of quantum states
//! and implements categorical structures for quantum computation.

use std::any::Any;
use std::fmt::{Debug};
use num_complex::{Complex64};
use ndarray::{array, Array1, Array2};

use crate::category::prelude::*;
use super::state::{QuantumState, StateVector};

use crate::quantum::circuit_to_gate;
use crate::quantum::QuantumCircuit;

/// Common complex numbers used in quantum gates
pub mod constants {
    use num_complex::Complex64;

    /// The imaginary unit i
    pub const I: Complex64 = Complex64::new(0.0, 1.0);

    /// 1/sqrt(2)
    pub const FRAC_1_SQRT_2: f64 = 0.7071067811865475;
}

/// The category of quantum gates
///
/// Objects are qubit counts, morphisms are quantum gates.
/// Forms a symmetric monoidal category with tensor products of gates.
#[derive(Clone)]
pub struct QuantumGateCategory;

impl Category for QuantumGateCategory {
    type Object = usize;  // Objects are qubit counts
    type Morphism = Box<dyn QuantumGate>;  // Morphisms are quantum gates

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count() // The domain is the number of qubits
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count() // For quantum gates, domain = codomain (they're endomorphisms)
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        Box::new(StandardGate::I(*obj)) // Identity gate with n qubits
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        // Can only compose if they have the same qubit count
        if f.qubit_count() != g.qubit_count() {
            return None;
        }
        // g after f (first apply f, then g)
        Some(Box::new(ComposedGate {
            gate1: f.clone_box(),
            gate2: g.clone_box(),
        }))
    }
}

impl MonoidalCategory for QuantumGateCategory {
    fn unit(&self) -> Self::Object {
        0 // The monoidal unit is 0 qubits (trivial system)
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b // Tensoring objects = adding qubit counts
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        Box::new(TensorProductGate {
            gate1: f.clone_box(),
            gate2: g.clone_box(),
        })
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Left unitor: 0 ⊗ a → a (isomorphism)
        Box::new(UnitIsomorphismGate {
            qubit_count: *a,
            direction: IsomorphismDirection::LeftUnitor,
        })
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Right unitor: a ⊗ 0 → a (isomorphism)
        Box::new(UnitIsomorphismGate {
            qubit_count: *a,
            direction: IsomorphismDirection::RightUnitor,
        })
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        // Associator: (a ⊗ b) ⊗ c → a ⊗ (b ⊗ c) (isomorphism)
        Box::new(AssociatorGate {
            a_qubits: *a,
            b_qubits: *b,
            c_qubits: *c,
        })
    }
}

impl SymmetricMonoidalCategory for QuantumGateCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Implements the braiding isomorphism a ⊗ b → b ⊗ a
        Box::new(BraidingGate {
            a_qubits: *a,
            b_qubits: *b,
        })
    }
}

impl DaggerCategory for QuantumGateCategory {
    fn dagger(&self, f: &Self::Morphism) -> Self::Morphism {
        f.adjoint()
    }
}

impl CompactClosedCategory for QuantumGateCategory {
    fn dual(&self, a: &Self::Object) -> Self::Object {
        *a  // Quantum systems are self-dual
    }

    fn unit_morphism(&self, a: &Self::Object) -> Self::Morphism {
        // Create bell pairs (unit morphism): η_A: I → A* ⊗ A
        let mut circuit = QuantumCircuit::new(2 * a);

        // Create bell pairs (maximally entangled states)
        for i in 0..*a {
            // Apply Hadamard to first qubit
            let _ = circuit.add_gate(Box::new(StandardGate::H), &[i]);
            // Apply CNOT with control on first qubit, target on second
            let _ = circuit.add_gate(Box::new(StandardGate::CNOT), &[i, i + a]);
        }

        circuit_to_gate(&circuit)
    }

    fn counit_morphism(&self, a: &Self::Object) -> Self::Morphism {
        // Bell measurement (counit morphism): ε_A: A ⊗ A* → I
        // This is the adjoint of the unit morphism
        self.dagger(&self.unit_morphism(a))
    }
}

impl DaggerCompactCategory for QuantumGateCategory {}

/// Direction of unit isomorphisms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsomorphismDirection {
    /// Left unitor: 0 ⊗ a → a
    LeftUnitor,
    /// Right unitor: a ⊗ 0 → a
    RightUnitor,
    /// Inverse left unitor: a → 0 ⊗ a
    InverseLeftUnitor,
    /// Inverse right unitor: a → a ⊗ 0
    InverseRightUnitor,
}

/// Trait for vector space enrichment of our category
///
/// This trait allows us to work with gates as elements of a complex vector space,
/// supporting operations like linear combinations.
pub trait VectorSpaceEnriched: Category {
    /// Returns the matrix representation of a morphism
    fn matrix_rep(&self, f: &Self::Morphism) -> Array2<Complex64>;

    /// Linear combination of morphisms
    fn linear_combination(&self,
        morphisms: &[Self::Morphism],
        coefficients: &[Complex64]
    ) -> Option<Self::Morphism>;
}

impl VectorSpaceEnriched for QuantumGateCategory {
    fn matrix_rep(&self, f: &Self::Morphism) -> Array2<Complex64> {
        f.matrix()
    }

    fn linear_combination(&self,
        morphisms: &[Self::Morphism],
        coefficients: &[Complex64]
    ) -> Option<Self::Morphism> {
        if morphisms.is_empty() || morphisms.len() != coefficients.len() {
            return None;
        }

        // All gates must have the same qubit count
        let qubit_count = morphisms[0].qubit_count();
        if !morphisms.iter().all(|g| g.qubit_count() == qubit_count) {
            return None;
        }

        Some(Box::new(LinearCombinationGate {
            gates: morphisms.iter().map(|g| g.clone_box()).collect(),
            coefficients: coefficients.to_vec(),
        }))
    }
}

/// A generic gate defined by its matrix
#[derive(Debug, Clone)]
pub struct CustomMatrixGate {
    pub matrix: Array2<Complex64>,
    pub name: String,
    pub qubits: usize,
}

impl QuantumGate for CustomMatrixGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.qubits
    }

    fn matrix(&self) -> Array2<Complex64> {
        self.matrix.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(self.clone())
    }
}

/// Trait for quantum gates
pub trait QuantumGate: Debug + Any + Send + Sync{
    /// Returns the number of qubits this gate acts on
    fn qubit_count(&self) -> usize;

    /// Returns the matrix representation of this gate
    fn matrix(&self) -> Array2<Complex64>;

    /// Returns a display name for this gate
    fn name(&self) -> String;

    /// Create a clone of this gate
    fn clone_box(&self) -> Box<dyn QuantumGate>;

    /// Convert to Any for runtime type checking
    fn as_any(&self) -> &dyn Any;

    /// Convert to mutable Any for runtime type checking
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Returns the adjoint (Hermitian conjugate) of this gate
    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // Default implementation: compute the adjoint matrix (conjugate transpose)
        let matrix = self.matrix();
        let mut adjoint_matrix = Array2::zeros(matrix.dim());

        for i in 0..matrix.shape()[0] {
            for j in 0..matrix.shape()[1] {
                adjoint_matrix[[j, i]] = matrix[[i, j]].conj();
            }
        }

        // Create a custom gate with the adjoint matrix
        Box::new(CustomMatrixGate {
            matrix: adjoint_matrix,
            name: format!("{}†", self.name()),
            qubits: self.qubit_count(),
        })
    }

    /// Compares this gate with another gate for equality
    fn equals(&self, other: &dyn QuantumGate) -> bool {
        // Default implementation compares matrices
        let m1 = self.matrix();
        let m2 = other.matrix();

        if m1.shape() != m2.shape() {
            return false;
        }

        // Compare matrix elements with tolerance for floating point
        for i in 0..m1.shape()[0] {
            for j in 0..m1.shape()[1] {
                let a = m1[[i, j]];
                let b = m2[[i, j]];
                if (a.re - b.re).abs() > 1e-10 || (a.im - b.im).abs() > 1e-10 {
                    return false;
                }
            }
        }

        true
    }

    /// Apply this gate to a quantum state
    fn apply(&self, state: &StateVector) -> Result<StateVector, String> {
        if self.qubit_count() > state.qubit_count() {
            return Err(format!(
                "Gate acts on {} qubits, but state has only {} qubits",
                self.qubit_count(), state.qubit_count()
            ));
        }

        // For gates that act on all qubits, we can apply them directly
        if self.qubit_count() == state.qubit_count() {
            return state.apply_matrix(&self.matrix());
        }

        Err("Application of partial gates not implemented directly. Use apply_to_qubits.".to_string())
    }

    /// Apply this gate to specific qubits in a state
    fn apply_to_qubits(
        &self,
        state: &StateVector,
        qubits: &[usize]
    ) -> Result<StateVector, String> {
        // Check that we have the right number of target qubits
        if qubits.len() != self.qubit_count() {
            return Err(format!(
                "Gate acts on {} qubits, but {} target qubits were specified",
                self.qubit_count(), qubits.len()
            ));
        }

        // Check that target qubits are in range
        for &q in qubits {
            if q >= state.qubit_count() {
                return Err(format!("Qubit index {} out of range", q));
            }
        }

        // Construct the full matrix for the entire system
        let full_matrix = self.tensor_to_full_system(state.qubit_count(), qubits);

        // Apply the full matrix to the state
        state.apply_matrix(&full_matrix)
    }

    fn tensor_to_full_system(
        &self,
        total_qubits: usize,
        target_qubits: &[usize]
    ) -> Array2<Complex64> {
        let gate_matrix = self.matrix();
        let dim = 1 << total_qubits;
        let mut result = Array2::zeros((dim, dim));

        // Special case for single-qubit gates on single-qubit systems
        if total_qubits == 1 && target_qubits.len() == 1 && target_qubits[0] == 0 {
            return gate_matrix.clone();
        }

        // Sort target_qubits to ensure consistent ordering
        let mut sorted_targets = target_qubits.to_vec();
        sorted_targets.sort();

        // Implementation for expanding the gate matrix to the full system
        for i in 0..dim {
            for j in 0..dim {
                let mut matches = true;
                // Check that non-target bits match
                for q in 0..total_qubits {
                    if !sorted_targets.contains(&q) {
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
                    // Extract the target qubits into a smaller index
                    let num_target = sorted_targets.len();
                    let mut sub_i = 0;
                    let mut sub_j = 0;

                    for (k, &q) in sorted_targets.iter().enumerate() {
                        let shift_full = total_qubits - 1 - q;
                        let bit_i = (i >> shift_full) & 1;
                        let bit_j = (j >> shift_full) & 1;
                        sub_i |= bit_i << ((num_target - 1) - k);
                        sub_j |= bit_j << ((num_target - 1) - k);
                    }

                    // Set the corresponding element from the gate's matrix
                    result[[i, j]] = gate_matrix[[sub_i, sub_j]];
                }
            }
        }

        result
    }
}

impl Clone for Box<dyn QuantumGate> {
    fn clone(&self) -> Box<dyn QuantumGate> {
        self.clone_box()
    }
}

impl PartialEq for Box<dyn QuantumGate> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other.as_ref())
    }
}

/// Standard quantum gates (Pauli, Hadamard, etc.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StandardGate {
    /// Identity gate
    I(usize), // number of qubits

    /// Pauli-X gate (NOT gate)
    X,

    /// Pauli-Y gate
    Y,

    /// Pauli-Z gate
    Z,

    /// Hadamard gate
    H,

    /// Phase gate (S gate)
    S,

    /// π/8 gate (T gate)
    T,

    /// CNOT gate
    CNOT,

    /// SWAP gate
    SWAP,

    /// Toffoli gate (CCNOT)
    Toffoli,

    /// Controlled-Z gate
    CZ,

    /// Controlled-Y gate
    CY,
}

impl QuantumGate for StandardGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        match self {
            StandardGate::I(n) => *n,
            StandardGate::X | StandardGate::Y | StandardGate::Z |
            StandardGate::H | StandardGate::S | StandardGate::T => 1,
            StandardGate::CNOT | StandardGate::SWAP | StandardGate::CZ |
            StandardGate::CY => 2,
            StandardGate::Toffoli => 3,
        }
    }

    fn matrix(&self) -> Array2<Complex64> {
        use constants::*;
        match self {
            StandardGate::I(n) => {
                let dim = 1 << n;
                Array2::from_diag(&Array1::from_elem(dim, Complex64::new(1.0, 0.0)))
            },
            StandardGate::X => {
                array![
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
                ]
            },
            StandardGate::Y => {
                array![
                    [Complex64::new(0.0, 0.0), -I],
                    [I, Complex64::new(0.0, 0.0)]
                ]
            },
            StandardGate::Z => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
                ]
            },
            StandardGate::H => {
                let factor = Complex64::new(FRAC_1_SQRT_2, 0.0);
                array![
                    [factor, factor],
                    [factor, -factor]
                ]
            },
            StandardGate::S => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), I]
                ]
            },
            StandardGate::T => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2)]
                ]
            },
            StandardGate::CNOT => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
                ]
            },
            StandardGate::SWAP => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
                ]
            },
            StandardGate::CZ => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
                ]
            },
            StandardGate::CY => {
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), I],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), -I, Complex64::new(0.0, 0.0)]
                ]
            },
            StandardGate::Toffoli => {
                let mut matrix = Array2::zeros((8, 8));
                for i in 0..8 {
                    let q0 = (i >> 2) & 1;  // MSB (big-endian)
                    let q1 = (i >> 1) & 1;  // Middle bit
                    let q2 = i & 1;         // LSB

                    // Flip q2 only if q0=1 and q1=1
                    let new_q2 = if q0 == 1 && q1 == 1 { q2 ^ 1 } else { q2 };

                    // Compute new index with big-endian ordering
                    let j = (q0 << 2) | (q1 << 1) | new_q2;

                    matrix[[i, j]] = Complex64::new(1.0, 0.0);
                }
                matrix
            }
        }
    }

    fn name(&self) -> String {
        match self {
            StandardGate::I(n) => format!("I({})", n),
            StandardGate::X => "X".to_string(),
            StandardGate::Y => "Y".to_string(),
            StandardGate::Z => "Z".to_string(),
            StandardGate::H => "H".to_string(),
            StandardGate::S => "S".to_string(),
            StandardGate::T => "T".to_string(),
            StandardGate::CNOT => "CNOT".to_string(),
            StandardGate::SWAP => "SWAP".to_string(),
            StandardGate::CZ => "CZ".to_string(),
            StandardGate::CY => "CY".to_string(),
            StandardGate::Toffoli => "Toffoli".to_string(),
        }
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(self.clone())
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        match self {
            // Gates that are self-adjoint (Hermitian)
            StandardGate::I(_) | StandardGate::X | StandardGate::Z |
            StandardGate::H | StandardGate::SWAP | StandardGate::CZ |
            StandardGate::Toffoli => self.clone_box(),

            // Gates with specific adjoints
            StandardGate::Y => self.clone_box(), // Y is self-adjoint
            StandardGate::S => Box::new(StandardGate::S), // S† = S* (complex conjugate)
            StandardGate::T => Box::new(StandardGate::T), // T† = T* (complex conjugate)
            StandardGate::CNOT => self.clone_box(), // CNOT is self-adjoint
            StandardGate::CY => self.clone_box(), // CY is self-adjoint
        }
    }
}

/// Parametrized quantum gates
#[derive(Clone, Debug)]
pub enum ParametrizedGate {
    /// Rotation around X-axis
    Rx(f64),

    /// Rotation around Y-axis
    Ry(f64),

    /// Rotation around Z-axis
    Rz(f64),

    /// General single-qubit unitary with Euler angles
    U3(f64, f64, f64),

    /// Controlled rotation around Z-axis
    CRz(f64),

    /// Controlled rotation around X-axis
    CRx(f64),

    /// Controlled rotation around Y-axis
    CRy(f64),

    /// Phase gate with arbitrary angle
    Phase(f64),

    /// Controlled phase gate with arbitrary angle
    CPhase(f64),
}

impl QuantumGate for ParametrizedGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        match self {
            ParametrizedGate::Rx(_) |
            ParametrizedGate::Ry(_) |
            ParametrizedGate::Rz(_) |
            ParametrizedGate::U3(_, _, _) |
            ParametrizedGate::Phase(_) => 1,
            ParametrizedGate::CRz(_) |
            ParametrizedGate::CRx(_) |
            ParametrizedGate::CRy(_) |
            ParametrizedGate::CPhase(_) => 2,
        }
    }

    fn matrix(&self) -> Array2<Complex64> {


        match self {
            ParametrizedGate::Rx(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                array![
                    [Complex64::new(cos, 0.0), Complex64::new(0.0, -sin)],
                    [Complex64::new(0.0, -sin), Complex64::new(cos, 0.0)]
                ]
            },
            ParametrizedGate::Ry(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                array![
                    [Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0)],
                    [Complex64::new(sin, 0.0), Complex64::new(cos, 0.0)]
                ]
            },
            ParametrizedGate::Rz(theta) => {
                let phase_pos = Complex64::new(0.0, theta / 2.0).exp();
                let phase_neg = Complex64::new(0.0, -theta / 2.0).exp();
                array![
                    [phase_neg, Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), phase_pos]
                ]
            },
            ParametrizedGate::U3(theta, phi, lambda) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                array![
                    [
                        Complex64::new(cos, 0.0),
                        -Complex64::new(lambda.cos(), lambda.sin()) * sin
                    ],
                    [
                        Complex64::new(phi.cos(), phi.sin()) * sin,
                        Complex64::new((phi + lambda).cos(), (phi + lambda).sin()) * cos
                    ]
                ]
            },
            ParametrizedGate::CRz(theta) => {
                let phase = Complex64::new(0.0, theta / 2.0).exp();
                let phase_conj = phase.conj();
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), phase_conj, Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), phase]
                ]
            },
            ParametrizedGate::CRx(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(cos, 0.0), Complex64::new(0.0, -sin)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -sin), Complex64::new(cos, 0.0)]
                ]
            },
            ParametrizedGate::CRy(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(sin, 0.0), Complex64::new(cos, 0.0)]
                ]
            },
            ParametrizedGate::CPhase(theta) => {
                let phase = Complex64::new(theta.cos(), theta.sin());
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), phase]
                ]
            },
            ParametrizedGate::Phase(theta) => {
                let phase = Complex64::new(theta.cos(), theta.sin());
                array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), phase]
                ]
            }
        }
    }

    fn name(&self) -> String {
        match self {
            ParametrizedGate::Rx(theta) => format!("Rx({:.2})", theta),
            ParametrizedGate::Ry(theta) => format!("Ry({:.2})", theta),
            ParametrizedGate::Rz(theta) => format!("Rz({:.2})", theta),
            ParametrizedGate::U3(theta, phi, lambda) => {
                format!("U3({:.2}, {:.2}, {:.2})", theta, phi, lambda)
            },
            ParametrizedGate::CRz(theta) => format!("CRz({:.2})", theta),
            ParametrizedGate::CRx(theta) => format!("CRx({:.2})", theta),
            ParametrizedGate::CRy(theta) => format!("CRy({:.2})", theta),
            ParametrizedGate::CPhase(theta) => format!("CPhase({:.2})", theta),
            ParametrizedGate::Phase(theta) => format!("P({:.2})", theta)
        }
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(self.clone())
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        match self {
            // Rotation gates have adjoint = rotation by negative angle
            ParametrizedGate::Rx(theta) => Box::new(ParametrizedGate::Rx(-*theta)),
            ParametrizedGate::Ry(theta) => Box::new(ParametrizedGate::Ry(-*theta)),
            ParametrizedGate::Rz(theta) => Box::new(ParametrizedGate::Rz(-*theta)),
            ParametrizedGate::U3(theta, phi, lambda) => {
                // U3 adjoint has complex conjugate parameters
                Box::new(ParametrizedGate::U3(*theta, -*lambda, -*phi))
            },
            ParametrizedGate::CRz(theta) => Box::new(ParametrizedGate::CRz(-*theta)),
            ParametrizedGate::CRx(theta) => Box::new(ParametrizedGate::CRx(-*theta)),
            ParametrizedGate::CRy(theta) => Box::new(ParametrizedGate::CRy(-*theta)),
            ParametrizedGate::Phase(theta) => Box::new(ParametrizedGate::Phase(-*theta)),
            ParametrizedGate::CPhase(theta) => Box::new(ParametrizedGate::CPhase(-*theta)),
        }
    }
}

/// A gate created by tensoring two gates together
#[derive(Debug)]
pub struct TensorProductGate {
    pub gate1: Box<dyn QuantumGate>,
    pub gate2: Box<dyn QuantumGate>,
}

impl QuantumGate for TensorProductGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.gate1.qubit_count() + self.gate2.qubit_count()
    }

    fn matrix(&self) -> Array2<Complex64> {
        let matrix1 = self.gate1.matrix();
        let matrix2 = self.gate2.matrix();

        // Compute the Kronecker product of the matrices
        let n1 = matrix1.shape()[0];
        let m1 = matrix1.shape()[1];
        let n2 = matrix2.shape()[0];
        let m2 = matrix2.shape()[1];

        let mut result = Array2::zeros((n1 * n2, m1 * m2));

        for i in 0..n1 {
            for j in 0..m1 {
                for k in 0..n2 {
                    for l in 0..m2 {
                        result[[i * n2 + k, j * m2 + l]] = matrix1[[i, j]] * matrix2[[k, l]];
                    }
                }
            }
        }

        result
    }
    fn name(&self) -> String {
        format!("{} ⊗ {}", self.gate1.name(), self.gate2.name())
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(TensorProductGate {
            gate1: self.gate1.clone_box(),
            gate2: self.gate2.clone_box(),
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // The adjoint of a tensor product is the tensor product of adjoints
        Box::new(TensorProductGate {
            gate1: self.gate1.adjoint(),
            gate2: self.gate2.adjoint(),
        })
    }
}

/// A gate created by composing two gates
#[derive(Debug)]
pub struct ComposedGate {
    pub gate1: Box<dyn QuantumGate>,
    pub gate2: Box<dyn QuantumGate>,
}

impl QuantumGate for ComposedGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        // Both gates should have the same qubit count
        self.gate1.qubit_count()
    }

    fn matrix(&self) -> Array2<Complex64> {
        let matrix1 = self.gate1.matrix();
        let matrix2 = self.gate2.matrix();

        // Matrix multiplication: gate2 * gate1
        // (We apply gate1 first, then gate2)
        matrix2.dot(&matrix1)
    }

    fn name(&self) -> String {
        format!("{} · {}", self.gate2.name(), self.gate1.name())
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(ComposedGate {
            gate1: self.gate1.clone_box(),
            gate2: self.gate2.clone_box(),
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // The adjoint of a composition is the reversed composition of adjoints
        // (g ∘ f)† = f† ∘ g†
        Box::new(ComposedGate {
            gate1: self.gate2.adjoint(),
            gate2: self.gate1.adjoint(),
        })
    }
}

/// A gate that applies to specific qubits in a larger system
#[derive(Debug)]
pub struct LocalizedGate {
    pub gate: Box<dyn QuantumGate>,
    pub target_qubits: Vec<usize>,
    pub total_qubits: usize,
}

impl LocalizedGate {
    pub fn new(
        gate: Box<dyn QuantumGate>,
        target_qubits: Vec<usize>,
        total_qubits: usize
    ) -> Result<Self, String> {
        if target_qubits.len() != gate.qubit_count() {
            return Err(format!(
                "Gate acts on {} qubits, but {} target qubits were specified",
                gate.qubit_count(), target_qubits.len()
            ));
        }

        for &q in &target_qubits {
            if q >= total_qubits {
                return Err(format!("Qubit index {} out of range", q));
            }
        }

        Ok(LocalizedGate {
            gate,
            target_qubits,
            total_qubits,
        })
    }
}

impl QuantumGate for LocalizedGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.total_qubits
    }

    fn matrix(&self) -> Array2<Complex64> {
        self.gate.tensor_to_full_system(self.total_qubits, &self.target_qubits)
    }

    fn name(&self) -> String {
        format!("{}({})", self.gate.name(),
            self.target_qubits.iter()
                .map(|q| q.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(LocalizedGate {
            gate: self.gate.clone_box(),
            target_qubits: self.target_qubits.clone(),
            total_qubits: self.total_qubits,
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // The adjoint of a localized gate is the localized adjoint
        Box::new(LocalizedGate {
            gate: self.gate.adjoint(),
            target_qubits: self.target_qubits.clone(),
            total_qubits: self.total_qubits,
        })
    }
}

/// Gate representing a linear combination of gates
/// This enables vector space enrichment of our category
#[derive(Debug)]
pub struct LinearCombinationGate {
    pub gates: Vec<Box<dyn QuantumGate>>,
    pub coefficients: Vec<Complex64>,
}

impl QuantumGate for LinearCombinationGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        // All gates should have the same qubit count
        if self.gates.is_empty() {
            return 0;
        }
        self.gates[0].qubit_count()
    }

    fn matrix(&self) -> Array2<Complex64> {
        if self.gates.is_empty() || self.gates.len() != self.coefficients.len() {
            // Return identity matrix for empty case
            let dim = 1 << self.qubit_count();
            return Array2::eye(dim).map(|x| Complex64::new(*x, 0.0));
        }

        let mut result = Array2::zeros(self.gates[0].matrix().dim());

        // Compute the linear combination
        for (gate, &coef) in self.gates.iter().zip(self.coefficients.iter()) {
            let gate_matrix = gate.matrix();
            result += &gate_matrix.mapv(|x| x * coef);
        }

        result
    }

    fn name(&self) -> String {
        if self.gates.is_empty() {
            return "EmptyLinearCombination".to_string();
        }

        let mut result = format!("{}⋅{}", self.coefficients[0], self.gates[0].name());
        for i in 1..self.gates.len() {
            result.push_str(&format!(" + {}⋅{}", self.coefficients[i], self.gates[i].name()));
        }
        result
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(LinearCombinationGate {
            gates: self.gates.iter().map(|g| g.clone_box()).collect(),
            coefficients: self.coefficients.clone(),
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // The adjoint of a linear combination is the linear combination of adjoints
        // with conjugated coefficients
        Box::new(LinearCombinationGate {
            gates: self.gates.iter().map(|g| g.adjoint()).collect(),
            coefficients: self.coefficients.iter().map(|c| c.conj()).collect(),
        })
    }
}

/// Gate implementing the braiding isomorphism in a symmetric monoidal category
/// Represents the swap operation between two quantum subsystems
#[derive(Debug)]
pub struct BraidingGate {
    pub a_qubits: usize,
    pub b_qubits: usize,
}

impl QuantumGate for BraidingGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.a_qubits + self.b_qubits
    }

    fn matrix(&self) -> Array2<Complex64> {
        // Handle special cases
        if self.a_qubits == 0 || self.b_qubits == 0 {
            // Trivial braiding is just identity
            let dim = 1 << self.qubit_count();
            return Array2::eye(dim).map(|x| Complex64::new(*x, 0.0));
        }

        if self.a_qubits == 1 && self.b_qubits == 1 {
            // Simple case: just a SWAP gate
            return StandardGate::SWAP.matrix();
        }

        // General case: implement braiding for arbitrary subsystems
        let dim_a = 1 << self.a_qubits;
        let dim_b = 1 << self.b_qubits;
        let dim_total = dim_a * dim_b;

        let mut result = Array2::zeros((dim_total, dim_total));

        // Braiding maps (a,b) to (b,a)
        for a in 0..dim_a {
            for b in 0..dim_b {
                // Map (a,b) -> (b,a)
                let input_idx = a * dim_b + b;
                let output_idx = b * dim_a + a;
                result[[input_idx, output_idx]] = Complex64::new(1.0, 0.0);
            }
        }

        result
    }

    fn name(&self) -> String {
        format!("Braiding({},{})", self.a_qubits, self.b_qubits)
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(BraidingGate {
            a_qubits: self.a_qubits,
            b_qubits: self.b_qubits,
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // Braiding is its own adjoint (self-adjoint)
        self.clone_box()
    }
}

/// Gate implementing the associator isomorphism in a monoidal category
/// Represents the isomorphism (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C)
#[derive(Debug)]
pub struct AssociatorGate {
    pub a_qubits: usize,
    pub b_qubits: usize,
    pub c_qubits: usize,
}

impl QuantumGate for AssociatorGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.a_qubits + self.b_qubits + self.c_qubits
    }

    fn matrix(&self) -> Array2<Complex64> {
        // In our case, the associator is trivial (identity matrix)
        // because tensoring is associative for qubit systems
        let dim = 1 << self.qubit_count();
        Array2::eye(dim).map(|x| Complex64::new(*x, 0.0))
    }

    fn name(&self) -> String {
        format!("Associator({},{},{})", self.a_qubits, self.b_qubits, self.c_qubits)
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(AssociatorGate {
            a_qubits: self.a_qubits,
            b_qubits: self.b_qubits,
            c_qubits: self.c_qubits,
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // Associator is its own adjoint (self-adjoint)
        self.clone_box()
    }
}

/// Gate implementing the unitor isomorphisms in a monoidal category
/// Represents the isomorphisms I ⊗ A ≅ A and A ⊗ I ≅ A
#[derive(Debug)]
pub struct UnitIsomorphismGate {
    pub qubit_count: usize,
    pub direction: IsomorphismDirection,
}

impl QuantumGate for UnitIsomorphismGate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn matrix(&self) -> Array2<Complex64> {
        // The unitor isomorphism is trivial (identity matrix)
        // because tensoring with the unit (0 qubits) is trivial
        let dim = 1 << self.qubit_count;
        Array2::eye(dim).map(|x| Complex64::new(*x, 0.0))
    }

    fn name(&self) -> String {
        match self.direction {
            IsomorphismDirection::LeftUnitor => format!("LeftUnitor({})", self.qubit_count),
            IsomorphismDirection::RightUnitor => format!("RightUnitor({})", self.qubit_count),
            IsomorphismDirection::InverseLeftUnitor => format!("InverseLeftUnitor({})", self.qubit_count),
            IsomorphismDirection::InverseRightUnitor => format!("InverseRightUnitor({})", self.qubit_count),
        }
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(UnitIsomorphismGate {
            qubit_count: self.qubit_count,
            direction: self.direction,
        })
    }

    fn adjoint(&self) -> Box<dyn QuantumGate> {
        // Unitor is its own adjoint (self-adjoint)
        self.clone_box()
    }
}

/// Helper function to create a swap network
/// This is useful for implementing braiding operations in larger systems
pub fn create_swap_network(system_a_size: usize, system_b_size: usize) -> Box<dyn QuantumGate> {
    if system_a_size == 0 || system_b_size == 0 {
        // Trivial case - identity gate on the non-empty system
        return Box::new(StandardGate::I(system_a_size + system_b_size));
    }

    if system_a_size == 1 && system_b_size == 1 {
        // Simple case - just a SWAP gate
        return Box::new(StandardGate::SWAP);
    }

    // General case - construct a network of SWAP gates
    // This could be implemented more efficiently, but this gives the idea
    let total_qubits = system_a_size + system_b_size;
    let mut circuit = QuantumCircuit::new(total_qubits);

    // Move each qubit from system A past all qubits in system B
    for a in 0..system_a_size {
        for b in 0..system_b_size {
            let q1 = a;
            let q2 = system_a_size + b;
            circuit.add_gate(Box::new(StandardGate::SWAP), &[q1, q2]).unwrap();
        }
    }

    // Convert the circuit back to a gate
    circuit_to_gate(&circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dagger_category_laws() {
        let category = QuantumGateCategory;

        // Create test gates as Box<dyn QuantumGate>
        let gate_x: Box<dyn QuantumGate> = Box::new(StandardGate::X);
        let gate_h: Box<dyn QuantumGate> = Box::new(StandardGate::H);
        let _gate_z: Box<dyn QuantumGate> = Box::new(StandardGate::Z);

        // Compose gates for testing
        let h_after_x = category.compose(&gate_x, &gate_h).unwrap();

        // Verify involution: (f†)† = f
        let gate_x_dagger = category.dagger(&gate_x);
        let gate_x_dagger_dagger = category.dagger(&gate_x_dagger);
        assert!(gate_x.equals(gate_x_dagger_dagger.as_ref()));

        // Verify contravariance: (g ∘ f)† = f† ∘ g†
        let h_after_x_dagger = category.dagger(&h_after_x);
        let gate_x_dagger = category.dagger(&gate_x);
        let gate_h_dagger = category.dagger(&gate_h);
        let x_dagger_after_h_dagger = category.compose(&gate_h_dagger, &gate_x_dagger).unwrap();
        assert!(h_after_x_dagger.equals(x_dagger_after_h_dagger.as_ref()));
    }

    #[test]
    fn test_compact_closed_snake_equations() {
        let category = QuantumGateCategory;

        // Test for 1 qubit
        let qubit_count = 1;

        // Get unit morphism: η_A: I → A* ⊗ A
        let unit = category.unit_morphism(&qubit_count);

        // Get counit morphism: ε_A: A ⊗ A* → I
        let counit = category.counit_morphism(&qubit_count);

        // Create identity morphism on A
        let _id_a = category.identity(&qubit_count);

        // Test that counit is the adjoint of unit
        assert!(counit.equals(category.dagger(&unit).as_ref()));
    }
}
