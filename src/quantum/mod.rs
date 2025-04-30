// src/quantum/mod.rs
//! Quantum computing abstractions using category theory
//!
//! This module implements quantum states, operations, and circuits
//! using the categorical framework established in the category module.

pub mod state;
pub mod gate;
pub mod gate_operations;
pub mod circuit;
pub mod optimizer;
pub mod density_matrix;
pub mod differentiation;

pub use density_matrix::{DensityMatrix};
pub use state::{QuantumState, StateVector, Qubit, QuantumStateCategory};
pub use gate::{QuantumGate, StandardGate, ParametrizedGate, QuantumGateCategory};
pub use circuit::{QuantumCircuit, QuantumCircuitCategory, CircuitBuilder, circuit_to_gate, gate_to_circuit, CircuitToGateFunctor, GateToCircuitFunctor};
pub use optimizer::{CircuitOptimizer, OptimizationEndofunctor};

/// Re-export commonly used types and traits
pub mod prelude {
    pub use super::{QuantumState, StateVector, Qubit};
    pub use super::{QuantumGate, StandardGate, ParametrizedGate};
    pub use super::{QuantumCircuit, CircuitBuilder};
    pub use super::state::QuantumStateCategory;
}
