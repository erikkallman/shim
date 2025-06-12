//! Quantum machine learning module

pub mod model;
pub mod circuit_model;
pub mod variational;
pub mod kernel;

// Re-exports for convenience
pub use model::QuantumModel;
pub use circuit_model::ParametrizedCircuitModel;
pub use variational::{VariationalQuantumModel, QuantumNeuralNetwork, VariationalQuantumClassifier};
pub use kernel::QuantumKernel;
