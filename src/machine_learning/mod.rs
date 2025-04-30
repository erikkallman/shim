//! Machine learning module for categorical quantum computing
//!
//! This module integrates classical machine learning with quantum computing
//! using category theory as a unifying framework.

pub mod core;
pub mod loss;
pub mod optimizer;
pub mod dataset;
pub mod quantum;
pub mod categorical;

/// Re-exports of commonly used components
pub mod prelude {
    // Core ML components
    pub use super::core::{Model, PredictiveModel, TrainableModel, ModelError};
    pub use super::loss::LossFunction;
    pub use super::optimizer::Optimizer;
    pub use super::dataset::{Dataset, TabularDataset};

    // Quantum ML components
    pub use super::quantum::model::QuantumModel;
    pub use super::quantum::circuit_model::ParametrizedCircuitModel;
    pub use super::quantum::kernel::QuantumKernel;
    pub use super::quantum::variational::{
        VariationalQuantumModel,
        QuantumNeuralNetwork,
        VariationalQuantumClassifier
    };

    // Categorical components
    pub use super::categorical::categories::{ModelCategory, DataCategory, CircuitCategory};
    pub use super::categorical::functors::{DataToCircuitFunctor, CircuitToModelFunctor};
    pub use super::categorical::{TrainingTransformation, OptimizationTransformation};

    pub use super::categorical::GradientOptimizationTransformation;
}

// Type aliases for convenience
pub type QNN = quantum::variational::QuantumNeuralNetwork;
pub type VQC = quantum::variational::VariationalQuantumClassifier;
