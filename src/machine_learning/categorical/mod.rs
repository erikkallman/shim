//! Categorical abstractions for machine learning

pub mod categories;
pub mod functors;
pub mod transformations;
pub mod prediction;

// Re-exports for convenience
pub use categories::{ModelCategory, DataCategory, CircuitCategory, IdentityTransformation};
pub use functors::{DataToCircuitFunctor, CircuitToModelFunctor};

pub use transformations::{
    CircuitPredictionTransformation,
    TrainingTransformation,
    OptimizationTransformation,
    GradientOptimizationTransformation,
};
pub use prediction::{PredictionTransformation,PredictionCategory};
