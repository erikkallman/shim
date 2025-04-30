//! Categorical Quantum Machine Learning Framework
//!
//! This crate provides a framework for implementing quantum machine learning
//! algorithms based on category theory. It includes abstractions for categories,
//! functors, monads, and other categorical structures, as well as their applications
//! to quantum computing and machine learning.

// Re-export the category module and its contents
pub mod category;
pub mod quantum;
pub mod simulators;
pub mod machine_learning;

// Create a prelude module for convenient imports
pub mod prelude {
    pub use crate::category::prelude::*;
}

// Version and crate information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
