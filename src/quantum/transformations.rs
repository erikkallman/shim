//! Natural transformations for quantum circuits
//!
//! This module provides implementations of natural transformations
//! for quantum circuits. These allow systematic transformation of
//! quantum circuits while preserving their categorical structure.

use std::fmt::Debug;
use std::any::Any;
use crate::category::NaturalTransformation;
use crate::category::Functor;
use crate::quantum::circuit::{QuantumCircuit, QuantumCircuitCategory};
use crate::quantum::optimizer::CircuitOptimizer;
use crate::quantum::circuit_to_gate;
use crate::quantum::gate_to_circuit;
use crate::category::monoidal::Category;

/// Base trait for quantum circuit transformations.
///
/// This trait represents transformations between quantum circuits
/// that can be applied to maintain specific properties or optimize circuits.
pub trait QuantumCircuitTransformation: Send + Sync + Debug {
    /// Transform a quantum circuit according to the implemented strategy
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit;

    /// Make a boxed clone of this transformation
    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation>;

    /// Returns self as Any for type checking
    fn as_any(&self) -> &dyn Any;

    /// Check if this transformation equals another one
    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool;
}

impl Clone for Box<dyn QuantumCircuitTransformation> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl PartialEq for Box<dyn QuantumCircuitTransformation> {
    fn eq(&self, other: &Box<dyn QuantumCircuitTransformation>) -> bool {
        self.equals(other.as_ref())
    }
}

/// An identity functor for quantum circuits.
///
/// Maps quantum circuits to themselves without changes.
#[derive(Clone, Debug)]
pub struct QuantumCircuitIdentityFunctor;

impl Functor<QuantumCircuitCategory, QuantumCircuitCategory> for QuantumCircuitIdentityFunctor {
    fn map_object(&self, _c: &QuantumCircuitCategory, _d: &QuantumCircuitCategory, obj: &usize) -> usize {
        *obj  // Objects (qubit counts) remain unchanged
    }

    fn map_morphism(&self, _c: &QuantumCircuitCategory, _d: &QuantumCircuitCategory,
                   circuit: &QuantumCircuit) -> QuantumCircuit {
        circuit.clone()  // Identity mapping
    }
}

/// A circuit optimizer implementation using natural transformations.
///
/// This transformation implements circuit optimization as a natural transformation
/// between the identity functor and itself.
#[derive(Clone, Debug)]
pub struct CircuitOptimizerTransformation {
    optimizer: CircuitOptimizer,
}

impl CircuitOptimizerTransformation {
    /// Create a new circuit optimizer transformation with the given optimizer
    pub fn new(optimizer: CircuitOptimizer) -> Self {
        CircuitOptimizerTransformation { optimizer }
    }

    /// Get a reference to the internal optimizer
    pub fn optimizer(&self) -> &CircuitOptimizer {
        &self.optimizer
    }
}

impl QuantumCircuitTransformation for CircuitOptimizerTransformation {
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        self.optimizer.optimize(circuit)
    }

    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
        // A more detailed comparison could look at the specific optimization patterns
    }
}

impl NaturalTransformation<QuantumCircuitIdentityFunctor, QuantumCircuitIdentityFunctor, QuantumCircuitCategory, QuantumCircuitCategory>
    for CircuitOptimizerTransformation
{
    fn component(
        &self,
        _c: &QuantumCircuitCategory,
        _d: &QuantumCircuitCategory,
        _f: &QuantumCircuitIdentityFunctor,
        _g: &QuantumCircuitIdentityFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        // For a given qubit count, return an optimized empty circuit
        // (which is just an empty circuit of the same size)
        QuantumCircuit::new(*obj)
    }
}

/// Transformation that converts a circuit to gates and back.
///
/// This can be useful for simplifying circuits by converting to their
/// gate representation and back, which can sometimes identify equivalent
/// but simpler representations.
#[derive(Clone, Debug)]
pub struct CircuitGateConversionTransformation;

impl QuantumCircuitTransformation for CircuitGateConversionTransformation {
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        // First convert to gate, then back to circuit
        let gate = circuit_to_gate(circuit);
        gate_to_circuit(&gate)
    }

    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }
}

impl NaturalTransformation<QuantumCircuitIdentityFunctor, QuantumCircuitIdentityFunctor, QuantumCircuitCategory, QuantumCircuitCategory>
    for CircuitGateConversionTransformation
{
    fn component(
        &self,
        _c: &QuantumCircuitCategory,
        _d: &QuantumCircuitCategory,
        _f: &QuantumCircuitIdentityFunctor,
        _g: &QuantumCircuitIdentityFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        // For a given qubit count, return an empty circuit
        QuantumCircuit::new(*obj)
    }
}

/// Composition of two quantum circuit transformations.
///
/// Applies one transformation followed by another, creating a new
/// transformation that is their composition.
#[derive(Clone, Debug)]
pub struct ComposedTransformation {
    first: Box<dyn QuantumCircuitTransformation>,
    second: Box<dyn QuantumCircuitTransformation>,
}

impl ComposedTransformation {
    /// Create a new composed transformation by applying first and then second
    pub fn new(
        first: Box<dyn QuantumCircuitTransformation>,
        second: Box<dyn QuantumCircuitTransformation>
    ) -> Self {
        ComposedTransformation { first, second }
    }
}

impl QuantumCircuitTransformation for ComposedTransformation {
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        // Apply the first transformation, then the second
        let intermediate = self.first.transform(circuit);
        self.second.transform(&intermediate)
    }

    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation> {
        Box::new(Self {
            first: self.first.clone(),
            second: self.second.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.first == other.first.clone() && self.second == other.second.clone()
        } else {
            false
        }
    }
}

/// A struct for composing and applying quantum circuit transformations.
///
/// This provides a convenient interface for creating and composing
/// transformations, and then applying them to quantum circuits.
#[derive(Clone, Debug)]
pub struct CircuitTransformer {
    transformations: Vec<Box<dyn QuantumCircuitTransformation>>,
}

impl CircuitTransformer {
    /// Create a new empty circuit transformer
    pub fn new() -> Self {
        CircuitTransformer { transformations: Vec::new() }
    }

    /// Add a transformation to the chain
    pub fn add_transformation(&mut self, transformation: Box<dyn QuantumCircuitTransformation>) -> &mut Self {
        self.transformations.push(transformation);
        self
    }

    /// Apply the chain of transformations to a circuit
    pub fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        let mut result = circuit.clone();

        for transformation in &self.transformations {
            result = transformation.transform(&result);
        }

        result
    }

    /// Create a transformation that combines all the transformations in this chain
    pub fn as_transformation(&self) -> Box<dyn QuantumCircuitTransformation> {
        if self.transformations.is_empty() {
            // If no transformations are added, return an identity transformation
            Box::new(IdentityTransformation)
        } else if self.transformations.len() == 1 {
            // If only one transformation, return a clone of it
            self.transformations[0].clone()
        } else {
            // Otherwise, create a composed transformation
            let mut result = self.transformations[0].clone();

            for transformation in self.transformations.iter().skip(1) {
                result = Box::new(ComposedTransformation::new(
                    result,
                    transformation.clone(),
                ));
            }

            result
        }
    }
}

impl Default for CircuitTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// A transformation that doesn't change the circuit (identity)
#[derive(Clone, Debug)]
pub struct IdentityTransformation;

impl QuantumCircuitTransformation for IdentityTransformation {
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        circuit.clone()
    }

    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }
}

impl NaturalTransformation<QuantumCircuitIdentityFunctor, QuantumCircuitIdentityFunctor, QuantumCircuitCategory, QuantumCircuitCategory>
    for IdentityTransformation
{
    fn component(
        &self,
        c: &QuantumCircuitCategory,
        _d: &QuantumCircuitCategory,
        _f: &QuantumCircuitIdentityFunctor,
        _g: &QuantumCircuitIdentityFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        c.identity(obj)
    }
}

/// A transformation that reverses the order of gates in a circuit
#[derive(Clone, Debug)]
pub struct ReverseTransformation;

impl QuantumCircuitTransformation for ReverseTransformation {
    fn transform(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        let mut result = QuantumCircuit::new(circuit.qubit_count);

        // Add gates in reverse order
        for (gate, qubits) in circuit.gates.iter().rev() {
            result.add_gate(gate.clone_box(), qubits).unwrap();
        }

        result
    }

    fn clone_box(&self) -> Box<dyn QuantumCircuitTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn QuantumCircuitTransformation) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }
}

impl NaturalTransformation<QuantumCircuitIdentityFunctor, QuantumCircuitIdentityFunctor, QuantumCircuitCategory, QuantumCircuitCategory>
    for ReverseTransformation
{
    fn component(
        &self,
        c: &QuantumCircuitCategory,
        _d: &QuantumCircuitCategory,
        _f: &QuantumCircuitIdentityFunctor,
        _g: &QuantumCircuitIdentityFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        c.identity(obj)
    }
}

/// A Functor that optimizes circuits by using the optimizer
#[derive(Clone, Debug)]
pub struct OptimizerFunctor {
    optimizer: CircuitOptimizer,
}

impl OptimizerFunctor {
    /// Create a new optimizer functor
    pub fn new(optimizer: CircuitOptimizer) -> Self {
        OptimizerFunctor { optimizer }
    }
}

impl Functor<QuantumCircuitCategory, QuantumCircuitCategory> for OptimizerFunctor {
    fn map_object(&self, _c: &QuantumCircuitCategory, _d: &QuantumCircuitCategory, obj: &usize) -> usize {
        *obj  // Objects (qubit counts) remain unchanged
    }

    fn map_morphism(&self, _c: &QuantumCircuitCategory, _d: &QuantumCircuitCategory,
                   circuit: &QuantumCircuit) -> QuantumCircuit {
        self.optimizer.optimize(circuit)
    }
}

/// A natural transformation that optimizes a quantum circuit
pub struct OptimizerNaturalTransformation {
    optimizer: CircuitOptimizer,
}

impl OptimizerNaturalTransformation {
    /// Create a new optimizer natural transformation
    pub fn new(optimizer: CircuitOptimizer) -> Self {
        OptimizerNaturalTransformation { optimizer }
    }
}

impl NaturalTransformation<QuantumCircuitIdentityFunctor, OptimizerFunctor, QuantumCircuitCategory, QuantumCircuitCategory>
    for OptimizerNaturalTransformation
{
    fn component(
        &self,
        _c: &QuantumCircuitCategory,
        _d: &QuantumCircuitCategory,
        _f: &QuantumCircuitIdentityFunctor,
        _g: &OptimizerFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        // For component at object obj, return an optimized identity circuit
        let identity = QuantumCircuit::new(*obj);
        self.optimizer.optimize(&identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::optimizer::{CircuitOptimizer, OptimizationPattern};
    use crate::quantum::gate::StandardGate;

    #[test]
    fn test_identity_transformation() {
        let transform = IdentityTransformation;

        // Create a simple circuit
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(Box::new(StandardGate::X), &[0]).unwrap();

        // Apply the transformation
        let result = transform.transform(&circuit);

        // Should be the same circuit
        assert_eq!(result.gate_count(), circuit.gate_count());
        assert_eq!(result.qubit_count, circuit.qubit_count);
    }

    #[test]
    fn test_circuit_transformer_composition() {
        // Create simple transformations
        let identity = Box::new(IdentityTransformation);
        let reverse = Box::new(ReverseTransformation);

        // Create a circuit
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(Box::new(StandardGate::X), &[0]).unwrap();
        circuit.add_gate(Box::new(StandardGate::H), &[0]).unwrap();

        // Create a transformer with identity and reverse
        let mut transformer = CircuitTransformer::new();
        transformer.add_transformation(identity)
                  .add_transformation(reverse);

        // Apply the transformer
        let result = transformer.transform(&circuit);

        // The result should be the reverse of the original circuit
        assert_eq!(result.gate_count(), circuit.gate_count());

        // First gate should be H (reversed order)
        let (first_gate, _) = &result.gates[0];
        assert_eq!(first_gate.name(), "H");
    }

    #[test]
    fn test_optimizer_transformation() {
        // Create an optimizer that cancels adjacent inverse gates
        let optimizer = CircuitOptimizer::new(
            vec![OptimizationPattern::CancelInverseGates],
            3
        );

        let transform = CircuitOptimizerTransformation::new(optimizer);

        // Create a circuit with adjacent X gates that should cancel
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(Box::new(StandardGate::X), &[0]).unwrap();
        circuit.add_gate(Box::new(StandardGate::X), &[0]).unwrap();

        // Apply the transformation
        let result = transform.transform(&circuit);

        // The X gates should cancel out, resulting in an empty circuit
        assert_eq!(result.gate_count(), 0);
    }
}
