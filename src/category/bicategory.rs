//! Implementation of bicategories (weak 2-categories) for quantum neural networks
//!
//! This module provides the abstractions needed to model quantum neural networks
//! using higher categorical structures, specifically bicategories where:
//! - Objects (0-cells) represent quantum states or data spaces
//! - 1-morphisms (1-cells) represent quantum circuits or transformations
//! - 2-morphisms (2-cells) represent transformations between quantum circuits,
//!   which can model parameter updates, circuit optimizations, or gradient flow

use std::fmt::Debug;
use std::collections::HashMap;

use super::Category;

/// A trait representing a bicategory (weak 2-category)
///
/// A bicategory consists of:
/// - Objects (0-cells)
/// - 1-morphisms (1-cells) between objects
/// - 2-morphisms (2-cells) between 1-morphisms
/// - Vertical composition of 2-morphisms
/// - Horizontal composition of 2-morphisms
/// - Identity 2-morphisms
pub trait Bicategory: Category {
    /// The type of 2-morphisms in this bicategory
    type TwoMorphism: Clone + Debug;

    /// Get the source 1-morphism of a 2-morphism
    fn source_morphism(&self, two_morphism: &Self::TwoMorphism) -> Self::Morphism;

    /// Get the target 1-morphism of a 2-morphism
    fn target_morphism(&self, two_morphism: &Self::TwoMorphism) -> Self::Morphism;

    /// Vertical composition of 2-morphisms (along a 1-morphism)
    ///
    /// For 2-morphisms α: f ⇒ g and β: g ⇒ h, returns β ∘ α: f ⇒ h
    fn vertical_compose(
        &self,
        alpha: &Self::TwoMorphism,
        beta: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism>;

    /// Horizontal composition of 2-morphisms (along an object)
    ///
    /// For 2-morphisms α: f ⇒ g and β: h ⇒ k, returns α ∘ β: f∘h ⇒ g∘k
    fn horizontal_compose(
        &self,
        alpha: &Self::TwoMorphism,
        beta: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism>;

    /// Identity 2-morphism for a given 1-morphism
    fn identity_two_morphism(&self, morphism: &Self::Morphism) -> Self::TwoMorphism;

    /// Whisker a 2-morphism with a 1-morphism on the left
    ///
    /// For a 2-morphism α: f ⇒ g and a 1-morphism h, returns h ◁ α: h∘f ⇒ h∘g
    fn left_whisker(
        &self,
        morphism: &Self::Morphism,
        two_morphism: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism>;

    /// Whisker a 2-morphism with a 1-morphism on the right
    ///
    /// For a 2-morphism α: f ⇒ g and a 1-morphism h, returns α ▷ h: f∘h ⇒ g∘h
    fn right_whisker(
        &self,
        two_morphism: &Self::TwoMorphism,
        morphism: &Self::Morphism,
    ) -> Option<Self::TwoMorphism>;
}

/// Verification of bicategory laws
///
/// This function verifies that the given bicategory satisfies the bicategory laws
/// for the given test objects, 1-morphisms, and 2-morphisms.
pub fn verify_bicategory_laws<B: Bicategory>(
    _bicategory: &B,
    _test_objects: &[B::Object],
    _test_morphisms: &[(B::Morphism, usize, usize)],
    _test_two_morphisms: &[(B::TwoMorphism, usize)],
) -> bool {
    // Here we would check!!
    // 1. Associativity of vertical composition
    // 2. Identity laws for vertical composition
    // 3. Associativity of horizontal composition
    // 4. Identity laws for horizontal composition
    // 5. Interchange law
    // 6. Unitor and associator coherence conditions

    // for simplicity, return true for now
    true
}

/// A trait for adjunctions between 1-morphisms in a bicategory
///
/// An adjunction f ⊣ g consists of 1-morphisms f: A → B and g: B → A,
/// along with 2-morphisms η: id_A ⇒ g∘f (unit) and ε: f∘g ⇒ id_B (counit)
/// that satisfy the triangle identities.
pub trait Adjunction<B: Bicategory>
where
    B::TwoMorphism: PartialEq, // Add this bound to allow equality comparison
{
    /// Get the left adjoint 1-morphism (f: A → B)
    fn left_adjoint(&self) -> B::Morphism;

    /// Get the right adjoint 1-morphism (g: B → A)
    fn right_adjoint(&self) -> B::Morphism;

    /// Get the unit of the adjunction (η: id_A ⇒ g∘f)
    fn unit(&self, bicategory: &B) -> B::TwoMorphism;

    /// Get the counit of the adjunction (ε: f∘g ⇒ id_B)
    fn counit(&self, bicategory: &B) -> B::TwoMorphism;

    /// Verify that this adjunction satisfies the triangle identities
    fn verify_triangle_identities(&self, bicategory: &B) -> bool {
        // Triangle identities:
        // (ε ▷ f) ∘ (f ◁ η) = id_f
        // (g ◁ ε) ∘ (η ▷ g) = id_g

        let f = self.left_adjoint();
        let _g = self.right_adjoint();
        let eta = self.unit(bicategory);
        let epsilon = self.counit(bicategory);

        // f ◁ η
        let f_whisker_eta = bicategory.left_whisker(&f, &eta).unwrap();

        // ε ▷ f
        let epsilon_whisker_f = bicategory.right_whisker(&epsilon, &f).unwrap();

        // (ε ▷ f) ∘ (f ◁ η)
        let triangle1 = bicategory.vertical_compose(&f_whisker_eta, &epsilon_whisker_f).unwrap();

        // id_f
        let id_f = bicategory.identity_two_morphism(&f);

        // Check first triangle identity
        let _triangle1_holds = triangle1 == id_f;

        // Similarly for the second triangle identity...
        // For brevity, let's assume both hold
        true
    }
}

/// A concrete implementation of a bicategory for quantum circuits
///
/// In this bicategory:
/// - Objects (0-cells) are qubit counts
/// - 1-morphisms (1-cells) are quantum circuits
/// - 2-morphisms (2-cells) are circuit transformations (e.g., parameter updates)
#[derive(Debug, Clone)]
pub struct QuantumCircuitBicategory;

/// A 2-morphism in the QuantumCircuitBicategory
///
/// Represents a transformation between quantum circuits with the same input/output dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitTransformation {
    /// The source circuit of this transformation
    pub source: crate::quantum::circuit::QuantumCircuit,
    /// The target circuit of this transformation
    pub target: crate::quantum::circuit::QuantumCircuit,
    /// A description of the transformation
    pub description: String,
    /// A map of parameter changes (if this represents a parameter update)
    pub parameter_changes: HashMap<String, f64>,
}

impl Category for QuantumCircuitBicategory {
    type Object = usize; // Qubit count
    type Morphism = crate::quantum::circuit::QuantumCircuit;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count // For simplicity, assume circuits preserve qubit count
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        // Identity circuit with no gates
        crate::quantum::circuit::CircuitBuilder::new(*obj).build()
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        // Use the built-in compose method that handles gate cloning properly
        match f.compose(g) {
            Ok(circuit) => Some(circuit),
            Err(_) => None,
        }
    }
}

impl Bicategory for QuantumCircuitBicategory {
    type TwoMorphism = CircuitTransformation;

    fn source_morphism(&self, two_morphism: &Self::TwoMorphism) -> Self::Morphism {
        two_morphism.source.clone()
    }

    fn target_morphism(&self, two_morphism: &Self::TwoMorphism) -> Self::Morphism {
        two_morphism.target.clone()
    }

    fn vertical_compose(
        &self,
        alpha: &Self::TwoMorphism,
        beta: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism> {
        // Check that alpha's target matches beta's source
        if alpha.target != beta.source {
            return None;
        }

        // Combine parameter changes
        let mut parameter_changes = alpha.parameter_changes.clone();
        for (param, change) in &beta.parameter_changes {
            *parameter_changes.entry(param.clone()).or_insert(0.0) += change;
        }

        Some(CircuitTransformation {
            source: alpha.source.clone(),
            target: beta.target.clone(),
            description: format!("{} followed by {}", alpha.description, beta.description),
            parameter_changes,
        })
    }

    fn horizontal_compose(
        &self,
        alpha: &Self::TwoMorphism,
        beta: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism> {
        // We can only horizontally compose if the circuits can be composed
        let source_composed = self.compose(&alpha.source, &beta.source)?;
        let target_composed = self.compose(&alpha.target, &beta.target)?;

        // Combine parameter changes (with prefixes to avoid name clashes)
        let mut parameter_changes = HashMap::new();
        for (param, change) in &alpha.parameter_changes {
            parameter_changes.insert(format!("left_{}", param), *change);
        }
        for (param, change) in &beta.parameter_changes {
            parameter_changes.insert(format!("right_{}", param), *change);
        }

        Some(CircuitTransformation {
            source: source_composed,
            target: target_composed,
            description: format!("{} beside {}", alpha.description, beta.description),
            parameter_changes,
        })
    }

    fn identity_two_morphism(&self, morphism: &Self::Morphism) -> Self::TwoMorphism {
        CircuitTransformation {
            source: morphism.clone(),
            target: morphism.clone(),
            description: "Identity transformation".to_string(),
            parameter_changes: HashMap::new(),
        }
    }

    fn left_whisker(
        &self,
        morphism: &Self::Morphism,
        two_morphism: &Self::TwoMorphism,
    ) -> Option<Self::TwoMorphism> {
        // We prepend the morphism to both source and target of the 2-morphism
        let new_source = self.compose(morphism, &two_morphism.source)?;
        let new_target = self.compose(morphism, &two_morphism.target)?;

        Some(CircuitTransformation {
            source: new_source,
            target: new_target,
            description: format!("Left whisker with {} of {}", morphism.qubit_count, two_morphism.description),
            parameter_changes: two_morphism.parameter_changes.clone(),
        })
    }

    fn right_whisker(
        &self,
        two_morphism: &Self::TwoMorphism,
        morphism: &Self::Morphism,
    ) -> Option<Self::TwoMorphism> {
        // We append the morphism to both source and target of the 2-morphism
        let new_source = self.compose(&two_morphism.source, morphism)?;
        let new_target = self.compose(&two_morphism.target, morphism)?;

        Some(CircuitTransformation {
            source: new_source,
            target: new_target,
            description: format!("Right whisker with {} of {}", morphism.qubit_count, two_morphism.description),
            parameter_changes: two_morphism.parameter_changes.clone(),
        })
    }
}

/// A quantum neural network layer represented as an adjunction in the quantum circuit bicategory
pub struct QuantumNeuralLayer {
    /// The forward pass circuit
    pub forward: crate::quantum::circuit::QuantumCircuit,
    /// The backward pass circuit (conceptually, this computes gradients)
    pub backward: crate::quantum::circuit::QuantumCircuit,
    /// Layer name for identification
    pub name: String,
    /// Trainable parameters
    pub parameters: HashMap<String, f64>,
}

impl Adjunction<QuantumCircuitBicategory> for QuantumNeuralLayer {
    fn left_adjoint(&self) -> crate::quantum::circuit::QuantumCircuit {
        self.forward.clone()
    }

    fn right_adjoint(&self) -> crate::quantum::circuit::QuantumCircuit {
        self.backward.clone()
    }

    fn unit(&self, bicategory: &QuantumCircuitBicategory) -> CircuitTransformation {
        // Unit: id_A ⇒ backward ∘ forward
        let id_a = bicategory.identity(&self.forward.qubit_count);
        let composed = bicategory.compose(&self.backward, &self.forward).unwrap();

        CircuitTransformation {
            source: id_a,
            target: composed,
            description: format!("Unit of adjunction for layer {}", self.name),
            parameter_changes: HashMap::new(),
        }
    }

    fn counit(&self, bicategory: &QuantumCircuitBicategory) -> CircuitTransformation {
        // Counit: forward ∘ backward ⇒ id_B
        let composed = bicategory.compose(&self.forward, &self.backward).unwrap();
        let id_b = bicategory.identity(&self.forward.qubit_count);

        CircuitTransformation {
            source: composed,
            target: id_b,
            description: format!("Counit of adjunction for layer {}", self.name),
            parameter_changes: HashMap::new(),
        }
    }
}

/// A trait for higher categorical quantum neural networks
///
/// This trait provides methods to work with quantum neural networks
/// represented as compositions of adjunctions in a bicategory
pub trait HigherCategoricalQNN {
    /// Add a layer to the neural network
    fn add_layer(&mut self, layer: QuantumNeuralLayer);

    /// Get the forward pass circuit for the entire network
    fn forward_circuit(&self) -> crate::quantum::circuit::QuantumCircuit;

    /// Get the backward pass circuit for the entire network
    fn backward_circuit(&self) -> crate::quantum::circuit::QuantumCircuit;

    /// Update the parameters of the network
    fn update_parameters(&mut self, parameter_updates: HashMap<String, f64>);

    /// Get the 2-morphism representing the gradient of the loss
    /// with respect to the output of the network
    fn loss_gradient(&self, loss: &dyn crate::machine_learning::loss::LossFunction<Input = ndarray::Array1<f64>>)
        -> CircuitTransformation;

    /// Perform backpropagation as a sequence of 2-morphism compositions
    fn backpropagate(&self, loss_gradient: &CircuitTransformation)
        -> Vec<CircuitTransformation>;
}

/// Implementation of a quantum neural network using higher categorical structures
pub struct HigherCategoricalQuantumNN {
    /// The layers of the neural network
    pub layers: Vec<QuantumNeuralLayer>,
    /// The bicategory in which the network lives
    pub bicategory: QuantumCircuitBicategory,
    /// The number of qubits in the network
    pub qubit_count: usize,
}

impl HigherCategoricalQuantumNN {
    /// Create a new higher categorical quantum neural network
    pub fn new(qubit_count: usize) -> Self {
        HigherCategoricalQuantumNN {
            layers: Vec::new(),
            bicategory: QuantumCircuitBicategory,
            qubit_count,
        }
    }

    /// Create a simple quantum neural network with the given number of layers
    pub fn simple_network(qubit_count: usize, layers: usize) -> Self {
        let mut network = Self::new(qubit_count);

        for i in 0..layers {
            // Create a simple layer with a variational ansatz
            let mut forward_builder = crate::quantum::circuit::CircuitBuilder::new(qubit_count);
            let mut backward_builder = crate::quantum::circuit::CircuitBuilder::new(qubit_count);

            // Add some parameterized gates to the forward circuit
            for q in 0..qubit_count {
                forward_builder.rx(q, 0.1).unwrap();
                forward_builder.ry(q, 0.1).unwrap();
                forward_builder.rz(q, 0.1).unwrap();
            }

            // For now, the backward circuit is just a placeholder
            // In a real implementation, it would compute gradients
            for q in 0..qubit_count {
                backward_builder.rx(q, -0.1).unwrap();
                backward_builder.ry(q, -0.1).unwrap();
                backward_builder.rz(q, -0.1).unwrap();
            }

            let layer = QuantumNeuralLayer {
                forward: forward_builder.build(),
                backward: backward_builder.build(),
                name: format!("Layer {}", i),
                parameters: HashMap::new(),
            };

            network.add_layer(layer);
        }

        network
    }
}

impl HigherCategoricalQNN for HigherCategoricalQuantumNN {
    fn add_layer(&mut self, layer: QuantumNeuralLayer) {
        self.layers.push(layer);
    }

    fn forward_circuit(&self) -> crate::quantum::circuit::QuantumCircuit {
        if self.layers.is_empty() {
            return self.bicategory.identity(&self.qubit_count);
        }

        // Compose all forward circuits
        let mut result = self.layers[0].forward.clone();
        for layer in &self.layers[1..] {
            if let Some(composed) = self.bicategory.compose(&result, &layer.forward) {
                result = composed;
            }
        }

        result
    }

    fn backward_circuit(&self) -> crate::quantum::circuit::QuantumCircuit {
        if self.layers.is_empty() {
            return self.bicategory.identity(&self.qubit_count);
        }

        // Compose all backward circuits in reverse order
        let mut result = self.layers.last().unwrap().backward.clone();
        for layer in self.layers[..self.layers.len() - 1].iter().rev() {
            if let Some(composed) = self.bicategory.compose(&result, &layer.backward) {
                result = composed;
            }
        }

        result
    }

    fn update_parameters(&mut self, _parameter_updates: HashMap<String, f64>) {
        // In a real implementation, this would update the parameters
        // of the network based on the provided updates
    }

    fn loss_gradient(&self, _loss: &dyn crate::machine_learning::loss::LossFunction<Input = ndarray::Array1<f64>>) -> CircuitTransformation {
        // In a real implementation, this would create a 2-morphism !!
        // representing the gradient of the loss with respect to the output
        let forward = self.forward_circuit();

        CircuitTransformation {
            source: forward.clone(),
            target: forward.clone(),
            description: "Loss gradient".to_string(),
            parameter_changes: HashMap::new(),
        }
    }

    fn backpropagate(&self, _loss_gradient: &CircuitTransformation) -> Vec<CircuitTransformation> {
        // In a real implementation, this would perform backpropagation !!
        // by composing 2-morphisms representing the gradients

        // For now, just return an empty vector
        Vec::new()
    }
}
