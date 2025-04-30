//! Category definitions for machine learning

use std::fmt::Debug;
use ndarray::{Array1, Array2};
use std::any::Any;

use crate::category::{Category, MonoidalCategory, SymmetricMonoidalCategory};
use crate::quantum::circuit::QuantumCircuit;

/// A type for data dimensions in ML models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDimension {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

pub trait ModelTransformation: Send + Sync + Debug {
    /// Gets the domain dimensions
    fn domain(&self) -> ModelDimension;

    /// Gets the codomain dimensions
    fn codomain(&self) -> ModelDimension;

    /// Applies the transformation to an input
    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>>;

    /// Clones the transformation into a boxed trait object
    fn clone_box(&self) -> Box<dyn ModelTransformation>;

    /// Returns a reference to self as Any for downcasting in PartialEq implementation
    fn as_any(&self) -> &dyn Any;

    /// Compares two ModelTransformations for equality
    fn equals(&self, other: &dyn ModelTransformation) -> bool;

}

impl PartialEq for Box<dyn ModelTransformation> {
    fn eq(&self, other: &Box<dyn ModelTransformation>) -> bool {
        self.equals(other.as_ref())
    }
}

// Add this implementation to allow Clone for Box<dyn ModelTransformation>
impl Clone for Box<dyn ModelTransformation> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Category of machine learning models
pub struct ModelCategory;

impl Category for ModelCategory {
    /// Objects are model dimensions (input_dim, output_dim)
    type Object = ModelDimension;

    /// Morphisms are transformations between dimensions
    type Morphism = Box<dyn ModelTransformation>;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.domain()
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.codomain()
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        Box::new(IdentityTransformation {
            dimension: obj.clone(),
        })
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        if f.codomain() != g.domain() {
            return None;
        }

        Some(Box::new(ComposedTransformation {
            first: f.clone_box(),
            second: g.clone_box(),
        }))
    }
}

impl MonoidalCategory for ModelCategory {
    fn unit(&self) -> Self::Object {
        ModelDimension {
            input_dim: 0,
            output_dim: 0,
        }
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        ModelDimension {
            input_dim: a.input_dim + b.input_dim,
            output_dim: a.output_dim + b.output_dim,
        }
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        Box::new(TensorProductTransformation {
            first: f.clone_box(),
            second: g.clone_box(),
        })
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        Box::new(LeftUnitorTransformation {
            object: a.clone(),
        })
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        Box::new(RightUnitorTransformation {
            object: a.clone(),
        })
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        Box::new(AssociatorTransformation {
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
        })
    }
}

impl SymmetricMonoidalCategory for ModelCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        Box::new(BraidingTransformation {
            a: a.clone(),
            b: b.clone(),
        })
    }
}

/// Category of data and data transformations
pub struct DataCategory;

impl Category for DataCategory {
    /// Objects are dimensions (sizes of data vectors)
    type Object = usize;

    /// Morphisms are linear transformations of data
    type Morphism = Array2<f64>;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.shape()[1]
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.shape()[0]
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        Array2::eye(*obj)
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        if f.shape()[0] != g.shape()[1] {
            return None;
        }

        Some(g.dot(f))
    }
}

impl MonoidalCategory for DataCategory {
    fn unit(&self) -> Self::Object {
        0
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        // For simplicity, we implement tensor product as block diagonal matrix
        let f_rows = f.shape()[0];
        let f_cols = f.shape()[1];
        let g_rows = g.shape()[0];
        let g_cols = g.shape()[1];

        let mut result = Array2::zeros((f_rows + g_rows, f_cols + g_cols));

        // Copy f to top-left block
        for i in 0..f_rows {
            for j in 0..f_cols {
                result[[i, j]] = f[[i, j]];
            }
        }

        // Copy g to bottom-right block
        for i in 0..g_rows {
            for j in 0..g_cols {
                result[[i + f_rows, j + f_cols]] = g[[i, j]];
            }
        }

        result
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Left unitor is trivial since tensor is just addition
        Array2::eye(*a)
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Right unitor is also trivial
        Array2::eye(*a)
    }

    fn associator(&self, _a: &Self::Object, _b: &Self::Object, _c: &Self::Object) -> Self::Morphism {
        // Associator is trivial for data (associativity of addition)
        // We just return identity on the tensor product
        Array2::eye(self.tensor_objects(_a, &self.tensor_objects(_b, _c)))
    }
}

impl SymmetricMonoidalCategory for DataCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Braiding is a permutation matrix that swaps the a and b parts
        let dim = a + b;
        let mut result = Array2::zeros((dim, dim));

        // Copy b part first
        for i in 0..*b {
            result[[i, i + a]] = 1.0;
        }

        // Then copy a part
        for i in 0..*a {
            result[[i + b, i]] = 1.0;
        }

        result
    }
}

/// Category of quantum circuits
pub struct CircuitCategory;

impl Category for CircuitCategory {
    /// Objects are qubit counts
    type Object = usize;

    /// Morphisms are quantum circuits
    type Morphism = QuantumCircuit;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.qubit_count
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        QuantumCircuit::new(*obj)
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        if f.qubit_count != g.qubit_count {
            return None;
        }

        // Compose by running f then g
        match f.compose(g) {
            Ok(circuit) => Some(circuit),
            Err(_) => None,
        }
    }
}

impl MonoidalCategory for CircuitCategory {
    fn unit(&self) -> Self::Object {
        0
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        match f.tensor(g) {
            Ok(circuit) => circuit,
            Err(_) => {
                // Fall back to manual tensor product
                let mut result = QuantumCircuit::new(f.qubit_count + g.qubit_count);

                // Copy gates from f with unchanged qubits
                for (gate, qubits) in &f.gates {
                    let _ = result.add_gate(gate.clone_box(), qubits);
                }

                // Copy gates from g with shifted qubits
                for (gate, qubits) in &g.gates {
                    let shifted_qubits: Vec<usize> = qubits.iter()
                        .map(|&q| q + f.qubit_count)
                        .collect();
                    let _ = result.add_gate(gate.clone_box(), &shifted_qubits);
                }

                result
            }
        }
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Left unitor maps 0 ⊗ a to a
        QuantumCircuit::new(*a)
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Right unitor maps a ⊗ 0 to a
        QuantumCircuit::new(*a)
    }

    fn associator(&self, _a: &Self::Object, _b: &Self::Object, _c: &Self::Object) -> Self::Morphism {
        // Associator is trivial for quantum circuits (tensor is just concatenation)
        QuantumCircuit::new(_a + _b + _c)
    }
}

impl SymmetricMonoidalCategory for CircuitCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Braiding is implemented using SWAP gates
        let mut circuit = QuantumCircuit::new(a + b);

        // Apply SWAP gates to exchange the a and b parts
        for i in 0..*a {
            for j in 0..*b {
                let _ = circuit.add_gate(
                    Box::new(crate::quantum::gate::StandardGate::SWAP),
                    &[i, j + a]
                );
            }
        }

        circuit
    }
}

// Concrete implementations of model transformations

/// Identity transformation
#[derive(Clone, Debug)]
pub struct IdentityTransformation {
    pub dimension: ModelDimension,
}

impl IdentityTransformation {
    pub fn new(dimension: ModelDimension) -> Self {
        IdentityTransformation { dimension }
    }
}

impl ModelTransformation for IdentityTransformation {
    fn domain(&self) -> ModelDimension {
        self.dimension.clone()
    }

    fn codomain(&self) -> ModelDimension {
        self.dimension.clone()
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        if input.len() != self.dimension.input_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.dimension.input_dim,
                input.len()
            ).into());
        }

        Ok(input.clone())
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

   fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.dimension == other.dimension
        } else {
            false
        }
    }
}

/// Composition of two transformations
#[derive(Clone, Debug)]
struct ComposedTransformation {
    first: Box<dyn ModelTransformation>,
    second: Box<dyn ModelTransformation>,
}

impl ModelTransformation for ComposedTransformation {
    fn domain(&self) -> ModelDimension {
        self.first.domain()
    }

    fn codomain(&self) -> ModelDimension {
        self.second.codomain()
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let intermediate = self.first.apply(input)?;
        self.second.apply(&intermediate)
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.first.equals(other.first.as_ref()) &&
            self.second.equals(other.second.as_ref())
        } else {
            false
        }
    }
}

/// Tensor product of two transformations
#[derive(Clone, Debug)]
struct TensorProductTransformation {
    first: Box<dyn ModelTransformation>,
    second: Box<dyn ModelTransformation>,
}

impl ModelTransformation for TensorProductTransformation {
    fn domain(&self) -> ModelDimension {
        let d1 = self.first.domain();
        let d2 = self.second.domain();

        ModelDimension {
            input_dim: d1.input_dim + d2.input_dim,
            output_dim: d1.output_dim + d2.output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        let c1 = self.first.codomain();
        let c2 = self.second.codomain();

        ModelDimension {
            input_dim: c1.input_dim + c2.input_dim,
            output_dim: c1.output_dim + c2.output_dim,
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let d1 = self.first.domain();
        let d2 = self.second.domain();

        if input.len() != d1.input_dim + d2.input_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                d1.input_dim + d2.input_dim,
                input.len()
            ).into());
        }

        // Split the input
        let input1 = input.slice(ndarray::s![0..d1.input_dim]).to_owned();
        let input2 = input.slice(ndarray::s![d1.input_dim..]).to_owned();

        // Apply each transformation
        let output1 = self.first.apply(&input1)?;
        let output2 = self.second.apply(&input2)?;

        // Concatenate the outputs
        let mut result = Array1::zeros(output1.len() + output2.len());
        for (i, &val) in output1.iter().enumerate() {
            result[i] = val;
        }

        for (i, &val) in output2.iter().enumerate() {
            result[output1.len() + i] = val;
        }

        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.first.equals(other.first.as_ref()) &&
            self.second.equals(other.second.as_ref())
        } else {
            false
        }
    }
}

/// Left unitor transformation
#[derive(Clone, Debug)]
struct LeftUnitorTransformation {
    object: ModelDimension,
}

impl ModelTransformation for LeftUnitorTransformation {
    fn domain(&self) -> ModelDimension {
        // Domain is 0 ⊗ a
        ModelDimension {
            input_dim: self.object.input_dim,
            output_dim: self.object.output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        // Codomain is a
        self.object.clone()
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        if input.len() != self.object.input_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.object.input_dim,
                input.len()
            ).into());
        }

        Ok(input.clone())
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.object == other.object
        } else {
            false
        }
    }
}

/// Right unitor transformation
#[derive(Clone, Debug)]
struct RightUnitorTransformation {
    object: ModelDimension,
}

impl ModelTransformation for RightUnitorTransformation {
    fn domain(&self) -> ModelDimension {
        // Domain is a ⊗ 0
        ModelDimension {
            input_dim: self.object.input_dim,
            output_dim: self.object.output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        // Codomain is a
        self.object.clone()
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        if input.len() != self.object.input_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.object.input_dim,
                input.len()
            ).into());
        }

        Ok(input.clone())
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.object == other.object
        } else {
            false
        }
    }
}

/// Associator transformation
#[derive(Clone, Debug)]
struct AssociatorTransformation {
    a: ModelDimension,
    b: ModelDimension,
    c: ModelDimension,
}

impl ModelTransformation for AssociatorTransformation {
    fn domain(&self) -> ModelDimension {
        // Domain is (a ⊗ b) ⊗ c
        ModelDimension {
            input_dim: self.a.input_dim + self.b.input_dim + self.c.input_dim,
            output_dim: self.a.output_dim + self.b.output_dim + self.c.output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        // Codomain is a ⊗ (b ⊗ c)
        ModelDimension {
            input_dim: self.a.input_dim + self.b.input_dim + self.c.input_dim,
            output_dim: self.a.output_dim + self.b.output_dim + self.c.output_dim,
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let total_dim = self.a.input_dim + self.b.input_dim + self.c.input_dim;

        if input.len() != total_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                total_dim,
                input.len()
            ).into());
        }

        // For vector spaces, associator is just identity
        Ok(input.clone())
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.a == other.a && self.b == other.b && self.c == other.c
        } else {
            false
        }
    }
}

/// Braiding transformation
#[derive(Clone, Debug)]
struct BraidingTransformation {
    a: ModelDimension,
    b: ModelDimension,
}

impl ModelTransformation for BraidingTransformation {
    fn domain(&self) -> ModelDimension {
        // Domain is a ⊗ b
        ModelDimension {
            input_dim: self.a.input_dim + self.b.input_dim,
            output_dim: self.a.output_dim + self.b.output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        // Codomain is b ⊗ a
        ModelDimension {
            input_dim: self.b.input_dim + self.a.input_dim,
            output_dim: self.b.output_dim + self.a.output_dim,
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let total_dim = self.a.input_dim + self.b.input_dim;

        if input.len() != total_dim {
            return Err(format!(
                "Input dimension mismatch: expected {}, got {}",
                total_dim,
                input.len()
            ).into());
        }

        // Split and reorder
        let a_part = input.slice(ndarray::s![0..self.a.input_dim]).to_owned();
        let b_part = input.slice(ndarray::s![self.a.input_dim..]).to_owned();

        // Reorder as [b_part, a_part]
        let mut result = Array1::zeros(total_dim);
        for (i, &val) in b_part.iter().enumerate() {
            result[i] = val;
        }

        for (i, &val) in a_part.iter().enumerate() {
            result[self.b.input_dim + i] = val;
        }

        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.a == other.a && self.b == other.b
        } else {
            false
        }
    }
}
