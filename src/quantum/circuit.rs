// src/quantum/circuit.rs
use num_complex::{Complex64};
use ndarray::Array2;
use crate::quantum::gate::{QuantumGate, StandardGate, ParametrizedGate};
use crate::quantum::state::{StateVector, QuantumState, QuantumStateCategory};
use crate::category::prelude::*;
use crate::quantum::gate_operations::{gate_operations::CategoryAware, gate_operations};
use crate::quantum::gate::ComposedGate;
use crate::quantum::gate::QuantumGateCategory;
/// A quantum circuit consisting of a sequence of gates
#[derive(Debug)]
pub struct QuantumCircuit {
    pub gates: Vec<(Box<dyn QuantumGate>, Vec<usize>)>,
    pub qubit_count: usize,
}

pub struct QuantumCircuitCategory;

impl Category for QuantumCircuitCategory {
    type Object = usize;  // Number of qubits
    type Morphism = QuantumCircuit;  // Circuits as morphisms

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
        match f.compose(g) {
            Ok(circuit) => Some(circuit),
            Err(_) => None
        }
    }
}


impl QuantumCircuit {
    /// Create a new empty quantum circuit
    pub fn new(qubit_count: usize) -> Self {
        QuantumCircuit {
            gates: Vec::new(),
            qubit_count,
        }
    }

    pub fn add_gate(&mut self, gate: Box<dyn QuantumGate>, qubits: &[usize]) -> Result<(), String> {

        // Validate qubit indices
        for &q in qubits {
            if q >= self.qubit_count {
                return Err(format!("Qubit index {} out of range", q));
            }
        }

        // Check gate's qubit count matches the specified qubits
        if gate.qubit_count() != qubits.len() {
            return Err(format!(
                "Gate acts on {} qubits, but {} qubits were specified",
                gate.qubit_count(), qubits.len()
            ));
        }

        // Add the gate to the circuit
        self.gates.push((gate, qubits.to_vec()));
        Ok(())
    }

    /// Get the number of gates in the circuit
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Apply the circuit to a quantum state
    pub fn apply(&self, state: &StateVector) -> Result<StateVector, String> {
        if state.qubit_count() < self.qubit_count {
            return Err(format!(
                "State has {} qubits, but circuit requires at least {} qubits",
                state.qubit_count(), self.qubit_count
            ));
        }

        // Apply each gate in sequence
        let mut current_state = state.clone();
        for (gate, qubits) in &self.gates {
            current_state = gate.apply_to_qubits(&current_state, qubits)?;
        }

        Ok(current_state)
    }

    /// Convert the circuit to a single composite gate using categorical composition
    pub fn as_single_gate(&self) -> Result<Box<dyn QuantumGate>, String> {
        let category = QuantumGateCategory;

        if self.gates.is_empty() {
            // Empty circuit is just the identity gate
            return Ok(category.identity(&self.qubit_count));
        }

        // Create localized gates for each gate in the circuit
        let mut localized_gates = Vec::with_capacity(self.gates.len());

        for (gate, qubits) in &self.gates {
            // Skip identity gates for efficiency
            if let Some(std_gate) = gate.as_any().downcast_ref::<StandardGate>() {
                if matches!(std_gate, StandardGate::I(1)) {
                    continue;
                }
            }

            // Create a localized version of the gate
            let category_aware = CategoryAware::new();
            let localized = category_aware.localize(gate.as_ref(), qubits, self.qubit_count)?;
            localized_gates.push(localized);
        }

        if localized_gates.is_empty() {
            // If all gates were identities, return identity
            return Ok(category.identity(&self.qubit_count));
        }

        // Compose all gates using categorical composition
        let mut result = localized_gates[0].clone_box();
        for gate in localized_gates.iter().skip(1) {
            match category.compose(&result, gate) {
                Some(composed) => result = composed,
                None => return Err("Failed to compose gates in circuit".to_string())
            }
        }

        Ok(result)
    }

    /// Compose this circuit with another circuit
    pub fn compose(&self, other: &QuantumCircuit) -> Result<QuantumCircuit, String> {
        if self.qubit_count != other.qubit_count {
            return Err(format!(
                "Cannot compose circuits with different qubit counts: {} and {}",
                self.qubit_count, other.qubit_count
            ));
        }

        // Create a new circuit with the same qubit count
        let mut result = QuantumCircuit::new(self.qubit_count);

        // First add all gates from this circuit
        for (gate, qubits) in &self.gates {
            result.add_gate(gate.clone_box(), qubits)?;
        }

        // Then add all gates from the other circuit
        for (gate, qubits) in &other.gates {
            result.add_gate(gate.clone_box(), qubits)?;
        }

        Ok(result)
    }

    /// Tensor this circuit with another circuit
    pub fn tensor(&self, other: &QuantumCircuit) -> Result<QuantumCircuit, String> {
        // Create a new circuit with combined qubit count
        let combined_qubits = self.qubit_count + other.qubit_count;
        let mut result = QuantumCircuit::new(combined_qubits);

        // Add gates from this circuit, keeping the same qubit indices
        for (gate, qubits) in &self.gates {
            result.add_gate(gate.clone_box(), qubits)?;
        }

        // Add gates from other circuit, offsetting qubit indices by this circuit's qubit count
        for (gate, qubits) in &other.gates {
            let offset_qubits: Vec<usize> = qubits.iter()
                .map(|&q| q + self.qubit_count)
                .collect();

            result.add_gate(gate.clone_box(), &offset_qubits)?;
        }

        Ok(result)
    }

}

impl PartialEq for QuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        // Two circuits are equal if they have:
        // 1. Same number of qubits
        if self.qubit_count != other.qubit_count {
            return false;
        }

        // 2. Same number of gates
        if self.gates.len() != other.gates.len() {
            return false;
        }

        // 3. Gates in the same order with the same qubits
        // This is a simplified comparison - ideally we'd check if gates are functionally equivalent
        for (i, (gate1, qubits1)) in self.gates.iter().enumerate() {
            let (gate2, qubits2) = &other.gates[i];

            // Compare qubit indices
            if qubits1 != qubits2 {
                return false;
            }

            // Compare gate types by name (this is a simplification)
            // In a full implementation, we would check if gates are functionally equivalent
            if gate1.name() != gate2.name() {
                return false;
            }
        }

        true
    }
}

impl MonoidalCategory for QuantumCircuitCategory {
    fn unit(&self) -> Self::Object {
        0  // Zero-qubit system
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b  // Tensor product combines qubit counts
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        f.tensor(g).unwrap_or_else(|_| self.identity(&0))
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        self.identity(a)  // Trivial for quantum circuits
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        self.identity(a)  // Trivial for quantum circuits
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        self.identity(&(a + b + c))  // Trivial for quantum circuits
    }
}

impl QuantumCircuit {
    /// Convert this circuit to a morphism in the quantum state category
    pub fn to_state_morphism(&self) -> <QuantumStateCategory as Category>::Morphism {
        let category = QuantumStateCategory;
        category.circuit_to_morphism(self)
    }
}

/// A builder for quantum circuits
pub struct CircuitBuilder {
    circuit: QuantumCircuit,
    category_aware: CategoryAware,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub fn new(qubit_count: usize) -> Self {
        CircuitBuilder {
            circuit: QuantumCircuit::new(qubit_count),
            category_aware: CategoryAware::new(),
        }
    }

    /// Build the quantum circuit
    pub fn build(self) -> QuantumCircuit {
        self.circuit
    }

    /// Internal helper to add a gate with categorical awareness
    pub fn add_gate<G: QuantumGate + 'static>(&mut self, gate: G, qubits: &[usize]) -> Result<(), String> {
        self.circuit.add_gate(Box::new(gate), qubits)
    }

    /// Add a Hadamard gate
    pub fn h(&mut self, qubit: usize) -> Result<(), String> {
        self.add_gate(StandardGate::H, &[qubit])
    }

    /// Add a Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> Result<(), String> {
        self.add_gate(StandardGate::X, &[qubit])
    }

    /// Add a Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> Result<(), String> {
        self.add_gate(StandardGate::Y, &[qubit])
    }

    /// Add a Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> Result<(), String> {
        self.add_gate(StandardGate::Z, &[qubit])
    }

    /// Add a CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        self.add_gate(StandardGate::CNOT, &[control, target])
    }

    /// Add a SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        self.add_gate(StandardGate::SWAP, &[qubit1, qubit2])
    }

    /// Add a Toffoli gate (CCNOT)
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> Result<(), String> {
        self.add_gate(StandardGate::Toffoli, &[control1, control2, target])
    }

    /// Add an Rx gate
    pub fn rx(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        self.add_gate(ParametrizedGate::Rx(theta), &[qubit])
    }

    /// Add an Ry gate
    pub fn ry(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        self.add_gate(ParametrizedGate::Ry(theta), &[qubit])
    }

    /// Add an Rz gate
    pub fn rz(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        self.add_gate(ParametrizedGate::Rz(theta), &[qubit])
    }

    /// Add a controlled Rz gate
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) -> Result<(), String> {
        self.add_gate(ParametrizedGate::CRz(theta), &[control, target])
    }

    /// Combine this circuit with another circuit using tensor product
    pub fn tensor(mut self, other: CircuitBuilder) -> Result<Self, String> {
        if self.circuit.gates.is_empty() {
            return Ok(other);
        }
        if other.circuit.gates.is_empty() {
            return Ok(self);
        }

        let self_gate = self.circuit.as_single_gate()?;
        let other_gate = other.circuit.as_single_gate()?;
        let combined_gate = self.category_aware.tensor(self_gate.as_ref(), other_gate.as_ref());
        let combined_qubits = self.circuit.qubit_count + other.circuit.qubit_count;

        let mut new_circuit = QuantumCircuit::new(combined_qubits);
        let all_qubits: Vec<usize> = (0..combined_qubits).collect();
        new_circuit.add_gate(combined_gate, &all_qubits)?;

        self.circuit = new_circuit;
        Ok(self)
    }

    /// Compose this circuit with another circuit
    pub fn compose(mut self, other: CircuitBuilder) -> Result<Self, String> {
        if self.circuit.qubit_count != other.circuit.qubit_count {
            return Err(format!(
                "Cannot compose circuits with different qubit counts: {} and {}",
                self.circuit.qubit_count, other.circuit.qubit_count
            ));
        }

        if self.circuit.gates.is_empty() {
            return Ok(other);
        }
        if other.circuit.gates.is_empty() {
            return Ok(self);
        }

        let self_gate = self.circuit.as_single_gate()?;
        let other_gate = other.circuit.as_single_gate()?;
        let combined_gate = self.category_aware.compose(self_gate.as_ref(), other_gate.as_ref())?;

        let mut new_circuit = QuantumCircuit::new(self.circuit.qubit_count);
        let all_qubits: Vec<usize> = (0..self.circuit.qubit_count).collect();
        new_circuit.add_gate(combined_gate, &all_qubits)?;

        self.circuit = new_circuit;
        Ok(self)
    }
}


// Add Clone implementation for QuantumCircuit
impl Clone for QuantumCircuit {
    fn clone(&self) -> Self {
        QuantumCircuit {
            gates: self.gates.iter()
                .map(|(gate, qubits)| (gate.clone_box(), qubits.clone()))
                .collect(),
            qubit_count: self.qubit_count,
        }
    }
}


pub struct CircuitMatrixFunctor;

impl Functor<QuantumCircuitCategory, QuantumStateCategory> for CircuitMatrixFunctor {
    fn map_object(&self, _c: &QuantumCircuitCategory, _d: &QuantumStateCategory, obj: &usize) -> usize {
        *obj  // Object mapping is identity (same qubit count)
    }

    fn map_morphism(
        &self,
        _c: &QuantumCircuitCategory,
        _d: &QuantumStateCategory,
        f: &QuantumCircuit
    ) -> Array2<Complex64> {
        f.to_state_morphism()  // Map circuit to matrix representation
    }
}

// Convert a quantum gate to a quantum circuit
pub fn gate_to_circuit(gate: &Box<dyn QuantumGate>) -> QuantumCircuit {
    let qubit_count = gate.qubit_count();
    let mut circuit = QuantumCircuit::new(qubit_count);

    // Add the gate to the circuit
    // We need to determine which qubits the gate operates on
    let qubits: Vec<usize> = (0..qubit_count).collect();
    circuit.add_gate(gate.clone_box(), &qubits).unwrap();

    circuit
}

// Convert a quantum circuit to a quantum gate
pub fn circuit_to_gate(circuit: &QuantumCircuit) -> Box<dyn QuantumGate> {
    // If circuit is empty, return identity
    if circuit.gates.is_empty() {
        return Box::new(StandardGate::I(circuit.qubit_count));
    }

    // If circuit has only one gate, return it directly
    if circuit.gates.len() == 1 {
        return circuit.gates[0].0.clone_box();
    }

    // Build the gate sequence
    let gates: Vec<Box<dyn QuantumGate>> = circuit.gates
        .iter()
        .map(|(gate, _)| gate.clone_box())
        .collect();

    // Create the composed gate
    let composed = gate_operations::sequence(gates)
        .unwrap_or_else(|_| Box::new(StandardGate::I(circuit.qubit_count)));

    // Check for known patterns and simplify
    if let Some(composed_gate) = composed.as_any().downcast_ref::<ComposedGate>() {
        // Try to identify and simplify known patterns
        if is_hxh_pattern(composed_gate) {
            return Box::new(StandardGate::Z);
        }
    }

    composed
}

// Helper function to identify H-X-H pattern
fn is_hxh_pattern(gate: &ComposedGate) -> bool {
    // Check if the gate structure matches H-(X-H)
    if let Some(h1) = gate.gate1.as_any().downcast_ref::<StandardGate>() {
        if let StandardGate::H = h1 {
            if let Some(inner) = gate.gate2.as_any().downcast_ref::<ComposedGate>() {
                if let Some(x) = inner.gate1.as_any().downcast_ref::<StandardGate>() {
                    if let Some(h2) = inner.gate2.as_any().downcast_ref::<StandardGate>() {
                        if let (StandardGate::X, StandardGate::H) = (x, h2) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // Also check for (H-X)-H pattern
    if let Some(inner) = gate.gate1.as_any().downcast_ref::<ComposedGate>() {
        if let Some(h1) = inner.gate1.as_any().downcast_ref::<StandardGate>() {
            if let Some(x) = inner.gate2.as_any().downcast_ref::<StandardGate>() {
                if let Some(h2) = gate.gate2.as_any().downcast_ref::<StandardGate>() {
                    if let (StandardGate::H, StandardGate::X, StandardGate::H) = (h1, x, h2) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

pub struct GateToCircuitFunctor;

impl Functor<QuantumGateCategory, QuantumCircuitCategory> for GateToCircuitFunctor {
    fn map_object(&self, _c: &QuantumGateCategory, _d: &QuantumCircuitCategory, obj: &usize) -> usize {
        *obj  // Qubit count stays the same
    }

    fn map_morphism(&self, _c: &QuantumGateCategory, _d: &QuantumCircuitCategory,
                   gate: &Box<dyn QuantumGate>) -> QuantumCircuit {
        gate_to_circuit(gate)
    }
}

pub struct CircuitToGateFunctor;

impl Functor<QuantumCircuitCategory, QuantumGateCategory> for CircuitToGateFunctor {
    fn map_object(&self, _c: &QuantumCircuitCategory, _d: &QuantumGateCategory, obj: &usize) -> usize {
        *obj  // Qubit count stays the same
    }

    fn map_morphism(&self, _c: &QuantumCircuitCategory, _d: &QuantumGateCategory,
                   circuit: &QuantumCircuit) -> Box<dyn QuantumGate> {
        circuit_to_gate(circuit)
    }
}
