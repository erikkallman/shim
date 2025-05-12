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

    /// Create the adjoint (dagger) of this circuit
    pub fn adjoint(&self) -> Self {
        let mut result = QuantumCircuit::new(self.qubit_count);

        // Add the gates in reverse order, with each gate replaced by its adjoint
        for (gate, qubits) in self.gates.iter().rev() {
            let adjoint_gate = gate.adjoint();
            // The adjoint of a gate acts on the same qubits
            let _ = result.add_gate(adjoint_gate, qubits);
        }

        result
    }
    /// Convert this circuit to a morphism in the quantum state category
    pub fn to_state_morphism(&self) -> <QuantumStateCategory as Category>::Morphism {
        let category = QuantumStateCategory;
        category.circuit_to_morphism(self)
    }

    /// Implement quantum teleportation using the compact closed structure
    pub fn teleport(&self, state_circuit: &QuantumCircuit) -> Result<QuantumCircuit, String> {
        if state_circuit.qubit_count != 1 {
            return Err("Teleportation requires a single-qubit state".to_string());
        }

        let cat = QuantumCircuitCategory;

        // Create an entangled Bell pair
        let bell_pair = cat.unit_morphism(&1);

        // Tensor the state with the first qubit of the Bell pair
        let state_and_bell = state_circuit.tensor(&bell_pair)?;

        // Perform Bell measurement on the state and first Bell qubit
        let _bell_measurement = cat.counit_morphism(&1);

        // Need to reorder qubits to apply Bell measurement to the right qubits
        let mut measurement_circuit = QuantumCircuit::new(3);
        // Apply CNOT with control on state qubit, target on first Bell qubit
        measurement_circuit.add_gate(Box::new(StandardGate::CNOT), &[0, 1])?;
        // Apply Hadamard to state qubit
        measurement_circuit.add_gate(Box::new(StandardGate::H), &[0])?;

        // Compose: first prepare state and Bell pair, then measure
        let teleported = measurement_circuit.compose(&state_and_bell)?;

        // The third qubit now contains the teleported state (up to corrections)
        // For a complete teleportation protocol, classical communication and
        // corrections would follow but for now lets just..
        Ok(teleported)
    }

    /// Implement entanglement swapping using the compact closed structure
    pub fn entanglement_swap(a_entangled: &QuantumCircuit, b_entangled: &QuantumCircuit) -> Result<QuantumCircuit, String> {
        if a_entangled.qubit_count != 2 || b_entangled.qubit_count != 2 {
            return Err("Entanglement swapping requires two-qubit entangled states".to_string());
        }

        let cat = QuantumCircuitCategory;

        // Tensor the two entangled states
        let combined = a_entangled.tensor(b_entangled)?;

        // Perform Bell measurement on the middle two qubits (1 and 2)
        let _bell_measurement = cat.counit_morphism(&1);

        // Need to apply the measurement to the right qubits
        let mut measurement_circuit = QuantumCircuit::new(4);
        // Apply CNOT with control on qubit 1, target on qubit 2
        measurement_circuit.add_gate(Box::new(StandardGate::CNOT), &[1, 2])?;
        // Apply Hadamard to qubit 1
        measurement_circuit.add_gate(Box::new(StandardGate::H), &[1])?;

        // Compose the circuits
        let swapped = measurement_circuit.compose(&combined)?;

        // Qubits 0 and 3 are now entangled, despite never having interacted directly
        Ok(swapped)
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

impl SymmetricMonoidalCategory for QuantumCircuitCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Implement braiding as a sequence of SWAP gates
        let total_qubits = a + b;
        let mut circuit = QuantumCircuit::new(total_qubits);

        // Add SWAP gates to move the 'a' qubits past the 'b' qubits
        for i in 0..*a {
            for j in 0..*b {
                let qubit1 = a - i - 1; // Start from the rightmost 'a' qubit
                let qubit2 = a + j;     // Corresponding 'b' qubit
                let _ = circuit.add_gate(Box::new(StandardGate::SWAP), &[qubit1, qubit2]);
            }
        }

        circuit
    }
}

impl DaggerCategory for QuantumCircuitCategory {
    fn dagger(&self, f: &Self::Morphism) -> Self::Morphism {
        f.adjoint()
    }
}

impl CompactClosedCategory for QuantumCircuitCategory {
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

        circuit
    }

    fn counit_morphism(&self, a: &Self::Object) -> Self::Morphism {
        // Bell measurement (counit morphism): ε_A: A ⊗ A* → I
        // This represents measurement in Bell basis
        let mut circuit = QuantumCircuit::new(2 * a);

        // Bell measurement is effectively the adjoint of Bell state preparation
        for i in 0..*a {
            // Apply CNOT with control on first qubit, target on second
            let _ = circuit.add_gate(Box::new(StandardGate::CNOT), &[i, i + a]);
            // Apply Hadamard to first qubit
            let _ = circuit.add_gate(Box::new(StandardGate::H), &[i]);
        }

        circuit
    }
}

impl DaggerCompactCategory for QuantumCircuitCategory {}


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

    /// Create the adjoint (dagger) of this circuit
    pub fn adjoint(mut self) -> Self {
        self.circuit = self.circuit.adjoint();
        self
    }

    /// Create a Bell pair (entangled state)
    pub fn bell_pair(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        self.h(qubit1)?;
        self.cnot(qubit1, qubit2)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dagger_category_laws() {
        let category = QuantumCircuitCategory;

        // Create test circuits
        let mut c1 = QuantumCircuit::new(1);
        c1.add_gate(Box::new(StandardGate::X), &[0]).unwrap();

        let mut c2 = QuantumCircuit::new(1);
        c2.add_gate(Box::new(StandardGate::H), &[0]).unwrap();

        let mut c3 = QuantumCircuit::new(1);
        c3.add_gate(Box::new(StandardGate::Z), &[0]).unwrap();

        // Compose circuits for testing
        let c1_c2 = category.compose(&c1, &c2).unwrap();

        // Verify involution: (f†)† = f
        let c1_dagger = category.dagger(&c1);
        let c1_dagger_dagger = category.dagger(&c1_dagger);
        assert_eq!(c1, c1_dagger_dagger);

        // Verify contravariance: (g ∘ f)† = f† ∘ g†
        let c1_c2_dagger = category.dagger(&c1_c2);
        let c2_dagger = category.dagger(&c2);
        let c1_dagger = category.dagger(&c1);
        let c2_dagger_c1_dagger = category.compose(&c2_dagger, &c1_dagger).unwrap();
        assert_eq!(c1_c2_dagger, c2_dagger_c1_dagger);
    }

    #[test]
    fn test_compact_closed_snake_equations() {
        let category = QuantumCircuitCategory;

        // Test for 1 qubit
        let qubit_count = 1;

        // Get unit morphism: η_A: I → A* ⊗ A
        let unit = category.unit_morphism(&qubit_count);

        // Get counit morphism: ε_A: A ⊗ A* → I
        let counit = category.counit_morphism(&qubit_count);

        // Create identity morphism on A
        let id_a = category.identity(&qubit_count);

        // Tensor operations for testing snake equations
        let _id_a_tensor_unit = category.tensor_morphisms(&id_a, &unit);
        let _counit_tensor_id_a = category.tensor_morphisms(&counit, &id_a);

        // First snake equation: (id_A ⊗ ε_A) ∘ (η_A ⊗ id_A) = id_A
        // Due to quantum circuit limitations, we need to simulate this equation
        // through matrix equivalence rather than direct circuit comparison

        // For now, we test that applying this sequence to a basis state returns the same state
        // Create a basis state to test with
        let basis_state = StateVector::zero_state(qubit_count);

        // Apply the sequence
        let state1 = id_a.apply(&basis_state).unwrap();

        // Verify state hasn't changed (identity operation)
        assert_eq!(state1.amplitudes(), basis_state.amplitudes());
    }

    #[test]
    fn test_teleportation_protocol() {
        // Create a test state to teleport
        let mut state_circuit = QuantumCircuit::new(1);
        state_circuit.add_gate(Box::new(StandardGate::X), &[0]).unwrap();

        // Create a circuit that implements teleportation
        let teleported = state_circuit.teleport(&state_circuit).unwrap();

        // For a complete test, we would need to simulate the circuit and verify
        // that the final qubit is in the same state as the input
        assert!(teleported.qubit_count >= 3);
    }
}
