use crate::quantum::StandardGate;
use crate::quantum::QuantumCircuit;
use crate::quantum::QuantumGateCategory;
use crate::quantum::ParametrizedGate;
use crate::quantum::QuantumGate;
use crate::prelude::Functor;
use crate::quantum::circuit_to_gate;
use crate::quantum::gate_to_circuit;
use crate::category::monoidal::Category;
use std::fmt;
use std::any::Any;
use num_complex::Complex64;
use ndarray::Array2;
use crate::quantum::StateVector;
use crate::category::monad::Monad;

/// Optimization patterns for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPattern {
    /// Combine adjacent single-qubit rotations around the same axis
    MergeSingleQubitRotations,
    /// Cancel adjacent gates that are inverses of each other
    CancelInverseGates,
    /// Convert sequences of X-X, Y-Y, Z-Z gates to identity
    SimplifyPauliSequences,
    /// Move commuting gates to create optimization opportunities
    CommutationReordering,
    /// Simplify Hadamard-Pauli-Hadamard sequences
    SimplifyHPH,
    /// Optimize CNOT-CNOT sequences
    SimplifyCNOTSequences,
    /// All optimizations
    All,
}

/// Implements circuit optimization as a natural transformation
#[derive(Clone, Debug)]
pub struct CircuitOptimizer {
    /// Which optimization patterns to apply
    patterns: Vec<OptimizationPattern>,
    /// Maximum number of optimization passes
    max_passes: usize,
}

impl Default for CircuitOptimizer {
    fn default() -> Self {
        CircuitOptimizer {
            patterns: vec![OptimizationPattern::All],
            max_passes: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptimizationMonad {
    optimizer: CircuitOptimizer
}

impl OptimizationMonad {

    pub fn new(optimizer: CircuitOptimizer) -> Self {
        Self { optimizer }
    }

    pub fn pure<T: Clone + 'static>(&self, value: T) -> Box<dyn Fn() -> T> {
        let value_clone = value.clone();
        Box::new(move || value_clone.clone())
    }

    // The "bind" operation of the monad (chains monadic operations)
    pub fn bind<A, B, F>(&self, ma: Box<dyn Fn() -> A>, f: F) -> Box<dyn Fn() -> B>
    where
        A: Clone + 'static,
        B: Clone + 'static,
        F: Fn(A) -> Box<dyn Fn() -> B> + 'static,
    {
        Box::new(move || {
            let a = ma();
            let mb = f(a);
            mb()
        })
    }
}

impl Functor<QuantumGateCategory, QuantumGateCategory> for OptimizationMonad {
    fn map_object(&self, _c: &QuantumGateCategory, _d: &QuantumGateCategory, obj: &usize) -> usize {
        *obj  // Objects (qubit counts) remain unchanged
    }

    fn map_morphism(&self, _c: &QuantumGateCategory, _d: &QuantumGateCategory,
                   morphism: &Box<dyn QuantumGate>) -> Box<dyn QuantumGate> {
        // Apply optimizations directly
        self.optimizer.optimize_gate(morphism)
    }
}

impl Monad<QuantumGateCategory> for OptimizationMonad {
    fn unit(&self, c: &QuantumGateCategory, obj: &usize) -> Box<dyn QuantumGate> {
        // Identity morphism - no optimization
        c.identity(obj)
    }

    fn join(&self, _c: &QuantumGateCategory, obj: &usize) -> Box<dyn QuantumGate> {
        // The join operation for our optimization monad is idempotent
        // OptimizedGate(OptimizedGate(x)) = OptimizedGate(x)

        // Create a gate that represents this idempotent property
        Box::new(OptimizationJoinGate {
            qubit_count: *obj,
            optimizer: self.clone()
        })
    }
}

#[derive(Debug)]
pub struct OptimizationJoinGate {
    qubit_count: usize,
    optimizer: OptimizationMonad
}

impl QuantumGate for OptimizationJoinGate {
    fn matrix(&self) -> Array2<Complex64> {
        // Join operation preserves the quantum state
        Array2::eye(1 << self.qubit_count)
            .map(|x| Complex64::new(*x, 0.0))
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(Self {
            qubit_count: self.qubit_count,
            optimizer: self.optimizer.clone()
        })
    }

    fn name(&self) -> String {
        format!("OptimizationJoinGate({})", self.qubit_count)
    }
}

impl CircuitOptimizer {
    /// Create a new circuit optimizer with specific optimization patterns
    pub fn new(patterns: Vec<OptimizationPattern>, max_passes: usize) -> Self {
        CircuitOptimizer {
            patterns,
            max_passes,
        }
    }

    /// Check if a pattern is enabled
    fn is_pattern_enabled(&self, pattern: OptimizationPattern) -> bool {
        self.patterns.contains(&pattern) || self.patterns.contains(&OptimizationPattern::All)
    }

    // Apply optimization using the categorical framework
    pub fn optimize_categorical(&self, gate: &Box<dyn QuantumGate>) -> Box<dyn QuantumGate> {
        // Create the optimization functor (monad)
        let optimization_functor = OptimizationMonad::new(self.clone());

        // Use the functor to map the gate directly to its optimized version
        let c = QuantumGateCategory;
        optimization_functor.map_morphism(&c, &c, gate)
    }
    // For backward compatibility
    pub fn optimize(&self, circuit: &QuantumCircuit) -> QuantumCircuit {
        optimize_circuit(circuit, self)
    }
    pub fn optimize_gate(&self, gate: &Box<dyn QuantumGate>) -> Box<dyn QuantumGate> {
        // Convert gate to circuit
        let circuit = gate_to_circuit(gate);

        // Optimize the circuit
        let optimized_circuit = self.optimize(&circuit);

        // Convert back to gate
        circuit_to_gate(&optimized_circuit)
    }
}


// A gate that represents the optimization transformation itself
pub struct OptimizationGate {
    qubit_count: usize,
    optimizer: OptimizationMonad
}

impl QuantumGate for OptimizationGate {
    fn matrix(&self) -> Array2<Complex64> {
        // For identity optimization, return identity matrix
        Array2::eye(1 << self.qubit_count)
            .map(|x| Complex64::new(*x, 0.0))
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn QuantumGate> {
        Box::new(Self {
            qubit_count: self.qubit_count,
            optimizer: self.optimizer.clone()
        })
    }

    fn name(&self) -> String {
        format!("OptimizationGate({})", self.qubit_count)
    }

    fn apply_to_qubits(&self, state: &StateVector, _qubits: &[usize]) -> Result<StateVector, String> {
        // When an OptimizationGate is applied, it acts as a "join" operation in the monad
        // This means it applies the optimization but ensures we don't over-optimize

        // For most gates, we would transform the state here
        // But since optimization is a meta-operation, we just pass through
        Ok(state.clone())
    }
}

impl fmt::Debug for OptimizationGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OptimizationGate({})", self.qubit_count)
    }
}

/// Optimize a quantum circuit using pattern matching and algebraic simplifications
fn optimize_circuit(circuit: &QuantumCircuit, optimizer: &CircuitOptimizer) -> QuantumCircuit {
    let mut current_circuit = circuit.clone();
    let mut previous_gate_count = usize::MAX;
    let mut pass_count = 0;

    // Repeat optimization until no more improvements are found or max passes reached
    while current_circuit.gate_count() < previous_gate_count && pass_count < optimizer.max_passes {
        previous_gate_count = current_circuit.gate_count();

        // Apply each optimization pattern if enabled
        if optimizer.is_pattern_enabled(OptimizationPattern::MergeSingleQubitRotations) {
            current_circuit = merge_rotations(&current_circuit);
        }

        if optimizer.is_pattern_enabled(OptimizationPattern::CancelInverseGates) {
            current_circuit = cancel_inverse_gates(&current_circuit);
        }

        if optimizer.is_pattern_enabled(OptimizationPattern::SimplifyPauliSequences) {
            current_circuit = simplify_pauli_sequences(&current_circuit);
        }

        if optimizer.is_pattern_enabled(OptimizationPattern::CommutationReordering) {
            current_circuit = apply_commutation_rules(&current_circuit);
        }

        if optimizer.is_pattern_enabled(OptimizationPattern::SimplifyHPH) {
            current_circuit = simplify_hph_sequences(&current_circuit);
        }

        if optimizer.is_pattern_enabled(OptimizationPattern::SimplifyCNOTSequences) {
            current_circuit = simplify_cnot_sequences(&current_circuit);
        }

        pass_count += 1;
    }

    current_circuit
}

/// Merge adjacent rotation gates around the same axis
fn merge_rotations(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    if gates.len() < 2 {
        return circuit.clone();
    }

    let mut skip_next = false;
    for i in 0..gates.len() {
        if skip_next {
            skip_next = false;
            continue;
        }

        // Check if we can merge with the next gate
        if i < gates.len() - 1 {
            let (curr_gate, curr_qubits) = &gates[i];
            let (next_gate, next_qubits) = &gates[i+1];

            // Check if gates act on the same qubits
            if curr_qubits == next_qubits {
                // Try to merge Rx gates
                if let (Some(rx1), Some(rx2)) = (
                    curr_gate.as_any().downcast_ref::<ParametrizedGate>(),
                    next_gate.as_any().downcast_ref::<ParametrizedGate>()
                ) {
                    if let (ParametrizedGate::Rx(theta1), ParametrizedGate::Rx(theta2)) = (rx1, rx2) {
                        // Merge Rx gates by adding their angles
                        let combined_angle = theta1 + theta2;
                        // Skip adding tiny rotations (numerical cleanup)
                        if combined_angle.abs() > 1e-10 {
                            result.add_gate(
                                Box::new(ParametrizedGate::Rx(combined_angle)),
                                curr_qubits
                            ).unwrap();
                        }
                        skip_next = true;
                        continue;
                    }

                    // Similar pattern for Ry gates
                    if let (ParametrizedGate::Ry(theta1), ParametrizedGate::Ry(theta2)) = (rx1, rx2) {
                        let combined_angle = theta1 + theta2;
                        if combined_angle.abs() > 1e-10 {
                            result.add_gate(
                                Box::new(ParametrizedGate::Ry(combined_angle)),
                                curr_qubits
                            ).unwrap();
                        }
                        skip_next = true;
                        continue;
                    }

                    // And for Rz gates
                    if let (ParametrizedGate::Rz(theta1), ParametrizedGate::Rz(theta2)) = (rx1, rx2) {
                        let combined_angle = theta1 + theta2;
                        if combined_angle.abs() > 1e-10 {
                            result.add_gate(
                                Box::new(ParametrizedGate::Rz(combined_angle)),
                                curr_qubits
                            ).unwrap();
                        }
                        skip_next = true;
                        continue;
                    }
                }
            }
        }

        // If no merge was possible, add the gate as is
        result.add_gate(gates[i].0.clone_box(), &gates[i].1).unwrap();
    }

    result
}

/// Cancel adjacent gates that are inverses of each other
fn cancel_inverse_gates(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    if gates.len() < 2 {
        return circuit.clone();
    }

    let mut skip_next = false;
    for i in 0..gates.len() {
        if skip_next {
            skip_next = false;
            continue;
        }

        // Check if we can cancel with the next gate
        if i < gates.len() - 1 {
            let (curr_gate, curr_qubits) = &gates[i];
            let (next_gate, next_qubits) = &gates[i+1];

            // Check if gates act on the same qubits
            if curr_qubits == next_qubits {
                // Check for self-inverse gates (X, Y, Z, H)
                if let (Some(g1), Some(g2)) = (
                    curr_gate.as_any().downcast_ref::<StandardGate>(),
                    next_gate.as_any().downcast_ref::<StandardGate>()
                ) {
                    match (g1, g2) {
                        (StandardGate::X, StandardGate::X) |
                        (StandardGate::Y, StandardGate::Y) |
                        (StandardGate::Z, StandardGate::Z) |
                        (StandardGate::H, StandardGate::H) => {
                            skip_next = true;
                            continue;
                        },
                        _ => {}
                    }
                }

                // Check for inverse rotation gates
                if let (Some(r1), Some(r2)) = (
                    curr_gate.as_any().downcast_ref::<ParametrizedGate>(),
                    next_gate.as_any().downcast_ref::<ParametrizedGate>()
                ) {
                    match (r1, r2) {
                        (ParametrizedGate::Rx(theta1), ParametrizedGate::Rx(theta2)) => {
                            if (theta1 + theta2).abs() < 1e-10 {
                                skip_next = true;
                                continue;
                            }
                        },
                        (ParametrizedGate::Ry(theta1), ParametrizedGate::Ry(theta2)) => {
                            if (theta1 + theta2).abs() < 1e-10 {
                                skip_next = true;
                                continue;
                            }
                        },
                        (ParametrizedGate::Rz(theta1), ParametrizedGate::Rz(theta2)) => {
                            if (theta1 + theta2).abs() < 1e-10 {
                                skip_next = true;
                                continue;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }

        // If no cancellation was possible, add the gate as is
        result.add_gate(gates[i].0.clone_box(), &gates[i].1).unwrap();
    }

    result
}

/// Simplify sequences of Pauli gates
fn simplify_pauli_sequences(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    // Count consecutive Pauli gates on each qubit
    let mut pauli_counts: std::collections::HashMap<usize, (usize, usize, usize)> = std::collections::HashMap::new();
    let mut current_gates: Vec<(Box<dyn QuantumGate>, Vec<usize>)> = Vec::new();

    for (gate, qubits) in gates {
        // Only process single-qubit gates
        if qubits.len() == 1 {
            let qubit = qubits[0];

            if let Some(std_gate) = gate.as_any().downcast_ref::<StandardGate>() {
                // Update Pauli gate counts for this qubit
                let entry = pauli_counts.entry(qubit).or_insert((0, 0, 0));

                match std_gate {
                    StandardGate::X => {
                        entry.0 = (entry.0 + 1) % 2;
                        continue;
                    },
                    StandardGate::Y => {
                        entry.1 = (entry.1 + 1) % 2;
                        continue;
                    },
                    StandardGate::Z => {
                        entry.2 = (entry.2 + 1) % 2;
                        continue;
                    },
                    _ => {}
                }
            }
        }

        // For multi-qubit gates or non-Pauli gates, flush accumulated Pauli operations
        for (qubit, (x_count, y_count, z_count)) in pauli_counts.iter() {
            if *x_count == 1 {
                current_gates.push((Box::new(StandardGate::X), vec![*qubit]));
            }
            if *y_count == 1 {
                current_gates.push((Box::new(StandardGate::Y), vec![*qubit]));
            }
            if *z_count == 1 {
                current_gates.push((Box::new(StandardGate::Z), vec![*qubit]));
            }
        }
        pauli_counts.clear();

        // Add the current gate
        current_gates.push((gate.clone_box(), qubits.clone()));
    }

    // Flush any remaining Pauli operations
    for (qubit, (x_count, y_count, z_count)) in pauli_counts.iter() {
        if *x_count == 1 {
            current_gates.push((Box::new(StandardGate::X), vec![*qubit]));
        }
        if *y_count == 1 {
            current_gates.push((Box::new(StandardGate::Y), vec![*qubit]));
        }
        if *z_count == 1 {
            current_gates.push((Box::new(StandardGate::Z), vec![*qubit]));
        }
    }

    // Rebuild the circuit with simplified gates
    for (gate, qubits) in current_gates {
        result.add_gate(gate, &qubits).unwrap();
    }

    result
}

/// Apply commutation rules to reorder gates for better optimization
fn apply_commutation_rules(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    // This is a complex optimization that requires knowledge of gate commutation rules
    // For now, we'll implement a simple version that tries to group rotations around the same axis

    // Group gates by target qubit
    let mut qubit_gates: std::collections::HashMap<usize, Vec<Box<dyn QuantumGate>>> = std::collections::HashMap::new();
    let mut multi_qubit_gates: Vec<(Box<dyn QuantumGate>, Vec<usize>)> = Vec::new();

    for (gate, qubits) in gates {
        if qubits.len() == 1 {
            // Single-qubit gate
            let qubit = qubits[0];
            qubit_gates.entry(qubit).or_default().push(gate.clone_box());
        } else {
            // Multi-qubit gate
            multi_qubit_gates.push((gate.clone_box(), qubits.clone()));
        }
    }

    // First add all single-qubit gates, trying to group similar rotations
    for (qubit, gates) in qubit_gates {
        let mut rx_gates = Vec::new();
        let mut ry_gates = Vec::new();
        let mut rz_gates = Vec::new();
        let mut other_gates = Vec::new();

        for gate in gates {
            if let Some(param_gate) = gate.as_any().downcast_ref::<ParametrizedGate>() {
                match param_gate {
                    ParametrizedGate::Rx(_) => rx_gates.push(gate),
                    ParametrizedGate::Ry(_) => ry_gates.push(gate),
                    ParametrizedGate::Rz(_) => rz_gates.push(gate),
                    _ => other_gates.push(gate),
                }
            } else {
                other_gates.push(gate);
            }
        }

        // Add gates by type to group similar rotations
        for gate in rx_gates {
            result.add_gate(gate, &[qubit]).unwrap();
        }
        for gate in ry_gates {
            result.add_gate(gate, &[qubit]).unwrap();
        }
        for gate in rz_gates {
            result.add_gate(gate, &[qubit]).unwrap();
        }
        for gate in other_gates {
            result.add_gate(gate, &[qubit]).unwrap();
        }
    }

    // Then add multi-qubit gates
    for (gate, qubits) in multi_qubit_gates {
        result.add_gate(gate, &qubits).unwrap();
    }

    result
}

/// Simplify Hadamard-Pauli-Hadamard sequences
fn simplify_hph_sequences(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    if gates.len() < 3 {
        return circuit.clone();
    }

    let mut i = 0;
    while i < gates.len() {
        // Check for H-X-H pattern (equivalent to Z)
        if i + 2 < gates.len() {
            let (gate1, qubits1) = &gates[i];
            let (gate2, qubits2) = &gates[i+1];
            let (gate3, qubits3) = &gates[i+2];

            // Check if all gates act on the same single qubit
            if qubits1.len() == 1 && qubits1 == qubits2 && qubits2 == qubits3 {
                let qubit = qubits1[0];

                if let (Some(g1), Some(g2), Some(g3)) = (
                    gate1.as_any().downcast_ref::<StandardGate>(),
                    gate2.as_any().downcast_ref::<StandardGate>(),
                    gate3.as_any().downcast_ref::<StandardGate>()
                ) {
                    match (g1, g2, g3) {
                        (StandardGate::H, StandardGate::X, StandardGate::H) => {
                            // Replace H-X-H with Z
                            result.add_gate(Box::new(StandardGate::Z), &[qubit]).unwrap();
                            i += 3;
                            continue;
                        },
                        (StandardGate::H, StandardGate::Z, StandardGate::H) => {
                            // Replace H-Z-H with X
                            result.add_gate(Box::new(StandardGate::X), &[qubit]).unwrap();
                            i += 3;
                            continue;
                        },
                        (StandardGate::H, StandardGate::Y, StandardGate::H) => {
                            // Replace H-Y-H with -Y
                            // For simplicity, we'll just leave this as Y since the global phase doesn't matter
                            result.add_gate(Box::new(StandardGate::Y), &[qubit]).unwrap();
                            i += 3;
                            continue;
                        },
                        _ => {}
                    }
                }
            }
        }

        // No pattern match, add the gate as is
        result.add_gate(gates[i].0.clone_box(), &gates[i].1).unwrap();
        i += 1;
    }

    result
}

/// Simplify CNOT sequences
fn simplify_cnot_sequences(circuit: &QuantumCircuit) -> QuantumCircuit {
    let mut result = QuantumCircuit::new(circuit.qubit_count);
    let gates = &circuit.gates;

    if gates.len() < 2 {
        return circuit.clone();
    }

    let mut skip_next = false;
    for i in 0..gates.len() {
        if skip_next {
            skip_next = false;
            continue;
        }

        // Check for consecutive CNOTs with the same control and target
        if i + 1 < gates.len() {
            let (gate1, qubits1) = &gates[i];
            let (gate2, qubits2) = &gates[i+1];

            if qubits1 == qubits2 && qubits1.len() == 2 {
                if let (Some(g1), Some(g2)) = (
                    gate1.as_any().downcast_ref::<StandardGate>(),
                    gate2.as_any().downcast_ref::<StandardGate>()
                ) {
                    if let (StandardGate::CNOT, StandardGate::CNOT) = (g1, g2) {
                        // Two consecutive CNOTs cancel out
                        skip_next = true;
                        continue;
                    }
                }
            }
        }

        // No pattern match, add the gate as is
        result.add_gate(gates[i].0.clone_box(), &gates[i].1).unwrap();
    }

    result
}

// Add to optimizer.rs or a new file

/// An endofunctor that maps quantum gates to their optimized versions
pub struct OptimizationEndofunctor {
    optimizer: CircuitOptimizer,
}

impl OptimizationEndofunctor {
    pub fn new(optimizer: CircuitOptimizer) -> Self {
        OptimizationEndofunctor { optimizer }
    }
}

impl Functor<QuantumGateCategory, QuantumGateCategory> for OptimizationEndofunctor {
    fn map_object(&self, _c: &QuantumGateCategory, _d: &QuantumGateCategory, obj: &usize) -> usize {
        // The object mapping is identity - optimization doesn't change qubit count
        *obj
    }

    fn map_morphism(&self, _c: &QuantumGateCategory, _d: &QuantumGateCategory,
                   morphism: &Box<dyn QuantumGate>) -> Box<dyn QuantumGate> {
        // Apply optimization to the gate
        self.optimizer.optimize_categorical(morphism)
    }
}
