// examples/quantum_transformations.rs
//! Example demonstrating the use of quantum circuit transformations
//!
//! This example shows how to use natural transformations to modify
//! quantum circuits while preserving their categorical structure.

use shim::quantum::{
    QuantumCircuit, CircuitBuilder,
    transformations::{CircuitTransformer, CircuitOptimizerTransformation},
    specialized_transformations::{ErrorDetectionTransformation, ZXCalculusTransformation},
    optimizer::{CircuitOptimizer, OptimizationPattern},
};
use shim::quantum::QuantumCircuitTransformation;

fn main() {
    println!("Quantum Circuit Transformations Example");
    println!("======================================\n");

    // Create a simple quantum circuit using the builder
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).unwrap();
    builder.cnot(0, 1).unwrap();
    builder.h(0).unwrap();
    builder.z(0).unwrap();
    builder.h(0).unwrap();

    let circuit = builder.build();

    println!("Original circuit:");
    print_circuit(&circuit);

    // Create a simple optimizer transformer
    let optimizer = CircuitOptimizer::new(
        vec![OptimizationPattern::SimplifyHPH],
        3
    );
    let optimizer_transform = CircuitOptimizerTransformation::new(optimizer);

    // Apply the optimizer transformation
    let optimized_circuit = optimizer_transform.transform(&circuit);

    println!("\nOptimized circuit (using H-Z-H -> X rule):");
    print_circuit(&optimized_circuit);

    // Create a ZX calculus transformer
    let zx_transform = ZXCalculusTransformation;

    // Apply the ZX calculus transformation
    let zx_optimized_circuit = zx_transform.transform(&circuit);

    println!("\nZX calculus optimized circuit:");
    print_circuit(&zx_optimized_circuit);

    // Creating a circuit transformer to chain transformations
    let mut transformer = CircuitTransformer::new();
    transformer
        .add_transformation(Box::new(optimizer_transform.clone()))
        .add_transformation(Box::new(ErrorDetectionTransformation::new(2)));

    // Apply the composed transformation
    let transformed_circuit = transformer.transform(&circuit);

    println!("\nCircuit with optimizer + error detection:");
    println!("(Original {} qubits, Transformed {} qubits, {} gates)",
        circuit.qubit_count,
        transformed_circuit.qubit_count,
        transformed_circuit.gate_count()
    );

    // Teleportation circuit example
    println!("\n\nQuantum Teleportation Example");
    println!("==============================\n");

    // Create state to teleport (|+> state)
    let mut state_builder = CircuitBuilder::new(1);
    state_builder.h(0).unwrap();
    let state_circuit = state_builder.build();

    println!("State to teleport (|+> state):");
    print_circuit(&state_circuit);

    // Apply teleportation
    let teleported = state_circuit.teleport(&state_circuit).unwrap();

    println!("\nTeleportation circuit:");
    print_circuit(&teleported);

    // Apply error detection to teleportation
    let error_detection = ErrorDetectionTransformation::new(2);
    let protected_teleportation = error_detection.transform(&teleported);

    println!("\nError-protected teleportation circuit:");
    println!("(Original {} qubits, Protected {} qubits, {} gates)",
        teleported.qubit_count,
        protected_teleportation.qubit_count,
        protected_teleportation.gate_count()
    );
}

// Helper function to print circuit information
fn print_circuit(circuit: &QuantumCircuit) {
    println!("Qubits: {}, Gates: {}", circuit.qubit_count, circuit.gate_count());

    for (i, (gate, qubits)) in circuit.gates.iter().enumerate() {
        println!("  Gate {}: {} on qubits {:?}", i, gate.name(), qubits);
    }
}
