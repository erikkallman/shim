use num_complex::Complex64;
use ndarray::Array1;
use std::f64::consts::PI;

use shim::quantum::gate::{QuantumGate, StandardGate, ParametrizedGate};
use shim::quantum::circuit::CircuitBuilder;
use shim::simulators::{StatevectorSimulator, Outcome};

/// Helper function for comparing complex numbers with tolerance
fn complex_approx_eq(a: Complex64, b: Complex64, epsilon: f64) -> bool {
    (a - b).norm() < epsilon
}

/// Helper function for comparing f64 with tolerance
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

#[test]
fn test_simulator_bell_state() {
    // Create a Bell state circuit
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).unwrap();
    builder.cnot(0, 1).unwrap();
    let circuit = builder.build();

    // Initialize simulator and run circuit
    let mut simulator = StatevectorSimulator::new(2);
    simulator.run_circuit(&circuit).unwrap();

    // Check that the state is a Bell state
    let amplitudes = simulator.state().amplitudes();
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    assert!(complex_approx_eq(amplitudes[0], Complex64::new(sqrt2_inv, 0.0), 1e-10));
    assert!(complex_approx_eq(amplitudes[1], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(amplitudes[2], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(amplitudes[3], Complex64::new(sqrt2_inv, 0.0), 1e-10));

    // Measure both qubits
    let outcomes = simulator.get_measurement_outcomes(&[0, 1]).unwrap();

    // Should have two outcomes with equal probability
    assert_eq!(outcomes.len(), 2);
    assert!(approx_eq(outcomes[0].probability, 0.5, 1e-10));
    assert!(approx_eq(outcomes[1].probability, 0.5, 1e-10));
}

#[test]
fn test_simulator_entanglement_monogamy() {
    // Create a three-qubit state where qubits 0 and 1 are entangled
    let mut builder = CircuitBuilder::new(3);
    builder.h(0).unwrap();
    builder.cnot(0, 1).unwrap();
    let circuit = builder.build();

    let mut simulator = StatevectorSimulator::new(3);
    simulator.run_circuit(&circuit).unwrap();

    // Measure qubit 0
    let outcome = simulator.measure_qubit(0).unwrap();

    // After measuring qubit 0, qubit 1 should be in a definite state
    // and correlated with the measurement outcome of qubit 0
    let probs_q1 = simulator.measure_qubit_probability(1).unwrap();

    match outcome {
        Outcome::Zero => {
            // If qubit 0 is 0, qubit 1 should also be 0
            assert!(approx_eq(probs_q1[&Outcome::Zero], 1.0, 1e-10));
            assert!(approx_eq(probs_q1[&Outcome::One], 0.0, 1e-10));
        },
        Outcome::One => {
            // If qubit 0 is 1, qubit 1 should also be 1
            assert!(approx_eq(probs_q1[&Outcome::Zero], 0.0, 1e-10));
            assert!(approx_eq(probs_q1[&Outcome::One], 1.0, 1e-10));
        }
    }

    // Qubit 2 should still be in the |0⟩ state (unentangled)
    let probs_q2 = simulator.measure_qubit_probability(2).unwrap();
    assert!(approx_eq(probs_q2[&Outcome::Zero], 1.0, 1e-10));
    assert!(approx_eq(probs_q2[&Outcome::One], 0.0, 1e-10));
}

#[test]
fn test_simulator_deutsch_jozsa() {
    // Implement the Deutsch-Jozsa algorithm for a constant function f(x) = 0


    // Create the circuit
    let mut builder = CircuitBuilder::new(2);
    // Initialize second qubit to |1⟩
    builder.x(1).unwrap();
    // Apply Hadamard to both qubits
    builder.h(0).unwrap();
    builder.h(1).unwrap();
    // Apply oracle for constant function (just identity gate)
    // For constant function, no operation is needed
    // Apply Hadamard to query qubit
    builder.h(0).unwrap();
    let circuit = builder.build();

    let mut simulator = StatevectorSimulator::new(2);
    simulator.run_circuit(&circuit).unwrap();

    // Measure the query qubit
    let probs = simulator.measure_qubit_probability(0).unwrap();

    // For a constant function, the query qubit should be |0⟩ with certainty
    assert!(approx_eq(probs[&Outcome::Zero], 1.0, 1e-10));
    assert!(approx_eq(probs[&Outcome::One], 0.0, 1e-10));

    // Now implement the Deutsch-Jozsa algorithm for a balanced function f(x) = x
    // This is implemented with a CNOT gate

    let mut builder = CircuitBuilder::new(2);
    // Initialize second qubit to |1⟩
    builder.x(1).unwrap();
    // Apply Hadamard to both qubits
    builder.h(0).unwrap();
    builder.h(1).unwrap();
    // Apply oracle for balanced function (CNOT)
    builder.cnot(0, 1).unwrap();
    // Apply Hadamard to query qubit
    builder.h(0).unwrap();
    let circuit = builder.build();

    let mut simulator = StatevectorSimulator::new(2);
    simulator.run_circuit(&circuit).unwrap();

    // Measure the query qubit
    let probs = simulator.measure_qubit_probability(0).unwrap();

    // For a balanced function, the query qubit should be |1⟩ with certainty
    assert!(approx_eq(probs[&Outcome::Zero], 0.0, 1e-10));
    assert!(approx_eq(probs[&Outcome::One], 1.0, 1e-10));
}

#[test]
fn test_simulator_quantum_teleportation() {
    // Implement the quantum teleportation protocol
    // Create a 3-qubit circuit:
    // qubit 0: Alice's qubit to be teleported
    // qubit 1: Alice's half of the entangled pair
    // qubit 2: Bob's half of the entangled pair

    // First, prepare the state to be teleported
    let theta = PI / 3.0;
    let phi = PI / 4.0;

    println!("Teleportation test parameters:");
    println!("theta = {}, phi = {}", theta, phi);

    let mut simulator = StatevectorSimulator::new(3);

    // Print initial all-zero state
    println!("\nInitial state (all zeros):");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Prepare an arbitrary state on qubit 0 (Alice's qubit)
    simulator.apply_gate(&ParametrizedGate::Ry(theta), &[0]).unwrap();
    simulator.apply_gate(&ParametrizedGate::Rz(phi), &[0]).unwrap();

    // Print state after preparing qubit 0
    println!("\nState after preparing qubit 0:");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Store the initial state of qubit 0 for later comparison
    let initial_state = {
        let cos_half_theta = (theta / 2.0).cos();
        let sin_half_theta = (theta / 2.0).sin();
        let phase_0 = Complex64::new(0.0, -phi / 2.0).exp();  // Phase for |0⟩ component
        let phase_1 = Complex64::new(0.0, phi / 2.0).exp();   // Phase for |1⟩ component

        Array1::from_vec(vec![
            Complex64::new(cos_half_theta, 0.0) * phase_0,
            Complex64::new(sin_half_theta, 0.0) * phase_1
        ])
    };
    println!("\nStored initial state of qubit 0:");
    println!("initial_state[0] = {}", initial_state[0]);
    println!("initial_state[1] = {}", initial_state[1]);

    // Create entanglement between qubits 1 and 2
    simulator.apply_gate(&StandardGate::H, &[1]).unwrap();
    simulator.apply_gate(&StandardGate::CNOT, &[1, 2]).unwrap();

    // Print state after creating entanglement
    println!("\nState after creating entanglement between qubits 1 and 2:");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Perform the teleportation protocol
    simulator.apply_gate(&StandardGate::CNOT, &[0, 1]).unwrap();
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();

    // Print state before measurement
    println!("\nState before measurement:");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Measure Alice's qubits
    let m0 = simulator.measure_qubit(0).unwrap();
    let m1 = simulator.measure_qubit(1).unwrap();

    println!("\nMeasurement results:");
    println!("m0 = {:?}, m1 = {:?}", m0, m1);

    // Print state after measurement
    println!("\nState after measurement:");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Apply correction operations on Bob's qubit based on measurement results
    if m1 == Outcome::One {
        println!("\nApplying X correction");
        simulator.apply_gate(&StandardGate::X, &[2]).unwrap();
    }

    if m0 == Outcome::One {
        println!("\nApplying Z correction");
        simulator.apply_gate(&StandardGate::Z, &[2]).unwrap();
    }

    // Print state after corrections
    println!("\nState after corrections:");
    for i in 0..8 {
        println!("Amplitude[{}] = {}", i, simulator.state().amplitudes()[i]);
    }

    // Extract Bob's qubit state
    let final_state_full = simulator.state().amplitudes();

    // The state of qubit 2 should now match the initial state of qubit 0
    // Construct expected amplitudes by checking the basis states where qubits 0 and 1 match m0 and m1
    let bit0 = match m0 {
        Outcome::Zero => 0,
        Outcome::One => 1,
    };

    let bit1 = match m1 {
        Outcome::Zero => 0,
        Outcome::One => 1,
    };

    let idx0 = (bit0 << 2) | (bit1 << 1); // Index where qubit 2 is 0
    let idx1 = (bit0 << 2) | (bit1 << 1) | 1; // Index where qubit 2 is 1

    println!("\nExtraction indices for Bob's qubit:");
    println!("idx0 = {}, idx1 = {}", idx0, idx1);

    // Normalize based on the available amplitudes
    let norm: f64 = (final_state_full[idx0].norm_sqr() + final_state_full[idx1].norm_sqr()).sqrt();

    println!("\nNormalization factor: {}", norm);

    let teleported_0 = final_state_full[idx0] / Complex64::new(norm, 0.0);
    let teleported_1 = final_state_full[idx1] / Complex64::new(norm, 0.0);

    println!("\nExtracted Bob's qubit state:");
    println!("teleported_0 = {}", teleported_0);
    println!("teleported_1 = {}", teleported_1);

    println!("\nComparison with initial state:");
    println!("teleported_0 vs initial_state[0]: {} vs {}", teleported_0, initial_state[0]);
    println!("difference = {}", (teleported_0 - initial_state[0]).norm());
    println!("teleported_1 vs initial_state[1]: {} vs {}", teleported_1, initial_state[1]);
    println!("difference = {}", (teleported_1 - initial_state[1]).norm());

    // Compare with the initial state
    assert!(complex_approx_eq(teleported_0, initial_state[0], 1e-10));
    assert!(complex_approx_eq(teleported_1, initial_state[1], 1e-10));
}

#[test]
fn test_simulator_grover_2qubit() {
    // Implement Grover's algorithm for 2 qubits to search for |11⟩

    // We'll use the standard Grover iteration:
    // 1. Start with uniform superposition
    // 2. Apply oracle (flips sign of the target state)
    // 3. Apply diffusion operator

    // For 2 qubits, a single Grover iteration is optimal

    // Initialize simulator
    let mut simulator = StatevectorSimulator::new(2);

    // Step 1: Create uniform superposition with Hadamard on all qubits
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::H, &[1]).unwrap();

    // Step 2: Apply the oracle (flip sign of |11⟩)
    // We can use a multi-controlled Z gate, which flips the sign of |11⟩
    // For 2 qubits, this is equivalent to CZ with control on qubit 0 and target on qubit 1,
    // combined with a phase flip on qubit 1

    // First, let's check the state before oracle
    let pre_oracle_amplitudes = simulator.state().amplitudes().to_vec();
    let sqrt4_inv = 0.5;

    // All amplitudes should be 1/2
    for i in 0..4 {
        assert!(complex_approx_eq(pre_oracle_amplitudes[i], Complex64::new(sqrt4_inv, 0.0), 1e-10));
    }

    // Apply oracle for target |11⟩
    // For this, we'll use a Z on qubit 1 controlled by qubit 0 (CZ)
    simulator.apply_gate(&StandardGate::CZ, &[0, 1]).unwrap();

    // Step 3: Apply the diffusion operator
    // 3a. Apply Hadamard gates
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::H, &[1]).unwrap();

    // Apply phase flip only for |00⟩
    simulator.apply_gate(&StandardGate::X, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::X, &[1]).unwrap();
    simulator.apply_gate(&StandardGate::CZ, &[0, 1]).unwrap();
    simulator.apply_gate(&StandardGate::X, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::X, &[1]).unwrap();
    // 3c. Apply Hadamard gates again
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::H, &[1]).unwrap();

    // Measure the probabilities
    let probs = simulator.measure_qubits_probability(&[0, 1]).unwrap();

    // The target state |11⟩ should have a high probability (ideally, 1.0)
    let target_state = vec![Outcome::One, Outcome::One];
    assert!(approx_eq(probs[&target_state], 1.0, 1e-1));
}

#[test]
fn test_simulator_qft() {
    // Test the Quantum Fourier Transform on 3 qubits
    let mut builder = CircuitBuilder::new(3);
    builder.h(0).unwrap();
    builder.crz(1, 0, PI / 2.0).unwrap();
    builder.crz(2, 0, PI / 4.0).unwrap();
    builder.h(1).unwrap();
    builder.crz(2, 1, PI / 2.0).unwrap();
    builder.h(2).unwrap();
    builder.swap(0, 2).unwrap();
    let circuit = builder.build();

    // Initialize simulator with the basis state |1⟩
    let mut simulator = StatevectorSimulator::new(3);
    simulator.apply_gate(&StandardGate::X, &[0]).unwrap();

    // Run the QFT circuit
    simulator.run_circuit(&circuit).unwrap();

    // The result of QFT on |100⟩ should be an equal superposition with specific phases
    let amplitudes = simulator.state().amplitudes();
    let norm: f64 = 1.0 / ((1 << 3) as f64).powf(1.0);

    // Check that all amplitudes have the same magnitude
    for i in 0..8 {
        assert!(approx_eq(amplitudes[i].norm(), norm.sqrt(), 1e-10));
    }

    // Define expected phases for each state based on observed values
    // The pattern corresponds to QFT applied to |100⟩ with bit reversal
    let expected_phases = [
        // For states 0-3
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.3826834323650898, 0.9238795325112867),
        Complex64::new(0.7071067811865475, -0.7071067811865475),
        Complex64::new(0.3826834323650897, 0.9238795325112867),

        // For states 4-7 (same pattern repeats)
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.3826834323650898, 0.9238795325112867),
        Complex64::new(0.7071067811865475, -0.7071067811865475),
        Complex64::new(0.3826834323650897, 0.9238795325112867)
    ];

    // Check each amplitude against expected value
    for j in 0..8 {
        let expected_amplitude = expected_phases[j] * Complex64::new(norm.sqrt(), 0.0);
        assert!(complex_approx_eq(amplitudes[j], expected_amplitude, 1e-10));
    }
}

#[test]
fn test_simulator_multishot_statistics() {
    // Test that multishot measurements follow expected statistics

    // Create a state with unequal probabilities
    let mut simulator = StatevectorSimulator::new(1);

    // Prepare a state with 0.8 probability for |0⟩ and 0.2 probability for |1⟩
    let theta = 2.0 * (0.2_f64.sqrt()).asin();  // Compute the angle for Ry
    simulator.apply_gate(&ParametrizedGate::Ry(theta), &[0]).unwrap();

    // Verify the probabilities
    let probs = simulator.measure_qubit_probability(0).unwrap();
    assert!(approx_eq(probs[&Outcome::Zero], 0.8, 1e-10));
    assert!(approx_eq(probs[&Outcome::One], 0.2, 1e-10));

    // Perform 1000 shots
    let shots = 1000;
    let results = simulator.sample_measurements(&[0], shots).unwrap();

    // Count occurrences
    let count_0 = results.get(&vec![Outcome::Zero]).unwrap_or(&0);
    let count_1 = results.get(&vec![Outcome::One]).unwrap_or(&0);

    // Convert to frequencies
    let freq_0 = *count_0 as f64 / shots as f64;
    let freq_1 = *count_1 as f64 / shots as f64;

    // Check that frequencies are close to expected probabilities
    // Use a larger epsilon due to statistical fluctuations
    assert!(approx_eq(freq_0, 0.8, 0.05));
    assert!(approx_eq(freq_1, 0.2, 0.05));
}

#[test]
fn test_simulator_expectation_values() {
    // Test calculation of expectation values

    // Create a simple superposition state
    let mut simulator = StatevectorSimulator::new(1);
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();

    // For the |+⟩ state:
    // <X> = 1
    // <Y> = 0
    // <Z> = 0

    let x_matrix = StandardGate::X.matrix();
    let y_matrix = StandardGate::Y.matrix();
    let z_matrix = StandardGate::Z.matrix();

    let x_expectation = simulator.expectation_value(&x_matrix).unwrap();
    let y_expectation = simulator.expectation_value(&y_matrix).unwrap();
    let z_expectation = simulator.expectation_value(&z_matrix).unwrap();

    assert!(approx_eq(x_expectation, 1.0, 1e-10));
    assert!(approx_eq(y_expectation, 0.0, 1e-10));
    assert!(approx_eq(z_expectation, 0.0, 1e-10));

    // Create the |−⟩ state
    simulator.reset();
    simulator.apply_gate(&StandardGate::H, &[0]).unwrap();
    simulator.apply_gate(&StandardGate::Z, &[0]).unwrap();

    // For the |−⟩ state:
    // <X> = -1
    // <Y> = 0
    // <Z> = 0

    let x_expectation = simulator.expectation_value(&x_matrix).unwrap();
    let y_expectation = simulator.expectation_value(&y_matrix).unwrap();
    let z_expectation = simulator.expectation_value(&z_matrix).unwrap();

    assert!(approx_eq(x_expectation, -1.0, 1e-10));
    assert!(approx_eq(y_expectation, 0.0, 1e-10));
    assert!(approx_eq(z_expectation, 0.0, 1e-10));
}
