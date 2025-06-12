use shim::quantum::StandardGate;
use shim::quantum::density_matrix::PureToDensityFunctor;
use shim::quantum::density_matrix::DensityMatrixCategory;
use shim::quantum::QuantumStateCategory;
use shim::quantum::StateVector;
use num_complex::Complex64;
use ndarray::Array1;
use shim::category::Functor;
use shim::quantum::QuantumGate;
use shim::quantum::QuantumState;
use shim::category::MonoidalCategory;
use shim::category::Category;
use shim::quantum::DensityMatrix;

/// Helper function for comparing complex numbers with tolerance
fn complex_approx_eq(a: Complex64, b: Complex64, epsilon: f64) -> bool {
    (a - b).norm() < epsilon
}

// Helper function for comparing state vectors with tolerance
fn state_approx_eq(a: &StateVector, b: &StateVector, epsilon: f64) -> bool {
    if a.qubit_count() != b.qubit_count() {
        return false;
    }

    let a_amp = a.amplitudes();
    let b_amp = b.amplitudes();

    if a_amp.len() != b_amp.len() {
        return false;
    }

    for i in 0..a_amp.len() {
        if !complex_approx_eq(a_amp[i], b_amp[i], epsilon) {
            return false;
        }
    }

    true
}


#[test]
fn test_density_matrix_creation_and_conversion() {
    // Create a pure state
    let state = StateVector::computational_basis(1, 0).unwrap(); // |0⟩
    println!("Original state: |0⟩ with amplitudes: {:?}", state.amplitudes());

    // Convert to density matrix
    let dm = DensityMatrix::from_state_vector(&state);
    println!("Density matrix from |0⟩:");
    for i in 0..2 {
        for j in 0..2 {
            println!("dm[{}, {}] = {}", i, j, dm.matrix()[[i, j]]);
        }
    }

    // Check dimensions
    assert_eq!(dm.qubit_count(), 1);
    assert_eq!(dm.dimension(), 2);

    // Check matrix elements for |0⟩⟨0|
    assert!(complex_approx_eq(dm.matrix()[[0, 0]], Complex64::new(1.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm.matrix()[[0, 1]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm.matrix()[[1, 0]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm.matrix()[[1, 1]], Complex64::new(0.0, 0.0), 1e-10));

    // Check that the state is pure
    let purity = dm.purity();
    println!("Purity of |0⟩⟨0|: {}", purity);
    assert!((purity - 1.0).abs() < 1e-10);

    // Debug the density matrix eigenvalues (if possible)
    println!("Density matrix diagonal elements:");
    for i in 0..2 {
        println!("diagonal[{}] = {}", i, dm.matrix()[[i, i]]);
    }

    // Convert back to state vector with detailed debugging
    println!("Attempting to convert back to state vector");
    let recovered_state_option = dm.to_state_vector();

    if recovered_state_option.is_none() {
        println!("Failed to convert density matrix back to state vector!");
        // Additional debugging to understand why conversion failed
        let is_diagonal = (dm.matrix()[[0, 1]].norm() < 1e-10) && (dm.matrix()[[1, 0]].norm() < 1e-10);
        println!("Is matrix diagonal? {}", is_diagonal);

        // Check if any diagonal element is significant
        let has_significant_diag = dm.matrix()[[0, 0]].norm() > 1e-10 || dm.matrix()[[1, 1]].norm() > 1e-10;
        println!("Has significant diagonal element? {}", has_significant_diag);

        // Force an assertion failure to see the debug output
        assert!(false, "Failed to convert |0⟩⟨0| back to state vector");
    }

    let recovered_state = recovered_state_option.unwrap();
    println!("Recovered state amplitudes: {:?}", recovered_state.amplitudes());

    // Check that we recover the original state
    assert!(state_approx_eq(&state, &recovered_state, 1e-10));

    // Create a superposition state
    let mut amplitudes = Array1::zeros(2);
    amplitudes[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    amplitudes[1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let plus_state = StateVector::new(1, amplitudes).unwrap(); // |+⟩
    println!("\nPlus state amplitudes: {:?}", plus_state.amplitudes());

    let plus_dm = DensityMatrix::from_state_vector(&plus_state);
    println!("Plus state density matrix:");
    for i in 0..2 {
        for j in 0..2 {
            println!("plus_dm[{}, {}] = {}", i, j, plus_dm.matrix()[[i, j]]);
        }
    }

    // Check matrix elements for |+⟩⟨+|
    assert!(complex_approx_eq(plus_dm.matrix()[[0, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(plus_dm.matrix()[[0, 1]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(plus_dm.matrix()[[1, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(plus_dm.matrix()[[1, 1]], Complex64::new(0.5, 0.0), 1e-10));

    // Check purity of |+⟩⟨+|
    let plus_purity = plus_dm.purity();
    println!("Purity of |+⟩⟨+|: {}", plus_purity);

    // Debug the density matrix eigenvalues
    println!("Plus state density matrix diagonal elements:");
    for i in 0..2 {
        println!("diagonal[{}] = {}", i, plus_dm.matrix()[[i, i]]);
    }

    // Convert back to state vector with detailed debugging
    println!("Attempting to convert |+⟩⟨+| back to state vector");
    let recovered_plus_option = plus_dm.to_state_vector();

    if recovered_plus_option.is_none() {
        println!("Failed to convert |+⟩⟨+| back to state vector!");
        // Additional debugging
        println!("Off-diagonal elements norms: {}, {}",
                 plus_dm.matrix()[[0, 1]].norm(), plus_dm.matrix()[[1, 0]].norm());

        // Force an assertion failure to see the debug output
        assert!(false, "Failed to convert |+⟩⟨+| back to state vector");
    }

    let recovered_plus = recovered_plus_option.unwrap();
    println!("Recovered plus state amplitudes: {:?}", recovered_plus.amplitudes());

    assert!(state_approx_eq(&plus_state, &recovered_plus, 1e-10));
}

#[test]
fn test_density_matrix_tensor_product() {
    // Create two 1-qubit states
    let state1 = StateVector::computational_basis(1, 0).unwrap(); // |0⟩
    let state2 = StateVector::computational_basis(1, 1).unwrap(); // |1⟩

    // Convert to density matrices
    let dm1 = DensityMatrix::from_state_vector(&state1);
    let dm2 = DensityMatrix::from_state_vector(&state2);

    // Tensor them together
    let dm12 = dm1.tensor(&dm2);

    // Check dimensions
    assert_eq!(dm12.qubit_count(), 2);
    assert_eq!(dm12.dimension(), 4);

    // Check that the resulting matrix represents |01⟩⟨01|
    assert!(complex_approx_eq(dm12.matrix()[[0, 0]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm12.matrix()[[0, 1]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm12.matrix()[[1, 1]], Complex64::new(1.0, 0.0), 1e-10));

    // Check that we can convert back to a state vector
    let recovered_state = dm12.to_state_vector().unwrap();
    let expected_state = state1.tensor(&state2);

    assert!(state_approx_eq(&expected_state, &recovered_state, 1e-10));
}

#[test]
fn test_partial_trace_pure_state() {
    // Create a 2-qubit Bell state (|00⟩ + |11⟩)/√2
    let mut amplitudes = Array1::zeros(4);
    amplitudes[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    amplitudes[3] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let bell_state = StateVector::new(2, amplitudes).unwrap();

    // Convert to density matrix
    let bell_dm = DensityMatrix::from_state_vector(&bell_state);

    // Trace out the first qubit
    let traced_dm = bell_dm.partial_trace(&[0]).unwrap();

    // Check dimensions
    assert_eq!(traced_dm.qubit_count(), 1);
    assert_eq!(traced_dm.dimension(), 2);

    // The result should be the maximally mixed state I/2
    assert!(complex_approx_eq(traced_dm.matrix()[[0, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[0, 1]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[1, 0]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[1, 1]], Complex64::new(0.5, 0.0), 1e-10));

    // Check that it's a mixed state with purity 0.5
    assert!((traced_dm.purity() - 0.5).abs() < 1e-10);

    // Verify that we can't convert this mixed state back to a pure state vector
    assert!(traced_dm.to_state_vector().is_none());

    // Try the same with tracing out the second qubit
    let traced_dm2 = bell_dm.partial_trace(&[1]).unwrap();

    // Should be the same mixed state
    assert!(complex_approx_eq(traced_dm2.matrix()[[0, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm2.matrix()[[1, 1]], Complex64::new(0.5, 0.0), 1e-10));
    assert!((traced_dm2.purity() - 0.5).abs() < 1e-10);
}

#[test]
fn test_partial_trace_separable_state() {
    // Create a 2-qubit separable state |0⟩⊗|+⟩
    let zero = StateVector::computational_basis(1, 0).unwrap();

    let mut plus_amplitudes = Array1::zeros(2);
    plus_amplitudes[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    plus_amplitudes[1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let plus = StateVector::new(1, plus_amplitudes).unwrap();

    let state = zero.tensor(&plus);
    let dm = DensityMatrix::from_state_vector(&state);

    // Trace out the first qubit
    let traced_dm = dm.partial_trace(&[0]).unwrap();

    // Should get |+⟩⟨+|
    assert!(complex_approx_eq(traced_dm.matrix()[[0, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[0, 1]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[1, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm.matrix()[[1, 1]], Complex64::new(0.5, 0.0), 1e-10));

    // Should be a pure state with purity 1
    assert!((traced_dm.purity() - 1.0).abs() < 1e-10);

    // Trace out the second qubit
    let traced_dm2 = dm.partial_trace(&[1]).unwrap();

    // Should get |0⟩⟨0|
    assert!(complex_approx_eq(traced_dm2.matrix()[[0, 0]], Complex64::new(1.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm2.matrix()[[0, 1]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm2.matrix()[[1, 0]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced_dm2.matrix()[[1, 1]], Complex64::new(0.0, 0.0), 1e-10));

    // Verify we can convert back to a state vector
    let recovered = traced_dm2.to_state_vector().unwrap();
    assert!(state_approx_eq(&zero, &recovered, 1e-10));
}

#[test]
fn test_density_matrix_operations() {
    // Create a state vector
    let state = StateVector::computational_basis(1, 0).unwrap(); // |0⟩
    let dm = DensityMatrix::from_state_vector(&state);

    // Apply X gate operation
    let x_gate = StandardGate::X;
    let x_matrix = x_gate.matrix();
    let dm_after_x = dm.apply_operation(&x_matrix).unwrap();

    // Should be |1⟩⟨1|
    assert!(complex_approx_eq(dm_after_x.matrix()[[0, 0]], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(dm_after_x.matrix()[[1, 1]], Complex64::new(1.0, 0.0), 1e-10));

    // Apply H gate operation
    let h_gate = StandardGate::H;
    let h_matrix = h_gate.matrix();
    let dm_after_h = dm.apply_operation(&h_matrix).unwrap();

    // Should be |+⟩⟨+|
    assert!(complex_approx_eq(dm_after_h.matrix()[[0, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(dm_after_h.matrix()[[0, 1]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(dm_after_h.matrix()[[1, 0]], Complex64::new(0.5, 0.0), 1e-10));
    assert!(complex_approx_eq(dm_after_h.matrix()[[1, 1]], Complex64::new(0.5, 0.0), 1e-10));
}

#[test]
fn test_category_operations() {
    // Create a DensityMatrixCategory
    let category = DensityMatrixCategory;

    // Create identity morphism
    let id_1 = category.identity(&1);

    // Check that it's a 2x2 identity matrix
    assert!(complex_approx_eq(id_1[[0, 0]], Complex64::new(1.0, 0.0), 1e-10));
    assert!(complex_approx_eq(id_1[[1, 1]], Complex64::new(1.0, 0.0), 1e-10));

    // Create X and H matrices
    let x_gate = StandardGate::X;
    let h_gate = StandardGate::H;
    let x_matrix = x_gate.matrix();
    let h_matrix = h_gate.matrix();

    // Compose them
    let h_x = category.compose(&x_matrix, &h_matrix).unwrap();

    // Check matrix multiplication correctness
    let expected = h_matrix.dot(&x_matrix);
    for i in 0..2 {
        for j in 0..2 {
            assert!(complex_approx_eq(h_x[[i, j]], expected[[i, j]], 1e-10));
        }
    }

    // Test tensor product of morphisms
    let tensor = category.tensor_morphisms(&x_matrix, &h_matrix);

    // Check dimensions
    assert_eq!(tensor.shape(), [4, 4]);

    // Test tensor product of objects
    let tensor_obj = category.tensor_objects(&1, &1);
    assert_eq!(tensor_obj, 2);
}

#[test]
fn test_state_vector_partial_trace() {
    // Create a 2-qubit product state |0⟩⊗|1⟩
    let state0 = StateVector::computational_basis(1, 0).unwrap();
    let state1 = StateVector::computational_basis(1, 1).unwrap();

    println!("state0 (|0⟩) amplitudes: {:?}", state0.amplitudes());
    println!("state1 (|1⟩) amplitudes: {:?}", state1.amplitudes());

    let product_state = state0.tensor(&state1);
    println!("product_state (|0⟩⊗|1⟩) amplitudes: {:?}", product_state.amplitudes());

    // Trace out first qubit
    let traced = product_state.partial_trace(&[0]).unwrap();

    println!("After tracing out first qubit:");
    println!("traced.qubit_count(): {}", traced.qubit_count());
    println!("traced amplitudes: {:?}", traced.amplitudes());

    // Should get |1⟩
    assert_eq!(traced.qubit_count(), 1);

    // Debug the first amplitude - this is where the test is failing
    println!("Expected amplitude[0]: {}", Complex64::new(0.0, 0.0));
    println!("Actual amplitude[0]: {}", traced.amplitudes()[0]);
    println!("Difference magnitude: {}", (traced.amplitudes()[0] - Complex64::new(0.0, 0.0)).norm());

    assert!(complex_approx_eq(traced.amplitudes()[0], Complex64::new(0.0, 0.0), 1e-10));
    assert!(complex_approx_eq(traced.amplitudes()[1], Complex64::new(1.0, 0.0), 1e-10));

    // Debug the second amplitude
    println!("Expected amplitude[1]: {}", Complex64::new(1.0, 0.0));
    println!("Actual amplitude[1]: {}", traced.amplitudes()[1]);
    println!("Difference magnitude: {}", (traced.amplitudes()[1] - Complex64::new(1.0, 0.0)).norm());

    // Let's examine the internal process of partial_trace
    // Convert to density matrix
    let dm = DensityMatrix::from_state_vector(&product_state);
    println!("\nDensity matrix of |0⟩⊗|1⟩:");
    for i in 0..4 {
        for j in 0..4 {
            println!("dm[{}, {}] = {}", i, j, dm.matrix()[[i, j]]);
        }
    }

    // Partial trace the density matrix directly
    let traced_dm = dm.partial_trace(&[0]).unwrap();
    println!("\nAfter tracing out first qubit (density matrix):");
    for i in 0..2 {
        for j in 0..2 {
            println!("traced_dm[{}, {}] = {}", i, j, traced_dm.matrix()[[i, j]]);
        }
    }

    // Convert back to state vector
    if let Some(traced_sv) = traced_dm.to_state_vector() {
        println!("\nDensity matrix converted back to state vector:");
        println!("traced_sv amplitudes: {:?}", traced_sv.amplitudes());
    } else {
        println!("\nCouldn't convert traced density matrix back to state vector");
    }

    // Create a 2-qubit Bell state (entangled)
    let mut amplitudes = Array1::zeros(4);
    amplitudes[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    amplitudes[3] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let bell_state = StateVector::new(2, amplitudes).unwrap();
    println!("\nBell state amplitudes: {:?}", bell_state.amplitudes());

    // Let's look at the density matrix for Bell state
    let bell_dm = DensityMatrix::from_state_vector(&bell_state);
    println!("\nBell state density matrix:");
    for i in 0..4 {
        for j in 0..4 {
            if bell_dm.matrix()[[i, j]].norm() > 1e-10 {
                println!("bell_dm[{}, {}] = {}", i, j, bell_dm.matrix()[[i, j]]);
            }
        }
    }

    // Trace out first qubit of Bell state density matrix
    let traced_bell_dm = bell_dm.partial_trace(&[0]).unwrap();
    println!("\nAfter tracing out first qubit of Bell state:");
    for i in 0..2 {
        for j in 0..2 {
            println!("traced_bell_dm[{}, {}] = {}", i, j, traced_bell_dm.matrix()[[i, j]]);
        }
    }

    // Check purity to verify it's mixed
    println!("Purity of traced Bell state: {}", traced_bell_dm.purity());

    // Attempt to convert to state vector (should fail)
    let traced_bell_sv = bell_state.partial_trace(&[0]);
    println!("\nBell state partial trace result: {:?}", traced_bell_sv.is_some());

    // This should return None since it's a mixed state
    assert!(bell_state.partial_trace(&[0]).is_none());
}

#[test]
fn test_functor_operations() {
    // Create QuantumStateCategory and DensityMatrixCategory
    let qsc = QuantumStateCategory;
    let dmc = DensityMatrixCategory;

    // Create the functor
    let functor = PureToDensityFunctor;

    // Map a simple object (1 qubit)
    let mapped_obj = functor.map_object(&qsc, &dmc, &1);
    assert_eq!(mapped_obj, 1);

    // Create a unitary operation (X gate)
    let x_gate = StandardGate::X;
    let x_matrix = x_gate.matrix();

    // Map the morphism
    let mapped_morphism = functor.map_morphism(&qsc, &dmc, &x_matrix);

    // Check that it's the same matrix
    for i in 0..2 {
        for j in 0..2 {
            assert!(complex_approx_eq(mapped_morphism[[i, j]], x_matrix[[i, j]], 1e-10));
        }
    }
}
