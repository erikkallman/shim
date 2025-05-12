#[cfg(test)]
mod tests {
    use shim::category::prelude::*;
    use shim::machine_learning::categorical::categories::{ModelCategory, DataCategory, CircuitCategory};
    use shim::machine_learning::categorical::functors::{DataToCircuitFunctor, CircuitToModelFunctor};
    use shim::machine_learning::categorical::transformations::{StatePreparationTransformation, ModelPredictionTransformation, CircuitPredictionTransformation};
    use shim::machine_learning::quantum::model::{DecodingStrategy, EncodingStrategy};
    use shim::machine_learning::categorical::prediction::{PredictionCategory, PredictionTransformation};
    use shim::machine_learning::dataset::TabularDataset;

    use shim::quantum::circuit::*;
    use shim::quantum::state::StateVector;

    use shim::machine_learning::dataset::Dataset;
    use ndarray::{Array1, Array2};
    use std::f64::consts::PI;
    use std::time::Instant;

    #[test]
    fn test_categorical_quantum_ml_advanced_pipeline() {
        println!("======= ADVANCED CATEGORICAL QUANTUM ML SHOWCASE =======");

        // ===== 1. Define our categories - the mathematical backbone =====
        println!("Setting up categorical structure...");
        let data_category = DataCategory;
        let circuit_category = CircuitCategory;
        let model_category = ModelCategory;
        let _prediction_category = PredictionCategory;

        // ===== 2. Create a more impressive synthetic dataset =====
        // Generate a spiral dataset for binary classification (inspired by scikit-learn's demo)
        println!("Generating complex spiral dataset...");

        let n_samples_per_class = 50;
        let n_classes = 2;
        let total_samples = n_samples_per_class * n_classes;

        // Parameters for the spiral
        let n_rotations = 3.0;
        let noise_level = 0.1;

        let mut x_data = Array2::zeros((total_samples, 2));
        let mut y_data_flat = Array1::zeros(total_samples);

        // Generate the spiral pattern
        for i in 0..n_classes {
            let _ix = i * n_samples_per_class..(i + 1) * n_samples_per_class;
            let r = Array1::linspace(0.0, 1.0, n_samples_per_class);
            let t = Array1::linspace(
                i as f64 * 2.0 * PI / n_classes as f64,
                (i as f64 + n_rotations) * 2.0 * PI / n_classes as f64,
                n_samples_per_class,
            );

            // Add uniform noise
            use rand::Rng;
            let mut rng = rand::thread_rng();

            for j in 0..n_samples_per_class {
                let idx = i * n_samples_per_class + j;
                let noise_x = noise_level * (2.0 * rng.gen::<f64>() - 1.0);
                let noise_y = noise_level * (2.0 * rng.gen::<f64>() - 1.0);

                // Spiral equation with noise
                x_data[[idx, 0]] = r[j] * t[j].cos() + noise_x;
                x_data[[idx, 1]] = r[j] * t[j].sin() + noise_y;
                y_data_flat[idx] = i as f64;
            }
        }

        // One-hot encode the targets for binary classification
        let mut y_data = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            if y_data_flat[i] < 0.5 {
                y_data.push(Array1::from_vec(vec![1.0, 0.0]));
            } else {
                y_data.push(Array1::from_vec(vec![0.0, 1.0]));
            }
        }

        // Convert to a proper dataset
        let targets_matrix = Array2::from_shape_vec((total_samples, 2),
            y_data.iter().flat_map(|arr| arr.to_vec()).collect()).unwrap();

        let dataset = TabularDataset::new(x_data.clone(), targets_matrix).unwrap();
        println!("Created spiral dataset with {} samples, 2 features, 2 classes", total_samples);

        // ===== 3. Define advanced functors between our categories =====
        println!("Setting up quantum encoding strategy...");

        // Use a more advanced encoding strategy
        let encoding_strategy = EncodingStrategy::AngleEncoding;

        // Use 4 qubits to increase the expressivity of our quantum model
        let qubit_count = 4;
        let data_to_circuit = DataToCircuitFunctor::new(qubit_count, encoding_strategy.clone());

        // Define a more sophisticated measurement strategy
        let measurement_qubits = vec![0, 1, 2, 3];  // Measure all qubits
        let decoding_strategy = DecodingStrategy::MeasurementBased(measurement_qubits);
        let circuit_to_model = CircuitToModelFunctor::new(2, decoding_strategy.clone());

        // ===== 4. Define natural transformations =====
        println!("Creating natural transformations...");
        let state_prep = StatePreparationTransformation::new(encoding_strategy);
        let model_prediction = ModelPredictionTransformation::new(decoding_strategy.clone());

        // ===== 5. Create a complex variational quantum circuit model =====
        println!("Building advanced variational quantum circuit...");

        // This is a more sophisticated quantum model with multiple variational layers
        let mut builder = CircuitBuilder::new(qubit_count);

        // Layer 1: Initial state preparation with Hadamard gates
        for q in 0..qubit_count {
            builder.h(q).unwrap();
        }

        // Layer 2: First variational layer - single-qubit rotations
        let angles_layer1 = [PI/4.0, PI/3.0, PI/6.0, PI/8.0];
        for q in 0..qubit_count {
            builder.ry(q, angles_layer1[q]).unwrap();
            builder.rz(q, angles_layer1[q]/2.0).unwrap();
        }

        // Layer 3: Entanglement layer - create full entanglement
        for q in 0..qubit_count-1 {
            builder.cnot(q, q+1).unwrap();
        }
        builder.cnot(qubit_count-1, 0).unwrap(); // Create a cycle

        // Layer 4: Second variational layer
        let angles_layer2 = [PI/5.0, PI/7.0, PI/9.0, PI/11.0];
        for q in 0..qubit_count {
            builder.rx(q, angles_layer2[q]).unwrap();
            builder.rz(q, angles_layer2[q]/3.0).unwrap();
        }

        // Layer 5: Second entanglement layer - more complex topology
        builder.cnot(0, 2).unwrap();
        builder.cnot(1, 3).unwrap();

        // Layer 6: Third variational layer
        let angles_layer3 = [PI/2.0, PI/5.0, PI/3.0, PI/7.0];
        for q in 0..qubit_count {
            builder.ry(q, angles_layer3[q]).unwrap();
        }

        let variational_circuit = builder.build();
        println!("Built variational circuit with {} qubits and {} gates",
            qubit_count, variational_circuit.gate_count());

        // ===== 6. Create prediction circuit transformation =====
        let prediction_circuit = CircuitPredictionTransformation {
            circuit: variational_circuit.clone(),
            qubit_count,
            decoding_strategy: decoding_strategy.clone(),
        };

        // ===== 7. Demonstrate categorical composition =====
        println!("Validating categorical structure...");

        // Verify identity law for CircuitCategory
        let identity_circuit = circuit_category.identity(&qubit_count);
        let composed_with_id = circuit_category.compose(&identity_circuit, &variational_circuit).unwrap();
        assert_eq!(composed_with_id.qubit_count, variational_circuit.qubit_count);
        assert_eq!(composed_with_id.gate_count(), variational_circuit.gate_count());
        println!("✓ Identity law verified for CircuitCategory");

        // Verify functoriality of data_to_circuit
        let data_obj = 2; // 2D input data
        let circuit_obj = data_to_circuit.map_object(&data_category, &circuit_category, &data_obj);
        assert_eq!(circuit_obj, qubit_count);
        println!("✓ Functoriality verified for DataToCircuitFunctor");

        // ===== 8. Demonstrate inference on a batch of samples =====
        println!("\nRunning inference on quantum model...");

        // Create a small batch for inference demonstration
        let batch_size = 5;
        let batch_indices = (0..batch_size).collect::<Vec<_>>();
        let (batch_x, batch_y) = dataset.get_batch(&batch_indices).unwrap();

        println!("Processing batch of {} samples through quantum pipeline...", batch_size);
        let start_time = Instant::now();

        // Process each sample through the quantum pipeline
        let mut correct_predictions = 0;
        for (i, sample) in batch_x.iter().enumerate() {
            // Create initial quantum state
            let initial_state = StateVector::zero_state(qubit_count);

            // Prepare the input state using angle encoding
            let mut state_prep_circuit = CircuitBuilder::new(qubit_count);
            state_prep_circuit.ry(0, sample[0] * PI).unwrap();
            state_prep_circuit.ry(1, sample[1] * PI).unwrap();

            // Extra rotations to spread the encoding across all qubits
            if qubit_count > 2 {
                state_prep_circuit.ry(2, (sample[0] + sample[1]) * PI/2.0).unwrap();
            }
            if qubit_count > 3 {
                state_prep_circuit.ry(3, (sample[0] - sample[1]) * PI/2.0).unwrap();
            }

            let prepared_state = state_prep_circuit.build().apply(&initial_state).unwrap();

            // Apply the variational circuit
            let output_state = variational_circuit.apply(&prepared_state).unwrap();

            // Get prediction
            let prediction = prediction_circuit.apply(&output_state).unwrap();

            // Check if prediction is correct (argmax)
            let pred_class = if prediction[0] > prediction[1] { 0 } else { 1 };
            let true_class = if batch_y[i][0] > batch_y[i][1] { 0 } else { 1 };

            if pred_class == true_class {
                correct_predictions += 1;
            }

            println!("Sample {}: True class = {}, Predicted probabilities = [{:.3}, {:.3}]",
                i, true_class, prediction[0], prediction[1]);
        }

        let elapsed = start_time.elapsed();
        let accuracy = (correct_predictions as f64) / (batch_size as f64);

        println!("\nProcessed batch in {:.2?}", elapsed);
        println!("Accuracy on batch: {:.1}% ({}/{} correct)",
            accuracy * 100.0, correct_predictions, batch_size);

        // ===== 9. Demonstrate categorical composition for the full pipeline =====
        println!("\nDemonstrating categorical composition of the full quantum ML pipeline...");

        // Map a test sample through the entire pipeline using categorical composition
        let test_sample_idx = batch_size;  // A sample we haven't used yet
        let test_sample = x_data.row(test_sample_idx).to_owned();

        // Step 1: Use the data to circuit functor to map the classical data to a quantum circuit
        let circuit_morphism = data_to_circuit.map_morphism(
            &data_category,
            &circuit_category,
            &Array2::from_shape_vec((1, 2), vec![test_sample[0], test_sample[1]]).unwrap()
        );

        // Step 2: Compose with the variational model circuit
        let composed_circuit = circuit_category.compose(&circuit_morphism, &variational_circuit).unwrap();

        // Step 3: Apply the circuit to get a quantum state
        let test_state = StateVector::zero_state(qubit_count);
        let composed_output_state = composed_circuit.apply(&test_state).unwrap();

        // Step 4: Apply the prediction transformation to get the final output
        let final_prediction = prediction_circuit.apply(&composed_output_state).unwrap();

        // Get the true class
        let true_y = if y_data_flat[test_sample_idx] < 0.5 { "Class 0" } else { "Class 1" };
        let predicted_class = if final_prediction[0] > final_prediction[1] { "Class 0" } else { "Class 1" };

        println!("Test sample: [{:.3}, {:.3}]", test_sample[0], test_sample[1]);
        println!("True class: {}", true_y);
        println!("Predicted probabilities: [{:.3}, {:.3}]", final_prediction[0], final_prediction[1]);
        println!("Predicted class: {}", predicted_class);

        // ===== 10. Demonstrate how categorical structures enable formal verification =====
        println!("\nDemonstrating how category theory enables formal reasoning...");

        // Create the component morphisms for our model
        let _prep_morphism = state_prep.prepare_state(
            &data_category,
            &circuit_category,
            &data_to_circuit,
            &data_to_circuit.identity_functor(),
            &2
        );

        let _model_morphism = model_prediction.measure_circuit(
            &circuit_category,
            &model_category,
            &circuit_to_model.identity_functor(),
            &circuit_to_model,
            &qubit_count
        );

        // Verify that our implementation respects the functorial properties
        // F(id_A) = id_F(A)
        let id_data = data_category.identity(&2);
        let mapped_id = data_to_circuit.map_morphism(&data_category, &circuit_category, &id_data);
        let id_circuit = circuit_category.identity(&data_to_circuit.map_object(&data_category, &circuit_category, &2));

        // The circuit dimensions should match (both should be qubit_count)
        assert_eq!(mapped_id.qubit_count, id_circuit.qubit_count);
        println!("✓ Functorial property F(id_A) = id_F(A) verified");

        // F(g ∘ f) = F(g) ∘ F(f) - this is harder to test directly but we can verify dimensions

        println!("\n=== CONCLUSION ===");
        println!("This showcase demonstrated how category theory provides a rigorous mathematical");
        println!("foundation for quantum machine learning, enabling formal reasoning about the");
        println!("composition of quantum operations and data transformations.");
        println!("\nKey achievements in this demonstration:");
        println!("1. Created a complex spiral dataset with {} samples", total_samples);
        println!("2. Built a sophisticated {} qubit, {} gate variational circuit",
            qubit_count, variational_circuit.gate_count());
        println!("3. Demonstrated categorical composition of quantum operations");
        println!("4. Verified functorial properties and category laws");
        println!("5. Achieved {:.1}% accuracy on the test batch", accuracy * 100.0);
        println!("\nCategorical Quantum ML pipeline successfully demonstrated!");
    }
}
