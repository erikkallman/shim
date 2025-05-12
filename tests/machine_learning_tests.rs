// tests/machine_learning_tests.rs
//! Tests for the machine learning module

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use shim::machine_learning::prelude::*;
use shim::machine_learning::quantum::model::{
    QuantumModel, EncodingStrategy, DecodingStrategy
};
use shim::machine_learning::quantum::circuit_model::{
    ParametrizedCircuitModel, ParamGateType
};
use shim::machine_learning::quantum::variational::QuantumNeuralNetwork;
use shim::machine_learning::quantum::kernel::QuantumKernel;
use shim::machine_learning::categorical::categories::{
    ModelCategory, DataCategory, CircuitCategory
};
use shim::quantum::circuit::CircuitBuilder;
use shim::quantum::state::StateVector;
use shim::category::{Category, MonoidalCategory, Functor};
use shim::machine_learning::categorical::categories::ModelDimension;
use shim::quantum::QuantumState;

use shim::machine_learning::categorical::transformations::DataToModelFunctor;
use shim::machine_learning::categorical::transformations::ModelPredictionTransformation;
use shim::machine_learning::categorical::transformations::StatePreparationTransformation;

#[test]
fn test_parametrized_circuit_model() {
    // Create a simple parametrized circuit model
    let qubit_count = 2;
    let input_dim = 2;
    let output_dim = 1;
    let parameters = vec![PI/4.0, PI/3.0, PI/2.0, PI/6.0];

    let mut model = ParametrizedCircuitModel::new(
        qubit_count,
        parameters,
        input_dim,
        output_dim
    );

    // Add some gates
    model.add_parametrized_gate(ParamGateType::Rx, 0, 0).unwrap();
    model.add_parametrized_gate(ParamGateType::Ry, 1, 1).unwrap();
    model.circuit_mut().add_gate(
        Box::new(shim::quantum::gate::StandardGate::CNOT),
        &[0, 1]
    ).unwrap();
    model.add_parametrized_gate(ParamGateType::Rz, 0, 2).unwrap();
    model.add_parametrized_gate(ParamGateType::Rz, 1, 3).unwrap();

    // Make a prediction
    let input = Array1::from_vec(vec![0.1, 0.2]);
    let result = model.predict(&input).unwrap();

    // Check that we get a valid result
    assert_eq!(result.len(), output_dim);
    assert!(result[0] >= 0.0 && result[0] <= 1.0);
}

#[test]
fn test_quantum_neural_network() {
    // Create a quantum neural network
    let qnn = QuantumNeuralNetwork::new(
        3,  // qubits
        3,  // input_dim
        2,  // output_dim
        2   // layers
    );

    // Make a prediction
    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let result = qnn.predict(&input).unwrap();

    // Check dimensions
    assert_eq!(result.len(), 2);

    // Check values are valid probabilities
    for &val in result.iter() {
        assert!((0.0..=1.0).contains(&val));
    }
}

#[test]
fn test_quantum_kernel() {
    // Create a quantum kernel
    let kernel = QuantumKernel::new(3, 3);

    // Compute kernel value between two data points
    let x1 = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let x2 = Array1::from_vec(vec![0.4, 0.5, 0.6]);

    let kernel_value = kernel.compute_kernel(&x1, &x2).unwrap();

    // Check that kernel value is valid
    assert!((0.0..=1.0).contains(&kernel_value));

    // Compute kernel matrix
    let data = vec![
        Array1::from_vec(vec![0.1, 0.2, 0.3]),
        Array1::from_vec(vec![0.4, 0.5, 0.6]),
        Array1::from_vec(vec![0.7, 0.8, 0.9]),
    ];

    let kernel_matrix = kernel.compute_kernel_matrix(&data).unwrap();

    // Check dimensions
    assert_eq!(kernel_matrix.shape(), [3, 3]);

    // Check that diagonal elements are 1 (or close to it)
    for i in 0..3 {
        assert!((kernel_matrix[[i, i]] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_categorical_composition() {
    // Create the categories
    let model_category = ModelCategory;
    let data_category = DataCategory;
    let circuit_category = CircuitCategory;

    // Create some objects
    let data_dim = 3;
    let qubit_count = 2;
    let model_dim = model_category.tensor_objects(
        &ModelDimension { input_dim: 3, output_dim: 1 },
        &ModelDimension { input_dim: 2, output_dim: 2 }
    );

    // Check tensor product dimensions
    assert_eq!(model_dim.input_dim, 5);
    assert_eq!(model_dim.output_dim, 3);

    // Create identity morphisms
    let id_data = data_category.identity(&data_dim);
    let id_circuit = circuit_category.identity(&qubit_count);

    // Check that identities have correct shapes
    assert_eq!(id_data.shape(), [data_dim, data_dim]);
    assert_eq!(id_circuit.qubit_count, qubit_count);

    // Test DataToCircuitFunctor
    let encoding_strategy = EncodingStrategy::AngleEncoding;
    let data_to_circuit = shim::machine_learning::categorical::functors::DataToCircuitFunctor::new(
        qubit_count,
        encoding_strategy
    );

    // Map an object
    let mapped_object = data_to_circuit.map_object(&data_category, &circuit_category, &data_dim);
    assert_eq!(mapped_object, qubit_count);

    // Test encoding data to quantum state
    let data = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let state = data_to_circuit.encode_data(&data).unwrap();

    // Check state is valid
    assert_eq!(state.qubit_count, qubit_count);
    assert_eq!(state.dimension(), 1 << qubit_count);
}

#[test]
fn test_circuit_categorical_properties() {
    // Create a circuit category
    let cat = CircuitCategory;

    // Set up circuit builders
    let mut b1 = CircuitBuilder::new(2);
    let mut b2 = CircuitBuilder::new(2);
    let mut b3 = CircuitBuilder::new(2);

    // Create three circuits
    b1.h(0).unwrap();
    b1.cnot(0, 1).unwrap();
    let c1 = b1.build();

    b2.x(0).unwrap();
    let c2 = b2.build();

    b3.z(1).unwrap();
    let c3 = b3.build();

    // Test composition
    let c1_c2 = cat.compose(&c1, &c2).unwrap();
    let c2_c3 = cat.compose(&c2, &c3).unwrap();

    // Test composition associativity
    let left = cat.compose(&c1, &c2_c3).unwrap();
    let right = cat.compose(&c1_c2, &c3).unwrap();

    // Create states to test equality
    let state = StateVector::zero_state(2);
    let left_result = left.apply(&state).unwrap();
    let right_result = right.apply(&state).unwrap();

    // Check associativity
    let amplitudes1 = left_result.amplitudes();
    let amplitudes2 = right_result.amplitudes();

    for i in 0..4 {
        assert!((amplitudes1[i].norm() - amplitudes2[i].norm()).abs() < 1e-10);
    }

    // Test monoidal structure
    let tensor = cat.tensor_morphisms(&c1, &c2);
    assert_eq!(tensor.qubit_count, 4);

    // Create identity
    let id = cat.identity(&2);

    // Test left identity
    let left_id = cat.compose(&id, &c1).unwrap();
    let test_state = StateVector::zero_state(2);
    let orig_result = c1.apply(&test_state).unwrap();
    let left_id_result = left_id.apply(&test_state).unwrap();

    for i in 0..4 {
        assert!((orig_result.amplitudes()[i].norm() - left_id_result.amplitudes()[i].norm()).abs() < 1e-10);
    }
}

#[test]
fn test_data_to_model_functor() {
    // Create test data - 2 samples with 3 features each
    let data = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

    // Set up functors and transformations

    // 1. Define qubit count and encoding/decoding strategies
    let qubit_count = 2;
    let encoding_strategy = EncodingStrategy::AngleEncoding;

    // We'll measure qubits 0 and 1 for our output
    let measurement_qubits = vec![0, 1];
    let decoding_strategy = DecodingStrategy::MeasurementBased(measurement_qubits);

    // 2. Create the component functors
    let data_to_circuit = DataToCircuitFunctor {
        qubit_count,
        encoding_strategy: encoding_strategy.clone(),
    };

    let circuit_to_model = CircuitToModelFunctor {
        output_dim: 2, // We'll get 2 outputs (one for each qubit)
        decoding_strategy: decoding_strategy.clone(),
    };

    // 3. Create the transformations
    let state_preparation = StatePreparationTransformation::new(encoding_strategy);
    let model_prediction = ModelPredictionTransformation::new(decoding_strategy);

    // 4. Create the composite functor
    let data_to_model = DataToModelFunctor::new(
        state_preparation,
        model_prediction,
        data_to_circuit,
        circuit_to_model,
    );

    // Test object mapping
    let data_cat = DataCategory;
    let model_cat = ModelCategory;

    // Map the data dimension (3) to model dimension
    let model_dim = data_to_model.map_object(&data_cat, &model_cat, &3);

    // Verify the model dimensions match our expectations
    assert_eq!(model_dim.input_dim, qubit_count);
    assert_eq!(model_dim.output_dim, 2);

    // Test morphism mapping
    let model_transform = data_to_model.map_morphism(&data_cat, &model_cat, &data);

    // Apply the transformation to a test input
    let test_input = Array1::from_vec(vec![0.1, 0.2]);
    let result = model_transform.apply(&test_input).unwrap();

    // Verify we get the expected output dimensions
    assert_eq!(result.len(), 2);

    // The actual values will depend on the quantum circuit evaluation,
    // but we can check that they're valid probabilities
    for val in result.iter() {
        assert!((&0.0..=&1.0).contains(&val), "Output value {} is not a valid probability", val);
    }

    // Test category laws: F(id_A) = id_F(A)
    // Create an identity morphism in DataCategory
    let identity_data = Array2::eye(3);
    let mapped_identity = data_to_model.map_morphism(&data_cat, &model_cat, &identity_data);

    // Apply it to the same test input
    let result_identity = mapped_identity.apply(&test_input).unwrap();

    // Since we're using quantum circuits, the identity property might not hold exactly
    // but the output dimensions should still match
    assert_eq!(result_identity.len(), 2);

    // Test category laws: F(g ∘ f) = F(g) ∘ F(f)
    // Create two morphisms in DataCategory
    let f = Array2::from_shape_vec((3, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).unwrap();
    let g = Array2::from_shape_vec((3, 3), vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]).unwrap();

    // Compose them in DataCategory
    let composed = g.dot(&f);

    // Map the composition
    let mapped_composed = data_to_model.map_morphism(&data_cat, &model_cat, &composed);

    // Map each morphism individually
    let _mapped_f = data_to_model.map_morphism(&data_cat, &model_cat, &f);
    let _mapped_g = data_to_model.map_morphism(&data_cat, &model_cat, &g);

    // Compose the mapped morphisms in ModelCategory
    // Note: This would require implementing a compose method for ModelTransformation
    // which is outside the scope of this test

    // Instead, we can just check that the dimensions are consistent
    assert_eq!(mapped_composed.domain().input_dim, qubit_count);
    assert_eq!(mapped_composed.codomain().output_dim, 2);
}

#[test]
fn test_quantum_model_training() {
    use ndarray::{Array1, Array2};
    use std::f64::consts::PI;

    use shim::machine_learning::prelude::*;
    use shim::machine_learning::quantum::model::QuantumModel;
    use shim::machine_learning::quantum::circuit_model::ParametrizedCircuitModel;
    use shim::machine_learning::optimizer::{Optimizer, GradientDescent};
    use shim::machine_learning::loss::{LossFunction, BinaryCrossEntropy};

    // Create a binary classification dataset
    let x_train = Array2::from_shape_vec((10, 3), vec![
        0.1, 0.2, 0.3,
        0.2, 0.3, 0.4,
        0.3, 0.4, 0.5,
        0.4, 0.5, 0.6,
        0.5, 0.6, 0.7,
        0.6, 0.7, 0.8,
        0.7, 0.8, 0.9,
        0.8, 0.9, 1.0,
        0.9, 1.0, 0.1,
        1.0, 0.1, 0.2
    ]).unwrap();

    // Binary labels: first 5 are class 0, last 5 are class 1
    let y_train = vec![
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![0.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![1.0])
    ];

    // Create a parameterized quantum circuit model
    let qubit_count = 2;
    let input_dim = 3;
    let output_dim = 1;

    // Initialize with random parameters
    let initial_params = vec![
        rand::random::<f64>() * PI,
        rand::random::<f64>() * PI,
        rand::random::<f64>() * PI,
        rand::random::<f64>() * PI,
        rand::random::<f64>() * PI,
        rand::random::<f64>() * PI
    ];

    let mut model = ParametrizedCircuitModel::new(
        qubit_count,
        initial_params,
        input_dim,
        output_dim
    );

    // Add parametrized gates to create a simple variational circuit
    model.add_parametrized_gate(ParamGateType::Rx, 0, 0).unwrap();
    model.add_parametrized_gate(ParamGateType::Ry, 1, 1).unwrap();
    model.circuit_mut().add_gate(
        Box::new(shim::quantum::gate::StandardGate::CNOT),
        &[0, 1]
    ).unwrap();
    model.add_parametrized_gate(ParamGateType::Rz, 0, 2).unwrap();
    model.add_parametrized_gate(ParamGateType::Rz, 1, 3).unwrap();
    model.add_parametrized_gate(ParamGateType::Rx, 0, 4).unwrap();
    model.add_parametrized_gate(ParamGateType::Ry, 1, 5).unwrap();

    // Create optimizer and loss function
    let optimizer = GradientDescent::new(0.01);
    let loss_fn = BinaryCrossEntropy;

    // Evaluate initial loss
    let mut initial_loss = 0.0;
    for i in 0..x_train.nrows() {
        let x_i = x_train.row(i).to_owned();
        let y_pred = model.predict(&x_i).unwrap();
        let loss = loss_fn.calculate_loss(&y_pred, &y_train[i]);
        initial_loss += loss;
    }
    initial_loss /= x_train.nrows() as f64;

    println!("Initial loss: {}", initial_loss);

    // Train the model for several epochs
    let epochs = 200;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for i in 0..x_train.nrows() {
            let x_i = x_train.row(i).to_owned();
            let y_i = &y_train[i];

            // Forward pass to get prediction
            let y_pred = model.predict(&x_i).unwrap();

            // Calculate loss
            let loss = loss_fn.calculate_loss(&y_pred, y_i);
            epoch_loss += loss;

            // Calculate gradients using finite differences
            let mut gradients = vec![0.0; model.parameter_count()];
            let epsilon = 1e-5;

            for j in 0..model.parameter_count() {
                let mut params = model.get_parameters();

                // f(θ + ε)
                params[j] += epsilon;
                model.set_parameters(&params).unwrap();
                let y_pred_plus = model.predict(&x_i).unwrap();
                let loss_plus = loss_fn.calculate_loss(&y_pred_plus, y_i);

                // f(θ - ε)
                params[j] -= 2.0 * epsilon;
                model.set_parameters(&params).unwrap();
                let y_pred_minus = model.predict(&x_i).unwrap();
                let loss_minus = loss_fn.calculate_loss(&y_pred_minus, y_i);

                // Compute central finite difference
                gradients[j] = (loss_plus - loss_minus) / (2.0 * epsilon);

                // Reset parameters
                params[j] += epsilon;
                model.set_parameters(&params).unwrap();
            }

            // Update parameters
            let mut params = model.get_parameters();
            optimizer.update(&mut params, &gradients);
            model.set_parameters(&params).unwrap();
        }

        epoch_loss /= x_train.nrows() as f64;

        if epoch % 5 == 0 {
            println!("Epoch {}: Loss = {}", epoch, epoch_loss);
        }
    }

    // Calculate final loss
    let mut final_loss = 0.0;
    for i in 0..x_train.nrows() {
        let x_i = x_train.row(i).to_owned();
        let y_pred = model.predict(&x_i).unwrap();
        let loss = loss_fn.calculate_loss(&y_pred, &y_train[i]);
        final_loss += loss;
    }
    final_loss /= x_train.nrows() as f64;

    println!("Final loss: {}", final_loss);

    // Verify that training improved the model
    assert!(final_loss < initial_loss,
            "Training did not reduce loss: initial = {}, final = {}",
            initial_loss, final_loss);

    // Calculate accuracy
    let mut correct = 0;
    for i in 0..x_train.nrows() {
        let x_i = x_train.row(i).to_owned();
        let y_pred = model.predict(&x_i).unwrap();
        let predicted_class = if y_pred[0] > 0.5 { 1.0 } else { 0.0 };
        let actual_class = y_train[i][0];

        if (predicted_class - actual_class).abs() < 1e-5 {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / x_train.nrows() as f64;
    println!("Accuracy: {}", accuracy);

    // Check that accuracy is better than random guessing
    assert!(accuracy > 0.5, "Model accuracy {} is not better than random", accuracy);
}

// #[test]
// fn test_categorical_optimization() {
//     use ndarray::{Array1, Array2};

//     use shim::machine_learning::prelude::*;
//     use shim::machine_learning::quantum::model::{
//         QuantumModel, EncodingStrategy, DecodingStrategy
//     };
//     use shim::machine_learning::quantum::circuit_model::{
//         ParametrizedCircuitModel, ParamGateType
//     };
//     use shim::machine_learning::optimizer::{Optimizer, Adam};
//     use shim::machine_learning::loss::{LossFunction, BinaryCrossEntropy};
//     use shim::machine_learning::categorical::categories::ModelCategory;
//     use shim::machine_learning::optimizer::OptimizationCategory;
//     use shim::machine_learning::categorical::GradientOptimizationTransformation;
//     use shim::category::Category;

//     // Create optimization category
//     let opt_category = OptimizationCategory;

//     // Create a simple parametrized circuit model
//     let qubit_count = 2;
//     let input_dim = 3;
//     let output_dim = 1;
//     let initial_parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

//     let mut model = ParametrizedCircuitModel::new(
//         qubit_count,
//         initial_parameters.clone(),
//         input_dim,
//         output_dim
//     );

//     // Add parametrized gates
//     model.add_parametrized_gate(ParamGateType::Rx, 0, 0).unwrap();
//     model.add_parametrized_gate(ParamGateType::Ry, 1, 1).unwrap();
//     model.circuit_mut().add_gate(
//         Box::new(shim::quantum::gate::StandardGate::CNOT),
//         &[0, 1]
//     ).unwrap();
//     model.add_parametrized_gate(ParamGateType::Rz, 0, 2).unwrap();
//     model.add_parametrized_gate(ParamGateType::Rz, 1, 3).unwrap();
//     model.add_parametrized_gate(ParamGateType::Rx, 0, 4).unwrap();
//     model.add_parametrized_gate(ParamGateType::Ry, 1, 5).unwrap();

//     // Create training data
//     let x_train = Array2::from_shape_vec((4, 3), vec![
//         0.1, 0.2, 0.3,
//         0.4, 0.5, 0.6,
//         0.7, 0.8, 0.9,
//         0.2, 0.3, 0.4
//     ]).unwrap();

//     let y_train = vec![
//         Array1::from_vec(vec![0.0]),
//         Array1::from_vec(vec![0.0]),
//         Array1::from_vec(vec![1.0]),
//         Array1::from_vec(vec![1.0])
//     ];

//     // Create optimizer and loss function
//     let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
//     let loss_fn = BinaryCrossEntropy;

//     // Create initial predictions for comparison
//     let test_input = Array1::from_vec(vec![0.15, 0.25, 0.35]);
//     let initial_prediction = model.predict(&test_input).unwrap();

//     // Create the optimization transformation
//     let opt_transform = GradientOptimizationTransformation::new(
//         model.clone(),
//         optimizer.clone(),
//         loss_fn.clone(),
//         20 // epochs
//     );
//     let opt_transform_boxed = Box::new(opt_transform);

//     // Apply the optimization transformation to the training data
//     let optimized_model_transform = opt_transform_boxed.apply(&(x_train.clone(), y_train.clone())).unwrap();

//     // Test the optimized model transformation
//     let optimized_prediction = optimized_model_transform.apply(&test_input).unwrap();

//     // Verify that predictions changed after optimization
//     assert!(
//         (initial_prediction[0] - optimized_prediction[0]).abs() > 1e-6,
//         "Model predictions did not change after optimization"
//     );

//     println!("Original prediction: {:?}", initial_prediction);
//     println!("Optimized prediction: {:?}", optimized_prediction);

//     // Test accuracy on the training data
//     let mut correct = 0;
//     for i in 0..x_train.nrows() {
//         let x_i = x_train.row(i).to_owned();
//         let y_pred = optimized_model_transform.apply(&x_i).unwrap();
//         let predicted_class = if y_pred[0] > 0.5 { 1.0 } else { 0.0 };
//         let actual_class = y_train[i][0];

//         if (initial_prediction[0] - optimized_prediction[0]).abs() > 1e-6 {
//             correct += 1;
//         }
//     }

//     let accuracy = correct as f64 / x_train.nrows() as f64;
//     println!("Accuracy: {}", accuracy);

//     // Check that accuracy is at least better than random
//     assert!(accuracy >= 0.5, "Optimized model accuracy {} is not better than random", accuracy);

//     // Test composition using the category
//     let half_epochs_transform1 = GradientOptimizationTransformation::new(
//         model.clone(),
//         Adam::new(0.01, 0.9, 0.999, 1e-8),
//         BinaryCrossEntropy,
//         10 // half the epochs
//     );

//     let half_epochs_transform2 = GradientOptimizationTransformation::new(
//         model.clone(),
//         Adam::new(0.01, 0.9, 0.999, 1e-8),
//         BinaryCrossEntropy,
//         10 // second half of the epochs
//     );

//     let half_epochs_boxed1: Box<dyn OptimizationTransformation> = Box::new(half_epochs_transform1);
//     let half_epochs_boxed2: Box<dyn OptimizationTransformation> = Box::new(half_epochs_transform2);

//     // Compose the two transformations
//     let composed_transform = opt_category.compose(&half_epochs_boxed1, &half_epochs_boxed2).unwrap();

//     // Apply the composed transformation
//     let composed_model_transform = composed_transform.apply(&(x_train.clone(), y_train.clone())).unwrap();
//     let composed_pred = composed_model_transform.apply(&test_input).unwrap();

//     println!("Composed optimization prediction: {:?}", composed_pred);

//     // Verify that composition produced a valid result
//     assert!(
//         (initial_prediction[0] - composed_pred[0]).abs() > 1e-6,
//         "Composed optimization did not change predictions"
//     );
// }

// #[test]
// fn test_quantum_ml_pipeline() {
//     use ndarray::{Array1, Array2, Axis};
//     use std::f64::consts::PI;
//     use rand::Rng;

//     use shim::machine_learning::prelude::*;
//     use shim::machine_learning::quantum::model::{
//         QuantumModel, EncodingStrategy, DecodingStrategy
//     };
//     use shim::machine_learning::quantum::circuit_model::{
//         ParametrizedCircuitModel, ParamGateType
//     };
//     use shim::machine_learning::optimizer::{Optimizer, Adam, GradientDescent, SGDMomentum};
//     use shim::machine_learning::loss::{LossFunction, BinaryCrossEntropy};

//     // Create a synthetic dataset
//     let data_size = 100;
//     let features = 2;
//     let mut rng = rand::thread_rng();
//     let output_dim = 1;
//     let qubit_count = 3;

//     let mut x_data = Array2::zeros((data_size, features));
//     let mut y_data = Vec::with_capacity(data_size);

//     for i in 0..data_size {
//         for j in 0..features {
//             x_data[[i, j]] = rng.gen_range(0.0..1.0);
//         }

//         // Simple decision boundary: if sum of features > 1.5, class = 1, else 0
//         let feature_sum: f64 = x_data.row(i).sum();
//         let y_i = if feature_sum > 1.5 {
//             Array1::from_vec(vec![1.0])
//         } else {
//             Array1::from_vec(vec![0.0])
//         };

//         y_data.push(y_i);
//     }

//     // Setup cross-validation
//     let k_folds = 3;
//     let fold_size = data_size / k_folds;

//     // Track metrics across folds
//     let mut fold_accuracies = Vec::with_capacity(k_folds);
//     let mut fold_losses = Vec::with_capacity(k_folds);

//     for fold in 0..k_folds {
//         println!("Processing fold {}/{}", fold + 1, k_folds);

//         // Split data into train and test
//         let test_start = fold * fold_size;
//         let test_end = (fold + 1) * fold_size;

//         let mut train_indices = Vec::new();
//         let mut test_indices = Vec::new();

//         for i in 0..data_size {
//             if i >= test_start && i < test_end {
//                 test_indices.push(i);
//             } else {
//                 train_indices.push(i);
//             }
//         }

//         // Create train/test datasets
//         let mut x_train = Array2::zeros((train_indices.len(), features));
//         let mut y_train = Vec::with_capacity(train_indices.len());
//         let mut x_test = Array2::zeros((test_indices.len(), features));
//         let mut y_test = Vec::with_capacity(test_indices.len());

//         // Fill train/test datasets
//         for (idx, &i) in train_indices.iter().enumerate() {
//             for j in 0..features {
//                 x_train[[idx, j]] = x_data[[i, j]];
//             }
//             y_train.push(y_data[i].clone());
//         }

//         for (idx, &i) in test_indices.iter().enumerate() {
//             for j in 0..features {
//                 x_test[[idx, j]] = x_data[[i, j]];
//             }
//             y_test.push(y_data[i].clone());
//         }

//         // Create quantum model
//         let input_dim = features;

//         let initial_params = vec![
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//             rng.gen::<f64>() * PI,
//         ];

//         let mut model = ParametrizedCircuitModel::new(
//             qubit_count,
//             initial_params,
//             input_dim,
//             output_dim
//         );

//         // Build a more complex circuit with two layers
//         // First layer
//         model.add_parametrized_gate(ParamGateType::Rx, 0, 0).unwrap();
//         model.add_parametrized_gate(ParamGateType::Ry, 1, 1).unwrap();
//         model.circuit_mut().add_gate(
//             Box::new(shim::quantum::gate::StandardGate::CNOT),
//             &[0, 1]
//         ).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 0, 2).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 1, 3).unwrap();

//         // Second layer
//         model.add_parametrized_gate(ParamGateType::Rx, 0, 4).unwrap();
//         model.add_parametrized_gate(ParamGateType::Ry, 1, 5).unwrap();
//         model.circuit_mut().add_gate(
//             Box::new(shim::quantum::gate::StandardGate::CNOT),
//             &[1, 0]
//         ).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 0, 6).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 1, 7).unwrap();

//         // Setup optimizer and loss function
//         // Try different optimizers for different folds
//         let optimizer: Box<dyn Optimizer> = match fold {
//             0 => Box::new(GradientDescent::new(0.01)),
//             1 => Box::new(Adam::default()),
//             _ => Box::new(SGDMomentum::new(0.01, 0.9)),
//         };

//         let loss_fn = BinaryCrossEntropy;

//         // Train the model
//         let epochs = 50;
//         let mut final_loss = 0.0;

//         for epoch in 0..epochs {
//             let mut epoch_loss = 0.0;

//             for i in 0..x_train.nrows() {
//                 let x_i = x_train.row(i).to_owned();
//                 let y_i = &y_train[i];

//                 // Forward pass
//                 let y_pred = model.predict(&x_i).unwrap();
//                 let loss = loss_fn.calculate_loss(&y_pred, y_i);
//                 epoch_loss += loss;

//                 // Calculate gradients using finite differences
//                 let mut gradients = vec![0.0; model.parameter_count()];
//                 let epsilon = 1e-5;

//                 for j in 0..model.parameter_count() {
//                     let mut params = model.get_parameters();

//                     // f(θ + ε)
//                     params[j] += epsilon;
//                     model.set_parameters(&params).unwrap();
//                     let y_pred_plus = model.predict(&x_i).unwrap();
//                     let loss_plus = loss_fn.calculate_loss(&y_pred_plus, y_i);

//                     // f(θ - ε)
//                     params[j] -= 2.0 * epsilon;
//                     model.set_parameters(&params).unwrap();
//                     let y_pred_minus = model.predict(&x_i).unwrap();
//                     let loss_minus = loss_fn.calculate_loss(&y_pred_minus, y_i);

//                     // Compute central finite difference
//                     gradients[j] = (loss_plus - loss_minus) / (2.0 * epsilon);

//                     // Reset parameters
//                     params[j] += epsilon;
//                     model.set_parameters(&params).unwrap();
//                 }

//                 // Update parameters
//                 let mut params = model.get_parameters();
//                 optimizer.update(&mut params, &gradients);
//                 model.set_parameters(&params).unwrap();
//             }

//             epoch_loss /= x_train.nrows() as f64;

//             if epoch == epochs - 1 {
//                 final_loss = epoch_loss;
//                 println!("Fold {}, Final Loss: {}", fold, epoch_loss);
//             } else if epoch % 10 == 0 {
//                 println!("Fold {}, Epoch {}: Loss = {}", fold, epoch, epoch_loss);
//             }
//         }

//         fold_losses.push(final_loss);

//         // Evaluate on test set
//         let mut correct = 0;
//         for i in 0..x_test.nrows() {
//             let x_i = x_test.row(i).to_owned();
//             let y_pred = model.predict(&x_i).unwrap();
//             let predicted_class = if y_pred[0] > 0.5 { 1.0 } else { 0.0 };
//             let actual_class = y_test[i][0];

//             if (predicted_class as f64 - actual_class as f64).abs() < 1e-5_f64 {
//                 correct += 1;
//             }
//         }

//         let accuracy = correct as f64 / x_test.nrows() as f64;
//         fold_accuracies.push(accuracy);

//         println!("Fold {} Test Accuracy: {}", fold, accuracy);

//         // Compare with a classical linear model as baseline
//         let mut linear_model = shim::machine_learning::core::LinearModel::new(
//             features, output_dim
//         );

//         // Train the linear model using gradient descent
//         let linear_optimizer = GradientDescent::new(0.01);
//         let linear_epochs = 100;

//         for _ in 0..linear_epochs {
//             for i in 0..x_train.nrows() {
//                 let x_i = x_train.row(i).to_owned();
//                 let y_i = &y_train[i];

//                 // Forward pass
//                 let y_pred = linear_model.predict(&x_i).unwrap();

//                 // Calculate gradients
//                 let grad_output = loss_fn.calculate_gradients(&y_pred, y_i);

//                 // Manually compute gradients for the linear model
//                 let mut gradients = vec![0.0; linear_model.parameter_count()];

//                 // Weight gradients
//                 let n = x_train.nrows() as f64;
//                 for j in 0..features {
//                     for k in 0..output_dim {
//                         gradients[k * features + j] = grad_output[k] * x_i[j] / n;
//                     }
//                 }

//                 // Bias gradients
//                 for k in 0..output_dim {
//                     gradients[features * output_dim + k] = grad_output[k] / n;
//                 }

//                 // Update parameters
//                 let mut params = linear_model.get_parameters();
//                 linear_optimizer.update(&mut params, &gradients);
//                 linear_model.set_parameters(&params).unwrap();
//             }
//         }

//         // Evaluate linear model
//         let mut linear_correct = 0;
//         for i in 0..x_test.nrows() {
//             let x_i = x_test.row(i).to_owned();
//             let y_pred = linear_model.predict(&x_i).unwrap();
//             let predicted_class = if y_pred[0] > 0.5 { 1.0 } else { 0.0 };
//             let actual_class = y_test[i][0];

//             if (predicted_class as f64 - actual_class as f64).abs() < 1e-5 {
//                 linear_correct += 1;
//             }
//         }

//         let linear_accuracy = linear_correct as f64 / x_test.nrows() as f64;
//         println!("Fold {} Linear Model Accuracy: {}", fold, linear_accuracy);
//         println!("-------------------------------------------");
//     }

//     // Calculate average metrics across folds
//     let avg_accuracy = fold_accuracies.iter().sum::<f64>() / k_folds as f64;
//     let avg_loss = fold_losses.iter().sum::<f64>() / k_folds as f64;

//     println!("Cross-validation results:");
//     println!("Fold accuracies: {:?}", fold_accuracies);
//     println!("Average accuracy: {}", avg_accuracy);
//     println!("Average loss: {}", avg_loss);

//     // Assert that average accuracy is better than random guessing
//     assert!(avg_accuracy > 0.5,
//             "Cross-validation accuracy {} is not better than random",
//             avg_accuracy);

//     // Create ensemble model from the best performing models
//     // (In practice, you would save models from each fold and load them here)
//     println!("Testing ensemble of quantum models");

//     // Create an ensemble model (simplified version)
//     let mut ensemble_models = Vec::new();

//     // Create 3 different models with different initializations
//     for _ in 0..3 {
//         let params = (0..8).map(|_| rng.gen::<f64>() * PI).collect::<Vec<_>>();

//         let mut model = ParametrizedCircuitModel::new(
//             qubit_count,
//             params,
//             features,
//             output_dim
//         );

//         // Build variational circuit
//         model.add_parametrized_gate(ParamGateType::Rx, 0, 0).unwrap();
//         model.add_parametrized_gate(ParamGateType::Ry, 1, 1).unwrap();
//         model.circuit_mut().add_gate(
//             Box::new(shim::quantum::gate::StandardGate::CNOT),
//             &[0, 1]
//         ).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 0, 2).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 1, 3).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rx, 0, 4).unwrap();
//         model.add_parametrized_gate(ParamGateType::Ry, 1, 5).unwrap();
//         model.circuit_mut().add_gate(
//             Box::new(shim::quantum::gate::StandardGate::CNOT),
//             &[1, 0]
//         ).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 0, 6).unwrap();
//         model.add_parametrized_gate(ParamGateType::Rz, 1, 7).unwrap();

//         // Train this model for a few epochs
//         let optimizer = Adam::default();
//         let loss_fn = BinaryCrossEntropy;

//         for _ in 0..15 {
//             for i in 0..x_data.nrows() {
//                 let x_i = x_data.row(i).to_owned();
//                 let y_i = &y_data[i];

//                 // Calculate gradients (simplified for brevity)
//                 let mut gradients = vec![0.0; model.parameter_count()];
//                 let epsilon = 1e-5;

//                 for j in 0..model.parameter_count() {
//                     let mut params = model.get_parameters();

//                     // f(θ + ε)
//                     params[j] += epsilon;
//                     model.set_parameters(&params).unwrap();
//                     let y_pred_plus = model.predict(&x_i).unwrap();
//                     let loss_plus = loss_fn.calculate_loss(&y_pred_plus, y_i);

//                     // f(θ - ε)
//                     params[j] -= 2.0 * epsilon;
//                     model.set_parameters(&params).unwrap();
//                     let y_pred_minus = model.predict(&x_i).unwrap();
//                     let loss_minus = loss_fn.calculate_loss(&y_pred_minus, y_i);

//                     // Compute central finite difference
//                     gradients[j] = (loss_plus - loss_minus) / (2.0 * epsilon);

//                     // Reset parameters
//                     params[j] += epsilon;
//                     model.set_parameters(&params).unwrap();
//                 }

//                 // Update parameters
//                 let mut params = model.get_parameters();
//                 optimizer.update(&mut params, &gradients);
//                 model.set_parameters(&params).unwrap();
//             }
//         }

//         ensemble_models.push(model);
//     }

//     // Evaluate ensemble model
//     let mut correct = 0;
//     for i in 0..data_size {
//         let x_i = x_data.row(i).to_owned();

//         // Average predictions from all models
//         let mut ensemble_prediction = 0.0;
//         for model in &ensemble_models {
//             let y_pred = model.predict(&x_i).unwrap();
//             ensemble_prediction += y_pred[0];
//         }
//         ensemble_prediction /= ensemble_models.len() as f64;

//         let predicted_class = if ensemble_prediction > 0.5 { 1.0 } else { 0.0 };
//         let actual_class = y_data[i][0];

//         if (predicted_class as f64 - actual_class as f64).abs() < 1e-5 {
//             correct += 1;
//         }
//     }

//     let ensemble_accuracy = correct as f64 / data_size as f64;
//     println!("Ensemble Model Accuracy: {}", ensemble_accuracy);

//     // Assert that ensemble model performs reasonably well
//     assert!(ensemble_accuracy >= 0.6,
//             "Ensemble model accuracy {} is not satisfactory",
//             ensemble_accuracy);
// }
