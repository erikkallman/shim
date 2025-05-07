use shim::category::bicategory::{
    QuantumCircuitBicategory, CircuitTransformation,
    HigherCategoricalQuantumNN, QuantumNeuralLayer, HigherCategoricalQNN,
    Bicategory, Adjunction
};
use shim::machine_learning::loss::{LossFunction, MeanSquaredError};
use shim::quantum::circuit::{QuantumCircuit, CircuitBuilder};
use shim::quantum::state::StateVector;
use shim::simulators::statevector::{StatevectorSimulator, Outcome};
use ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("========================================================");
    println!("       Higher Categorical Quantum ML Demonstration       ");
    println!("========================================================\n");

    println!("This example explores how higher category theory — particularly the use");
    println!("of bicategories, adjunctions, and 2-morphisms — can be applied to build,");
    println!("train, and reason about quantum neural networks (QNNs) in a principled way.");

    println!("\nWhy higher category theory?");
    println!("Traditional quantum machine learning models often rely on heuristics or ad hoc");
    println!("optimization procedures. By using categorical structures, we gain a formal");
    println!("language to express the architecture, learning rules, and transformations of");
    println!("quantum models. This makes the logic of training and compositionality explicit,");
    println!("rigorous, and extensible.\n");

    println!("Throughout this example, we will:");
    println!("  • Build a QNN using layers defined as adjunctions (1-morphisms with structure)");
    println!("    => This ensures each layer has a principled forward and backward process,");
    println!("       satisfying triangle identities that formalize information flow.");

    println!("  • Represent parameter updates and gradients as 2-morphisms");
    println!("    => This allows us to model learning not just as numeric tuning,");
    println!("       but as morphisms between morphisms — i.e., structured updates");

    println!("  • Perform backpropagation via vertical composition of 2-morphisms");
    println!("    => This mirrors how gradients flow back through layers during training,");
    println!("       with clear composition laws ensuring consistency across updates.");

    println!("  • Perform parallel updates via horizontal composition");
    println!("    => This represents simultaneous (independent) updates across different layers,");
    println!("       modeling concurrent learning steps or disentangled subsystems.");

    println!("  • Apply 'whiskering' to propagate gradients through specific layers");
    println!("    => Whiskering captures the notion of how a change in one part of the");
    println!("       network affects the whole, enabling modular update strategies.\n");

    println!("The key insight is that QML architectures and learning processes naturally");
    println!("form a bicategory: qubit systems (objects), quantum circuits (1-morphisms),");
    println!("and gradient transformations (2-morphisms) all interact in a coherent way.");

    println!("By leveraging this structure, we can reason about QML models not just");
    println!("numerically, but algebraically — enabling better abstraction, verification,");
    println!("and reusability of learning components.\n");

    println!("Let's begin by creating our QNN and exploring its categorical structure.");
    println!("========================================================\n");

    // Create a bicategory for quantum circuits
    let bicategory = QuantumCircuitBicategory;

    // Create a quantum neural network with 2 qubits
    let qubit_count = 2;
    let mut qnn = HigherCategoricalQuantumNN::new(qubit_count);

    // Build variational layers as adjunctions
    let layer1 = build_variational_layer(qubit_count, "layer1", 0.1);
    qnn.add_layer(layer1);

    let layer2 = build_variational_layer(qubit_count, "layer2", 0.2);
    qnn.add_layer(layer2);

    println!("Created a quantum neural network with {} layers and {} qubits",
             qnn.layers.len(), qnn.qubit_count);

    // Get the forward circuit for the entire network
    let forward_circuit = qnn.forward_circuit();
    println!("Forward circuit has {} gates", forward_circuit.gate_count());

    // Verify that our layers satisfy the adjunction properties
    println!("\n===== Verifying Adjunction Properties =====");
    let layer = &qnn.layers[0];
    let verification = layer.verify_triangle_identities(&bicategory);
    println!("First layer satisfies triangle identities: {}", verification);

    // Demonstrate how 2-morphisms represent parameter updates
    println!("\n===== Demonstrating 2-morphisms as Parameter Updates =====");

    // Create a parameter update as a 2-morphism
    let parameter_update = create_parameter_update(&forward_circuit, "layer1_rx0", 0.05);
    println!("Created parameter update 2-morphism:");
    println!("  - Description: {}", parameter_update.description);
    println!("  - Parameter changes: {:?}", parameter_update.parameter_changes);

    // Apply the update to get an updated circuit
    let updated_params = apply_parameter_update(&parameter_update, &mut qnn);
    println!("Updated parameters: {:?}", updated_params);

    // Demonstrate loss calculation and gradient 2-morphism
    println!("\n===== Demonstrating Loss Gradients as 2-morphisms =====");

    // Create a test dataset
    let inputs = Array1::from_vec(vec![0.5, 0.3]);
    let targets = Array1::from_vec(vec![1.0, 0.0]);

    // Get predictions from our network
    let predictions = simulate_circuit(&forward_circuit, &inputs, qubit_count);
    println!("Input: [{:.2}, {:.2}]", inputs[0], inputs[1]);
    println!("Target: [{:.2}, {:.2}]", targets[0], targets[1]);
    println!("Prediction: [{:.2}, {:.2}]", predictions[0], predictions[1]);

    // Calculate loss
    let loss_fn = MeanSquaredError;
    let loss_value = loss_fn.calculate_loss(&predictions, &targets);
    println!("MSE Loss: {:.4}", loss_value);

    // Create a 2-morphism representing the loss gradient
    let loss_gradient = create_loss_gradient_2morphism(
        &forward_circuit, &loss_fn, &predictions, &targets);

    println!("Loss gradient 2-morphism:");
    println!("  - Description: {}", loss_gradient.description);
    println!("  - Parameter gradients: {:?}", loss_gradient.parameter_changes);

    // Demonstrate backpropagation through 2-morphism composition
    println!("\n===== Demonstrating Backpropagation Through Composition =====");

    // Create layer gradients using the same circuit as source and target
    // This ensures they can be composed
    let shared_circuit = forward_circuit.clone();

    // Create gradients that can be composed (target of layer1 = source of layer2)
    let layer1_gradient = CircuitTransformation {
        source: shared_circuit.clone(),
        target: shared_circuit.clone(),
        description: "Gradient for layer1".to_string(),
        parameter_changes: {
            let mut params = HashMap::new();
            params.insert("layer1_rx0".to_string(), -0.01);
            params.insert("layer1_ry0".to_string(), -0.02);
            if qubit_count > 1 {
                params.insert("layer1_rx1".to_string(), -0.005);
                params.insert("layer1_ry1".to_string(), -0.015);
            }
            params
        }
    };

    let layer2_gradient = CircuitTransformation {
        source: shared_circuit.clone(),
        target: shared_circuit.clone(),
        description: "Gradient for layer2".to_string(),
        parameter_changes: {
            let mut params = HashMap::new();
            params.insert("layer2_rx0".to_string(), -0.008);
            params.insert("layer2_ry0".to_string(), -0.012);
            if qubit_count > 1 {
                params.insert("layer2_rx1".to_string(), -0.004);
                params.insert("layer2_ry1".to_string(), -0.006);
            }
            params
        }
    };

    // Vertically compose the gradients (representing backpropagation)
    let composed_gradient = bicategory.vertical_compose(&layer2_gradient, &layer1_gradient).unwrap();

    println!("Composed gradient 2-morphism:");
    println!("  - Description: {}", composed_gradient.description);
    println!("  - Parameter changes: {:?}", composed_gradient.parameter_changes);

    // Demonstrate horizontal composition (parallel layers)
    println!("\n===== Demonstrating Horizontal Composition =====");

    // Create two parallel gradients
    let parallel_grad1 = create_parameter_update(&qnn.layers[0].forward, "parallel_rx0", 0.01);
    let parallel_grad2 = create_parameter_update(&qnn.layers[1].forward, "parallel_rx0", 0.02);

    // Horizontally compose them (happens in parallel)
    let horizontal_composed = bicategory.horizontal_compose(&parallel_grad1, &parallel_grad2).unwrap();

    println!("Horizontally composed 2-morphism:");
    println!("  - Description: {}", horizontal_composed.description);
    println!("  - Parameter changes: {:?}", horizontal_composed.parameter_changes);

    // Demonstrate whiskers (gradient propagation)
    println!("\n===== Demonstrating Whiskers =====");

    // Create a gradient for the first layer
    let grad1 = create_parameter_update(&qnn.layers[0].forward, "grad_rx0", 0.03);

    // Add a right whisker (propagate through the second layer)
    let whisker = bicategory.right_whisker(&grad1, &qnn.layers[1].forward).unwrap();

    println!("Right whisker 2-morphism (gradient through layer 2):");
    println!("  - Description: {}", whisker.description);
    println!("  - Parameter changes: {:?}", whisker.parameter_changes);

    println!("\n===== Conclusion =====");
    println!("This example demonstrated how higher category theory provides a formal");
    println!("mathematical framework for quantum machine learning, where:");
    println!("  - Objects (0-cells) are qubit counts");
    println!("  - Morphisms (1-cells) are quantum circuits");
    println!("  - 2-morphisms are parameter updates and gradients");
    println!("  - Vertical composition represents backpropagation");
    println!("  - Horizontal composition represents parallel operations");
    println!("  - Whiskers represent gradient propagation through layers");
    println!("  - Adjunctions formalize the connection between forward and backward passes");
}

/// Build a variational quantum layer as an adjunction
fn build_variational_layer(qubit_count: usize, name: &str, init_param: f64) -> QuantumNeuralLayer {
    // Build the forward circuit
    let mut forward_builder = CircuitBuilder::new(qubit_count);

    // Add parameterized rotation gates
    for q in 0..qubit_count {
        forward_builder.rx(q, init_param).unwrap();
        forward_builder.ry(q, init_param).unwrap();
    }

    // Add entanglement
    if qubit_count > 1 {
        forward_builder.cnot(0, 1).unwrap();
    }

    // Build the backward circuit (conceptually for gradient computation)
    let mut backward_builder = CircuitBuilder::new(qubit_count);

    // In a real implementation, this would compute parameter gradients
    // For simplicity, we just use the transpose of the forward circuit
    for q in 0..qubit_count {
        backward_builder.ry(q, -init_param).unwrap();
        backward_builder.rx(q, -init_param).unwrap();
    }

    // Store parameters
    let mut parameters = HashMap::new();
    parameters.insert(format!("{}_rx0", name), init_param);
    parameters.insert(format!("{}_ry0", name), init_param);
    if qubit_count > 1 {
        parameters.insert(format!("{}_rx1", name), init_param);
        parameters.insert(format!("{}_ry1", name), init_param);
    }

    // Create the layer as an adjunction
    QuantumNeuralLayer {
        forward: forward_builder.build(),
        backward: backward_builder.build(),
        name: name.to_string(),
        parameters,
    }
}

/// Create a 2-morphism representing a parameter update
fn create_parameter_update(
    circuit: &QuantumCircuit,
    param_name: &str,
    update_value: f64
) -> CircuitTransformation {
    // In a real implementation, we would create a new circuit
    // with updated parameters. For now, just clone the original.
    let mut parameter_changes = HashMap::new();
    parameter_changes.insert(param_name.to_string(), update_value);

    CircuitTransformation {
        source: circuit.clone(),
        target: circuit.clone(),  // Would be different in a real implementation
        description: format!("Parameter update for {}", param_name),
        parameter_changes,
    }
}

/// Apply a parameter update to the network
fn apply_parameter_update(
    update: &CircuitTransformation,
    qnn: &mut HigherCategoricalQuantumNN
) -> HashMap<String, f64> {
    // In a real implementation, this would create new circuits
    // For now, just return the updated parameter values
    let mut updated_params = HashMap::new();

    for layer in &qnn.layers {
        for (name, &value) in &layer.parameters {
            if let Some(&change) = update.parameter_changes.get(name) {
                updated_params.insert(name.clone(), value + change);
            } else {
                updated_params.insert(name.clone(), value);
            }
        }
    }

    updated_params
}

/// Simulate running a quantum circuit with the given inputs
fn simulate_circuit(
    circuit: &QuantumCircuit,
    inputs: &Array1<f64>,
    qubit_count: usize
) -> Array1<f64> {
    // Create initial state
    let initial_state = StateVector::zero_state(qubit_count);

    // Encode inputs using rotation gates
    let mut encoding_builder = CircuitBuilder::new(qubit_count);
    for (i, &value) in inputs.iter().enumerate().take(qubit_count) {
        encoding_builder.ry(i, value * PI).unwrap();
    }
    let encoding_circuit = encoding_builder.build();

    // Apply encoding circuit to initial state
    let encoded_state = match encoding_circuit.apply(&initial_state) {
        Ok(state) => state,
        Err(e) => {
            println!("Error applying encoding circuit: {}", e);
            return Array1::zeros(2);
        }
    };

    // Apply the model circuit
    let output_state = match circuit.apply(&encoded_state) {
        Ok(state) => state,
        Err(e) => {
            println!("Error applying model circuit: {}", e);
            return Array1::zeros(2);
        }
    };

    // Create a simulator to help with measurements
    let simulator = StatevectorSimulator::from_state(output_state);

    // Get measurement probabilities for the first two qubits
    let mut predictions = Array1::zeros(2);

    // Measure each qubit (non-collapsing) to get probabilities
    if let Ok(prob_q0) = simulator.measure_qubit_probability(0) {
        predictions[0] = *prob_q0.get(&Outcome::Zero).unwrap_or(&0.5);
    }

    // For the second prediction, either use qubit 1 or compute from qubit 0
    if qubit_count > 1 {
        if let Ok(prob_q1) = simulator.measure_qubit_probability(1) {
            predictions[1] = *prob_q1.get(&Outcome::Zero).unwrap_or(&0.5);
        }
    } else {
        // If only one qubit, use complement of first probability
        predictions[1] = 1.0 - predictions[0];
    }

    predictions
}

/// Create a 2-morphism representing the gradient of the loss
fn create_loss_gradient_2morphism(
    circuit: &QuantumCircuit,
    loss_fn: &dyn LossFunction<Input = Array1<f64>>,
    predictions: &Array1<f64>,
    targets: &Array1<f64>
) -> CircuitTransformation {
    // Calculate the gradient of the loss with respect to predictions
    let output_gradient = loss_fn.calculate_gradients(predictions, targets);

    // In a real implementation, we would map this to parameter gradients
    // For simplicity, we just create dummy parameter gradients
    let mut parameter_changes = HashMap::new();
    parameter_changes.insert("layer1_rx0".to_string(), -0.01 * output_gradient[0]);
    parameter_changes.insert("layer1_ry0".to_string(), -0.01 * output_gradient[1]);
    parameter_changes.insert("layer2_rx0".to_string(), -0.005 * output_gradient[0]);
    parameter_changes.insert("layer2_ry0".to_string(), -0.005 * output_gradient[1]);

    CircuitTransformation {
        source: circuit.clone(),
        target: circuit.clone(),
        description: "Loss gradient".to_string(),
        parameter_changes,
    }
}

/// Create a 2-morphism representing a layer's gradient
fn create_layer_gradient(circuit: &QuantumCircuit, name: &str) -> CircuitTransformation {
    // For demonstration, create a simple gradient 2-morphism
    // Use the same circuit for source and target to allow composition
    let mut parameter_changes = HashMap::new();

    // Add dummy gradients
    parameter_changes.insert(format!("{}_rx0", name), -0.01);
    parameter_changes.insert(format!("{}_ry0", name), -0.02);
    if circuit.qubit_count > 1 {
        parameter_changes.insert(format!("{}_rx1", name), -0.005);
        parameter_changes.insert(format!("{}_ry1", name), -0.015);
    }

    CircuitTransformation {
        source: circuit.clone(),  // Same circuit as source
        target: circuit.clone(),  // Same circuit as target
        description: format!("Gradient for {}", name),
        parameter_changes,
    }
}
