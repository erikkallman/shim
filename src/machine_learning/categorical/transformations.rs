//! Natural transformations between functors

use ndarray::Array1;
use std::fmt::Debug;
use std::any::Any;

use crate::category::NaturalTransformation;
use crate::machine_learning::categorical::categories::{
    ModelCategory, DataCategory, CircuitCategory, ModelDimension
};
use crate::machine_learning::categorical::functors::{
    DataToCircuitFunctor, CircuitToModelFunctor, ClassicalToQuantumFunctor
};
use crate::machine_learning::core::{Model, PredictiveModel, TrainableModel};
use crate::machine_learning::optimizer::Optimizer;
use crate::machine_learning::loss::LossFunction;
use crate::machine_learning::quantum::model::QuantumModel;
use crate::machine_learning::quantum::model::DecodingStrategy;
use crate::quantum::QuantumCircuit;
use crate::machine_learning::categorical::prediction::PredictionTransformation;
use crate::quantum::StateVector;
use crate::category::monoidal::Category;
use crate::quantum::CircuitBuilder;
use crate::machine_learning::quantum::model::EncodingStrategy;
use std::f64::consts::PI;
use ndarray::Array2;
use crate::machine_learning::categorical::categories::ModelTransformation;
use crate::machine_learning::categorical::functors::DataToCircuitIdentityFunctor;
use crate::machine_learning::categorical::functors::CircuitToModelIdentityFunctor;
use crate::machine_learning::categorical::IdentityTransformation;

/// Natural transformation representing the training process
pub struct TrainingTransformation<M, O, L>
where
    M: Model,
    O: Optimizer,
    L: LossFunction,
{
    /// The model being trained
    model: M,
    /// The optimizer
    optimizer: O,
    /// The loss function
    loss: L,
}

impl<M, O, L> TrainingTransformation<M, O, L>
where
    M: Model,
    O: Optimizer,
    L: LossFunction,
{
    /// Creates a new training transformation
    pub fn new(model: M, optimizer: O, loss: L) -> Self {
        TrainingTransformation {
            model,
            optimizer,
            loss,
        }
    }

    /// Gets a reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Gets a mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Gets a reference to the optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Gets a reference to the loss function
    pub fn loss(&self) -> &L {
        &self.loss
    }
}

impl<M, O, L, F1, F2> NaturalTransformation<F1, F2, ModelCategory, ModelCategory> for TrainingTransformation<M, O, L>
where
    M: TrainableModel<LossFunction = L>,
    O: Optimizer,
    L: LossFunction,
    F1: crate::category::Functor<ModelCategory, ModelCategory>,
    F2: crate::category::Functor<ModelCategory, ModelCategory>,
{
    fn component(
        &self,
        c: &ModelCategory,
        _d: &ModelCategory,
        _f: &F1,
        _g: &F2,
        obj: &ModelDimension
    ) -> <ModelCategory as crate::category::Category>::Morphism {
        // This is a placeholder implementation - in a real system, this would
        // represent how training transforms the model's behavior
        c.identity(obj)
    }
}

/// Natural transformation for state preparation from data to circuits
pub struct StatePreparationTransformation<E> {
    #[allow(dead_code)]
    encoding_strategy: E,
}

impl<E> StatePreparationTransformation<E> {
    pub fn new(encoding_strategy: E) -> Self {
        StatePreparationTransformation {
            encoding_strategy,
        }
    }

    // Add this helper method
    pub fn prepare_state(
        &self,
        data_cat: &DataCategory,
        circuit_cat: &CircuitCategory,
        functor: &DataToCircuitFunctor,
        id_functor: &DataToCircuitIdentityFunctor,
        obj: &usize
    ) -> QuantumCircuit {
        // Use the fully qualified syntax to ensure the correct implementation is used
        <Self as NaturalTransformation<DataToCircuitFunctor, DataToCircuitIdentityFunctor, DataCategory, CircuitCategory>>::component(
            self,
            data_cat,
            circuit_cat,
            functor,
            id_functor,
            obj
        )
    }
}

// For StatePreparationTransformation
impl<E> NaturalTransformation<DataToCircuitFunctor, DataToCircuitIdentityFunctor, DataCategory, CircuitCategory>
    for StatePreparationTransformation<E>
{
    fn component(
        &self,
        _c: &DataCategory,
        _d: &CircuitCategory,
        f: &DataToCircuitFunctor,
        _id: &DataToCircuitIdentityFunctor,
        obj: &usize
    ) -> <CircuitCategory as Category>::Morphism {
        // Create a circuit for state preparation
        let mut builder = CircuitBuilder::new(f.qubit_count);

        // Add gates based on the encoding strategy
        match &f.encoding_strategy {
            EncodingStrategy::AmplitudeEncoding => {
                // Simple Hadamard gates for superposition
                for q in 0..f.qubit_count {
                    builder.h(q).unwrap();
                }
            },
            EncodingStrategy::AngleEncoding => {
                // Apply Ry gates with dummy angles for preparation
                for q in 0..f.qubit_count {
                    if q < *obj {
                        builder.ry(q, PI/4.0).unwrap(); // Dummy angle
                    }
                }

                // Add entanglement
                if f.qubit_count > 1 {
                    for q in 0..f.qubit_count-1 {
                        builder.cnot(q, q+1).unwrap();
                    }
                }
            },
            // Add other strategies as needed
            _ => {
                // Default: just prepare |0⟩ state (empty circuit)
            }
        }

        builder.build()
    }
}

/// Natural transformation for model prediction
pub struct ModelPredictionTransformation {
    decoding_strategy: DecodingStrategy,
}

impl ModelPredictionTransformation {
    pub fn new(decoding_strategy: DecodingStrategy) -> Self {
        ModelPredictionTransformation {
            decoding_strategy,
        }
    }

    // Add this helper method to wrap the trait call
    pub fn measure_circuit(
        &self,
        circuit_cat: &CircuitCategory,
        model_cat: &ModelCategory,
        id_functor: &CircuitToModelIdentityFunctor,
        functor: &CircuitToModelFunctor,
        obj: &usize
    ) -> Box<dyn ModelTransformation> {
        // Use the fully qualified syntax to ensure the correct implementation is used
        <Self as NaturalTransformation<CircuitToModelIdentityFunctor, CircuitToModelFunctor, CircuitCategory, ModelCategory>>::component(
            self,
            circuit_cat,
            model_cat,
            id_functor,
            functor,
            obj
        )
    }
}

/// Model transformation that measures quantum circuit output
#[derive(Clone, Debug)]
pub struct CircuitMeasurementTransformation {
    /// Number of qubits in the circuit
    qubit_count: usize,
    /// Output dimension of the model
    output_dim: usize,
    /// Strategy for decoding quantum states to classical outputs
    decoding_strategy: DecodingStrategy,
}

impl ModelTransformation for CircuitMeasurementTransformation {
    fn domain(&self) -> ModelDimension {
        ModelDimension {
            input_dim: self.qubit_count,
            output_dim: 0, // Not used for domain
        }
    }

    fn codomain(&self) -> ModelDimension {
        ModelDimension {
            input_dim: 0, // Not used for codomain
            output_dim: self.output_dim,
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Create an encoding circuit
        let mut builder = CircuitBuilder::new(self.qubit_count);

        // Encode input data as quantum state
        for (i, &value) in input.iter().enumerate() {
            if i < self.qubit_count {
                builder.ry(i, value).map_err(|e| e.to_string())?;
            }
        }

        // Run the circuit
        let circuit = builder.build();
        let initial_state = StateVector::zero_state(self.qubit_count);
        let final_state = circuit.apply(&initial_state)
            .map_err(|e| e.to_string())?;

        // Decode using the existing decoding logic from CircuitToPredictionFunctor
        // We can reuse the decode_state method logic from CircuitToPredictionFunctor
        match &self.decoding_strategy {
            DecodingStrategy::MeasurementBased(qubits) => {
                // Get measurement probabilities for the specified qubits
                let amplitudes = final_state.amplitudes();
                let mut result = Array1::zeros(qubits.len());

                for (i, &qubit) in qubits.iter().enumerate() {
                    if qubit < self.qubit_count {
                        let mut prob_1 = 0.0;
                        for (j, amp) in amplitudes.iter().enumerate() {
                            if (j >> qubit) & 1 == 1 {
                                prob_1 += amp.norm_sqr();
                            }
                        }
                        result[i] = prob_1;
                    }
                }

                Ok(result)
            },

            DecodingStrategy::ExpectationBased(observables) => {
                let mut result = Array1::zeros(observables.len());
                let amplitude_vec = Array1::from_iter(final_state.amplitudes().iter().cloned());

                for (i, observable) in observables.iter().enumerate() {
                    let expectation = amplitude_vec.dot(&observable.dot(&amplitude_vec.clone())).re;
                    result[i] = expectation;
                }

                Ok(result)
            },

            DecodingStrategy::Custom(decoder) => {
                decoder(&final_state).map_err(|e| e.into())
            },
        }
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.qubit_count == other.qubit_count &&
            self.output_dim == other.output_dim &&
            match (&self.decoding_strategy, &other.decoding_strategy) {
                (DecodingStrategy::MeasurementBased(qubits1), DecodingStrategy::MeasurementBased(qubits2)) => {
                    qubits1 == qubits2
                },
                (DecodingStrategy::ExpectationBased(obs1), DecodingStrategy::ExpectationBased(obs2)) => {
                    // Simple check for dimensions, full matrix comparison would be expensive
                    obs1.len() == obs2.len() &&
                    obs1.iter().zip(obs2.iter()).all(|(a, b)| a.shape() == b.shape())
                },
                // For custom decodings, we can't compare the closures directly
                _ => false
            }
        } else {
            false
        }
    }
}

impl NaturalTransformation<CircuitToModelIdentityFunctor, CircuitToModelFunctor, CircuitCategory, ModelCategory>
    for ModelPredictionTransformation
{
    fn component(
        &self,
        _c: &CircuitCategory,
        _d: &ModelCategory,
        _id: &CircuitToModelIdentityFunctor,
        g: &CircuitToModelFunctor,
        obj: &usize
    ) -> <ModelCategory as Category>::Morphism {
        // Create a model transformation that measures circuit output
        Box::new(CircuitMeasurementTransformation {
            qubit_count: *obj,
            output_dim: g.output_dim,
            decoding_strategy: match &self.decoding_strategy {
                DecodingStrategy::MeasurementBased(qubits) =>
                    DecodingStrategy::MeasurementBased(qubits.clone()),
                DecodingStrategy::ExpectationBased(obs) =>
                    DecodingStrategy::ExpectationBased(obs.clone()),
                DecodingStrategy::Custom(_func) =>
                    DecodingStrategy::MeasurementBased(vec![0]), // Simple default
            },
        })
    }
}

pub struct DataToModelFunctor<E> {
    state_preparation: StatePreparationTransformation<E>,
    model_prediction: ModelPredictionTransformation,
    data_to_circuit: DataToCircuitFunctor,
    circuit_to_model: CircuitToModelFunctor,
    // Add the identity functors
    data_to_circuit_id: DataToCircuitIdentityFunctor,
    circuit_to_model_id: CircuitToModelIdentityFunctor,
}

impl<E> DataToModelFunctor<E> {
    pub fn new(
        state_preparation: StatePreparationTransformation<E>,
        model_prediction: ModelPredictionTransformation,
        data_to_circuit: DataToCircuitFunctor,
        circuit_to_model: CircuitToModelFunctor,
    ) -> Self {
        DataToModelFunctor {
            state_preparation,
            model_prediction,
            data_to_circuit: data_to_circuit.clone(),
            circuit_to_model: circuit_to_model.clone(),
            // Initialize the identity functors
            data_to_circuit_id: DataToCircuitIdentityFunctor {
                qubit_count: data_to_circuit.clone().qubit_count,
            },
            circuit_to_model_id: CircuitToModelIdentityFunctor {
                output_dim: circuit_to_model.clone().output_dim,
            },
        }
    }
}


impl<E> crate::category::Functor<DataCategory, ModelCategory> for DataToModelFunctor<E> {

    fn map_object(&self, _source: &DataCategory, _target: &ModelCategory, obj: &usize) -> ModelDimension {
        // Map from data dimension to model dimension
        let circuit_obj = self.data_to_circuit.map_object(&DataCategory, &CircuitCategory, obj);
        self.circuit_to_model.map_object(&CircuitCategory, &ModelCategory, &circuit_obj)
    }

    fn map_morphism(&self, _source: &DataCategory, _target: &ModelCategory, f: &Array2<f64>) -> Box<dyn ModelTransformation> {
        let data_cat = DataCategory;
        let circuit_cat = CircuitCategory;
        let model_cat = ModelCategory;

        // Get the circuit from data transformation
        let circuit_morphism = self.data_to_circuit.map_morphism(&data_cat, &circuit_cat, f);

        let preparation_circuit = self.state_preparation.prepare_state(
            &data_cat,
            &circuit_cat,
            &self.data_to_circuit,
            &self.data_to_circuit_id,
            &f.shape()[1]
        );

        // Compose the circuits
        let composed_circuit = circuit_cat.compose(&preparation_circuit, &circuit_morphism)
            .expect("Failed to compose circuits");

        // Apply measurement/prediction using the proper identity functor
        self.model_prediction.measure_circuit(
            &circuit_cat,
            &model_cat,
            &self.circuit_to_model_id,
            &self.circuit_to_model,
            &composed_circuit.qubit_count
        )
    }
}

/// Natural transformation for circuit optimization
pub struct CircuitOptimizationTransformation {
    /// Optimization level (0-3)
    #[allow(dead_code)]
    level: usize,
}

impl CircuitOptimizationTransformation {
    /// Creates a new circuit optimization transformation
    pub fn new(level: usize) -> Self {
        CircuitOptimizationTransformation {
            level: level.min(3),  // Clamp to max level 3
        }
    }
}

impl NaturalTransformation<ClassicalToQuantumFunctor, ClassicalToQuantumFunctor, ModelCategory, ModelCategory>
    for CircuitOptimizationTransformation
{
    fn component(
        &self,
        c: &ModelCategory,
        _d: &ModelCategory,
        _f: &ClassicalToQuantumFunctor,
        _g: &ClassicalToQuantumFunctor,
        obj: &ModelDimension
    ) -> <ModelCategory as crate::category::Category>::Morphism {
        // Circuit optimization doesn't change the dimensions, so we return identity
        c.identity(obj)
    }
}

/// Helper to compose quantum circuits categorically
pub struct CategoricalCircuitComposition;

impl CategoricalCircuitComposition {
    /// Composes two quantum models using categorical composition
    pub fn compose<M1, M2>(
        _model1: &M1,
        _model2: &M2
    ) -> Result<Box<dyn std::error::Error>, Box<dyn std::error::Error>>
    where
        M1: QuantumModel + Clone + Send + Sync + 'static,
        M2: QuantumModel<Input=M1::Output> + Clone + Send + Sync + 'static,
    {
        // This would create a new model that composes the two models!!
        Err("Categorical circuit composition not fully implemented".into())
    }

    /// Combines two quantum models using tensor product
    pub fn tensor_product<M1, M2>(
        _model1: &M1,
        _model2: &M2
    ) -> Result<Box<dyn std::error::Error>, Box<dyn std::error::Error>>
    where
        M1: QuantumModel + Clone + Send + Sync + 'static,
        M2: QuantumModel + Clone + Send + Sync + 'static,
        M1::Input: AsRef<Array1<f64>>,
        M1::Output: AsRef<Array1<f64>>,
        M2::Input: AsRef<Array1<f64>>,
        M2::Output: AsRef<Array1<f64>>,
    {
        // This would create a new model that combines the two models using tensor product !!
        Err("Categorical tensor product not fully implemented".into())
    }
}


/// Prediction transformation backed by a quantum circuit
#[derive(Debug, Clone)]
pub struct CircuitPredictionTransformation {
    /// The underlying quantum circuit
    pub circuit: QuantumCircuit,
    /// Number of qubits
    pub qubit_count: usize,
    /// Decoding strategy
    pub decoding_strategy: DecodingStrategy,
}

impl PredictionTransformation for CircuitPredictionTransformation {
    fn domain(&self) -> usize {
        self.qubit_count
    }

    fn codomain(&self) -> usize {
        match &self.decoding_strategy {
            DecodingStrategy::MeasurementBased(qubits) => qubits.len(),
            DecodingStrategy::ExpectationBased(observables) => observables.len(),
            DecodingStrategy::Custom(_) => self.qubit_count, // Default for custom decoders
        }
    }

    fn apply(&self, input_state: &StateVector) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Run the circuit on the input state
        let final_state = self.circuit.apply(input_state)
            .map_err(|e| e.to_string())?;

        // Decode the final state based on the decoding strategy
        match &self.decoding_strategy {
            DecodingStrategy::MeasurementBased(qubits) => {
                let amplitudes = final_state.amplitudes();
                let mut result = Array1::zeros(qubits.len());

                // Calculate measurement probabilities for each specified qubit
                for (i, &qubit) in qubits.iter().enumerate() {
                    if qubit < self.qubit_count {
                        let mut prob_1 = 0.0;
                        for (j, amp) in amplitudes.iter().enumerate() {
                            if (j >> qubit) & 1 == 1 {
                                prob_1 += amp.norm_sqr();
                            }
                        }
                        result[i] = prob_1;
                    }
                }

                Ok(result)
            },

            DecodingStrategy::ExpectationBased(observables) => {
                let mut result = Array1::zeros(observables.len());
                let amplitude_vec = Array1::from_iter(final_state.amplitudes().iter().cloned());

                // Calculate expectation values for each observable
                for (i, observable) in observables.iter().enumerate() {
                    let expectation = amplitude_vec.dot(&observable.dot(&amplitude_vec.clone())).re;
                    result[i] = expectation;
                }

                Ok(result)
            },

            DecodingStrategy::Custom(decoder) => {
                decoder(&final_state).map_err(|e| e.into())
            },
        }
    }

    fn clone_box(&self) -> Box<dyn PredictionTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn PredictionTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.qubit_count == other.qubit_count &&
            // Compare circuits - simple comparison of qubit count
            self.circuit.qubit_count == other.circuit.qubit_count &&
            // Compare decoding strategies
            match (&self.decoding_strategy, &other.decoding_strategy) {
                (DecodingStrategy::MeasurementBased(qubits1), DecodingStrategy::MeasurementBased(qubits2)) => {
                    qubits1 == qubits2
                },
                (DecodingStrategy::ExpectationBased(obs1), DecodingStrategy::ExpectationBased(obs2)) => {
                    // Simple check for dimensions, full matrix comparison would be expensive
                    obs1.len() == obs2.len() &&
                    obs1.iter().zip(obs2.iter()).all(|(a, b)| a.shape() == b.shape())
                },
                // For custom decodings, we can't compare the closures directly
                _ => false
            }
        } else {
            false
        }
    }
}

/// Trait for optimization transformations
pub trait OptimizationTransformation: Send + Sync + Debug {
    /// Gets the domain dimensions
    fn domain(&self) -> ModelDimension;

    /// Gets the codomain dimensions
    fn codomain(&self) -> ModelDimension;

    /// Applies the optimization to training data and returns an optimized model transformation
    fn apply(&self, input: &(Array2<f64>, Vec<Array1<f64>>)) -> Result<Box<dyn ModelTransformation>, Box<dyn std::error::Error>>;

    /// Clones the transformation into a boxed trait object
    fn clone_box(&self) -> Box<dyn OptimizationTransformation>;

    /// Returns a reference to self as Any for downcasting in PartialEq implementation
    fn as_any(&self) -> &dyn Any;

    /// Compares two OptimizationTransformations for equality
    fn equals(&self, other: &dyn OptimizationTransformation) -> bool;
}

// Add this implementation to allow Clone for Box<dyn OptimizationTransformation>
impl Clone for Box<dyn OptimizationTransformation> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// Implement PartialEq for Box<dyn OptimizationTransformation>
impl PartialEq for Box<dyn OptimizationTransformation> {
    fn eq(&self, other: &Box<dyn OptimizationTransformation>) -> bool {
        self.equals(other.as_ref())
    }
}

/// Struct for gradient-based optimization transformations
#[derive(Debug, Clone)]
pub struct GradientOptimizationTransformation<M, O, L>
where
    M: Model + Clone + Debug,
    O: Optimizer + Clone + Debug,
    L: LossFunction + Clone + Debug,
{
    model: M,
    optimizer: O,
    loss_fn: L,
    epochs: usize,
}

impl<M, O, L> GradientOptimizationTransformation<M, O, L>
where
    M: Model + Clone + Debug,
    O: Optimizer + Clone + Debug,
    L: LossFunction + Clone + Debug,
{
    /// Creates a new optimization transformation
    pub fn new(model: M, optimizer: O, loss_fn: L, epochs: usize) -> Self {
        GradientOptimizationTransformation {
            model,
            optimizer,
            loss_fn,
            epochs,
        }
    }
}

impl<M, O, L> OptimizationTransformation for GradientOptimizationTransformation<M, O, L>
where
    M: Model<Input = Array1<f64>, Output = Array1<f64>> + PredictiveModel + Clone + Debug + Send + Sync + 'static,
    M::Error: Into<Box<dyn std::error::Error>>, // Add this to accept any error type
    O: Optimizer + Clone + Debug + Send + Sync + 'static,
    L: LossFunction<Input = Array1<f64>> + Clone + Debug + Send + Sync + 'static,
{
    fn domain(&self) -> ModelDimension {
        let (input_dim, output_dim) = self.model.dimensions();
        ModelDimension {
            input_dim,
            output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        let (input_dim, output_dim) = self.model.dimensions();
        ModelDimension {
            input_dim,
            output_dim,
        }
    }

    fn apply(&self, input: &(Array2<f64>, Vec<Array1<f64>>)) -> Result<Box<dyn ModelTransformation>, Box<dyn std::error::Error>> {
        let (x_data, y_data) = input;

        // Clone the model for training
        let mut model = self.model.clone();
        let optimizer = self.optimizer.clone();
        let loss_fn = self.loss_fn.clone();

        // Training loop
        for _ in 0..self.epochs {
            for i in 0..x_data.nrows() {
                let x_i = x_data.row(i).to_owned();
                let y_i = &y_data[i];

                // Calculate gradients using finite differences
                let mut gradients = vec![0.0; model.parameter_count()];
                let epsilon = 1e-5_f64;

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
        }

        // Return a model transformation that uses the optimized model
        Ok(Box::new(OptimizedModelTransformation {
            model: Box::new(model),
        }))
    }

    fn clone_box(&self) -> Box<dyn OptimizationTransformation> {
        Box::new(Self {
            model: self.model.clone(),
            optimizer: self.optimizer.clone(),
            loss_fn: self.loss_fn.clone(),
            epochs: self.epochs,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn OptimizationTransformation) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
        // In practice, comparing two gradient optimizations would be more complex
        // since we would need to compare models, optimizers, and loss functions
    }
}

#[derive(Debug)]
pub struct OptimizedModelTransformation<M>
where
    M: PredictiveModel + Debug + ?Sized,
{
    model: Box<M>,
}

impl<M> ModelTransformation for OptimizedModelTransformation<M>
where
    M: PredictiveModel<Input = Array1<f64>, Output = Array1<f64>> + Debug + ?Sized + 'static + std::marker::Sync + std::marker::Send,
{
    fn domain(&self) -> ModelDimension {
        let (input_dim, output_dim) = self.model.dimensions();
        ModelDimension {
            input_dim,
            output_dim,
        }
    }

    fn codomain(&self) -> ModelDimension {
        let (input_dim, output_dim) = self.model.dimensions();
        ModelDimension {
            input_dim,
            output_dim,
        }
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        self.model.predict(input).map_err(|e| e.into())
    }

    fn clone_box(&self) -> Box<dyn ModelTransformation> {
        // Use a constructor method to create a new instance
        OptimizedModelTransformation::from_model(self.model.as_ref())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ModelTransformation) -> bool {
        // For optimized model transformations, we consider them equal
        // if they refer to models of the same type
        // Note: This is a simplified equality check that only checks types
        other.as_any().type_id() == self.as_any().type_id()
    }
}

impl<M> OptimizedModelTransformation<M>
where
    M: PredictiveModel<Input = Array1<f64>, Output = Array1<f64>> + Debug + ?Sized + 'static + std::marker::Sync + std::marker::Send,
{
    // Create a new instance from a model reference
    pub fn from_model(model: &M) -> Box<dyn ModelTransformation> {

        // probably use clone_box_predictive or other methods to properly clone the model

        // For now, let's create a new IdentityTransformation as a placeholder
        let (input_dim, output_dim) = model.dimensions();
        let dimension = ModelDimension { input_dim, output_dim };

        Box::new(IdentityTransformation::new(dimension))
    }
}

// Add this identity model transformation
#[derive(Debug, Clone)]
pub struct IdentityModelTransformation {
    dim: ModelDimension,
}

impl IdentityModelTransformation {
    pub fn new(dim: ModelDimension) -> Self {
        IdentityModelTransformation { dim }
    }
}

impl ModelTransformation for IdentityModelTransformation {
    fn domain(&self) -> ModelDimension {
        self.dim.clone()
    }

    fn codomain(&self) -> ModelDimension {
        self.dim.clone()
    }

    fn apply(&self, input: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Identity just returns the input
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
            self.dim == other.dim
        } else {
            false
        }
    }
}
