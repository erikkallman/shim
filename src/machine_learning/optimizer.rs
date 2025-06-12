//! Optimization algorithms for machine learning models
use ndarray::Array1;
use ndarray::Array2;
use std::any::Any;
use crate::machine_learning::categorical::categories::ModelDimension;
use crate::machine_learning::categorical::categories::ModelTransformation;
use std::sync::Mutex;
use crate::prelude::Category;
use std::fmt::Debug;
use crate::machine_learning::categorical::IdentityTransformation;
use crate::machine_learning::categorical::OptimizationTransformation;

/// Trait for optimization algorithms
pub trait Optimizer: Send + Sync {
    /// Update parameters using gradients
    fn update(&self, parameters: &mut [f64], gradients: &[f64]);

    /// Reset the optimizer's internal state
    fn reset(&mut self);
}

/// Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct GradientDescent {
    learning_rate: f64,
}

impl GradientDescent {
    /// Creates a new Gradient Descent optimizer
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn update(&self, parameters: &mut [f64], gradients: &[f64]) {
        assert_eq!(parameters.len(), gradients.len(), "Parameter and gradient dimensions must match");

        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            *param -= self.learning_rate * grad;
        }
    }

    fn reset(&mut self) {
        // Gradient descent has no state to reset
    }
}

/// Adaptive Moment Estimation (Adam) optimizer
#[derive(Debug)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Mutex<Vec<f64>>,        // First moment estimate
    v: Mutex<Vec<f64>>,        // Second moment estimate
    t: Mutex<usize>,           // Timestep
}

impl Adam {
    /// Creates a new Adam optimizer
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Mutex::new(Vec::new()),
            v: Mutex::new(Vec::new()),
            t: Mutex::new(0),
        }
    }

    /// Creates a new Adam optimizer with default hyperparameters
    pub fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Clone for Adam {
    fn clone(&self) -> Self {
        let m = self.m.lock().unwrap().clone();
        let v = self.v.lock().unwrap().clone();
        let t = *self.t.lock().unwrap();

        Adam {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            m: Mutex::new(m),
            v: Mutex::new(v),
            t: Mutex::new(t),
        }
    }
}

impl Optimizer for Adam {
    fn update(&self, parameters: &mut [f64], gradients: &[f64]) {
        let n = parameters.len();
        assert_eq!(n, gradients.len(), "Parameter and gradient dimensions must match");

        // Get locks for mutable access to internal state
        let mut m = self.m.lock().unwrap();
        let mut v = self.v.lock().unwrap();
        let mut t = self.t.lock().unwrap();

        // Initialize moment estimates if not already done
        if m.is_empty() {
            *m = vec![0.0; n];
        }

        if v.is_empty() {
            *v = vec![0.0; n];
        }

        // Increment timestep
        *t += 1;
        let t_value = *t;

        // Update parameters
        for i in 0..n {
            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * gradients[i];

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            let m_hat = m[i] / (1.0 - self.beta1.powi(t_value as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = v[i] / (1.0 - self.beta2.powi(t_value as i32));

            // Update parameters
            parameters[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.m.lock().unwrap().clear();
        self.v.lock().unwrap().clear();
        *self.t.lock().unwrap() = 0;
    }
}

/// Stochastic Gradient Descent with momentum
#[derive(Debug)]
pub struct SGDMomentum {
    learning_rate: f64,
    momentum: f64,
    velocity: Mutex<Vec<f64>>,
}

impl SGDMomentum {
    /// Creates a new SGD with momentum optimizer
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        SGDMomentum {
            learning_rate,
            momentum,
            velocity: Mutex::new(Vec::new()),
        }
    }
}

impl Clone for SGDMomentum {
    fn clone(&self) -> Self {
        let velocity = self.velocity.lock().unwrap().clone();

        SGDMomentum {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            velocity: Mutex::new(velocity),
        }
    }
}

impl Optimizer for SGDMomentum {
    fn update(&self, parameters: &mut [f64], gradients: &[f64]) {
        let n = parameters.len();
        assert_eq!(n, gradients.len(), "Parameter and gradient dimensions must match");

        // Get lock for mutable access to velocity
        let mut velocity = self.velocity.lock().unwrap();

        // Initialize velocity if not already done
        if velocity.is_empty() {
            *velocity = vec![0.0; n];
        }

        // Update parameters
        for i in 0..n {
            // Update velocity
            velocity[i] = self.momentum * velocity[i] - self.learning_rate * gradients[i];

            // Update parameters
            parameters[i] += velocity[i];
        }
    }

    fn reset(&mut self) {
        self.velocity.lock().unwrap().clear();
    }
}

/// RMSProp optimizer
#[derive(Debug)]
pub struct RMSProp {
    learning_rate: f64,
    decay_rate: f64,
    epsilon: f64,
    cache: Mutex<Vec<f64>>,
}

impl RMSProp {
    /// Creates a new RMSProp optimizer
    pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64) -> Self {
        RMSProp {
            learning_rate,
            decay_rate,
            epsilon,
            cache: Mutex::new(Vec::new()),
        }
    }

    /// Creates a new RMSProp optimizer with default hyperparameters
    pub fn default() -> Self {
        RMSProp::new(0.001, 0.9, 1e-8)
    }
}

impl Clone for RMSProp {
    fn clone(&self) -> Self {
        let cache = self.cache.lock().unwrap().clone();

        RMSProp {
            learning_rate: self.learning_rate,
            decay_rate: self.decay_rate,
            epsilon: self.epsilon,
            cache: Mutex::new(cache),
        }
    }
}

impl Optimizer for RMSProp {
    fn update(&self, parameters: &mut [f64], gradients: &[f64]) {
        let n = parameters.len();
        assert_eq!(n, gradients.len(), "Parameter and gradient dimensions must match");

        // Get lock for mutable access to cache
        let mut cache = self.cache.lock().unwrap();

        // Initialize cache if not already done
        if cache.is_empty() {
            *cache = vec![0.0; n];
        }

        // Update parameters
        for i in 0..n {
            // Update cache
            cache[i] = self.decay_rate * cache[i] + (1.0 - self.decay_rate) * gradients[i] * gradients[i];

            // Update parameters
            parameters[i] -= self.learning_rate * gradients[i] / (cache[i].sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.cache.lock().unwrap().clear();
    }
}

/// Category for optimization problems
#[derive(Debug)]
pub struct OptimizationCategory;

impl Category for OptimizationCategory {
    /// Objects are model dimensions
    type Object = ModelDimension;

    /// Morphisms are optimization transformations
    type Morphism = Box<dyn OptimizationTransformation>;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.domain()
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.codomain()
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        Box::new(IdentityOptimizationTransformation {
            dimension: obj.clone(),
        })
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        if f.codomain() != g.domain() {
            return None;
        }

        Some(Box::new(ComposedOptimizationTransformation {
            first: f.clone(),
            second: g.clone(),
        }))
    }
}

// Identity optimization transformation
#[derive(Clone, Debug)]
pub struct IdentityOptimizationTransformation {
    pub dimension: ModelDimension,
}

impl OptimizationTransformation for IdentityOptimizationTransformation {
    fn domain(&self) -> ModelDimension {
        self.dimension.clone()
    }

    fn codomain(&self) -> ModelDimension {
        self.dimension.clone()
    }

    fn apply(&self, _input: &(Array2<f64>, Vec<Array1<f64>>)) -> Result<Box<dyn ModelTransformation>, Box<dyn std::error::Error>> {
        // Return identity model transformation
        Ok(Box::new(IdentityTransformation::new(self.dimension.clone())))
    }

    fn clone_box(&self) -> Box<dyn OptimizationTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn OptimizationTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.dimension == other.dimension
        } else {
            false
        }
    }
}

// Composed optimization transformation
#[derive(Clone, Debug)]
pub struct ComposedOptimizationTransformation {
    first: Box<dyn OptimizationTransformation>,
    second: Box<dyn OptimizationTransformation>,
}

impl OptimizationTransformation for ComposedOptimizationTransformation {
    fn domain(&self) -> ModelDimension {
        self.first.domain()
    }

    fn codomain(&self) -> ModelDimension {
        self.second.codomain()
    }

    fn apply(&self, input: &(Array2<f64>, Vec<Array1<f64>>)) -> Result<Box<dyn ModelTransformation>, Box<dyn std::error::Error>> {
        let _intermediate = self.first.apply(input)?;
        // This is a simplified composition - in reality would need to apply the intermediate transformation
        self.second.apply(input)
    }

    fn clone_box(&self) -> Box<dyn OptimizationTransformation> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn OptimizationTransformation) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self.first.equals(other.first.as_ref()) &&
            self.second.equals(other.second.as_ref())
        } else {
            false
        }
    }
}
