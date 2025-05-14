# Shim - A Categorical Quantum Machine Learning Framework in Rust

Shim is a Rust library that provides a mathematical foundation for quantum
machine learning using category theory. It aims to formalize quantum computation
and machine learning through categorical structures, enabling composition of
quantum operations and data transformations.

## Features

- **Category Theory Foundations**: Implementation of category theory primitives including categories, functors, monads, natural transformations, and bicategories
- **Quantum Circuit Simulation**: Tools for building and simulating quantum circuits
- **Categorical Quantum ML**: Models quantum machine learning as composition of categorical structures
- **Higher-Order Transformations**: Represents parameter updates and gradients as 2-morphisms in bicategories
- **Natural Transformations Framework**: System for implementing and composing quantum circuit transformations like optimization, error correction, and noise simulation
- **Dagger Compact Closed Categories**: Support for quantum teleportation and entanglement swapping through categorical structure
- **ZX-Calculus**: Support for reasoning about quantum operations using ZX-diagrams

## Current State of Development

Shim has many basic structures in place but certain functions are still shell
implementations. This is a hobby project (evening and weekend time), meaning I
typically work with less structure and priority.

That means that there is no roadmap. Im just exploring ideas and see the code
evolve as a result of that.

## Examples

The repository contains several examples demonstrating the capabilities of the framework:

### Categorical Quantum ML Showcase (`examples/categorical_quantum_ml_showcase.rs`)

Demonstrates how category theory can be used to formalize quantum machine learning pipelines:

- Creates a spiral dataset with configurable parameters
- Builds variational quantum circuits with customizable designs
- Demonstrates functorial mappings between data, circuits, and models
- Verifies category laws and functorial properties
- Runs inference on quantum models with categorical composition

### Categorical Verification (`examples/categorical_verification.rs`)

Demonstrates the verification of categorical laws with detailed explanations:

- Verification of basic category laws (identity, associativity)
- Testing monoidal category laws (unit laws, associativity)
- Verification of symmetric monoidal categories (symmetry law, hexagon identities)
- Testing naturality of braidings
- Verification of functor laws and monad laws

### Higher Categorical QML (`examples/higher_categorical_qml.rs`)

Explores how higher category theory can formalize quantum machine learning:

- Building quantum neural networks as compositions of adjunctions
- Representing parameter updates as 2-morphisms
- Formalizing backpropagation using vertical and horizontal composition
- Demonstrating whiskers for gradient propagation through layers

### Quantum Transformations (`examples/quantum_transformations.rs`)

Demonstrates the natural transformation framework for quantum circuits:

- Creating and applying circuit transformations including optimization, error detection, and ZX calculus
- Composing transformations to create complex circuit manipulation pipelines
- Implementing quantum teleportation with error protection
- Verifying naturality conditions for quantum transformations

## Getting Started

To use Shim in your project, add it to your `Cargo.toml`:

```toml
[dependencies]
categorical_qc = { git = "https://github.com/erikkallman/shim.git" }
```

Then import the necessary modules:

```rust
use categorical_qc::category::prelude::*;
use categorical_qc::quantum::circuit::*;
use categorical_qc::machine_learning::categorical::*;
```

## Building and Testing

Clone the repository and build:

```bash
git clone https://github.com/yourusername/shim.git
cd shim
cargo build
```

Run examples:

```bash
cargo run --example categorical_quantum_ml_showcase
cargo run --example categorical_verification
cargo run --example higher_categorical_qml
cargo run --example quantum_transformations
```

Run tests:

```bash
cargo test
```

## License

All files in the Shim Rust library are distributed using the MIT license. The LICENSE file contains the license details.
