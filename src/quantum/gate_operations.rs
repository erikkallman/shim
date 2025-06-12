// src/quantum/gate_operations.rs
use crate::quantum::QuantumGate;
use crate::quantum::gate::LocalizedGate;
use crate::category::prelude::*;
use crate::quantum::QuantumGateCategory;

// Enhanced gate_operations module that leverages the categorical structure
pub mod gate_operations {
    use super::*;

    // Our CategoryAware wrapper for working with categorical operations
    #[derive(Clone)]
    pub struct CategoryAware {
        pub category: QuantumGateCategory,
    }

    impl Default for CategoryAware {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CategoryAware {
        pub fn new() -> Self {
            CategoryAware {
                category: QuantumGateCategory,
            }
        }

        /// Create a tensor product of two gates using categorical operation
        pub fn tensor(
            &self,
            gate1: &dyn QuantumGate,
            gate2: &dyn QuantumGate
        ) -> Box<dyn QuantumGate> {
            self.category.tensor_morphisms(
                &Box::new(gate1.clone_box()),
                &Box::new(gate2.clone_box())
            )
        }

        /// Compose two gates (gate2 . gate1) using categorical operation
        pub fn compose(
            &self,
            gate1: &dyn QuantumGate,
            gate2: &dyn QuantumGate
        ) -> Result<Box<dyn QuantumGate>, String> {
            match self.category.compose(
                &Box::new(gate1.clone_box()),
                &Box::new(gate2.clone_box())
            ) {
                Some(composed) => Ok(composed),
                None => Err(format!(
                    "Cannot compose gates with different qubit counts: {} and {}",
                    gate1.qubit_count(), gate2.qubit_count()
                ))
            }
        }

        /// Create a gate that applies to specific qubits in a larger system
        pub fn localize(
            &self,
            gate: &dyn QuantumGate,
            target_qubits: &[usize],
            total_qubits: usize
        ) -> Result<Box<dyn QuantumGate>, String> {
            if target_qubits.len() != gate.qubit_count() {
                return Err(format!(
                    "Gate acts on {} qubits, but {} target qubits were specified",
                    gate.qubit_count(), target_qubits.len()
                ));
            }

            for &q in target_qubits {
                if q >= total_qubits {
                    return Err(format!("Qubit index {} out of range", q));
                }
            }

            Ok(Box::new(LocalizedGate {
                gate: gate.clone_box(),
                target_qubits: target_qubits.to_vec(),
                total_qubits,
            }))
        }

        /// Create a sequence of gates using categorical composition
        pub fn sequence(&self, gates: Vec<Box<dyn QuantumGate>>) -> Result<Box<dyn QuantumGate>, String> {
            if gates.is_empty() {
                return Err("Cannot create an empty gate sequence".to_string());
            }

            let qubit_count = gates[0].qubit_count();

            // Check that all gates have the same qubit count
            for (i, gate) in gates.iter().enumerate().skip(1) {
                if gate.qubit_count() != qubit_count {
                    return Err(format!(
                        "Gate at index {} has different qubit count: {} != {}",
                        i, gate.qubit_count(), qubit_count
                    ));
                }
            }

            // Use categorical composition to build the sequence
            let mut result = gates[0].clone_box();

            for gate in gates.iter().skip(1) {
                // Compose with next gate (g ∘ f applies f then g)
                match self.category.compose(&result, gate) {
                    Some(composed) => result = composed,
                    None => return Err("Composition failed".to_string())
                }
            }

            Ok(result)
        }
    }

    // Functions from the original gate_operations module, now using CategoryAware

    /// Create a tensor product of two gates
    pub fn tensor(
        gate1: &dyn QuantumGate,
        gate2: &dyn QuantumGate
    ) -> Box<dyn QuantumGate> {
        CategoryAware::new().tensor(gate1, gate2)
    }

    /// Compose two gates (gate2 . gate1)
    /// This applies gate1 first, then gate2
    pub fn compose(
        gate1: &dyn QuantumGate,
        gate2: &dyn QuantumGate
    ) -> Result<Box<dyn QuantumGate>, String> {
        CategoryAware::new().compose(gate1, gate2)
    }

    /// Create a gate that applies to specific qubits in a larger system
    pub fn localize(
        gate: &dyn QuantumGate,
        target_qubits: &[usize],
        total_qubits: usize
    ) -> Result<Box<dyn QuantumGate>, String> {
        CategoryAware::new().localize(gate, target_qubits, total_qubits)
    }

    /// Create a sequence of gates to be applied one after another
    pub fn sequence(gates: Vec<Box<dyn QuantumGate>>) -> Result<Box<dyn QuantumGate>, String> {
        CategoryAware::new().sequence(gates)
    }
}
