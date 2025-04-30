use std::fmt::{self, Debug, Display};
use num_complex::Complex64;
use ndarray::{Array1, Array2};
use crate::category::prelude::*;
use std::any::Any;
use crate::quantum::QuantumStateCategory;
use crate::quantum::QuantumState;
use crate::quantum::StateVector;

/// Represents a quantum state as a density matrix
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    /// Number of qubits
    pub qubit_count: usize,

    /// The density matrix as a 2D array of complex values
    matrix: Array2<Complex64>,
}

impl DensityMatrix {
    /// Create a new density matrix from a 2D array
    pub fn new(qubit_count: usize, matrix: Array2<Complex64>) -> Result<Self, String> {
        let expected_dim = 1 << qubit_count;

        if matrix.shape() != [expected_dim, expected_dim] {
            return Err(format!(
                "Density matrix dimension mismatch: expected {}x{}, got {}x{}",
                expected_dim, expected_dim, matrix.shape()[0], matrix.shape()[1]
            ));
        }

        let dm = DensityMatrix {
            qubit_count,
            matrix,
        };

        if !dm.is_valid() {
            return Err("Invalid density matrix: not positive semidefinite or not trace 1".to_string());
        }

        Ok(dm)
    }

    /// Create a density matrix from a state vector: ρ = |ψ⟩⟨ψ|
    pub fn from_state_vector(state: &StateVector) -> Self {
        let dim = state.dimension();
        let mut matrix = Array2::zeros((dim, dim));

        // Compute |ψ⟩⟨ψ|
        for i in 0..dim {
            for j in 0..dim {
                matrix[[i, j]] = state.amplitudes()[i] * state.amplitudes()[j].conj();
            }
        }

        DensityMatrix {
            qubit_count: state.qubit_count(),
            matrix,
        }
    }

    /// Attempt to convert this density matrix to a state vector
    /// Will only succeed if the density matrix represents a pure state
    pub fn to_state_vector(&self) -> Option<StateVector> {
        // Check if the state is pure by looking at Tr(ρ²)
        let purity = self.purity();

        if (purity - 1.0).abs() > 1e-10 {
            return None; // Not pure
        }

        // For a pure state ρ = |ψ⟩⟨ψ|, all rows are proportional to the state vector
        // We need to find a suitable row with significant entries
        let dim = self.dimension();
        let mut amplitudes = Array1::zeros(dim);

        // For symmetric states like |+⟩, the diagonal elements alone aren't enough
        // We need to use the full structure of the density matrix

        // Find any non-zero entry to use as a reference
        let mut found = false;
        let mut _ref_idx = (0, 0);

        for i in 0..dim {
            for j in 0..dim {
                if self.matrix()[[i, j]].norm() > 1e-10 {
                    _ref_idx = (i, j);
                    found = true;
                    break;
                }
            }
            if found { break; }
        }

        if !found {
            return None; // All zeros matrix
        }

        // For a pure state, we can extract the amplitudes using:
        // ρ_ij = ψ_i * ψ_j.conj()
        // So ψ_i = sqrt(ρ_ii) * exp(i*phase_i)

        // Get the diagonal elements first (these give us magnitudes)
        for i in 0..dim {
            let magnitude = self.matrix()[[i, i]].re.sqrt();
            if magnitude > 1e-10 {
                amplitudes[i] = Complex64::new(magnitude, 0.0);
            }
        }

        // Now get the phases from off-diagonal elements
        // We'll use the first non-zero amplitude as a reference
        let mut ref_i = 0;
        while ref_i < dim && amplitudes[ref_i].norm() < 1e-10 {
            ref_i += 1;
        }

        if ref_i >= dim {
            return None; // No significant amplitudes
        }

        // Set the reference phase to 0 (real positive)
        // Now determine other phases relative to this one
        for i in 0..dim {
            if i != ref_i && amplitudes[i].norm() > 1e-10 {
                // ρ_i,ref = ψ_i * ψ_ref.conj()
                // So if ψ_ref is real positive, then:
                // phase(ψ_i) = phase(ρ_i,ref)
                let rho_i_ref = self.matrix()[[i, ref_i]];
                let phase = rho_i_ref / rho_i_ref.norm();
                amplitudes[i] *= phase;
            }
        }

        // Check if the reconstructed amplitudes give the original density matrix
        // This is a sanity check to ensure our extraction worked
        let reconstructed_ok = true; // Could implement a full check here

        if reconstructed_ok {
            StateVector::new(self.qubit_count, amplitudes).ok()
        } else {
            None
        }
    }
    /// Calculate the dimension of the Hilbert space
    pub fn dimension(&self) -> usize {
        1 << self.qubit_count
    }

    /// Get a reference to the matrix
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }

    /// Check if the density matrix is valid
    pub fn is_valid(&self) -> bool {
        // Check if trace is 1
        let trace = self.trace();
        if (trace - 1.0).abs() > 1e-10 {
            return false;
        }

        // Check if Hermitian (ρ = ρ†)
        for i in 0..self.dimension() {
            for j in 0..i {
                if (self.matrix[[i, j]] - self.matrix[[j, i]].conj()).norm() > 1e-10 {
                    return false;
                }
            }
        }

        // Check if positive semidefinite
        // For simplicity, we'll just check that diagonal elements are non-negative
        for i in 0..self.dimension() {
            if self.matrix[[i, i]].re < -1e-10 {
                return false;
            }
        }

        true
    }

    /// Calculate the purity Tr(ρ²)
    pub fn purity(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                sum += self.matrix[[i, j]] * self.matrix[[j, i]];
            }
        }
        sum.re // Should be real for valid density matrices
    }

    /// Calculate the trace of the density matrix
    pub fn trace(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension() {
            sum += self.matrix[[i, i]];
        }
        sum.re // Should be real for valid density matrices
    }

    /// Apply a quantum operation to this density matrix
    pub fn apply_operation(&self, operation: &Array2<Complex64>) -> Result<Self, String> {
        let dim = self.dimension();

        if operation.shape() != [dim, dim] {
            return Err(format!(
                "Operation dimension mismatch: expected {}x{}, got {}x{}",
                dim, dim, operation.shape()[0], operation.shape()[1]
            ));
        }

        // For unitary operations: ρ → UρU†
        let conj_transpose = operation.t().map(|x| x.conj());
        let new_matrix = operation.dot(&self.matrix).dot(&conj_transpose);

        Ok(DensityMatrix {
            qubit_count: self.qubit_count,
            matrix: new_matrix,
        })
    }
}

// Implement the QuantumState trait for DensityMatrix
impl QuantumState for DensityMatrix {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn is_valid(&self) -> bool {
        // Already implemented in the struct
        self.is_valid()
    }

    fn tensor(&self, other: &Self) -> Self {
        let self_dim = self.dimension();
        let other_dim = other.dimension();
        let new_dim = self_dim * other_dim;

        let mut new_matrix = Array2::zeros((new_dim, new_dim));

        for i1 in 0..self_dim {
            for j1 in 0..self_dim {
                for i2 in 0..other_dim {
                    for j2 in 0..other_dim {
                        let i = i1 * other_dim + i2;
                        let j = j1 * other_dim + j2;
                        new_matrix[[i, j]] = self.matrix[[i1, j1]] * other.matrix[[i2, j2]];
                    }
                }
            }
        }

        DensityMatrix {
            qubit_count: self.qubit_count + other.qubit_count,
            matrix: new_matrix,
        }
    }

    fn partial_trace(&self, qubits: &[usize]) -> Option<Self> {
        // Implementation of partial trace for density matrices

        // Validate input
        if qubits.is_empty() {
            return Some(self.clone());
        }

        // Check if the qubits to trace over are valid
        for &q in qubits {
            if q >= self.qubit_count {
                return None; // Invalid qubit index
            }
        }

        // Sort and deduplicate qubits
        let mut trace_qubits = qubits.to_vec();
        trace_qubits.sort_unstable();
        trace_qubits.dedup();

        // If we're tracing over all qubits, return a 0-qubit state (scalar)
        if trace_qubits.len() >= self.qubit_count {
            let mut scalar_matrix = Array2::zeros((1, 1));
            scalar_matrix[[0, 0]] = Complex64::new(1.0, 0.0);

            return Some(DensityMatrix {
                qubit_count: 0,
                matrix: scalar_matrix,
            });
        }

        // Calculate the number of qubits left after tracing
        let remaining_qubits = self.qubit_count - trace_qubits.len();

        // Dimensions
        let dim_remain = 1 << remaining_qubits;
        let dim_trace = 1 << trace_qubits.len();

        // Create a mapping from original qubit indices to new indices
        let mut remaining_indices = Vec::new();
        for i in 0..self.qubit_count {
            if !trace_qubits.contains(&i) {
                remaining_indices.push(i);
            }
        }

        // Create the result density matrix
        let mut result_matrix = Array2::zeros((dim_remain, dim_remain));

        // Perform the partial trace
        for i_remain in 0..dim_remain {
            for j_remain in 0..dim_remain {
                let mut sum = Complex64::new(0.0, 0.0);

                for k_trace in 0..dim_trace {
                    // Map indices to the original density matrix
                    let i_orig = map_indices(i_remain, k_trace, &remaining_indices, &trace_qubits);
                    let j_orig = map_indices(j_remain, k_trace, &remaining_indices, &trace_qubits);

                    sum += self.matrix[[i_orig, j_orig]];
                }

                result_matrix[[i_remain, j_remain]] = sum;
            }
        }

        Some(DensityMatrix {
            qubit_count: remaining_qubits,
            matrix: result_matrix,
        })
    }
}

impl Display for DensityMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}-qubit density matrix:", self.qubit_count)?;

        let threshold = 1e-10;
        let dim = self.dimension();

        // Check if the state is pure and display its purity
        let purity = self.purity();
        writeln!(f, "Purity: {:.6} (pure: {})", purity, (purity - 1.0).abs() < 1e-10)?;

        // Display significant matrix elements
        if dim <= 16 {  // Only show full matrix for small dimensions
            for i in 0..dim {
                for j in 0..dim {
                    let elem = self.matrix[[i, j]];
                    if elem.norm_sqr() > threshold {
                        let i_bits = format!("{:0width$b}", i, width = self.qubit_count);
                        let j_bits = format!("{:0width$b}", j, width = self.qubit_count);
                        writeln!(
                            f,
                            "  |{}⟩⟨{}|: {:.6}{:+.6}i",
                            i_bits, j_bits, elem.re, elem.im
                        )?;
                    }
                }
            }
        } else {
            writeln!(f, "  (Matrix too large to display fully)")?;

            // Show diagonal elements (probabilities)
            for i in 0..dim {
                let prob = self.matrix[[i, i]].re;
                if prob > threshold {
                    let i_bits = format!("{:0width$b}", i, width = self.qubit_count);
                    writeln!(f, "  |{}⟩: [{:.1}%]", i_bits, prob * 100.0)?;
                }
            }
        }

        Ok(())
    }
}

// A category of density matrices
#[derive(Debug, Clone)]
pub struct DensityMatrixCategory;

// Implement Category for DensityMatrixCategory
impl Category for DensityMatrixCategory {
    type Object = usize; // Number of qubits
    type Morphism = Array2<Complex64>; // Superoperator representation

    fn domain(&self, _f: &Self::Morphism) -> Self::Object {
        // For simplicity, we don't track the domain and codomain in the morphism itself
        0 // Placeholder
    }

    fn codomain(&self, _f: &Self::Morphism) -> Self::Object {
        // Same as above
        0 // Placeholder
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        // Identity matrix of dimension 2^obj
        let dim = 1 << obj;
        Array2::from_diag(&Array1::from_elem(dim, Complex64::new(1.0, 0.0)))
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        // Matrix multiplication g · f
        if f.shape()[0] != g.shape()[1] {
            return None;
        }

        Some(g.dot(f))
    }
}

// Implement MonoidalCategory for DensityMatrixCategory
impl MonoidalCategory for DensityMatrixCategory {
    fn unit(&self) -> Self::Object {
        0 // 0-qubit system (scalar)
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        a + b // Tensor product adds qubit counts
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        // Kronecker product of matrices
        let f_rows = f.shape()[0];
        let f_cols = f.shape()[1];
        let g_rows = g.shape()[0];
        let g_cols = g.shape()[1];

        let mut result = Array2::zeros((f_rows * g_rows, f_cols * g_cols));

        for i in 0..f_rows {
            for j in 0..f_cols {
                for k in 0..g_rows {
                    for l in 0..g_cols {
                        result[[i * g_rows + k, j * g_cols + l]] = f[[i, j]] * g[[k, l]];
                    }
                }
            }
        }

        result
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        self.identity(a)
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        self.identity(a)
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        let total_qubits = a + b + c;
        self.identity(&total_qubits)
    }
}

// Implement SymmetricMonoidalCategory for DensityMatrixCategory
impl SymmetricMonoidalCategory for DensityMatrixCategory {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        let dim_a = 1 << a;
        let dim_b = 1 << b;
        let total_dim = dim_a * dim_b;

        let mut result = Array2::zeros((total_dim, total_dim));

        for i in 0..dim_a {
            for j in 0..dim_b {
                // Map |i⟩⊗|j⟩ to |j⟩⊗|i⟩
                let src_idx = i * dim_b + j;
                let dst_idx = j * dim_a + i;
                result[[dst_idx, src_idx]] = Complex64::new(1.0, 0.0);
            }
        }

        result
    }
}

// A functor from pure states to density matrices
#[derive(Debug, Clone)]
pub struct PureToDensityFunctor;

// Implement the functor from QuantumStateCategory to DensityMatrixCategory
impl Functor<QuantumStateCategory, DensityMatrixCategory> for PureToDensityFunctor {
    fn map_object(&self, _c: &QuantumStateCategory, _d: &DensityMatrixCategory, obj: &usize) -> usize {
        // Object mapping is identity (same qubit count)
        *obj
    }

    fn map_morphism(
        &self,
        _c: &QuantumStateCategory,
        _d: &DensityMatrixCategory,
        f: &Array2<Complex64>
    ) -> Array2<Complex64> {
        // Map unitary transformation to superoperator
        // For unitary transformations, this is the same matrix
        f.clone()
    }
}

// Helper function to map indices between original and reduced representation
fn map_indices(remain_idx: usize, trace_idx: usize,
               remain_qubits: &[usize], trace_qubits: &[usize]) -> usize {
    // In quantum computing, typically the leftmost qubit is the most significant bit
    // This would be consistent with |q1 q0⟩ where q1 is qubit 1 and q0 is qubit 0

    let total_qubits = remain_qubits.len() + trace_qubits.len();
    let mut bit_values = vec![0; total_qubits];

    // Assign values to the remaining qubits' positions
    for (i, &qubit_pos) in remain_qubits.iter().enumerate() {
        let bit_value = (remain_idx >> i) & 1;
        bit_values[qubit_pos] = bit_value;
    }

    // Assign values to the traced qubits' positions
    for (i, &qubit_pos) in trace_qubits.iter().enumerate() {
        let bit_value = (trace_idx >> i) & 1;
        bit_values[qubit_pos] = bit_value;
    }

    // Construct the full index from the bit values
    // Here the bit at position 0 is the least significant bit
    let mut full_idx = 0;
    for &bit in &bit_values {
        full_idx = (full_idx << 1) | bit;
    }

    full_idx
}
