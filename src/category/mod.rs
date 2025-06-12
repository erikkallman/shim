//! Category theory abstractions for quantum computing
//!
//! This module provides the fundamental category theory structures
//! that will be used throughout the categorical quantum machine learning
//! framework.

pub mod monoidal;
pub mod functor;
pub mod monad;
pub mod examples;
pub mod bicategory;

pub use monoidal::{
    Category, MonoidalCategory, SymmetricMonoidalCategory,
    CompactClosedCategory, DaggerCategory, DaggerCompactCategory
};

pub use functor::{
    Functor, NaturalTransformation, Endofunctor,
    IdentityFunctor, ConstantFunctor
};

pub use monad::{Monad, Kleisli};

pub use bicategory::{Bicategory, QuantumCircuitBicategory, CircuitTransformation};

/// Module for verification of category theory laws
pub mod laws {
    pub use super::monoidal::laws::{
        verify_category_laws,
        verify_monoidal_laws,
        verify_symmetric_monoidal_laws,
        verify_braiding_naturality,
        verify_compact_closed_laws,
        verify_dagger_laws
    };

    /// Helper struct that combines all law verifications
    pub struct LawVerifier;

    impl Default for LawVerifier {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LawVerifier {
        pub fn new() -> Self {
            LawVerifier
        }

        /// Verify category laws
        pub fn verify_category<C: super::Category>(
            &self,
            category: &C,
            test_objects: &[C::Object],
            test_morphisms: &[(C::Morphism, usize, usize)]
        ) -> bool {
            super::monoidal::laws::verify_category_laws(category, test_objects, test_morphisms)
        }

        /// Verify monoidal category laws
        pub fn verify_monoidal<C: super::MonoidalCategory>(
            &self,
            category: &C,
            test_objects: &[C::Object]
        ) -> bool {
            super::monoidal::laws::verify_monoidal_laws(category, test_objects)
        }

        /// Verify symmetric monoidal category laws
        pub fn verify_symmetric_monoidal<C: super::SymmetricMonoidalCategory>(
            &self,
            category: &C,
            test_objects: &[C::Object]
        ) -> bool {
            super::monoidal::laws::verify_symmetric_monoidal_laws(category, test_objects)
        }

        /// Verify compact closed category laws
        pub fn verify_compact_closed<C: super::CompactClosedCategory>(
            &self,
            category: &C,
            test_objects: &[C::Object]
        ) -> bool {
            super::monoidal::laws::verify_compact_closed_laws(category, test_objects)
        }

        /// Verify dagger category laws
        pub fn verify_dagger<C: super::DaggerCategory>(
            &self,
            category: &C,
            test_morphisms: &[C::Morphism]
        ) -> bool {
            super::monoidal::laws::verify_dagger_laws(category, test_morphisms)
        }

        /// Verify functor laws
        pub fn verify_functor<F, C, D>(&self,
            functor: &F,
            c: &C,
            d: &D,
            test_objects: &[C::Object],
            test_morphisms: &[C::Morphism]
        ) -> bool
        where
            F: super::Functor<C, D>,
            C: super::Category,
            D: super::Category
        {
            functor.verify_functor_laws(c, d, test_objects, test_morphisms)
        }

        /// Verify monad laws
        pub fn verify_monad<M, C>(&self,
            monad: &M,
            category: &C,
            test_objects: &[C::Object]
        ) -> bool
        where
            M: super::Monad<C>,
            C: super::Category
        {
            monad.verify_all_monad_laws(category, test_objects)
        }
    }
}

/// Re-export commonly used types and traits
pub mod prelude {
    pub use super::{
        Category, MonoidalCategory, SymmetricMonoidalCategory,
        CompactClosedCategory, DaggerCategory, DaggerCompactCategory
    };

    pub use super::{
        Functor, NaturalTransformation, Endofunctor,
        IdentityFunctor, ConstantFunctor
    };

    pub use super::{Monad, Kleisli};

    // Export law verification utilities
    pub use super::laws::LawVerifier;

    // Convenient constructor for law verifier
    pub fn verify_laws() -> super::laws::LawVerifier {
        super::laws::LawVerifier::new()
    }
}
