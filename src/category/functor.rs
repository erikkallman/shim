//! Functors and natural transformations
//!
//! This module implements functors, which are mappings between categories
//! that preserve the categorical structure, and natural transformations,
//! which are mappings between functors.

use std::marker::PhantomData;
use super::Category;

/// A functor maps objects and morphisms from one category to another,
/// preserving identity morphisms and composition.
///
/// In quantum computing, functors can represent transformations between
/// different representations of quantum processes.
pub trait Functor<C: Category, D: Category> {
    /// Map an object from category C to category D
    fn map_object(&self, c: &C, d: &D, obj: &C::Object) -> D::Object;

    /// Map a morphism from category C to category D
    fn map_morphism(&self, c: &C, d: &D, f: &C::Morphism) -> D::Morphism;

    /// Check if the functor properly maps domains
    fn preserves_domain(&self, c: &C, d: &D, f: &C::Morphism) -> bool {
        let c_domain = c.domain(f);
        let d_domain = d.domain(&self.map_morphism(c, d, f));
        let mapped_c_domain = self.map_object(c, d, &c_domain);

        mapped_c_domain == d_domain
    }

    /// Check if the functor properly maps codomains
    fn preserves_codomain(&self, c: &C, d: &D, f: &C::Morphism) -> bool {
        let c_codomain = c.codomain(f);
        let d_codomain = d.codomain(&self.map_morphism(c, d, f));
        let mapped_c_codomain = self.map_object(c, d, &c_codomain);

        mapped_c_codomain == d_codomain
    }

    /// Verify that the functor preserves identity
    fn preserves_identity(&self, c: &C, d: &D, obj: &C::Object) -> bool {
        let c_id = c.identity(obj);
        let d_obj = self.map_object(c, d, obj);
        let d_id = d.identity(&d_obj);
        let mapped_c_id = self.map_morphism(c, d, &c_id);

        // Now we can properly check equality since D::Morphism requires PartialEq
        mapped_c_id == d_id
    }

    /// Verify that the functor preserves composition
    fn preserves_composition(
        &self,
        c: &C,
        d: &D,
        f: &C::Morphism,
        g: &C::Morphism
    ) -> bool {
        // First check that f and g can be composed in category C
        if !c.can_compose(f, g) {
            return true; // vacuously true
        }

        // Compute F(g ∘ f)
        if let Some(comp_fg) = c.compose(f, g) {
            let mapped_comp = self.map_morphism(c, d, &comp_fg);

            // Compute F(g) ∘ F(f)
            let mapped_f = self.map_morphism(c, d, f);
            let mapped_g = self.map_morphism(c, d, g);

            // Check that mapped_f and mapped_g can be composed in category D
            if !d.can_compose(&mapped_f, &mapped_g) {
                return false; // Functor failed to preserve composability
            }

            if let Some(comp_mapped) = d.compose(&mapped_f, &mapped_g) {
                // Compare F(g ∘ f) with F(g) ∘ F(f)
                mapped_comp == comp_mapped
            } else {
                false
            }
        } else {
            // This shouldn't happen if can_compose is true
            false
        }
    }

    /// Verify all functor laws at once for a collection of test objects and morphisms
    fn verify_functor_laws(
        &self,
        c: &C,
        d: &D,
        test_objects: &[C::Object],
        test_morphisms: &[C::Morphism]
    ) -> bool {
        // 1. Check identity preservation for all test objects
        let identity_preservation = test_objects.iter().all(|obj|
            self.preserves_identity(c, d, obj)
        );

        // 2. Check composition preservation for all valid pairs of test morphisms
        let mut composition_preservation = true;
        for f in test_morphisms.iter() {
            for g in test_morphisms.iter() {
                if c.can_compose(f, g) {
                    composition_preservation = composition_preservation &&
                        self.preserves_composition(c, d, f, g);
                }
            }
        }

        // 3. Check domain and codomain preservation
        let domain_preservation = test_morphisms.iter().all(|f|
            self.preserves_domain(c, d, f)
        );

        let codomain_preservation = test_morphisms.iter().all(|f|
            self.preserves_codomain(c, d, f)
        );

        identity_preservation && composition_preservation &&
        domain_preservation && codomain_preservation
    }
}

/// An endofunctor is a functor from a category to itself.
///
/// Endofunctors are particularly important for defining monads.
pub trait Endofunctor<C: Category>: Functor<C, C> {}

// Default implementation for any type that implements Functor<C, C>
impl<T, C: Category> Endofunctor<C> for T where T: Functor<C, C> {}

/// A natural transformation is a morphism between functors.
///
/// It consists of a family of morphisms that satisfy the naturality condition.
pub trait NaturalTransformation<F, G, C: Category, D: Category>
where
    F: Functor<C, D>,
    G: Functor<C, D>,
{
    /// The component of the natural transformation at object A
    fn component(
        &self,
        c: &C,
        d: &D,
        f: &F,
        g: &G,
        obj: &C::Object
    ) -> D::Morphism;

    /// Verify the naturality condition for a given morphism
    fn is_natural(
        &self,
        c: &C,
        d: &D,
        f: &F,
        g: &G,
        morphism: &C::Morphism
    ) -> bool {
        // The naturality condition states that for any morphism h: A → B in C,
        // the following diagram commutes:
        //
        //     F(A) ---F(h)---> F(B)
        //      |                |
        //  η_A |                | η_B
        //      v                v
        //     G(A) ---G(h)---> G(B)
        //
        // That is, G(h) ∘ η_A = η_B ∘ F(h)

        let a = &c.domain(morphism);
        let b = &c.codomain(morphism);

        let fa = f.map_object(c, d, a);
        let fb = f.map_object(c, d, b);
        let ga = g.map_object(c, d, a);
        let gb = g.map_object(c, d, b);

        let fh = f.map_morphism(c, d, morphism);
        let gh = g.map_morphism(c, d, morphism);

        let eta_a = self.component(c, d, f, g, a);
        let eta_b = self.component(c, d, f, g, b);

        // Verify domain and codomain are correct for components
        let eta_a_domain_ok = d.domain(&eta_a) == fa;
        let eta_a_codomain_ok = d.codomain(&eta_a) == ga;
        let eta_b_domain_ok = d.domain(&eta_b) == fb;
        let eta_b_codomain_ok = d.codomain(&eta_b) == gb;

        if !eta_a_domain_ok || !eta_a_codomain_ok || !eta_b_domain_ok || !eta_b_codomain_ok {
            return false;
        }

        // We need to check that G(h) ∘ η_A = η_B ∘ F(h)
        if let Some(lhs) = d.compose(&gh, &eta_a) {
            if let Some(rhs) = d.compose(&eta_b, &fh) {
                // Now we can compare the morphisms directly
                return lhs == rhs;
            }
        }

        // If we can't compose, there's a problem with the naturality condition
        false
    }

    /// Verify the naturality condition for all morphisms in a test set
    fn verify_naturality(
        &self,
        c: &C,
        d: &D,
        f: &F,
        g: &G,
        test_morphisms: &[C::Morphism]
    ) -> bool {
        test_morphisms.iter().all(|morphism| {
            self.is_natural(c, d, f, g, morphism)
        })
    }

    /// Verify that the components have the correct domains and codomains
    fn has_valid_components(
        &self,
        c: &C,
        d: &D,
        f: &F,
        g: &G,
        test_objects: &[C::Object]
    ) -> bool {
        test_objects.iter().all(|obj| {
            let fa = f.map_object(c, d, obj);
            let ga = g.map_object(c, d, obj);
            let eta = self.component(c, d, f, g, obj);

            d.domain(&eta) == fa && d.codomain(&eta) == ga
        })
    }
}

/// A type-safe implementation of the identity functor
pub struct IdentityFunctor<C: Category> {
    _phantom: PhantomData<C>,
}

impl<C: Category> Default for IdentityFunctor<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Category> IdentityFunctor<C> {
    pub fn new() -> Self {
        IdentityFunctor { _phantom: PhantomData }
    }
}

impl<C: Category> Functor<C, C> for IdentityFunctor<C> {
    fn map_object(&self, _c: &C, _d: &C, obj: &C::Object) -> C::Object {
        obj.clone()
    }

    fn map_morphism(&self, _c: &C, _d: &C, f: &C::Morphism) -> C::Morphism {
        f.clone()
    }
}

/// A type-safe implementation of the constant functor
pub struct ConstantFunctor<C: Category, D: Category> {
    object: D::Object,
    morphism: D::Morphism,
    _phantom_c: PhantomData<C>,
}

impl<C: Category, D: Category> ConstantFunctor<C, D> {
    pub fn new(d: &D, object: D::Object) -> Self {
        let morphism = d.identity(&object);
        ConstantFunctor {
            object,
            morphism,
            _phantom_c: PhantomData,
        }
    }
}

impl<C: Category, D: Category> Functor<C, D> for ConstantFunctor<C, D> {
    fn map_object(&self, _c: &C, _d: &D, _obj: &C::Object) -> D::Object {
        self.object.clone()
    }

    fn map_morphism(&self, _c: &C, _d: &D, _f: &C::Morphism) -> D::Morphism {
        self.morphism.clone()
    }
}

/// A type-safe implementation of the identity natural transformation
pub struct IdentityNaturalTransformation<F, C: Category, D: Category>
where
    F: Functor<C, D>,
{
    _phantom_f: PhantomData<F>,
    _phantom_c: PhantomData<C>,
    _phantom_d: PhantomData<D>,
}

impl<F, C: Category, D: Category> Default for IdentityNaturalTransformation<F, C, D>
where
    F: Functor<C, D>,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, C: Category, D: Category> IdentityNaturalTransformation<F, C, D>
where
    F: Functor<C, D>,
{
    pub fn new() -> Self {
        IdentityNaturalTransformation {
            _phantom_f: PhantomData,
            _phantom_c: PhantomData,
            _phantom_d: PhantomData,
        }
    }
}

impl<F, C: Category, D: Category> NaturalTransformation<F, F, C, D>
    for IdentityNaturalTransformation<F, C, D>
where
    F: Functor<C, D>,
{
    fn component(
        &self,
        _c: &C,
        d: &D,
        f: &F,
        _g: &F,
        obj: &C::Object
    ) -> D::Morphism {
        let f_obj = f.map_object(_c, d, obj);
        d.identity(&f_obj)
    }
}
