//! Concrete examples of categorical structures
//!
//! This module provides concrete implementations of categories, functors,
//! natural transformations, and monads to demonstrate and test the
//! categorical abstractions defined in the library.

use std::collections::HashMap;
use std::fmt::Debug;

use super::{
    Category, MonoidalCategory, SymmetricMonoidalCategory,
    Functor, NaturalTransformation, Monad,
    laws::{
        verify_category_laws, verify_monoidal_laws,
        verify_symmetric_monoidal_laws
    }
};

/// A category of finite sets and functions between them.
///
/// This is a concrete implementation of the mathematical category Set,
/// where objects are finite sets and morphisms are functions between them.
#[derive(Debug, Clone)]
pub struct FinSet;

/// A representation of a finite set as an object in FinSet
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteSet {
    /// The elements of the set, represented as a vector of integers
    pub elements: Vec<usize>,
}

impl FiniteSet {
    /// Create a new finite set with the given elements
    pub fn new(elements: Vec<usize>) -> Self {
        // Remove duplicates and sort for canonical representation
        let mut unique_elements = elements.clone();
        unique_elements.sort();
        unique_elements.dedup();

        FiniteSet { elements: unique_elements }
    }

    /// Create an empty set
    pub fn empty() -> Self {
        FiniteSet { elements: Vec::new() }
    }

    /// Create a singleton set containing just one element
    pub fn singleton(element: usize) -> Self {
        FiniteSet { elements: vec![element] }
    }

    /// Create a set containing a range of integers
    pub fn range(start: usize, end: usize) -> Self {
        FiniteSet { elements: (start..end).collect() }
    }

    /// Calculate the cardinality (size) of the set
    pub fn cardinality(&self) -> usize {
        self.elements.len()
    }

    /// Check if the set contains a given element
    pub fn contains(&self, element: usize) -> bool {
        self.elements.contains(&element)
    }
}

/// A morphism in the FinSet category, representing a function between finite sets
#[derive(Debug, Clone, PartialEq)]
pub struct SetFunction {
    /// The domain (source) of the function
    pub domain: FiniteSet,
    /// The codomain (target) of the function
    pub codomain: FiniteSet,
    /// The mapping of elements from domain to codomain, represented as a HashMap
    pub mapping: HashMap<usize, usize>,
}

impl SetFunction {
    /// Create a new function between finite sets
    ///
    /// Returns None if the mapping is not well-defined (i.e., it maps elements
    /// outside the domain or codomain)
    pub fn new(domain: FiniteSet, codomain: FiniteSet, mapping: HashMap<usize, usize>) -> Option<Self> {
        // Check that the mapping is well-defined
        for (&x, &y) in mapping.iter() {
            if !domain.contains(x) || !codomain.contains(y) {
                return None;
            }
        }

        // Check that all elements in the domain are mapped
        for &x in domain.elements.iter() {
            if !mapping.contains_key(&x) {
                return None;
            }
        }

        Some(SetFunction { domain, codomain, mapping })
    }

    /// Apply the function to an element in the domain
    ///
    /// Returns None if the element is not in the domain
    pub fn apply(&self, element: usize) -> Option<usize> {
        self.mapping.get(&element).copied()
    }

    /// Create the identity function on a set
    pub fn identity(set: &FiniteSet) -> Self {
        let mut mapping = HashMap::new();
        for &x in set.elements.iter() {
            mapping.insert(x, x);
        }

        SetFunction {
            domain: set.clone(),
            codomain: set.clone(),
            mapping,
        }
    }
}

impl Category for FinSet {
    type Object = FiniteSet;
    type Morphism = SetFunction;

    fn domain(&self, f: &Self::Morphism) -> Self::Object {
        f.domain.clone()
    }

    fn codomain(&self, f: &Self::Morphism) -> Self::Object {
        f.codomain.clone()
    }

    fn identity(&self, obj: &Self::Object) -> Self::Morphism {
        SetFunction::identity(obj)
    }

    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism> {
        // Check that f's codomain equals g's domain
        if f.codomain != g.domain {
            return None;
        }

        // Compose the functions
        let mut mapping = HashMap::new();
        for &x in f.domain.elements.iter() {
            if let Some(y) = f.apply(x) {
                if let Some(z) = g.apply(y) {
                    mapping.insert(x, z);
                }
            }
        }

        Some(SetFunction {
            domain: f.domain.clone(),
            codomain: g.codomain.clone(),
            mapping,
        })
    }
}

impl MonoidalCategory for FinSet {
    fn unit(&self) -> Self::Object {
        // The monoidal unit is the singleton set {1}
        FiniteSet::singleton(1)
    }

    fn tensor_objects(&self, a: &Self::Object, b: &Self::Object) -> Self::Object {
        // The tensor product of sets is their Cartesian product
        // We'll represent pairs (a,b) as a*N + b where N is some large number
        const N: usize = 10000; // Large enough for our examples

        let mut elements = Vec::new();
        for &x in a.elements.iter() {
            for &y in b.elements.iter() {
                // Encode the pair (x,y) as x*N + y
                elements.push(x * N + y);
            }
        }

        FiniteSet::new(elements)
    }

    fn tensor_morphisms(&self, f: &Self::Morphism, g: &Self::Morphism) -> Self::Morphism {
        // The tensor product of functions applies each function to its respective component
        const N: usize = 10000;

        let domain = self.tensor_objects(&f.domain, &g.domain);
        let codomain = self.tensor_objects(&f.codomain, &g.codomain);

        let mut mapping = HashMap::new();
        for &x1 in f.domain.elements.iter() {
            for &x2 in g.domain.elements.iter() {
                // Compute the encoded pairs
                let x = x1 * N + x2;

                // Apply f and g individually
                if let (Some(y1), Some(y2)) = (f.apply(x1), g.apply(x2)) {
                    // Compute the result pair
                    let y = y1 * N + y2;
                    mapping.insert(x, y);
                }
            }
        }

        // This should always be well-defined for well-defined tensor products
        SetFunction {
            domain,
            codomain,
            mapping,
        }
    }

    fn left_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Left unitor: I ⊗ A → A
        const N: usize = 10000;

        let unit = self.unit();
        let domain = self.tensor_objects(&unit, a);

        let mut mapping = HashMap::new();
        for &x in a.elements.iter() {
            // Map the pair (1,x) to x
            let pair_encoded = N + x;
            mapping.insert(pair_encoded, x);
        }

        SetFunction {
            domain,
            codomain: a.clone(),
            mapping,
        }
    }

    fn right_unitor(&self, a: &Self::Object) -> Self::Morphism {
        // Right unitor: A ⊗ I → A
        const N: usize = 10000;

        let unit = self.unit();
        let domain = self.tensor_objects(a, &unit);

        let mut mapping = HashMap::new();
        for &x in a.elements.iter() {
            // Map the pair (x,1) to x
            let pair_encoded = x * N + 1;
            mapping.insert(pair_encoded, x);
        }

        SetFunction {
            domain,
            codomain: a.clone(),
            mapping,
        }
    }

    fn associator(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Self::Morphism {
        // Associator: (A ⊗ B) ⊗ C → A ⊗ (B ⊗ C)
        const N: usize = 10000;

        let a_tensor_b = self.tensor_objects(a, b);
        let domain = self.tensor_objects(&a_tensor_b, c);

        let b_tensor_c = self.tensor_objects(b, c);
        let codomain = self.tensor_objects(a, &b_tensor_c);

        let mut mapping = HashMap::new();
        for &x in a.elements.iter() {
            for &y in b.elements.iter() {
                for &z in c.elements.iter() {
                    // Compute the encoded value for ((x,y),z)
                    let xy = x * N + y;
                    let xyz_left = xy * N + z;

                    // Compute the encoded value for (x,(y,z))
                    let yz = y * N + z;
                    let xyz_right = x * N + yz;

                    // Map ((x,y),z) to (x,(y,z))
                    mapping.insert(xyz_left, xyz_right);
                }
            }
        }

        SetFunction {
            domain,
            codomain,
            mapping,
        }
    }
}

impl SymmetricMonoidalCategory for FinSet {
    fn braiding(&self, a: &Self::Object, b: &Self::Object) -> Self::Morphism {
        // Braiding: A ⊗ B → B ⊗ A
        // Swaps the components of each pair
        const N: usize = 10000;

        let domain = self.tensor_objects(a, b);
        let codomain = self.tensor_objects(b, a);

        let mut mapping = HashMap::new();
        for &x in a.elements.iter() {
            for &y in b.elements.iter() {
                // Map (x,y) to (y,x)
                let xy = x * N + y;
                let yx = y * N + x;

                mapping.insert(xy, yx);
            }
        }

        SetFunction {
            domain,
            codomain,
            mapping,
        }
    }
}

/// A functor from FinSet to itself that maps each set to its power set
pub struct PowerSetFunctor;

impl Functor<FinSet, FinSet> for PowerSetFunctor {
    fn map_object(&self, _c: &FinSet, _d: &FinSet, obj: &FiniteSet) -> FiniteSet {
        // Map a set A to its power set P(A)
        // We represent subsets as their characteristic functions encoded as integers

        // Calculate the number of subsets: 2^|A|
        let n = obj.cardinality();
        let subset_count = 1 << n;

        // Generate all subsets as integers from 0 to 2^|A| - 1
        let elements: Vec<usize> = (0..subset_count).collect();

        FiniteSet::new(elements)
    }

    fn map_morphism(&self, c: &FinSet, d: &FinSet, f: &SetFunction) -> SetFunction {
        // Map a function f: A → B to P(f): P(A) → P(B)
        // For a subset S of A, P(f)(S) = { f(x) | x ∈ S }

        let domain = self.map_object(c, d, &f.domain);
        let codomain = self.map_object(c, d, &f.codomain);

        let mut mapping = HashMap::new();

        // For each subset of the domain
        for subset_encoded in domain.elements.iter() {
            // Convert the encoded subset back to a set of elements
            let mut subset_elements = Vec::new();
            for (i, &element) in f.domain.elements.iter().enumerate() {
                if (*subset_encoded & (1 << i)) != 0 {
                    subset_elements.push(element);
                }
            }

            // Apply f to each element in the subset
            let mut image_elements = Vec::new();
            for &element in subset_elements.iter() {
                if let Some(image) = f.apply(element) {
                    image_elements.push(image);
                }
            }

            // Convert the image back to an encoded subset
            let mut image_encoded = 0;
            for &element in image_elements.iter() {
                let position = f.codomain.elements.iter().position(|&x| x == element);
                if let Some(pos) = position {
                    image_encoded |= 1 << pos;
                }
            }

            mapping.insert(*subset_encoded, image_encoded);
        }

        SetFunction {
            domain,
            codomain,
            mapping,
        }
    }
}

/// A natural transformation from the identity functor to the power set functor
pub struct ElementsToSingleton;

impl NaturalTransformation<super::IdentityFunctor<FinSet>, PowerSetFunctor, FinSet, FinSet>
    for ElementsToSingleton
{
    fn component(
        &self,
        _c: &FinSet,
        _d: &FinSet,
        _f: &super::IdentityFunctor<FinSet>,
        _g: &PowerSetFunctor,
        obj: &FiniteSet
    ) -> SetFunction {
        // For each element x in A, map it to {x} in P(A)
        let mut mapping = HashMap::new();

        for (i, &element) in obj.elements.iter().enumerate() {
            // The singleton {x} is represented by the integer with only the i-th bit set
            let singleton_encoded = 1 << i;
            mapping.insert(element, singleton_encoded);
        }

        let power_obj = _g.map_object(_c, _d, obj);

        SetFunction {
            domain: obj.clone(),
            codomain: power_obj,
            mapping,
        }
    }
}

/// The list monad on FinSet
pub struct ListMonad;

impl Functor<FinSet, FinSet> for ListMonad {
    fn map_object(&self, _c: &FinSet, _d: &FinSet, obj: &FiniteSet) -> FiniteSet {
        // Map A to List(A), the set of all finite lists of elements from A
        // For simplicity, we'll limit ourselves to lists of length at most 3

        // Calculate the number of possible lists:
        // 1 + |A| + |A|^2 + |A|^3
        let n = obj.cardinality();
        let list_count = 1 + n + n*n + n*n*n;

        // Generate a list of dummy encodings for these lists
        let elements: Vec<usize> = (0..list_count).collect();

        FiniteSet::new(elements)
    }

    fn map_morphism(&self, _c: &FinSet, _d: &FinSet, f: &SetFunction) -> SetFunction {
        // Map f: A → B to List(f): List(A) → List(B)
        // This applies f to each element of a list

        // Simplified implementation for demonstration purposes
        let domain = self.map_object(_c, _d, &f.domain);
        let codomain = self.map_object(_c, _d, &f.codomain);

        // Create a dummy mapping that maintains the list structure
        let mut mapping = HashMap::new();
        for (i, &element) in domain.elements.iter().enumerate() {
            mapping.insert(element, codomain.elements[i % codomain.elements.len()]);
        }

        SetFunction {
            domain,
            codomain,
            mapping,
        }
    }
}

impl Monad<FinSet> for ListMonad {
    fn unit(&self, c: &FinSet, obj: &FiniteSet) -> SetFunction {
        // Unit: A → List(A) maps each element to a singleton list
        let codomain = self.map_object(c, c, obj);

        let mut mapping = HashMap::new();
        for (i, &element) in obj.elements.iter().enumerate() {
            // Map element to a singleton list (encoded as element + offset)
            mapping.insert(element, obj.cardinality() + i);
        }

        SetFunction {
            domain: obj.clone(),
            codomain,
            mapping,
        }
    }

    fn join(&self, c: &FinSet, obj: &FiniteSet) -> SetFunction {
        // Join: List(List(A)) → List(A) concatenates nested lists
        let list_obj = self.map_object(c, c, obj);
        let list_list_obj = self.map_object(c, c, &list_obj);

        // Simplified implementation for demonstration purposes
        let mut mapping = HashMap::new();
        for (i, &element) in list_list_obj.elements.iter().enumerate() {
            // Map nested lists to flattened lists (using a simple modulo operation for demonstration)
            mapping.insert(element, list_obj.elements[i % list_obj.elements.len()]);
        }

        SetFunction {
            domain: list_list_obj,
            codomain: list_obj,
            mapping,
        }
    }
}

/// Example function to demonstrate the use of categories
pub fn demonstrate_category_laws() {
    println!("Demonstrating category laws for FinSet...");

    // Create the category
    let set_category = FinSet;

    // Create some test objects
    let set_a = FiniteSet::range(1, 4);  // {1, 2, 3}
    let set_b = FiniteSet::range(10, 13); // {10, 11, 12}
    let set_c = FiniteSet::range(20, 23); // {20, 21, 22}

    let test_objects = vec![set_a.clone(), set_b.clone(), set_c.clone()];

    // Create some test morphisms
    let mut f_mapping = HashMap::new();
    f_mapping.insert(1, 10);
    f_mapping.insert(2, 11);
    f_mapping.insert(3, 12);

    let f = SetFunction::new(set_a.clone(), set_b.clone(), f_mapping).unwrap();

    let mut g_mapping = HashMap::new();
    g_mapping.insert(10, 20);
    g_mapping.insert(11, 21);
    g_mapping.insert(12, 22);

    let g = SetFunction::new(set_b.clone(), set_c.clone(), g_mapping).unwrap();

    let test_morphisms = vec![(f.clone(), 0, 1), (g.clone(), 1, 2)];

    // Verify category laws
    let laws_satisfied = verify_category_laws(&set_category, &test_objects, &test_morphisms);

    println!("Category laws satisfied: {}", laws_satisfied);

    // Test composition
    if let Some(g_f) = set_category.compose(&f, &g) {
        println!("Composed f and g successfully!");

        // Check that g ∘ f maps 1 to 20
        if let Some(result) = g_f.apply(1) {
            println!("(g ∘ f)(1) = {}", result);
            assert_eq!(result, 20);
        }
    }

    // Verify monoidal category laws
    let monoidal_laws_satisfied = verify_monoidal_laws(&set_category, &test_objects);

    println!("Monoidal category laws satisfied: {}", monoidal_laws_satisfied);

    // Verify symmetric monoidal category laws
    let symmetric_laws_satisfied = verify_symmetric_monoidal_laws(&set_category, &test_objects);

    println!("Symmetric monoidal category laws satisfied: {}", symmetric_laws_satisfied);

    // Demonstrate tensor product
    let tensor_ab = set_category.tensor_objects(&set_a, &set_b);
    println!("Tensor product of A and B has {} elements", tensor_ab.cardinality());

    // Demonstrate functor
    let power_set_functor = PowerSetFunctor;
    let power_a = power_set_functor.map_object(&set_category, &set_category, &set_a);

    println!("Power set of A has {} elements", power_a.cardinality());
    assert_eq!(power_a.cardinality(), 8); // 2^3 = 8 subsets

    // Demonstrate natural transformation
    let elements_to_singleton = ElementsToSingleton;
    let eta_a = elements_to_singleton.component(
        &set_category,
        &set_category,
        &super::IdentityFunctor::new(),
        &power_set_functor,
        &set_a
    );

    println!("Natural transformation maps element 1 to singleton 1");
    if let Some(singleton_1) = eta_a.apply(1) {
        // The singleton {1} should be encoded as the integer with the 0th bit set (since 1 is at index 0)
        assert_eq!(singleton_1, 1);
    }

    // Demonstrate list monad
    let list_monad = ListMonad;
    let list_a = list_monad.map_object(&set_category, &set_category, &set_a);

    println!("List monad maps set A to List(A) with {} elements", list_a.cardinality());

    let _unit_a = list_monad.unit(&set_category, &set_a);
    println!("Unit natural transformation maps elements to singleton lists");
}
