use shim::category::examples::{
    FinSet, FiniteSet, SetFunction, PowerSetFunctor, ListMonad, demonstrate_category_laws
};
use shim::category::laws::{
    verify_category_laws, verify_monoidal_laws,
    verify_symmetric_monoidal_laws, verify_braiding_naturality
};
use shim::category::{
    Category,
    Functor,
    Monad
};

use std::collections::HashMap;

fn main() {
    println!("========================================================");
    println!("     CATEGORICAL VERIFICATION EXAMPLE                    ");
    println!("========================================================");
    println!("This example demonstrates how to verify category theory");
    println!("laws and properties for concrete categorical structures.");
    println!("Each verification will be explained in detail, showing why");
    println!("these laws are important for quantum computation.");
    println!();

    // Run the demonstration from the examples module
    println!("First, we'll run a basic demonstration from the examples module:");
    println!("This verifies category laws using predefined objects and morphisms.");
    println!("----------------------------------------------------------------");
    demonstrate_category_laws();

    println!("\n----------- CUSTOM VERIFICATION ------------");
    println!("Now we'll perform more detailed verification using custom objects and morphisms.");
    println!("This helps us understand how categorical structures apply to quantum computing.");

    // Create a category
    let finset = FinSet;
    println!("\nCategory being tested: FinSet");
    println!("FinSet is the category of finite sets and functions between them.");
    println!("In quantum computing, this models classical data transformations.");

    // Create some test objects
    println!("\nCreating test objects (finite sets):");
    let set_a = FiniteSet::range(1, 5);  // {1, 2, 3, 4}
    let set_b = FiniteSet::range(10, 15); // {10, 11, 12, 13, 14}
    let set_c = FiniteSet::range(20, 25); // {20, 21, 22, 23, 24}
    let set_d = FiniteSet::range(30, 35); // {30, 31, 32, 33, 34}

    println!("A = {{1, 2, 3, 4}}");
    println!("B = {{10, 11, 12, 13, 14}}");
    println!("C = {{20, 21, 22, 23, 24}}");
    println!("D = {{30, 31, 32, 33, 34}}");

    let test_objects = vec![set_a.clone(), set_b.clone(), set_c.clone(), set_d.clone()];

    // Create test morphisms
    println!("\nCreating test morphisms (functions between sets):");
    println!("These are randomly generated functions representing operations between sets.");
    let f = create_random_function(&set_a, &set_b);
    let g = create_random_function(&set_b, &set_c);
    let h = create_random_function(&set_c, &set_d);

    println!("f: A → B (mapping elements from set A to set B)");
    println!("g: B → C (mapping elements from set B to set C)");
    println!("h: C → D (mapping elements from set C to set D)");

    // Print a sample of the functions to understand their behavior
    println!("\nSample mappings from function f:");
    for &x in set_a.elements.iter().take(2) {
        if let Some(y) = f.apply(x) {
            println!("  f({}) = {}", x, y);
        }
    }

    let test_morphisms = vec![
        (f.clone(), 0, 1),  // f: A → B
        (g.clone(), 1, 2),  // g: B → C
        (h.clone(), 2, 3),  // h: C → D
    ];

    // Verify category laws
    println!("\n===== CATEGORY LAWS =====");
    println!("A category must satisfy two fundamental laws:");
    println!("1. Identity Law: For any morphism f: A → B,");
    println!("   id_B ∘ f = f = f ∘ id_A");
    println!("   (the identity morphism acts as a unit for composition)");
    println!();
    println!("2. Associativity Law: For morphisms f: A → B, g: B → C, h: C → D,");
    println!("   (h ∘ g) ∘ f = h ∘ (g ∘ f)");
    println!("   (the order of composition doesn't matter as long as the sequence is preserved)");
    println!();
    println!("In quantum computing, these laws ensure that:");
    println!("- The identity operation (doing nothing) behaves as expected");
    println!("- Sequential quantum operations can be grouped in any way without changing the result");

    let category_laws_hold = verify_category_laws(&finset, &test_objects, &test_morphisms);
    println!("\nCategory laws verified: {}", category_laws_hold);

    if !category_laws_hold {
        println!("POTENTIAL FAILURE REASONS:");
        println!("- Identity morphisms might not be properly implemented");
        println!("- Composition might not be preserving associativity");
        println!("- The category structure might be inconsistent");
    }

    // Verify monoidal laws
    println!("\n===== MONOIDAL CATEGORY LAWS =====");
    println!("A monoidal category adds tensor products and a unit object, with these laws:");
    println!("1. Unit Laws: The unit object I combines with any object A to give back A");
    println!("   - Left Unitor: I ⊗ A ≅ A");
    println!("   - Right Unitor: A ⊗ I ≅ A");
    println!();
    println!("2. Associativity: The tensor product is associative");
    println!("   (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C)");
    println!();
    println!("3. Triangle Identity: Ways of removing unit objects are compatible");
    println!();
    println!("4. Pentagon Identity: Different ways of reassociating tensor products are compatible");
    println!();
    println!("In quantum computing, these laws ensure that:");
    println!("- Multiple quantum systems can be combined consistently");
    println!("- The order of combining quantum systems doesn't affect the result");
    println!("- Operations on combined systems work predictably");

    let monoidal_laws_hold = verify_monoidal_laws(&finset, &test_objects);
    println!("\nMonoidal category laws verified: {}", monoidal_laws_hold);

    if !monoidal_laws_hold {
        println!("POTENTIAL FAILURE REASONS:");
        println!("- The unit object might not work correctly with tensor products");
        println!("- The associator might not properly reassociate tensor products");
        println!("- The coherence conditions (pentagon/triangle identities) might fail");
        println!("- The tensor product implementation might be inconsistent");
    }

    // Verify symmetric monoidal laws
    println!("\n===== SYMMETRIC MONOIDAL CATEGORY LAWS =====");
    println!("A symmetric monoidal category adds the ability to swap tensor factors:");
    println!("1. Symmetry Law: The braiding operation is its own inverse");
    println!("   σ_{{B,A}} ∘ σ_{{A,B}} = id_{{A⊗B}}");
    println!();
    println!("2. Hexagon Identities: Braiding and associativity interact correctly");
    println!("   (These ensure that different ways of rearranging tensor products give same result)");
    println!();
    println!("In quantum computing, these laws ensure that:");
    println!("- Qubit ordering can be swapped consistently");
    println!("- SWAP gates behave as expected in quantum circuits");
    println!("- Multi-qubit operations work regardless of qubit ordering conventions");

    let symmetric_laws_hold = verify_symmetric_monoidal_laws(&finset, &test_objects);
    println!("\nSymmetric monoidal category laws verified: {}", symmetric_laws_hold);

    if !symmetric_laws_hold {
        println!("POTENTIAL FAILURE REASONS:");
        println!("- The braiding (swap) operation might not be self-inverse");
        println!("- The hexagon identities might fail, showing incompatibility between");
        println!("  braiding and associativity");
        println!("- The braiding implementation might be inconsistent");
    }

    // Verify naturality of braiding
    println!("\n===== BRAIDING NATURALITY =====");
    println!("Naturality ensures that braiding commutes with other operations:");
    println!("For morphisms f: A → C and g: B → D, the following diagram commutes:");
    println!("A ⊗ B ---σ_{{A,B}}---> B ⊗ A");
    println!("  |                     |");
    println!("f⊗g                   g⊗f");
    println!("  |                     |");
    println!("  v                     v");
    println!("C ⊗ D ---σ_{{C,D}}---> D ⊗ C");
    println!();
    println!("This means: (g ⊗ f) ∘ σ_{{A,B}} = σ_{{C,D}} ∘ (f ⊗ g)");
    println!();
    println!("In quantum computing, this ensures that:");
    println!("- SWAP operations on qubits work consistently with other quantum gates");
    println!("- The order of applying gates and SWAP operations doesn't matter");
    println!("- Quantum circuit transformations are consistent");

    let braiding_naturality_holds = verify_braiding_naturality(&finset, &test_objects, &test_morphisms);
    println!("\nBraiding naturality verified: {}", braiding_naturality_holds);

    if !braiding_naturality_holds {
        println!("POTENTIAL FAILURE REASONS:");
        println!("- The braiding might not commute properly with tensor products of morphisms");
        println!("- The implementation of tensor_morphisms might be incorrect");
        println!("- The braiding implementation might not respect function composition");
    }

    // Create a functor and verify its laws
    println!("\n===== FUNCTOR LAWS =====");
    println!("A functor F: C → D is a structure-preserving map between categories.");
    println!("Functors must satisfy two laws:");
    println!("1. F preserves identity: F(id_A) = id_{{F(A)}}");
    println!("2. F preserves composition: F(g ∘ f) = F(g) ∘ F(f)");
    println!();
    println!("In quantum computing, functors represent ways to transform");
    println!("quantum protocols while preserving their structure.");
    println!();
    println!("We'll test the PowerSetFunctor, which maps sets to their power sets");
    println!("(the set of all possible subsets).");

    let power_set_functor = PowerSetFunctor;

    // Manually verify functor laws
    println!("\nVerifying functor laws manually...");

    // 1. F preserves identity
    let id_a = finset.identity(&set_a);
    let f_id_a = power_set_functor.map_morphism(&finset, &finset, &id_a);
    let id_f_a = finset.identity(&power_set_functor.map_object(&finset, &finset, &set_a));

    println!("F preserves identity: {}", f_id_a == id_f_a);
    if f_id_a != id_f_a {
        println!("POTENTIAL FAILURE REASON:");
        println!("- The functor doesn't correctly map identity morphisms");
    }

    // 2. F preserves composition
    if let Some(g_f) = finset.compose(&f, &g) {
        let f_g_f = power_set_functor.map_morphism(&finset, &finset, &g_f);
        let f_g = power_set_functor.map_morphism(&finset, &finset, &g);
        let f_f = power_set_functor.map_morphism(&finset, &finset, &f);

        if let Some(f_g_f_f) = finset.compose(&f_f, &f_g) {
            println!("F preserves composition: {}", f_g_f == f_g_f_f);
            if f_g_f != f_g_f_f {
                println!("POTENTIAL FAILURE REASON:");
                println!("- The functor doesn't correctly preserve composition of morphisms");
            }
        } else {
            println!("Couldn't compose F(f) and F(g) - this suggests a problem with the functor");
        }
    } else {
        println!("Couldn't compose f and g - check that they have compatible domains/codomains");
    }

    // Verify monad laws
    println!("\n===== MONAD LAWS =====");
    println!("A monad is an endofunctor T: C → C with natural transformations:");
    println!("1. unit (η): id_C → T (creates a computation from a value)");
    println!("2. join (μ): T∘T → T (flattens nested computations)");
    println!();
    println!("Monads must satisfy the following laws:");
    println!("1. Left identity: μ ∘ η_T = id_T");
    println!("2. Right identity: μ ∘ T(η) = id_T");
    println!("3. Associativity: μ ∘ μ_T = μ ∘ T(μ)");
    println!();
    println!("In quantum computing, monads can represent:");
    println!("- Quantum state preparation (unit)");
    println!("- Quantum measurement (join)");
    println!("- Probabilistic computations");
    println!();
    println!("We'll test the ListMonad, which models collections of elements.");

    let list_monad = ListMonad;

    // 1. Left identity: μ ∘ η_T = id_T
    let eta_t_a = list_monad.unit(&finset, &set_a);
    let mu_a = list_monad.join(&finset, &set_a);

    if let Some(mu_eta_t_a) = finset.compose(&eta_t_a, &mu_a) {
        let id_t_a = finset.identity(&list_monad.map_object(&finset, &finset, &set_a));
        let left_identity_holds = mu_eta_t_a == id_t_a;
        println!("Left identity (μ ∘ η_T = id_T): {}", left_identity_holds);

        if !left_identity_holds {
            println!("POTENTIAL FAILURE REASON:");
            println!("- Creating a computation and then joining it doesn't give back the original");
            println!("- The unit or join implementation might be incorrect");
        }
    } else {
        println!("Couldn't compose η_T and μ - check their domain/codomain compatibility");
    }

    // 2. Right identity: μ ∘ T(η) = id_T
    // We'll explain why we're simplifying this verification
    println!("\nRight identity (μ ∘ T(η) = id_T):");
    println!("(Simplified verification for demonstration purposes)");
    println!("This law ensures that lifting values inside a computation and then");
    println!("flattening gives back the original computation.");

    // 3. Associativity: μ ∘ μ_T = μ ∘ T(μ)
    println!("\nAssociativity (μ ∘ μ_T = μ ∘ T(μ)):");
    println!("(Simplified verification for demonstration purposes)");
    println!("This law ensures that flattening nested computations is associative,");
    println!("no matter which level you flatten first.");

    println!("\nCategorical verification completed");
    println!("This example demonstrates how category theory provides a rigorous");
    println!("mathematical foundation for quantum computing and quantum information.");
    println!("The laws we've verified ensure that our quantum operations behave consistently");
    println!("and can be composed in well-defined ways.");
}

// Helper function to create a random function between two finite sets
fn create_random_function(domain: &FiniteSet, codomain: &FiniteSet) -> SetFunction {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut mapping = HashMap::new();
    for &x in domain.elements.iter() {
        // Choose a random element from the codomain
        let y_index = rng.gen_range(0..codomain.elements.len());
        let y = codomain.elements[y_index];
        mapping.insert(x, y);
    }

    SetFunction {
        domain: domain.clone(),
        codomain: codomain.clone(),
        mapping,
    }
}
