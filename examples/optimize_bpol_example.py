"""
Example: Optimizing B_pol parameter for polarization damping

This example demonstrates how to use the getPotentialPol() interface
to optimize the B_pol parameter (Tang-Toennies damping for polarization).

The B_pol parameter controls the damping of induced dipole interactions.
By computing the polarization energy separately, you can optimize B_pol
independently of other force field parameters.

Note: Due to the Feynman-Hellman approximation used in the ADMP implementation,
automatic differentiation gradients with respect to B_pol are not available.
This example uses finite differences instead, which is the recommended approach.
"""

import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp
from dmff import Hamiltonian, NeighborList


def optimize_bpol_example():
    """
    Example of optimizing B_pol using the polarization energy interface.
    """
    
    # Load force field and structure
    rc = 4.0
    H = Hamiltonian('path/to/your/forcefield.xml')  # Should have B_pol in Polarize tags
    pdb = app.PDBFile('path/to/your/structure.pdb')
    
    # Create potential
    potential = H.createPotential(
        pdb.topology, 
        nonbondedMethod=app.CutoffPeriodic, 
        nonbondedCutoff=rc*unit.angstrom, 
        ethresh=5e-4, 
        step_pol=5
    )
    
    # Get the ADMPPmeForce generator
    generator = None
    for gen in H.getGenerators():
        if gen.getName() == "ADMPPmeForce":
            generator = gen
            break
    
    if generator is None:
        print("No ADMPPmeForce found in the force field")
        return
    
    # Get the polarization energy function
    # This computes ONLY the polarization energy (including damping effects)
    potential_pol = generator.getPotentialPol()
    
    # Setup positions and neighbor list
    positions = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    box = jnp.array([a, b, c])
    
    covalent_map = potential.meta["cov_map"]
    nblist = NeighborList(box, 0.4, covalent_map)
    nblist.allocate(positions)
    pairs = nblist.pairs
    
    # ===== Method 1: Compute damping contribution =====
    # Get polarization energy with current B_pol
    energy_with_damping = potential_pol(positions, box, pairs, H.paramset.parameters)
    
    # Get polarization energy without B_pol (set B_pol = 0)
    params_no_damping = H.paramset.parameters.copy()
    params_no_damping["ADMPPmeForce"]["B_pol"] = jnp.zeros_like(
        params_no_damping["ADMPPmeForce"]["B_pol"]
    )
    energy_without_damping = potential_pol(positions, box, pairs, params_no_damping)
    
    # The damping contribution is the difference
    damping_contribution = energy_with_damping - energy_without_damping
    
    print(f"Polarization energy (with damping): {energy_with_damping:.4f} kJ/mol")
    print(f"Polarization energy (no damping):   {energy_without_damping:.4f} kJ/mol")
    print(f"Damping contribution:                {damping_contribution:.4f} kJ/mol")
    
    # ===== Method 2: Optimize B_pol using finite differences =====
    def compute_objective(b_pol_values):
        """
        Objective function for B_pol optimization.
        This could be, for example, fitting to reference energies.
        """
        params = H.paramset.parameters.copy()
        params["ADMPPmeForce"]["B_pol"] = b_pol_values
        energy = potential_pol(positions, box, pairs, params)
        # Your objective function here - e.g., compare to reference
        return energy
    
    # Get current B_pol values
    b_pol_current = H.paramset.parameters["ADMPPmeForce"]["B_pol"]
    print(f"\nCurrent B_pol values: {b_pol_current}")
    
    # Compute finite difference gradient
    epsilon = 0.01  # Small perturbation
    energy_base = compute_objective(b_pol_current)
    
    gradient = jnp.zeros_like(b_pol_current)
    for i in range(len(b_pol_current)):
        b_pol_perturbed = b_pol_current.at[i].add(epsilon)
        energy_perturbed = compute_objective(b_pol_perturbed)
        gradient = gradient.at[i].set((energy_perturbed - energy_base) / epsilon)
    
    print(f"Finite difference gradient: {gradient}")
    
    # ===== Method 3: Use with long-range correction =====
    # If you're doing optimization where you subtract long-range interactions first,
    # you can now add/subtract the polarization damping contribution separately
    
    # 1. Compute total energy
    total_pot = potential.getPotentialFunc()
    total_energy = total_pot(positions, box, pairs, H.paramset.parameters)
    
    # 2. Compute long-range contribution (however you do this)
    # long_range_energy = your_long_range_function(...)
    
    # 3. Subtract long-range, then add back polarization with updated B_pol
    # short_range_base = total_energy - long_range_energy - energy_with_damping
    # optimized_energy = short_range_base + compute_objective(optimized_b_pol)
    
    print("\nOptimization interface is ready to use!")
    print("You can now optimize B_pol independently using finite differences.")


if __name__ == "__main__":
    # This is just a template - you need to provide actual files
    print("This is a template example for B_pol optimization.")
    print("Please modify the file paths and objective function for your use case.")
    # optimize_bpol_example()
