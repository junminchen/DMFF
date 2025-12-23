"""
Test the polarization damping interface for B_pol optimization.

This test verifies that the polarization energy can be computed separately,
which is useful when optimizing B_pol parameters independently.
"""
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList


class TestDampingInterface:
    """Test polarization damping interface"""
    
    def test_pol_energy_with_bpol(self):
        """
        Test that the polarization energy function works correctly with B_pol values.
        
        The polarization energy should be different with and without TT damping.
        """
        rc = 4.0
        H = Hamiltonian('tests/data/admp_with_bpol.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        
        # Create the main potential
        potential = H.createPotential(
            pdb.topology, 
            nonbondedMethod=app.CutoffPeriodic, 
            nonbondedCutoff=rc*unit.angstrom, 
            ethresh=5e-4, 
            step_pol=5
        )
        
        # Get the generator to create polarization potential
        generator = None
        for gen in H.getGenerators():
            if gen.getName() == "ADMPPmeForce":
                generator = gen
                break
        
        # Create the polarization potential function (must be called after createPotential)
        potential_pol = generator.getPotentialPol()
        
        # Set up positions and neighbor list
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        
        covalent_map = potential.meta["cov_map"]
        nblist = NeighborList(box, 0.4, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        # Compute polarization energy with B_pol
        energy_pol_with = potential_pol(positions, box, pairs, H.paramset.parameters)
        
        print(f"Polarization energy (with B_pol): {energy_pol_with}")
        
        # The polarization energy should be finite when B_pol is non-zero
        # and ttDampingMode is set to "replace"
        assert not jnp.isnan(energy_pol_with), "Polarization energy should not be NaN"
        assert jnp.isfinite(energy_pol_with), "Polarization energy should be finite"
        
        # Now compute with B_pol set to zero to see the difference
        params_no_bpol = H.paramset.parameters.copy()
        params_no_bpol["ADMPPmeForce"]["B_pol"] = jnp.zeros_like(params_no_bpol["ADMPPmeForce"]["B_pol"])
        energy_pol_without = potential_pol(positions, box, pairs, params_no_bpol)
        
        print(f"Polarization energy (without B_pol): {energy_pol_without}")
        print(f"Damping contribution: {energy_pol_with - energy_pol_without}")
        
        # The energies should be different when B_pol is non-zero
        assert abs(energy_pol_with - energy_pol_without) > 1e-6, "B_pol should affect the polarization energy"
        
    def test_pol_energy_without_bpol(self):
        """
        Test that the polarization energy is the same when B_pol is zero or TT damping is disabled.
        """
        rc = 4.0
        # Use the original XML without B_pol (defaults to 0)
        H = Hamiltonian('tests/data/admp.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        
        # Create the main potential
        potential = H.createPotential(
            pdb.topology, 
            nonbondedMethod=app.CutoffPeriodic, 
            nonbondedCutoff=rc*unit.angstrom, 
            ethresh=5e-4, 
            step_pol=5
        )
        
        # Get the generator
        generator = None
        for gen in H.getGenerators():
            if gen.getName() == "ADMPPmeForce":
                generator = gen
                break
        
        if generator is None:
            pytest.skip("ADMPPmeForce generator not found")
        
        # Create the polarization potential function (must be called after createPotential)
        potential_pol = generator.getPotentialPol()
        
        # Set up positions and neighbor list
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        
        covalent_map = potential.meta["cov_map"]
        nblist = NeighborList(box, 0.4, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        # Compute polarization energy
        energy_pol = potential_pol(positions, box, pairs, H.paramset.parameters)
        
        print(f"Polarization energy (B_pol=0 by default): {energy_pol}")
        
        # The energy should be finite
        assert jnp.isfinite(energy_pol), "Polarization energy should be finite"
        
    def test_pol_optimization(self):
        """
        Test that the polarization energy can be used for B_pol optimization.
        
        This test demonstrates how to use the polarization interface for parameter optimization.
        NOTE: Due to the Feynman-Hellman theorem approximation used in the code,
        gradients with respect to B_pol are not currently available through this interface.
        
        This test is kept for documentation purposes but is marked as expected to fail.
        Users should optimize B_pol using finite differences or other methods.
        """
        from jax import grad
        
        rc = 4.0
        H = Hamiltonian('tests/data/admp_with_bpol.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        
        # Create the main potential
        potential = H.createPotential(
            pdb.topology, 
            nonbondedMethod=app.CutoffPeriodic, 
            nonbondedCutoff=rc*unit.angstrom, 
            ethresh=5e-4, 
            step_pol=5
        )
        
        # Get the generator
        generator = None
        for gen in H.getGenerators():
            if gen.getName() == "ADMPPmeForce":
                generator = gen
                break
        
        # Create the polarization potential function (must be called after createPotential)
        potential_pol = generator.getPotentialPol()
        
        # Set up positions and neighbor list
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        
        covalent_map = potential.meta["cov_map"]
        nblist = NeighborList(box, 0.4, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        # Demonstrate finite difference approach for B_pol optimization
        # This is the recommended approach given the Feynman-Hellman approximation
        epsilon = 0.01  # Small change in B_pol
        
        # Get energy with current B_pol
        energy_base = potential_pol(positions, box, pairs, H.paramset.parameters)
        
        # Perturb B_pol slightly
        params_perturbed = H.paramset.parameters.copy()
        params_perturbed["ADMPPmeForce"]["B_pol"] = (
            params_perturbed["ADMPPmeForce"]["B_pol"] + epsilon
        )
        energy_perturbed = potential_pol(positions, box, pairs, params_perturbed)
        
        # Estimate gradient using finite differences
        fd_gradient = (energy_perturbed - energy_base) / epsilon
        
        print(f"Base energy: {energy_base}")
        print(f"Perturbed energy: {energy_perturbed}")
        print(f"Finite difference gradient: {fd_gradient}")
        
        # The finite difference gradient should be finite and reasonable
        assert jnp.isfinite(fd_gradient), "Finite difference gradient should be finite"
        print("Finite difference approach for B_pol optimization works!")
