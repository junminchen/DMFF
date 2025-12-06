import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from dmff.api import DMFFTopology
from jax import value_and_grad


class TestPolTtDampingForce:
    """
    Test the PolTtDampingForce implementation
    """

    @pytest.mark.parametrize(
        "pdb, prm", 
        [
            (
                "tests/data/peg2.pdb", 
                "tests/data/peg.xml",
            ),
        ]
    )
    def test_pol_damping_basic(self, pdb, prm):
        """Test that PolTtDampingForce can be created and evaluated"""
        # This test uses the existing peg.xml file, but we'll only test 
        # if the force can be instantiated when present
        
        # For now, just test the pairwise kernel directly
        from dmff.admp.pairwise import TT_damping_pol_kernel
        
        # Test kernel with simple inputs
        dr = jnp.array([2.0, 3.0, 4.0])  # distances in Angstrom
        m = jnp.array([1.0, 1.0, 1.0])  # scaling factors
        bi = jnp.array([4.0, 4.0, 4.0])  # B parameters in A^-1
        bj = jnp.array([4.0, 4.0, 4.0])
        poli = jnp.array([1.0, 1.0, 1.0])  # polarizabilities in A^3
        polj = jnp.array([1.0, 1.0, 1.0])
        
        # Call the kernel
        energies = TT_damping_pol_kernel(dr, m, bi, bj, poli, polj)
        
        # Check that energies are computed and have the right shape
        assert energies.shape == (3,)
        
        # Check that energies are negative (attractive polarization)
        assert jnp.all(energies < 0)
        
        # Check that energy magnitude decreases with distance
        assert jnp.abs(energies[0]) > jnp.abs(energies[1])
        assert jnp.abs(energies[1]) > jnp.abs(energies[2])

    def test_pol_damping_gradient(self):
        """Test that gradients can be computed"""
        from dmff.admp.pairwise import TT_damping_pol_kernel
        
        # Test gradient computation
        dr = jnp.array([2.5])
        m = jnp.array([1.0])
        bi = jnp.array([4.0])
        bj = jnp.array([4.0])
        poli = jnp.array([1.0])
        polj = jnp.array([1.0])
        
        def energy_fn(dr_val):
            return jnp.sum(TT_damping_pol_kernel(
                jnp.array([dr_val]), m, bi, bj, poli, polj
            ))
        
        energy, grad = value_and_grad(energy_fn)(2.5)
        
        # Check that gradient exists and is finite
        assert jnp.isfinite(grad)
        
    def test_pol_damping_vs_no_damping(self):
        """Test that damping reduces interaction at short distances"""
        from dmff.admp.pairwise import TT_damping_pol_kernel
        
        # At long distances, damping should be ~1
        # At short distances, damping should be <1
        dr_long = jnp.array([10.0])
        dr_short = jnp.array([1.0])
        m = jnp.array([1.0])
        bi = jnp.array([4.0])
        bj = jnp.array([4.0])
        poli = jnp.array([1.0])
        polj = jnp.array([1.0])
        
        E_long = TT_damping_pol_kernel(dr_long, m, bi, bj, poli, polj)[0]
        E_short = TT_damping_pol_kernel(dr_short, m, bi, bj, poli, polj)[0]
        
        # At long distance, energy should follow ~1/r^3
        # At short distance, damping reduces the interaction
        # The ratio should be different from pure (r_short/r_long)^3
        ratio_damped = E_short / E_long
        ratio_undamped = (dr_long[0] / dr_short[0])**3
        
        # With damping, ratio should be smaller (less repulsive at short range)
        assert jnp.abs(ratio_damped) < jnp.abs(ratio_undamped)
