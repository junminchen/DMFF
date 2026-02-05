import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad
from ase.io import read, write
from dmff.mdtools.asetools import *

class TestADMPAPI:
    
    """ Test EANN related generators
    """
    
    @pytest.mark.parametrize(
            "pdb, xml",
            [
                (
                    "tests/data/water_new.pdb",
                    "tests/data/water_eann.xml",
                    )
                ]
            )

    def test_EANN_energy(self, pdb, xml):
        atoms = read(pdb)
        rc = 0.4
        kwargs = {}
        print(pdb)
        atoms.calc = DMFFCalculator(pdb=pdb, 
                                    ff_xml=xml, rc=rc, **kwargs)
        energy = atoms.get_potential_energy()
        print(energy)
        np.testing.assert_almost_equal(energy, -337.4403390507297, decimal=4)


