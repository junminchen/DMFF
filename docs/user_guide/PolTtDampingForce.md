# PolTtDampingForce

## Overview

`PolTtDampingForce` is a new force component in DMFF that provides an independent interface for polarization damping interactions. This force allows users to optimize polarization damping parameters (B_pol) separately from the main `ADMPPmeForce`, which is particularly useful when fitting force field parameters.

## Background

In the ADMP (Automatic Differentiable Multipolar Polarizable) force field, polarization damping is integrated into the `ADMPPmeForce` using Thole-type damping functions. However, during parameter optimization, it can be challenging to adjust the polarization damping parameters independently, especially when working with short-range (SR) components after subtracting long-range interactions.

The `PolTtDampingForce` addresses this issue by:
- Exposing polarization damping as a separate, optimizable force component
- Using Tang-Toennies (TT) style damping similar to `QqTtDampingForce`
- Allowing independent tuning of B_pol parameters

## Mathematical Form

The polarization damping energy is computed as:

```
E_pol = Σ_ij f_damp(r_ij) * sqrt(pol_i * pol_j) / r_ij^3
```

where:
- `f_damp(r) = 1 - exp(-B*r) * (1 + B*r + 0.5*(B*r)^2)` is the Tang-Toennies damping function
- `B = sqrt(B_i * B_j)` is the combined damping parameter
- `pol_i` and `pol_j` are the atomic polarizabilities
- The factor includes the dielectric constant for proper units

## XML Format

To use `PolTtDampingForce` in your force field XML file:

```xml
<PolTtDampingForce
    pScale12="0.00" pScale13="0.00" pScale14="1.00" pScale15="1.00" pScale16="1.00">
    <Atom type="1" B="3.977508e+01" Pol="1.072970e-03"/>
    <Atom type="2" B="4.596271e+01" Pol="3.680091e-04"/>
    <Atom type="3" B="4.637414e+01" Pol="6.192140e-04"/>
</PolTtDampingForce>
```

### Parameters

- **pScale12, pScale13, pScale14, etc.**: Scaling factors for 1-2, 1-3, 1-4 bonded pairs, etc.
  - Typically set to 0.00 for bonded pairs (no polarization damping)
  - Set to 1.00 for non-bonded pairs (full polarization damping)

- **Atom attributes**:
  - `type` or `class`: Atom type identifier
  - `B`: Damping parameter in nm^-1 (will be converted internally to Å^-1)
  - `Pol`: Polarizability in nm^3 (will be converted internally to Å^3)

## Usage Example

### Python Code

```python
import openmm.app as app
import openmm.unit as unit
from dmff import Hamiltonian, NeighborList
from dmff.api import DMFFTopology
import jax.numpy as jnp

# Load force field and structure
pdb = app.PDBFile('system.pdb')
H = Hamiltonian('forcefield.xml')  # Contains PolTtDampingForce

# Create potentials
rc = 1.2  # cutoff in nm
pots = H.createPotential(
    pdb.topology, 
    nonbondedCutoff=rc*unit.nanometer, 
    nonbondedMethod=app.CutoffPeriodic,
)

# Access the polarization damping potential
pot_pol = pots.dmff_potentials['PolTtDampingForce']

# Build neighbor list
dmfftop = DMFFTopology(from_top=pdb.topology)
covalent_map = dmfftop.buildCovMat()

pos = jnp.array(pdb.positions._value)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)

# ... create neighbor list and pairs ...

# Calculate energy
E_pol = pot_pol(pos, box, pairs, H.paramset)
print(f"Polarization damping energy: {E_pol} kJ/mol")
```

### Parameter Optimization

The main advantage of `PolTtDampingForce` is that B and Pol parameters can be optimized independently:

```python
# Access and modify parameters
params = H.paramset.parameters['PolTtDampingForce']
B_params = params['B']  # Damping parameters
Pol_params = params['Pol']  # Polarizabilities

# Optimize these parameters using your preferred method
# (gradient descent, least squares, etc.)
```

## Relationship to Other Forces

- **ADMPPmeForce**: Contains full electrostatic and polarization calculations including long-range effects. The polarization damping in ADMPPmeForce uses Thole damping.

- **QqTtDampingForce**: Similar in structure but for charge-charge interactions with Tang-Toennies damping.

- **SlaterSrPolForce**: Short-range polarization term using Slater-type functions.

The `PolTtDampingForce` can be used:
1. **Standalone**: For parameter optimization with Tang-Toennies style damping
2. **With ADMPPmeForce**: As a correction term when fitting B_pol separately
3. **With SlaterSrPolForce**: As an alternative damping scheme

## Implementation Details

The force is implemented using:
- **Kernel**: `TT_damping_pol_kernel` in `dmff/admp/pairwise.py`
- **Generator**: `PolTtDampingGenerator` in `dmff/generators/admp.py`
- **Registration**: Automatically registered in `_DMFFGenerators`

The energy calculation is fully differentiable with respect to:
- Atomic positions (for forces)
- B parameters (for optimization)
- Pol parameters (for optimization)

## Notes

1. The energy from this force is **negative** (attractive) due to induced polarization effects.
2. Units are automatically converted: input in nm/nm^3, internal calculations in Å/Å^3.
3. The damping function ensures the interaction goes to zero at short distances, preventing singularities.
4. The force inherits the efficient JAX-based implementation for GPU acceleration.

## See Also

- [ADMPPmeForce](4.2ADMPPmeForce.md)
- [ADMP Force Field Module](4.2ADMPPmeForce.md)
- [Optimization Guide](4.5Optimization.md)
