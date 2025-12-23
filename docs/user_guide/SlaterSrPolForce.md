# SlaterSrPolForce with Polarization Damping

## Overview

`SlaterSrPolForce` in DMFF now includes integrated polarization damping functionality. This force combines the original Slater-type short-range polarization term with Tang-Toennies (TT) style damping, allowing users to optimize polarization damping parameters (B_pol) while sharing the same B parameter for both components.

## Background

In the ADMP (Automatic Differentiable Multipolar Polarizable) force field, polarization damping is integrated into the `ADMPPmeForce` using Thole-type damping functions. However, during parameter optimization, it can be challenging to adjust the polarization damping parameters independently, especially when working with short-range (SR) components after subtracting long-range interactions.

The enhanced `SlaterSrPolForce` addresses this issue by:
- Integrating polarization damping directly into the short-range polarization force
- Using Tang-Toennies (TT) style damping for the polarization term
- Sharing the B parameter between the Slater SR term and the damping term
- Allowing optimization of both A, B, and Pol parameters together

## Mathematical Form

The total energy from `SlaterSrPolForce` includes two terms:

**1. Slater SR term (original):**
```
E_sr = -Σ_ij A_i * A_j * P(Br) * exp(-Br)
```
where `P(Br) = 1 + Br + (Br)²/3`

**2. Polarization damping term (new):**
```
E_pol = Σ_ij f_damp(r) * sqrt(pol_i * pol_j) / r³
```
where `f_damp(r) = 1 - exp(-Br) * (1 + Br + 0.5*(Br)²)` is the Tang-Toennies damping function

**Combined:**
```
E_total = E_sr + E_pol
```

Both terms share the same B parameter:
- `B = sqrt(B_i * B_j)` is the combined damping parameter
- `pol_i` and `pol_j` are the atomic polarizabilities  
- The factor includes the dielectric constant for proper units

## XML Format

To use `SlaterSrPolForce` with polarization damping in your force field XML file:

```xml
<SlaterSrPolForce
    mScale12="0.00" mScale13="0.00" mScale14="1.00" mScale15="1.00" mScale16="1.00">
    <Atom type="1" A="1" B="3.977508e+01" Pol="1.072970e-03"/>
    <Atom type="2" A="1" B="4.596271e+01" Pol="3.680091e-04"/>
    <Atom type="3" A="1" B="4.637414e+01" Pol="6.192140e-04"/>
</SlaterSrPolForce>
```

### Parameters

- **mScale12, mScale13, mScale14, etc.**: Scaling factors for 1-2, 1-3, 1-4 bonded pairs, etc.
  - Typically set to 0.00 for bonded pairs
  - Set to 1.00 for non-bonded pairs

- **Atom attributes**:
  - `type` or `class`: Atom type identifier
  - `A`: Amplitude parameter for Slater SR term
  - `B`: Shared damping parameter in nm^-1 (converted internally to Å^-1)
  - `Pol`: Polarizability in nm^3 (converted internally to Å^3) - **optional**, defaults to 0.0

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
H = Hamiltonian('forcefield.xml')  # Contains SlaterSrPolForce with Pol parameters

# Create potentials
rc = 1.2  # cutoff in nm
pots = H.createPotential(
    pdb.topology, 
    nonbondedCutoff=rc*unit.nanometer, 
    nonbondedMethod=app.CutoffPeriodic,
)

# Access the SlaterSrPolForce potential (includes both SR and damping)
pot_sr_pol = pots.dmff_potentials['SlaterSrPolForce']

# Build neighbor list
dmfftop = DMFFTopology(from_top=pdb.topology)
covalent_map = dmfftop.buildCovMat()

pos = jnp.array(pdb.positions._value)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)

# ... create neighbor list and pairs ...

# Calculate combined energy (SR + polarization damping)
E_total = pot_sr_pol(pos, box, pairs, H.paramset)
print(f"Total SlaterSrPol energy: {E_total} kJ/mol")
```

### Parameter Optimization

The main advantage is that A, B, and Pol parameters share the same force and can be optimized together:

```python
# Access and modify parameters
params = H.paramset.parameters['SlaterSrPolForce']
A_params = params['A']    # Amplitude parameters
B_params = params['B']    # Shared damping parameters
Pol_params = params['Pol'] # Polarizabilities

# Optimize these parameters using your preferred method
# (gradient descent, least squares, etc.)
# The B parameter affects both the SR and damping terms
```

## Relationship to Other Forces

- **ADMPPmeForce**: Contains full electrostatic and polarization calculations including long-range effects. The polarization damping in ADMPPmeForce uses Thole damping.

- **QqTtDampingForce**: Similar in structure but for charge-charge interactions with Tang-Toennies damping.

- **SlaterSrEsForce, SlaterSrDispForce, SlaterDhfForce**: Other Slater-type short-range forces that follow the same pattern.

The enhanced `SlaterSrPolForce` can be used:
1. **For parameter optimization**: Optimize A, B, and Pol together with shared B parameter
2. **With ADMPPmeForce**: As a correction term when fitting short-range components after subtracting long-range
3. **Backward compatible**: Pol parameter is optional; if omitted, only the original Slater SR term is computed

## Implementation Details

The force is implemented in:
- **Kernels**: 
  - `slater_sr_kernel` in `dmff/admp/pairwise.py` (original SR term)
  - `TT_damping_pol_kernel` in `dmff/admp/pairwise.py` (damping term)
- **Generator**: `SlaterSrPolGenerator` in `dmff/generators/admp.py`
- **Registration**: Automatically registered as `"SlaterSrPolForce"` in `_DMFFGenerators`

The energy calculation is fully differentiable with respect to:
- Atomic positions (for forces)
- A parameters (for optimization)
- B parameters (for optimization) - affects both SR and damping terms
- Pol parameters (for optimization)

## Notes

1. The energy includes both positive (SR) and negative (damping) contributions.
2. Units are automatically converted: input in nm/nm^3, internal calculations in Å/Å^3.
3. The damping function ensures proper behavior at short distances.
4. The force inherits the efficient JAX-based implementation for GPU acceleration.
5. **Backward compatible**: If `Pol` attribute is not specified in XML, it defaults to 0.0 (no damping term).

## See Also

- [ADMPPmeForce](4.2ADMPPmeForce.md)
- [ADMP Force Field Module](4.2ADMPPmeForce.md)
- [Optimization Guide](4.5Optimization.md)
