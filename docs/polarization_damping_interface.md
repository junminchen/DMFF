# Polarization Damping Interface for B_pol Optimization

## Overview

The ADMP force field includes Tang-Toennies (TT) damping for polarization interactions through the `B_pol` parameter. This document describes how to use the polarization energy interface to optimize `B_pol` independently of other force field parameters.

## Background

### What is B_pol?

`B_pol` is a parameter that controls Tang-Toennies damping in polarization interactions. It affects how induced dipoles interact with permanent multipoles at short range. The damping function prevents unphysical behavior at short distances.

### TT Damping Modes

The force field supports three TT damping modes (set via `ttDampingMode` in the XML):

1. **"none"** (default): No TT damping, uses only Thole damping
2. **"multiply"**: Multiply TT damping with Thole damping  
3. **"replace"**: Replace Thole damping with TT damping

## Force Field XML Configuration

To use TT damping, add `B_pol` to your `Polarize` tags and set the damping mode:

```xml
<ADMPPmeForce lmax="2" ttDampingMode="replace" ...>
  <Atom type="380" ... />
  <Polarize type="380" 
            polarizabilityXX="0.00088" 
            polarizabilityYY="0.00088" 
            polarizabilityZZ="0.00088" 
            thole="8.0" 
            B_pol="20.0"/>
</ADMPPmeForce>
```

The `B_pol` parameter is in units of nm⁻¹.

## Using the Polarization Energy Interface

### Basic Usage

```python
from dmff import Hamiltonian, NeighborList
import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp

# Load force field
H = Hamiltonian('forcefield.xml')
pdb = app.PDBFile('structure.pdb')

# Create potential (required before using getPotentialPol)
potential = H.createPotential(
    pdb.topology, 
    nonbondedMethod=app.CutoffPeriodic, 
    nonbondedCutoff=4.0*unit.angstrom
)

# Get the ADMPPmeForce generator
generator = None
for gen in H.getGenerators():
    if gen.getName() == "ADMPPmeForce":
        generator = gen
        break

# Get the polarization energy function
potential_pol = generator.getPotentialPol()

# Setup positions and neighbor list
positions = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
box = jnp.array(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
covalent_map = potential.meta["cov_map"]
nblist = NeighborList(box, 0.4, covalent_map)
nblist.allocate(positions)
pairs = nblist.pairs

# Compute polarization energy
energy_pol = potential_pol(positions, box, pairs, H.paramset.parameters)
```

### Computing Damping Contribution

To isolate the damping contribution, compute the energy with and without `B_pol`:

```python
# Energy with B_pol
energy_with = potential_pol(positions, box, pairs, H.paramset.parameters)

# Energy without B_pol
params_no_damping = H.paramset.parameters.copy()
params_no_damping["ADMPPmeForce"]["B_pol"] = jnp.zeros_like(
    params_no_damping["ADMPPmeForce"]["B_pol"]
)
energy_without = potential_pol(positions, box, pairs, params_no_damping)

# Damping contribution
damping_energy = energy_with - energy_without
```

### Optimizing B_pol

Due to the Feynman-Hellman approximation used in the code, automatic differentiation gradients with respect to `B_pol` are not available. Instead, use finite differences:

```python
def compute_objective(b_pol_values):
    """Objective function for optimization"""
    params = H.paramset.parameters.copy()
    params["ADMPPmeForce"]["B_pol"] = b_pol_values
    energy = potential_pol(positions, box, pairs, params)
    # Add your fitting objective here (e.g., compare to reference)
    return energy

# Current B_pol values
b_pol_current = H.paramset.parameters["ADMPPmeForce"]["B_pol"]

# Compute finite difference gradient
epsilon = 0.01
gradient = jnp.zeros_like(b_pol_current)
energy_base = compute_objective(b_pol_current)

for i in range(len(b_pol_current)):
    b_pol_perturbed = b_pol_current.at[i].add(epsilon)
    energy_perturbed = compute_objective(b_pol_perturbed)
    gradient = gradient.at[i].set((energy_perturbed - energy_base) / epsilon)

# Use gradient for optimization (e.g., gradient descent, L-BFGS, etc.)
```

## Use Cases

### 1. Independent Optimization

When optimizing force field parameters, you may want to:
1. Optimize short-range parameters with long-range interactions subtracted
2. Then optimize `B_pol` separately to tune polarization damping

The `getPotentialPol()` interface allows you to compute just the polarization energy, making this workflow easier.

### 2. Parameter Fitting

Fit `B_pol` to match reference quantum chemistry calculations or experimental data:

```python
# Reference energy from QM
E_ref = -50.0  # kJ/mol

def objective(b_pol):
    params = H.paramset.parameters.copy()
    params["ADMPPmeForce"]["B_pol"] = b_pol
    E_pol = potential_pol(positions, box, pairs, params)
    # You might also need other energy components
    return (E_pol - E_ref)**2

# Optimize using your preferred method (scipy, jaxopt, etc.)
```

### 3. Sensitivity Analysis

Analyze how `B_pol` affects the energy:

```python
import matplotlib.pyplot as plt

b_pol_values = jnp.linspace(0, 50, 20)  # nm^-1
energies = []

for b_pol in b_pol_values:
    params = H.paramset.parameters.copy()
    params["ADMPPmeForce"]["B_pol"] = jnp.array([b_pol, 0.0])  # Assuming 2 atom types
    energy = potential_pol(positions, box, pairs, params)
    energies.append(energy)

plt.plot(b_pol_values, energies)
plt.xlabel('B_pol (nm^-1)')
plt.ylabel('Polarization Energy (kJ/mol)')
plt.show()
```

## Technical Notes

### Feynman-Hellman Theorem

The implementation uses the Feynman-Hellman theorem to compute forces efficiently:
- During induced dipole optimization, gradients are not tracked
- After optimization, gradients flow through the energy calculation with fixed induced dipoles
- This is accurate for forces and energies, but means `dE/dB_pol` is not available via automatic differentiation

This is why we recommend finite differences for `B_pol` optimization.

### Performance

The `getPotentialPol()` function:
- Optimizes induced dipoles internally (same as regular energy calculation)
- Computes only the polarization energy contribution
- Has similar performance to the full energy calculation
- Can be JIT-compiled for better performance

### Units

- `B_pol` in XML: nm⁻¹
- `B_pol` in internal calculations: Å⁻¹ (automatically converted)
- Positions: nm (input) → Å (internal)
- Energy: kJ/mol

## Example

See `examples/optimize_bpol_example.py` for a complete working example.

## References

- Tang & Toennies damping: J. Chem. Phys. 80, 3726 (1984)
- MPID force field: J. Chem. Theory Comput. 2018, 14, 2, 1442–1455
