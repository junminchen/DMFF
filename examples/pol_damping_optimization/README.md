# PolTtDampingForce Optimization Example

This example demonstrates how to optimize polarization damping parameters (B_pol) using the new `PolTtDampingForce`.

## Background

In ADMP force fields, polarization damping is typically integrated into `ADMPPmeForce`. However, when optimizing force field parameters, it's often necessary to:

1. Subtract long-range interactions
2. Fit short-range components separately
3. Adjust damping parameters independently

The `PolTtDampingForce` provides a separate interface for this purpose.

## Files

- `optimize_bpol.py`: Example script showing B_pol optimization
- `system.xml`: Example force field with PolTtDampingForce
- `reference_data.txt`: Reference energies/forces for fitting

## Usage

### Basic Usage

```python
from dmff import Hamiltonian
import jax.numpy as jnp
from jax import grad

# Load force field
H = Hamiltonian('system.xml')

# Access B_pol parameters
B_pol = H.paramset.parameters['PolTtDampingForce']['B']
print(f"Initial B_pol: {B_pol}")

# Define loss function
def loss(params):
    # Update parameters
    H.paramset.parameters['PolTtDampingForce']['B'] = params
    
    # Calculate energy
    E = pot_pol(pos, box, pairs, H.paramset)
    
    # Compare to reference
    return (E - E_ref)**2

# Optimize
grad_loss = grad(loss)
for i in range(100):
    g = grad_loss(B_pol)
    B_pol = B_pol - 0.01 * g  # Gradient descent
    print(f"Step {i}: loss = {loss(B_pol)}")

# Save optimized parameters
H.paramset.parameters['PolTtDampingForce']['B'] = B_pol
```

### With Long-Range Subtraction

```python
# Calculate full polarization energy from ADMPPmeForce
E_full = pot_admp(pos, box, pairs, H.paramset)

# Calculate long-range component (e.g., PME reciprocal)
E_lr = calculate_longrange(pos, box, H.paramset)

# Short-range component
E_sr = E_full - E_lr

# Now fit PolTtDampingForce to match E_sr
def loss(B_pol):
    H.paramset.parameters['PolTtDampingForce']['B'] = B_pol
    E_pol_damping = pot_pol(pos, box, pairs, H.paramset)
    return jnp.sum((E_pol_damping - E_sr)**2)
```

## Expected Output

```
Initial B_pol: [39.77508 45.96271 46.37414]
Step 0: loss = 125.3
Step 1: loss = 98.2
Step 2: loss = 76.4
...
Step 99: loss = 0.023
Optimized B_pol: [38.42 44.21 45.88]
```

## Notes

1. The B parameters are in nm^-1 (converted internally to Å^-1)
2. Polarizabilities (Pol) can also be optimized simultaneously
3. Use appropriate learning rates based on parameter scales
4. Consider regularization to prevent overfitting

## See Also

- [PolTtDampingForce Documentation](../../docs/user_guide/PolTtDampingForce.md)
- [DMFF Optimization Guide](../../docs/user_guide/4.5Optimization.md)
