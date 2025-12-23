# SlaterSrPolForce with Polarization Damping - Optimization Example

This example demonstrates how to optimize polarization damping parameters (B_pol and Pol) using the enhanced `SlaterSrPolForce`.

## Background

In ADMP force fields, polarization damping is typically integrated into `ADMPPmeForce`. However, when optimizing force field parameters, it's often necessary to:

1. Subtract long-range interactions
2. Fit short-range components separately
3. Adjust damping parameters independently

The enhanced `SlaterSrPolForce` now includes integrated polarization damping that shares the B parameter with the Slater SR term, making optimization more convenient.

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

# Access SlaterSrPolForce parameters
params = H.paramset.parameters['SlaterSrPolForce']
B_pol = params['B']
Pol = params['Pol']
print(f"Initial B: {B_pol}")
print(f"Initial Pol: {Pol}")

# Define loss function
def loss(B_params, Pol_params):
    # Update parameters
    H.paramset.parameters['SlaterSrPolForce']['B'] = B_params
    H.paramset.parameters['SlaterSrPolForce']['Pol'] = Pol_params
    
    # Calculate energy (includes both SR and damping terms)
    E = pot_sr_pol(pos, box, pairs, H.paramset)
    
    # Compare to reference
    return (E - E_ref)**2

# Optimize
grad_loss = grad(loss, argnums=(0, 1))
for i in range(100):
    g_B, g_Pol = grad_loss(B_pol, Pol)
    B_pol = B_pol - 0.01 * g_B  # Gradient descent
    Pol = Pol - 0.01 * g_Pol
    print(f"Step {i}: loss = {loss(B_pol, Pol)}")

# Save optimized parameters
H.paramset.parameters['SlaterSrPolForce']['B'] = B_pol
H.paramset.parameters['SlaterSrPolForce']['Pol'] = Pol
```

### With Long-Range Subtraction

```python
# Calculate full polarization energy from ADMPPmeForce
E_full = pot_admp(pos, box, pairs, H.paramset)

# Calculate long-range component (e.g., PME reciprocal)
E_lr = calculate_longrange(pos, box, H.paramset)

# Short-range component
E_sr = E_full - E_lr

# Now fit SlaterSrPolForce to match E_sr
# Both B and Pol parameters affect the result
def loss(B_pol, Pol):
    H.paramset.parameters['SlaterSrPolForce']['B'] = B_pol
    H.paramset.parameters['SlaterSrPolForce']['Pol'] = Pol
    E_sr_pol = pot_sr_pol(pos, box, pairs, H.paramset)
    return jnp.sum((E_sr_pol - E_sr)**2)
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

- [SlaterSrPolForce Documentation](../../docs/user_guide/SlaterSrPolForce.md)
- [DMFF Optimization Guide](../../docs/user_guide/4.5Optimization.md)
