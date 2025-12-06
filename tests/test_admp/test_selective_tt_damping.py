"""
Test selective TT damping for electrolyte systems (e.g., Li/Na salts).

This test verifies that TT damping is applied to atom pairs where 
AT LEAST ONE atom has B_pol > 0. This enables selective damping for 
Li/Na containing pairs in electrolyte systems, including:
- Li-Li, Na-Na, Li-Na pairs (both atoms have B_pol > 0)
- Li-solvent, Na-solvent pairs (one atom has B_pol > 0)
Only solvent-solvent pairs (both B_pol = 0) do not have TT damping.
"""

import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff.admp.pme import calc_tt_damping_pol, TT_DAMPING_MODE_NONE, TT_DAMPING_MODE_MULTIPLY, TT_DAMPING_MODE_REPLACE, B_POL_THRESH


class TestSelectiveTTDamping:
    """Test selective TT damping functionality for electrolyte systems."""
    
    def test_both_b_pol_nonzero(self):
        """When both atoms have B_pol > 0, TT damping should be applied (factors < 1)."""
        dr = 3.0  # Angstrom
        b1, b2 = 2.0, 2.0  # Both non-zero
        tt_factors = calc_tt_damping_pol(dr, b1, b2)
        
        # All TT damping factors should be < 1.0 (damping is applied)
        for i, name in enumerate(['tt_c', 'tt_d0', 'tt_d1', 'tt_q0', 'tt_q1', 'tt_o0', 'tt_o1']):
            assert float(tt_factors[i]) < 1.0, f"{name} should be < 1.0 when both B_pol > 0"
    
    def test_one_b_pol_zero_still_has_damping(self):
        """When one atom has B_pol = 0, TT damping SHOULD still be applied (factors < 1).
        
        This is the key behavior for electrolyte systems: Li-solvent and Na-solvent
        pairs should have TT damping applied.
        """
        dr = 3.0  # Angstrom
        b1, b2 = 2.0, 0.0  # One zero (e.g., Li-solvent pair)
        tt_factors = calc_tt_damping_pol(dr, b1, b2)
        
        # All TT damping factors should be < 1.0 (damping IS applied)
        for i, name in enumerate(['tt_c', 'tt_d0', 'tt_d1', 'tt_q0', 'tt_q1', 'tt_o0', 'tt_o1']):
            assert float(tt_factors[i]) < 1.0, f"{name} should be < 1.0 when one B_pol > 0 (Li-solvent pair)"
    
    def test_one_b_pol_zero_reversed_still_has_damping(self):
        """When one atom has B_pol = 0 (reversed order), TT damping SHOULD still be applied."""
        dr = 3.0  # Angstrom
        b1, b2 = 0.0, 2.0  # Reversed order (e.g., solvent-Na pair)
        tt_factors = calc_tt_damping_pol(dr, b1, b2)
        
        # All TT damping factors should be < 1.0 (damping IS applied)
        for i, name in enumerate(['tt_c', 'tt_d0', 'tt_d1', 'tt_q0', 'tt_q1', 'tt_o0', 'tt_o1']):
            assert float(tt_factors[i]) < 1.0, f"{name} should be < 1.0 when one B_pol > 0 (solvent-Na pair)"
    
    def test_both_b_pol_zero_no_damping(self):
        """When both atoms have B_pol = 0, TT damping should NOT be applied (factors = 1).
        
        This is for solvent-solvent pairs where no TT damping is needed.
        """
        dr = 3.0  # Angstrom
        b1, b2 = 0.0, 0.0  # Both zero (solvent-solvent pair)
        tt_factors = calc_tt_damping_pol(dr, b1, b2)
        
        # All TT damping factors should be 1.0 (no damping)
        for i, name in enumerate(['tt_c', 'tt_d0', 'tt_d1', 'tt_q0', 'tt_q1', 'tt_o0', 'tt_o1']):
            npt.assert_almost_equal(float(tt_factors[i]), 1.0, decimal=10,
                                    err_msg=f"{name} should be 1.0 when both B_pol = 0 (solvent-solvent pair)")
    
    def test_selective_damping_li_na_case(self):
        """
        Simulate electrolyte system where Li/Na containing pairs have TT damping.
        
        In this scenario:
        - Li/Na atoms have B_pol > 0 (e.g., 2.0 A^-1)
        - Solvent atoms have B_pol = 0
        
        Expected behavior (TT damping applied to ALL Li/Na containing pairs):
        - Li-Li, Na-Na, Li-Na pairs: TT damping applied (factors < 1)
        - Li-Solvent, Na-Solvent pairs: TT damping applied (factors < 1)
        - Solvent-Solvent pairs: no TT damping (factors = 1)
        """
        dr = 3.0  # Angstrom
        b_li = 2.0  # TT damping parameter for Li
        b_na = 1.8  # TT damping parameter for Na
        b_solvent = 0.0  # No TT damping for solvent
        
        # Li-Li pair: should have damping
        tt_li_li = calc_tt_damping_pol(dr, b_li, b_li)
        assert float(tt_li_li[0]) < 1.0, "Li-Li pair should have TT damping"
        
        # Na-Na pair: should have damping
        tt_na_na = calc_tt_damping_pol(dr, b_na, b_na)
        assert float(tt_na_na[0]) < 1.0, "Na-Na pair should have TT damping"
        
        # Li-Na pair: should have damping
        tt_li_na = calc_tt_damping_pol(dr, b_li, b_na)
        assert float(tt_li_na[0]) < 1.0, "Li-Na pair should have TT damping"
        
        # Li-Solvent pair: should have damping (key change!)
        tt_li_solv = calc_tt_damping_pol(dr, b_li, b_solvent)
        assert float(tt_li_solv[0]) < 1.0, "Li-Solvent pair should have TT damping"
        
        # Na-Solvent pair: should have damping (key change!)
        tt_na_solv = calc_tt_damping_pol(dr, b_na, b_solvent)
        assert float(tt_na_solv[0]) < 1.0, "Na-Solvent pair should have TT damping"
        
        # Solvent-Solvent pair: should NOT have damping
        tt_solv_solv = calc_tt_damping_pol(dr, b_solvent, b_solvent)
        npt.assert_almost_equal(float(tt_solv_solv[0]), 1.0, decimal=10,
                                err_msg="Solvent-Solvent pair should NOT have TT damping")
    
    def test_distance_dependence(self):
        """Verify TT damping factors depend on distance when B_pol > 0."""
        b1, b2 = 2.0, 2.0
        
        tt_close = calc_tt_damping_pol(2.0, b1, b2)  # Close distance
        tt_mid = calc_tt_damping_pol(4.0, b1, b2)    # Medium distance
        tt_far = calc_tt_damping_pol(8.0, b1, b2)    # Far distance
        
        # At close distance, damping should be stronger (factor closer to 0)
        # At far distance, damping should be weaker (factor closer to 1)
        assert float(tt_close[0]) < float(tt_mid[0]) < float(tt_far[0]), \
            "TT damping should decrease (factor increase) with distance"
    
    def test_very_small_b_pol_treated_as_zero(self):
        """Very small B_pol values (< threshold) should be treated as zero."""
        dr = 3.0
        # Use values smaller than B_POL_THRESH
        b_tiny = B_POL_THRESH / 10.0  # Well below threshold
        
        # Both tiny -> should be treated as both zero -> no damping
        tt_factors = calc_tt_damping_pol(dr, b_tiny, b_tiny)
        
        # Should be treated as if both B_pol are zero -> no damping
        for i, name in enumerate(['tt_c', 'tt_d0', 'tt_d1', 'tt_q0', 'tt_q1', 'tt_o0', 'tt_o1']):
            npt.assert_almost_equal(float(tt_factors[i]), 1.0, decimal=5,
                                    err_msg=f"{name} should be ~1.0 for very small B_pol")
    
    def test_ion_solvent_uses_ion_b_pol(self):
        """When one B_pol is zero, verify the non-zero B_pol value is used correctly."""
        dr = 3.0
        b_ion = 2.0
        b_solvent = 0.0
        
        # Ion-solvent pair should use the ion's B_pol value
        tt_ion_solv = calc_tt_damping_pol(dr, b_ion, b_solvent)
        
        # Compare with ion-ion pair using the same B_pol
        # For ion-ion: b = sqrt(b_ion * b_ion) = b_ion
        # For ion-solvent: b = max(b_ion, b_solvent) = b_ion
        tt_ion_ion = calc_tt_damping_pol(dr, b_ion, b_ion)
        
        # Both should give the same damping factors since they use the same effective b
        npt.assert_almost_equal(float(tt_ion_solv[0]), float(tt_ion_ion[0]), decimal=10,
                                err_msg="Ion-solvent should use ion's B_pol value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
