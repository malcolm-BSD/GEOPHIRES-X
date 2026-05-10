"""Fluid properties adapter that uses CoolProp when available.

This adapter exposes cp, density, enthalpy functions for common fluids and
falls back to simple constant approximations if CoolProp is not installed.
"""
from typing import Optional

try:
    import CoolProp.CoolProp as CP  # type: ignore

    _HAS_COOLPROP = True
except Exception:  # pragma: no cover - optional dependency
    CP = None  # type: ignore
    _HAS_COOLPROP = False


class FluidPropsAdapter:
    """Adapter to obtain thermophysical properties.

    The adapter attempts to use CoolProp when initialized with use_coolprop=True;
    otherwise it returns fallback constants.
    """

    has_coolprop: bool

    def __init__(self, use_coolprop: bool = True) -> None:
        """Initialize and detect CoolProp availability.

        Parameters
        ----------
        use_coolprop:
            Whether to attempt to use CoolProp. If CoolProp is not installed the
            adapter will fall back to simple approximations.
        """
        self.has_coolprop = bool(use_coolprop and _HAS_COOLPROP)

    def cp(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return specific heat capacity [J/kg/K] at temperature t_c [degC].

        For water returns ~4186 J/kg/K as fallback.
        """
        if self.has_coolprop:
            # CoolProp expects Kelvin and standard names
            try:
                return CP.PropsSI("C", "T", t_c + 273.15, "P", 101325, fluid)
            except Exception:
                pass
        # Fallback approximations
        if fluid is None:
            return 4186.0
        fl = fluid.lower()
        # LiBr-water solution approximate cp
        if "libr" in fl or "libr" in fluid.lower() or "lib" in fl:
            # approximate specific heat capacity for LiBr solution [J/kg/K]
            return 3500.0
        if "nh3" in fl or "ammonia" in fl:
            # ammonia/water mixture cp fallback
            return 4200.0
        # default water
        return 4186.0

    def density(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return density [kg/m3]."""
        if self.has_coolprop:
            try:
                return CP.PropsSI("D", "T", t_c + 273.15, "P", 101325, fluid)
            except Exception:
                pass
        fl = (fluid or "").lower()
        if "libr" in fl or "lib" in fl:
            return 1200.0
        if "nh3" in fl or "ammonia" in fl:
            return 682.0
        # fallback for water at ambient
        return 997.0

    def enthalpy(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return specific enthalpy [J/kg].

        For fallback use cp * T (relative to 0 °C) approximation.
        """
        if self.has_coolprop:
            try:
                # Returns J/kg
                return CP.PropsSI("H", "T", t_c + 273.15, "P", 101325, fluid)
            except Exception:
                pass
        cp_val = self.cp(fluid, t_c, x)
        # approximate enthalpy relative to 0 degC [J/kg]
        return cp_val * (t_c)
