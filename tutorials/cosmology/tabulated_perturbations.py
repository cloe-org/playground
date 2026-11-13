"""A `cloelib.cosmology.cosmology.Perturbations`-compliant wrapper around
a tabulated `matter_power_lin`/`matter_power_nl`-style P(k, z) grid from
any external source (Boltzmann code, emulator, simulation pipeline, ...).

Lets you drop a tabulated P(k, z) straight into `PositionsTracer`,
`ShearTracer`, `AngularTwoPoint`, `PBJTATTLoopComputer`, etc. in place of an
emulator/Boltzmann-code-backed perturbations object, instead of going
through an intermediate cloelib backend (CAMB, HMcode2020Emu, ...).

Built on top of `TabulatedMatterPowerInterpolator` (log-log interpolation in
(z, log10 k), linear extrapolation past the grid edges) and adds the extra
Perturbations-protocol surface that interpolator alone doesn't cover:
`background`, `matter_power_spectrum_cb`, `growth_factor`, `growth_rate`,
`sigma8_0`, and the informally-required `.k`/`.z` attributes.
"""

import numpy as np
from scipy.integrate import quad

from tabulated_pk_interpolator import TabulatedMatterPowerInterpolator


class TabulatedPerturbations:
    """Perturbations object built directly from a tabulated P(k, z) table.

    Parameters
    ----------
    background : cloelib Background
        The background object used to convert your source's h-units (if
        any) to cloelib's (k in 1/Mpc, P(k) in Mpc^3) convention -- i.e.
        the same one you used to build `k = k_h * background.h` and
        `p_k = p_k_h / background.h**3` in the notebook, if your source
        used h-based units in the first place.
    z, k, p_k : np.ndarray
        Redshift grid, wavenumber grid [Mpc^-1], and P(k, z) [Mpc^3]
        already converted to those units. `p_k` must have shape
        (len(z), len(k)) (same orientation as loaded with `np.loadtxt` in
        the notebook).
    k_cb, p_k_cb : np.ndarray, optional
        Same, but for the cold-dark-matter+baryon-only spectrum, if your
        source provides one separately. If omitted,
        `matter_power_spectrum_cb` falls back to `matter_power_spectrum`
        -- only correct when mnu=0.
    linearperturbations : Perturbations, optional
        The *linear* counterpart of this spectrum, needed only when `p_k`
        is itself nonlinear. Downstream code that needs a linear source
        (e.g. `PBJTATTLoopComputer` for the TATT one-loop kernels) reads it
        off via `getattr(perturbations, "linearperturbations",
        perturbations)`. If omitted, `self.linearperturbations` is set to
        `self` -- correct only if `p_k` is linear.

    Notes
    -----
    `growth_factor`/`growth_rate`/`sigma8_0` are *derived* from the
    tabulated P(k, z) (ratios and a top-hat-window integral), not read from
    a separate growth/sigma8 output your source pipeline may have computed
    independently -- if you want an exact match to that, load it the same
    way as the Pk tables and override these methods/attributes instead.
    """

    def __init__(
        self,
        background,
        z: np.ndarray,
        k: np.ndarray,
        p_k: np.ndarray,
        k_cb: np.ndarray = None,
        p_k_cb: np.ndarray = None,
        linearperturbations=None,
    ) -> None:
        self.background = background
        self.z = np.asarray(z, dtype=float)
        self.k = np.asarray(k, dtype=float)
        self.Pk = np.asarray(p_k, dtype=float)

        self._interp = TabulatedMatterPowerInterpolator(self.z, self.k, self.Pk)

        if k_cb is not None and p_k_cb is not None:
            self._interp_cb = TabulatedMatterPowerInterpolator(
                self.z, np.asarray(k_cb, dtype=float), np.asarray(p_k_cb, dtype=float)
            )
        else:
            self._interp_cb = None

        self.linearperturbations = (
            linearperturbations if linearperturbations is not None else self
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Matter P(k, z), shape (len(zs), len(ks))."""
        return self._interp(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        """CDM+baryon P(k, z) (falls back to total matter if no cb table given)."""
        if self._interp_cb is not None:
            return self._interp_cb(zs, ks)
        return self.matter_power_spectrum(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""D(z, k) = sqrt(P(z, k) / P(0, k)), from the tabulated Pk ratio."""
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        P_z = self.matter_power_spectrum(zs, ks)
        P_0 = self.matter_power_spectrum(np.array([0.0]), ks)[0]
        return np.sqrt(P_z / P_0)

    def growth_rate(self, zs=None, ks=None) -> np.ndarray:
        r"""f(z) = dlnD/dlna, via a small central finite difference in ln a.

        Evaluated at a fixed large-scale k (the smallest tabulated k,
        or the first entry of `ks` if given) since f(z) is scale-independent
        to good approximation on the scales this class is normally used for.
        """
        if zs is None:
            zs = self.z
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        k_lin = np.atleast_1d(ks)[:1] if ks is not None else np.array([self.k.min()])

        h = 1e-3
        a = 1.0 / (1.0 + zs)
        a_hi = np.minimum(a * (1.0 + h), 1.0)
        a_lo = a * (1.0 - h)
        z_hi = 1.0 / a_hi - 1.0
        z_lo = 1.0 / a_lo - 1.0

        D_hi = self.growth_factor(z_hi, k_lin)[:, 0]
        D_lo = self.growth_factor(z_lo, k_lin)[:, 0]
        return (np.log(D_hi) - np.log(D_lo)) / (np.log(a_hi) - np.log(a_lo))

    def sigma8_0(self) -> float:
        r"""sigma8 at z=0, from a top-hat-window integral over the tabulated P(k, 0)."""
        R = 8.0 / self.background.h  # Mpc

        def integrand(k):
            x = k * R
            W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
            Pk = self.matter_power_spectrum(np.array([0.0]), np.array([k]))[0, 0]
            return k**2 * Pk * W**2

        kmin, kmax = self.k.min(), self.k.max()
        integral, _ = quad(integrand, kmin, kmax, limit=200)
        return float(np.sqrt(integral / (2.0 * np.pi**2)))
