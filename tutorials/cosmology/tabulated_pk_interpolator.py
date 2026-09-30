"""P(k, z) interpolator/extrapolator for a tabulated P(k, z) grid from any
external source (e.g. dumped to text files as z.txt, k_h.txt, p_k.txt by a
Boltzmann code, emulator, or simulation pipeline).

Expected array convention: `p_k` has shape (len(z), len(k)) - rows are
redshifts, columns are k. If your source's `k`/`p_k` are in h-based units
(h/Mpc and (Mpc/h)^3, a common convention for these codes), convert to
physical units (Mpc^-1, Mpc^3) before passing them in - see the tutorial
notebook for the `k *= h`, `p_k /= h**3` conversion.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class TabulatedMatterPowerInterpolator:
    """Interpolator (linear in z, log-spaced in k) for a tabulated P(k, z) grid.

    By default (`log_pk=True`) interpolates log10(P) over (z, log10(k))
    with `RegularGridInterpolator` - the standard, numerically well-behaved
    way to handle a power spectrum that spans many decades in k, but it
    requires P(k, z) > 0 everywhere (true for the matter power spectrum,
    but *not* for TATT's P_II^EE(k,z): it's a sum of terms with mixed
    signs - e.g. cross terms like C1*C1delta, C1*C2 - and legitimately
    goes negative at some (z, k), unlike P_mm). Pass `log_pk=False` for
    those: it interpolates the raw (signed) P(k,z) values directly instead
    (still on a log10(k) axis, since that's about grid resolution in k,
    independent of sign), at the cost of the usual log-log accuracy
    advantage for steeply-falling, strictly-positive spectra.

    Either way, `bounds_error=False, fill_value=None` makes
    `RegularGridInterpolator` *extrapolate* linearly past the edges of the
    input grid rather than raising or clamping - needed for k or z values a
    downstream Limber/PT integral asks for that fall just past the grid.

    Args:
      z: 1D array of redshifts (as loaded from z.txt), any order.
      k_h: 1D array of wavenumbers in h/Mpc (as loaded from k_h.txt).
      p_k: 2D array, shape (len(z), len(k_h)) (as loaded from p_k.txt) -
        rows are z, columns are k.
      log_pk: whether to log-transform `p_k` before interpolating (default
        True, matching the matter power spectrum's use case). Set False
        for a spectrum that can be negative (e.g. TATT's intrinsic_power).
    """

    def __init__(
        self, z: np.ndarray, k_h: np.ndarray, p_k: np.ndarray, log_pk: bool = True
    ) -> None:
        z = np.asarray(z)
        k_h = np.asarray(k_h)
        p_k = np.asarray(p_k)

        if p_k.shape == (k_h.size, z.size) and p_k.shape != (z.size, k_h.size):
            # defensive: accept the transposed layout too, in case the
            # files were produced/loaded in the other orientation.
            p_k = p_k.T
        if p_k.shape != (z.size, k_h.size):
            raise ValueError(
                f"p_k shape {p_k.shape} doesn't match (len(z), len(k_h)) = "
                f"({z.size}, {k_h.size})."
            )

        # RegularGridInterpolator requires strictly increasing axes.
        z_order = np.argsort(z)
        k_order = np.argsort(k_h)
        self.z = z[z_order]
        self.k_h = k_h[k_order]
        self.p_k = p_k[np.ix_(z_order, k_order)]

        self._log_pk_mode = log_pk
        if log_pk:
            if np.any(self.p_k <= 0):
                raise ValueError(
                    "p_k contains non-positive values; log-log interpolation "
                    "requires P(k,z) > 0 everywhere. Pass log_pk=False for a "
                    "spectrum that can be negative (e.g. TATT's intrinsic_power)."
                )
            values = np.log10(self.p_k)
        else:
            values = self.p_k

        self._log_k = np.log10(self.k_h)

        self._interp = RegularGridInterpolator(
            (self.z, self._log_k),
            values,
            method="linear",       # or "cubic" for a smoother fit
            bounds_error=False,
            fill_value=None,       # None => linear extrapolation outside the grid
        )

    def __call__(self, zs, ks) -> np.ndarray:
        """P(k, z) on the outer product of `zs` and `ks`.

        Args:
          zs: scalar or 1D array of redshifts.
          ks: scalar or 1D array of wavenumbers in h/Mpc.

        Returns:
          2D array of shape (len(zs), len(ks)) - matching cloelib's
          `Perturbations.matter_power_spectrum(zs, ks)` convention.
        """
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        Z, LOGK = np.meshgrid(zs, np.log10(ks), indexing="ij")
        pts = np.column_stack([Z.ravel(), LOGK.ravel()])
        out = self._interp(pts).reshape(len(zs), len(ks))
        return 10.0**out if self._log_pk_mode else out

    # alias matching cloelib's Perturbations protocol, so this can be
    # dropped in wherever a `perturbations.matter_power_spectrum(zs, ks)`
    # call is expected (e.g. PBJTATTLoopComputer, or your own comparison
    # code against cloelib's CAMB/HMemu perturbations objects).
    matter_power_spectrum = __call__


if __name__ == "__main__":
    # --- usage ---
    # z_ext = np.loadtxt(input_path_external_matter_power_lin + 'z.txt')
    # k_ext = np.loadtxt(input_path_external_matter_power_lin + 'k_h.txt')
    # p_k_lin_ext = np.loadtxt(input_path_external_matter_power_lin + 'p_k.txt')
    #
    # pk_interp = TabulatedMatterPowerInterpolator(z_ext, k_ext, p_k_lin_ext)
    # pk_interp(zs=np.array([0.3, 1.2]), ks=np.logspace(-3, 1, 50))  # shape (2, 50)

    # smoke test with synthetic data mimicking the real shapes
    rng = np.random.default_rng(0)
    z = np.linspace(0, 3, 30)
    k = np.logspace(-4, 2, 400)
    Z, K = np.meshgrid(z, k, indexing="ij")
    p_k = 2000 * K ** 0.95 * np.exp(-K) / (1 + Z) ** 2  # toy, always positive

    interp = TabulatedMatterPowerInterpolator(z, k, p_k)
    out_in_range = interp(zs=[0.5, 1.0], ks=[1e-3, 1e-1, 1.0])
    out_extrap = interp(zs=[5.0], ks=[1e-6, 500.0])  # outside both grids
    print("in-range shape:", out_in_range.shape, "values:\n", out_in_range)
    print("extrapolated shape:", out_extrap.shape, "values:\n", out_extrap)
    assert out_in_range.shape == (2, 3)
    assert out_extrap.shape == (1, 2)
    assert np.all(np.isfinite(out_extrap))
    print("OK")
