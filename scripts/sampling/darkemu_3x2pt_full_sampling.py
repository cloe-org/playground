"""Full sampling for Dark Emulator 3x2pt (GC + GGL + WL) likelihood.

This script performs joint sampling of cosmological parameters and HOD parameters
using the Dark Emulator-based 3x2pt likelihood:
- Galaxy Clustering (GC): w_p from Dark Emulator + HOD
- Galaxy-Galaxy Lensing (GGL): ΔΣ from Dark Emulator + HOD
- Weak Lensing (WL): ξ±(θ) from HMcode2020Emu + ShearTracer + Wigner transform

Sampled parameters:
- Cosmology: Omega_cdm0, As (or sigma8)
- HOD: logMmin, sigma_sq, logM1, alpha

Fixed parameters:
- H0, Omega_b0, ns, w0, wa, Omega_k0 (cosmology)
- kappa, poff, Roff (HOD)

Usage:
    Run from the CLOE repository root, e.g.:
        python playground/scripts/sampling/darkemu_3x2pt_full_sampling.py [--n_live N] [--n_workers N] [--output_dir DIR]
        e.g: python playground/scripts/sampling/darkemu_3x2pt_full_sampling.py --n_live 500 --output_dir playground/results
Expected runtime: ~4-12 hours depending on n_live and n_workers.

Output files (written to output_dir, default playground/results/):
    - checkpoint_3x2pt_full.hdf5 : Nautilus checkpoint (for resuming or inspection)
    - chain_3x2pt_full.npz       : Posterior chain, weights, log L, and derived (Omega_m, sigma8, S8) for corner plots

Corner plot: The command below creates a corner (triangle) plot for Omega_m, sigma8, and S8 from the chain saved in step 1:
python playground/scripts/plotting/corner_plot.py \
  --chain playground/results/chain_3x2pt_full.npz \
  --output playground/results/corner_3x2pt.pdf \
  --params Omega_m sigma8 S8
"""

# Force unbuffered output for progress visibility (skip in Jupyter where stdout has no reconfigure)
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import time
import argparse
from pathlib import Path

# =============================================================================
# Setup paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_DIR = SCRIPT_DIR.parent.parent
CLOE_ROOT = PLAYGROUND_DIR.parent
RESULTS_DIR = PLAYGROUND_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Add CLOE packages to path
import sys
sys.path.insert(0, str(CLOE_ROOT / "cloelib"))
sys.path.insert(0, str(CLOE_ROOT / "cloelike"))
sys.path.insert(0, str(CLOE_ROOT / "dark_emulator_public"))

# =============================================================================
# Imports
# =============================================================================

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelike.EuclidLikelihood_DarkEmu_RealSpace import EuclidLikelihood_DarkEmu_RealSpace
from cloelib.observables.darkemu_hod import DarkEmuHODParameters
from cloelib.cosmology.darkemu_cosmology import DarkEmuHODPerturbations

# Import nautilus sampler
from nautilus import Prior, Sampler


# =============================================================================
# Fiducial Parameters (used for mock data generation and fixed params)
# =============================================================================

# Cosmological parameters
# -----------------------
# Sampled: Omega_cdm0, As (primary parameters affecting clustering & lensing)
# Fixed parameters and reasons:
FIDUCIAL_COSMO = {
    # H0: Degeneracy with Omega_cdm0; GC+GGL primarily constrain Omega_m*h.
    #     Fixing H0 breaks this degeneracy for cleaner Omega_cdm0 constraints.
    'H0': 67.0,
    # Omega_cdm0: SAMPLED - directly affects matter clustering and lensing signal
    'Omega_cdm0': 0.27,
    # Omega_b0: Weakly constrained by GC+GGL alone; well-determined by CMB/BBN.
    #           Fixing to Planck value reduces parameter space.
    'Omega_b0': 0.049,
    # Omega_k0: Dark Emulator requires flat universe (Omega_k = 0).
    'Omega_k0': 0.0,
    # w0: Dark energy EoS has minimal impact on low-z GC+GGL.
    #     Fixing to LCDM value (-1) simplifies analysis.
    'w0': -1.0,
    # wa: Dark Emulator only supports constant w (wa = 0 required).
    'wa': 0.0,
    # ns: Spectral index affects large-scale power; weakly constrained by GC+GGL.
    #     Well-determined by CMB, so fixed to Planck value.
    'ns': 0.96,
    # As: SAMPLED - amplitude directly scales the clustering and lensing signals
    'As': 2.1e-9,
    # gamma_MG: Modified gravity growth index; fixed to GR value for standard analysis.
    'gamma_MG': 0.55,
    # mnu: Dark Emulator uses fixed omega_nu = 0.00064 internally.
    #      This corresponds to sum(mnu) ~ 0.06 eV.
    'mnu': 0.06,
    'N_mnu': 1,
}

# HOD parameters
# --------------
# Sampled: logMmin, sigma_sq, logM1, alpha (primary HOD parameters)
# Fixed parameters and reasons:
FIDUCIAL_HOD = {
    # logMmin: SAMPLED - minimum halo mass for central galaxy occupation
    'logMmin': 13.0,
    # sigma_sq: SAMPLED - scatter in central occupation (transition width)
    'sigma_sq': 0.3,
    # logM1: SAMPLED - characteristic mass for satellite galaxies
    'logM1': 14.0,
    # alpha: SAMPLED - power-law slope for satellite occupation
    'alpha': 1.0,
    # kappa: Satellite threshold parameter; typically fixed as it is
    #        degenerate with logM1 and poorly constrained.
    'kappa': 1.0,
    # poff: Off-centering fraction; set to 0 (all centrals at halo center).
    #       Off-centering is a second-order effect for most analyses.
    'poff': 0.0,
    # Roff: Off-centering scale; irrelevant when poff = 0.
    'Roff': 0.0,
}


# =============================================================================
# Mock Data Generation
# =============================================================================

def generate_mock_data(
    R_bins: np.ndarray,
    z_sample: float,
    pimax: float = 100.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Generate mock 3x2pt (GC + GGL) data from fiducial model.

    Parameters
    ----------
    R_bins : np.ndarray
        Projected radii in h^-1 Mpc.
    z_sample : float
        Sample redshift (lens redshift for GGL, clustering redshift for GC).
    pimax : float
        Line-of-sight integration limit for w_p [h^-1 Mpc].
    noise_level : float
        Fractional noise level (default 10%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_gc : dict
        Galaxy clustering data dictionary.
    data_ggl : dict
        Galaxy-galaxy lensing data dictionary.
    """
    print("\n[1] Generating mock 3x2pt data (GC + GGL)...")

    # Create fiducial cosmology and perturbations
    background = CAMBBackground(**FIDUCIAL_COSMO)
    hod_params = DarkEmuHODParameters(**FIDUCIAL_HOD)

    pert = DarkEmuHODPerturbations(
        background=background,
        redshifts=np.array([z_sample]),
        hod_params=hod_params,
    )

    # Generate fiducial predictions
    # GC: projected correlation function w_p(R)
    wp_fid = pert.projected_correlation(R_bins, z_sample, pimax=pimax)

    # GGL: excess surface mass density ΔΣ(R)
    ds_fid = pert.delta_sigma(R_bins, z_sample)

    # Add noise
    np.random.seed(seed)

    # GC noise
    sigma_wp = noise_level * wp_fid
    cov_wp = np.diag(sigma_wp ** 2)
    wp_obs = wp_fid + np.random.randn(len(R_bins)) * sigma_wp

    # GGL noise
    sigma_ds = noise_level * ds_fid
    cov_ds = np.diag(sigma_ds ** 2)
    ds_obs = ds_fid + np.random.randn(len(R_bins)) * sigma_ds

    print(f"  - Generated {len(R_bins)} radial bins at z = {z_sample}")
    print(f"  - R range: [{R_bins.min():.2f}, {R_bins.max():.2f}] h^-1 Mpc")
    print(f"  - w_p range: [{wp_obs.min():.1f}, {wp_obs.max():.1f}] (h^-1 Mpc)")
    print(f"  - ΔΣ range: [{ds_obs.min():.1f}, {ds_obs.max():.1f}] h M_sun/pc^2")

    data_gc = {
        'wp': wp_obs,
        'R_bins': R_bins,
        'z_sample': z_sample,
        'covariance': cov_wp,
        'pimax': pimax,
        'fiducial_wp': wp_fid,
    }

    data_ggl = {
        'delta_sigma': ds_obs,
        'R_bins': R_bins,
        'z_lens': z_sample,
        'covariance': cov_ds,
        'fiducial_ds': ds_fid,
    }

    return data_gc, data_ggl


def generate_mock_wl_data(
    theta_arcmin: np.ndarray,
    z_arr: np.ndarray,
    dndz: np.ndarray,
    noise_level: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate mock WL (cosmic shear) data: xi_+(theta) and xi_-(theta).

    This function generates mock weak lensing correlation functions using
    HMcode2020Emu for the matter power spectrum and AngularCorrelationFunctionWigner
    for the C_l -> xi_pm transformation.

    Parameters
    ----------
    theta_arcmin : np.ndarray
        Angular separations in arcminutes.
    z_arr : np.ndarray
        Redshift array for source n(z).
    dndz : np.ndarray
        Source redshift distribution, shape (N_tomo, N_z).
        Should be normalized to integrate to 1 per bin.
    noise_level : float
        Fractional noise level (default 10%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_wl : dict
        WL data dictionary containing:
        - 'theta': angular separations [arcmin]
        - 'theta_unit': 'arcmin'
        - 'dndz': redshift distributions
        - 'z_arr': redshift array
        - 'xi_plus': dict of {(i,j): array} for each bin pair
        - 'xi_minus': dict of {(i,j): array} for each bin pair
        - 'covariance': combined covariance matrix
        - 'fiducial_xi_plus': fiducial xi_+ predictions
        - 'fiducial_xi_minus': fiducial xi_- predictions
    """
    print("\n[WL] Generating mock weak lensing data (xi_pm)...")

    # Import required modules
    import jax.numpy as jnp
    from cloelib.cosmology.HMcode2020Emu_cosmology import (
        HMemuLinearPerturbations,
        HMemuNonLinearPerturbations,
    )
    from cloelib.observables.photo import ShearTracer
    from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
    from cloelib.summary_statistics.angular_correlation_function_wigner import (
        AngularCorrelationFunctionWigner,
    )

    # Create fiducial cosmology
    background = CAMBBackground(**FIDUCIAL_COSMO)

    # Create perturbations using HMcode2020Emu
    lp = HMemuLinearPerturbations(background, z_arr)
    nlp = HMemuNonLinearPerturbations(background, lp, z_arr, log10TAGN=7.8)

    # Fiducial WL nuisance parameters
    n_tomo = dndz.shape[0]
    nuisance_params = {
        'AIA': 1.0,
        'EtaIA': 0.0,
        'CIA': 0.0134,
    }
    for i in range(1, n_tomo + 1):
        nuisance_params[f'multiplicative_bias_{i}'] = 0.0
        nuisance_params[f'dz_shear_{i}'] = 0.0

    # Create ShearTracer
    dndz_jax = jnp.asarray(dndz)
    z_arr_jax = jnp.asarray(z_arr)
    shear_tracer = ShearTracer(
        perturbations=nlp,
        dndz=dndz_jax,
        z=z_arr_jax,
        nuisance_params=nuisance_params,
    )

    # ell and k grids
    ells = jnp.unique(jnp.geomspace(2, 15000, 500).astype(int))
    ks = jnp.geomspace(1e-4, 50.0, 200)

    # Compute angular power spectrum
    atp = AngularTwoPoint(shear_tracer, shear_tracer)

    # Compute xi_pm via Wigner d-matrices
    acf = AngularCorrelationFunctionWigner(atp, ells, ks)
    theta_radians = theta_arcmin * np.pi / 180.0 / 60.0
    xi_dict = acf.get_xi(jnp.asarray(theta_radians))

    # Extract xi_+ and xi_- for each bin pair
    xi_plus_fid = {}
    xi_minus_fid = {}
    bin_pairs = []

    for i in range(1, n_tomo + 1):
        for j in range(i, n_tomo + 1):
            key = ("SHE", "SHE", i, j)
            tpcf = xi_dict[key]
            xi_plus_fid[(i, j)] = np.asarray(tpcf.array[0, 0, :])
            xi_minus_fid[(i, j)] = np.asarray(tpcf.array[1, 1, :])
            bin_pairs.append((i, j))

    print(f"  - Generated {len(theta_arcmin)} theta bins")
    print(f"  - theta range: [{theta_arcmin.min():.2f}, {theta_arcmin.max():.2f}] arcmin")
    print(f"  - {n_tomo} tomographic bins -> {len(bin_pairs)} bin pairs")

    # Add noise
    np.random.seed(seed)

    xi_plus_obs = {}
    xi_minus_obs = {}
    sigma_plus = {}
    sigma_minus = {}

    for (i, j) in bin_pairs:
        # xi_+ noise
        sigma_p = noise_level * np.abs(xi_plus_fid[(i, j)])
        sigma_plus[(i, j)] = sigma_p
        xi_plus_obs[(i, j)] = xi_plus_fid[(i, j)] + np.random.randn(len(theta_arcmin)) * sigma_p

        # xi_- noise
        sigma_m = noise_level * np.abs(xi_minus_fid[(i, j)])
        sigma_minus[(i, j)] = sigma_m
        xi_minus_obs[(i, j)] = xi_minus_fid[(i, j)] + np.random.randn(len(theta_arcmin)) * sigma_m

    # Build covariance matrix
    # Data vector ordering: [xi_+(all pairs)] | [xi_-(all pairs)]
    n_theta = len(theta_arcmin)
    n_pairs = len(bin_pairs)
    n_total = 2 * n_pairs * n_theta  # xi_+ and xi_- for all pairs

    cov = np.zeros((n_total, n_total))

    # Fill diagonal blocks (assuming independent bins for simplicity)
    offset = 0
    for (i, j) in bin_pairs:
        idx = slice(offset, offset + n_theta)
        cov[idx, idx] = np.diag(sigma_plus[(i, j)] ** 2)
        offset += n_theta

    for (i, j) in bin_pairs:
        idx = slice(offset, offset + n_theta)
        cov[idx, idx] = np.diag(sigma_minus[(i, j)] ** 2)
        offset += n_theta

    print(f"  - xi_+ range: [{min(xi_plus_obs[(1,1)]):.2e}, {max(xi_plus_obs[(1,1)]):.2e}]")
    print(f"  - xi_- range: [{min(xi_minus_obs[(1,1)]):.2e}, {max(xi_minus_obs[(1,1)]):.2e}]")

    data_wl = {
        'theta': theta_arcmin,
        'theta_unit': 'arcmin',
        'dndz': dndz,
        'z_arr': z_arr,
        'xi_plus': xi_plus_obs,
        'xi_minus': xi_minus_obs,
        'covariance': cov,
        'fiducial_xi_plus': xi_plus_fid,
        'fiducial_xi_minus': xi_minus_fid,
    }

    return data_wl


def generate_simple_dndz(z_arr: np.ndarray, n_tomo: int = 4) -> np.ndarray:
    """Generate simple tomographic source n(z) distributions.

    Creates Gaussian-like n(z) distributions for each tomographic bin,
    similar to typical photometric surveys like HSC.

    Parameters
    ----------
    z_arr : np.ndarray
        Redshift array.
    n_tomo : int
        Number of tomographic bins.

    Returns
    -------
    dndz : np.ndarray
        Shape (n_tomo, len(z_arr)), normalized per bin.
    """
    dndz = np.zeros((n_tomo, len(z_arr)))

    # Tomographic bin centers (typical for HSC-like survey)
    z_centers = np.linspace(0.3, 1.5, n_tomo)
    z_widths = 0.15 * np.ones(n_tomo)

    for i in range(n_tomo):
        # Gaussian distribution
        dndz[i, :] = np.exp(-0.5 * ((z_arr - z_centers[i]) / z_widths[i]) ** 2)
        # Normalize to integrate to 1
        dz = z_arr[1] - z_arr[0] if len(z_arr) > 1 else 1.0
        dndz[i, :] /= np.sum(dndz[i, :]) * dz

    return dndz


# =============================================================================
# Sampler Setup
# =============================================================================

def setup_prior() -> Prior:
    """Setup prior distributions for sampled parameters.

    Returns
    -------
    prior : Prior
        Nautilus Prior object.
    """
    prior = Prior()

    # Cosmological parameters
    # Dark Emulator supports omega_cdm (= Omega_cdm * h^2) in [0.10782, 0.13178]
    # With H0 = 67 (h = 0.67), this translates to:
    #   Omega_cdm0 in [0.10782/0.67^2, 0.13178/0.67^2] = [0.240, 0.294]
    # Add margin to avoid edge warnings: 0.241 → omegac=0.1082, 0.293 → omegac=0.1315
    prior.add_parameter('Omega_cdm0', dist=(0.241, 0.293))
    prior.add_parameter('As', dist=(1.7e-9, 2.5e-9))

    # HOD parameters
    prior.add_parameter('logMmin', dist=(12.0, 14.0))
    prior.add_parameter('sigma_sq', dist=(0.05, 0.6))
    prior.add_parameter('logM1', dist=(13.0, 15.0))
    prior.add_parameter('alpha', dist=(0.5, 1.5))

    return prior


class LikelihoodWrapper:
    """Pickle-compatible likelihood wrapper for Nautilus sampler.

    Nautilus uses multiprocessing, so the likelihood callable must be picklable.
    This wrapper holds the likelihood and fixed params, merges sampled params,
    and calls EuclidLikelihood_DarkEmu_RealSpace.loglike(parameters).
    """

    def __init__(self, likelihood: EuclidLikelihood_DarkEmu_RealSpace, fixed_cosmo: dict, fixed_hod: dict):
        self.likelihood = likelihood
        self.fixed_cosmo = fixed_cosmo
        self.fixed_hod = fixed_hod

    def __call__(self, param_dict: dict) -> float:
        """Evaluate likelihood: merge fixed + sampled params and call loglike."""
        parameters = {**self.fixed_cosmo, **self.fixed_hod, **param_dict}
        try:
            logL = self.likelihood.loglike(parameters)
            return float(logL) if np.isfinite(logL) else -np.inf
        except (ValueError, RuntimeError):
            return -np.inf


# =============================================================================
# Main
# =============================================================================

def main(n_live: int = 500, n_workers: int = 1, output_dir: Path = None):
    """Run full cosmology + HOD sampling for 3x2pt.

    Parameters
    ----------
    n_live : int
        Number of live points for nautilus sampler.
    n_workers : int
        Number of parallel workers for likelihood evaluation.
        Set to number of CPU cores for parallel speedup.
        Note: Each worker loads Dark Emulator separately (~1GB memory each).
    output_dir : Path
        Output directory for results.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Dark Emulator 3x2pt Sampling: Cosmology + HOD Joint Fit")
    print("  Probes: GC (w_p) + GGL (ΔΣ) + WL (ξ±)")
    print("=" * 70)
    print(f"\nSampler settings:")
    print(f"  - n_live: {n_live}")
    print(f"  - n_workers: {n_workers}")
    print(f"  - Output: {output_dir}")

    # -------------------------------------------------------------------------
    # Generate mock data: GC + GGL
    # (Replace the two generate_* blocks below with your own data loaders for real observations.)
    # -------------------------------------------------------------------------
    R_bins = np.logspace(-0.5, 1.5, 15)  # 0.3 to 30 h^-1 Mpc
    z_sample = 0.5
    pimax = 100.0  # h^-1 Mpc

    data_gc, data_ggl = generate_mock_data(R_bins, z_sample, pimax=pimax, noise_level=0.1)

    # -------------------------------------------------------------------------
    # Generate mock data: WL (cosmic shear)
    # -------------------------------------------------------------------------
    theta_arcmin = np.logspace(np.log10(1.0), np.log10(200.0), 20)  # 1 to 200 arcmin
    z_arr = np.linspace(0.01, 2.5, 100)  # Redshift array for source n(z)
    n_tomo = 4  # Number of tomographic bins
    dndz = generate_simple_dndz(z_arr, n_tomo=n_tomo)

    data_wl = generate_mock_wl_data(
        theta_arcmin=theta_arcmin,
        z_arr=z_arr,
        dndz=dndz,
        noise_level=0.1,
        seed=43,  # Different seed for WL
    )

    # -------------------------------------------------------------------------
    # Settings with scale cuts
    # -------------------------------------------------------------------------
    settings = {
        # GC scale cuts (HSC-Y3 style)
        'R_min_gc': 2.0,    # GC: exclude small scales (1-halo dominated)
        'R_max_gc': 30.0,
        # GGL scale cuts (HSC-Y3 style)
        'R_min_ggl': 3.0,   # GGL: minimum scale for Dark Emulator
        'R_max_ggl': 30.0,
        # WL scale cuts (HSC-Y3 style, in arcmin)
        'wl_settings': {
            'theta_min_plus': 7.0,    # xi_+ minimum
            'theta_max_plus': 56.0,   # xi_+ maximum
            'theta_min_minus': 28.0,  # xi_- minimum
            'theta_max_minus': 178.0, # xi_- maximum
        },
    }

    # -------------------------------------------------------------------------
    # Initialize 3x2pt likelihood (cloelike EuclidLikelihood_DarkEmu_RealSpace)
    # -------------------------------------------------------------------------
    print("\n[2] Initializing 3x2pt likelihood (cloelike)...")
    t0 = time.time()

    # Fixed cosmological and HOD parameters
    fixed_cosmo = {k: v for k, v in FIDUCIAL_COSMO.items()
                   if k not in ['Omega_cdm0', 'As']}
    fixed_hod = {k: v for k, v in FIDUCIAL_HOD.items()
                 if k not in ['logMmin', 'sigma_sq', 'logM1', 'alpha']}
    settings['hod_params_fixed'] = fixed_hod

    likelihood = EuclidLikelihood_DarkEmu_RealSpace(
        data_gc=data_gc,
        data_ggl=data_ggl,
        data_wl=data_wl,
        settings=settings,
        Background=CAMBBackground,
    )
    print(f"  - Initialization time: {time.time() - t0:.1f} s")

    # Count data points
    n_gc = likelihood.n_gc
    n_ggl = likelihood.n_ggl
    n_wl = likelihood.n_wl
    n_data_total = n_gc + n_ggl + n_wl
    print(f"  - GC data points: {n_gc}")
    print(f"  - GGL data points: {n_ggl}")
    print(f"  - WL data points: {n_wl} (xi_+: {likelihood.n_xi_plus}, xi_-: {likelihood.n_xi_minus})")
    print(f"  - Total data points: {n_data_total}")

    # -------------------------------------------------------------------------
    # Setup prior and wrapper
    # -------------------------------------------------------------------------
    print("\n[3] Setting up priors...")
    prior = setup_prior()
    # Nautilus Prior.keys is a property (list), not a method
    param_names = list(prior.keys)
    print(f"  - Free parameters ({len(param_names)}): {param_names}")

    wrapper = LikelihoodWrapper(likelihood, fixed_cosmo, fixed_hod)

    # Test likelihood evaluation
    print("\n[4] Testing likelihood evaluation...")
    test_params = {name: FIDUCIAL_COSMO.get(name, FIDUCIAL_HOD.get(name))
                   for name in param_names}

    t0 = time.time()
    test_logL = wrapper(test_params)
    eval_time = time.time() - t0
    print(f"  - Test log-likelihood: {test_logL:.2f}")
    print(f"  - Evaluation time: {eval_time:.3f} s")

    # Estimate total time (nautilus typically needs many evaluations)
    avg_eval_time = eval_time
    print(f"  - Estimated total time: {avg_eval_time * n_live * 50 / 3600:.1f} hours (single worker)")
    if n_workers > 1:
        print(f"  - With {n_workers} workers: ~{avg_eval_time * n_live * 50 / 3600 / n_workers:.1f} hours")

    # -------------------------------------------------------------------------
    # Run sampler
    # -------------------------------------------------------------------------
    print("\n[5] Running nautilus sampler...")
    print(f"    n_live = {n_live}")
    print(f"    n_workers = {n_workers}")
    if n_workers > 1:
        print(f"    Parallel mode: {n_workers} workers (using multiprocessing pool)")
    print("    This may take several hours...")

    checkpoint_file = output_dir / 'checkpoint_3x2pt_full.hdf5'

    # Configure sampler
    # Note: Nautilus 1.0.x uses 'pool' for parallelization, not 'n_eff_workers'
    sampler_kwargs = {
        'n_live': n_live,
        'filepath': str(checkpoint_file),
    }

    # Setup parallel pool if n_workers > 1
    pool = None
    if n_workers > 1:
        from multiprocessing import Pool
        pool = Pool(n_workers)
        sampler_kwargs['pool'] = pool

    sampler = Sampler(
        prior,
        wrapper,
        **sampler_kwargs,
    )

    t_start = time.time()
    try:
        # Run with convergence settings for faster completion
        # n_eff: target effective sample size (lower = faster)
        # For testing with n_live<200, use smaller n_eff; for production use 1000+
        if n_live < 50:
            target_n_eff = 50   # Quick test
        elif n_live < 200:
            target_n_eff = 200  # Moderate (n_live=100 → ~30min)
        else:
            target_n_eff = 1000  # Production (n_live=500)
        print(f"    Target n_eff = {target_n_eff}")
        sampler.run(verbose=True, n_eff=target_n_eff, discard_exploration=True)
    finally:
        # Clean up pool
        if pool is not None:
            pool.close()
            pool.join()
    t_total = time.time() - t_start

    print(f"\n[6] Sampling complete!")
    print(f"    Total time: {t_total:.1f} s ({t_total/3600:.2f} hours)")

    # -------------------------------------------------------------------------
    # Save results
    # checkpoint_3x2pt_full.hdf5 : Nautilus state (resume / debug). chain_3x2pt_full.npz : chain + derived params for plotting.
    # -------------------------------------------------------------------------
    points, log_w, log_l = sampler.posterior()

    # -------------------------------------------------------------------------
    # Compute derived parameters (Omega_m, sigma8, S8)
    # These are the key parameters for comparison with other surveys (HSC, DES, KiDS, Planck)
    # -------------------------------------------------------------------------
    print("\n[7] Computing derived parameters (Omega_m, sigma8, S8)...")

    n_samples = len(points)
    Omega_m_samples = np.zeros(n_samples)
    sigma8_samples = np.zeros(n_samples)
    S8_samples = np.zeros(n_samples)

    # Get indices
    idx_Omega_cdm0 = param_names.index('Omega_cdm0')
    idx_As = param_names.index('As')

    # Compute derived parameters for each sample
    # Use Dark Emulator's get_sigma8 for accurate sigma8 calculation
    # Dark Emulator fixed neutrino density: omega_nu = 0.00064
    from cloelib.auxiliary.darkemu_utils import DARKEMU_OMEGA_NU
    h = FIDUCIAL_COSMO['H0'] / 100.0
    Omega_nu0 = DARKEMU_OMEGA_NU / h**2  # Convert omega_nu to Omega_nu

    print(f"    Computing sigma8 for {n_samples} samples...")
    print(f"    (Using Omega_nu = {Omega_nu0:.6f} from Dark Emulator's omega_nu = {DARKEMU_OMEGA_NU})")
    print(f"    Computing sigma8 for each sample...")

    # Cache for sigma8 computation (key = (Omega_cdm0, As))
    sigma8_cache = {}
    cache_hits = 0

    for i in range(n_samples):
        if i % 1000 == 0 and i > 0:
            print(f"    ... {i}/{n_samples} samples processed (cache hits: {cache_hits})")

        # Omega_m = Omega_cdm + Omega_b + Omega_nu (total matter density)
        Omega_cdm0 = points[i, idx_Omega_cdm0]
        Omega_m = Omega_cdm0 + FIDUCIAL_COSMO['Omega_b0'] + Omega_nu0
        Omega_m_samples[i] = Omega_m

        # Build cosmology for this sample
        As = points[i, idx_As]

        # Use cache for sigma8
        cache_key = (round(float(Omega_cdm0), 8), round(float(As), 12))
        if cache_key in sigma8_cache:
            sigma8 = sigma8_cache[cache_key]
            cache_hits += 1
        else:
            cosmo_params = fixed_cosmo.copy()
            cosmo_params['Omega_cdm0'] = Omega_cdm0
            cosmo_params['As'] = As

            try:
                # Use Dark Emulator to compute sigma8
                background = CAMBBackground(**cosmo_params)
                pert = DarkEmuHODPerturbations(
                    background=background,
                    redshifts=np.array([0.0]),
                    hod_params=DarkEmuHODParameters(**FIDUCIAL_HOD),
                    validate_params=False,  # Skip validation for speed
                )
                sigma8 = pert.get_sigma8(z=0.0)
            except Exception:
                # Fallback to approximate scaling if Dark Emulator fails
                sigma8 = 0.81 * (As / 2.1e-9) ** 0.5 * (Omega_m / 0.3) ** 0.25

            sigma8_cache[cache_key] = sigma8

        sigma8_samples[i] = sigma8

        # S8 = sigma8 * (Omega_m / 0.3)^0.5
        # This is the standard definition used by weak lensing surveys
        S8_samples[i] = sigma8 * (Omega_m / 0.3) ** 0.5

    print(f"    ... {n_samples}/{n_samples} samples processed")
    print(f"    Cache stats: {cache_hits}/{n_samples} hits ({100*cache_hits/n_samples:.1f}%)")

    # Save with derived parameters as primary output
    # Format compatible with corner plot: Omega_m, sigma8, S8
    output_file = output_dir / "chain_3x2pt_full.npz"
    np.savez_compressed(
        str(output_file),
        # Primary chain (sampled parameters)
        chain=points,
        weights=np.exp(log_w),
        logl=log_l,
        param_names=param_names,
        fiducial=np.array([test_params[k] for k in param_names]),
        # Derived cosmological parameters (for corner plot)
        Omega_m=Omega_m_samples,
        sigma8=sigma8_samples,
        S8=S8_samples,
        # Fiducial derived values (including neutrino contribution)
        fiducial_Omega_m=FIDUCIAL_COSMO['Omega_cdm0'] + FIDUCIAL_COSMO['Omega_b0'] + Omega_nu0,
        fiducial_sigma8=0.81,  # Approximate fiducial
        fiducial_S8=0.81 * ((FIDUCIAL_COSMO['Omega_cdm0'] + FIDUCIAL_COSMO['Omega_b0'] + Omega_nu0) / 0.3) ** 0.5,
    )
    print(f"    Results saved to {output_file}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Parameter Constraints (median +/- 1 sigma)")
    print("=" * 70)

    fiducial_values = [test_params[k] for k in param_names]

    print(f"{'Parameter':<15} {'Fiducial':>12} {'Median':>12} {'Std':>10} {'(Fid-Med)/Std':>15}")
    print("-" * 65)

    for i, (name, fid) in enumerate(zip(param_names, fiducial_values)):
        weights = np.exp(log_w)
        weights /= weights.sum()
        median = np.average(points[:, i], weights=weights)
        # Weighted std
        std = np.sqrt(np.average((points[:, i] - median)**2, weights=weights))
        delta = (fid - median) / std if std > 0 else 0

        if 'As' in name:
            print(f"{name:<15} {fid:>12.2e} {median:>12.2e} {std:>10.2e} {delta:>15.2f}")
        else:
            print(f"{name:<15} {fid:>12.4f} {median:>12.4f} {std:>10.4f} {delta:>15.2f}")

    # Derived parameters
    print("\nDerived parameters:")
    print(f"  sigma8: {np.median(sigma8_samples):.4f} +/- {np.std(sigma8_samples):.4f}")
    print(f"  S8:     {np.median(S8_samples):.4f} +/- {np.std(S8_samples):.4f}")
    print(f"  Omega_m:{np.median(Omega_m_samples):.4f} +/- {np.std(Omega_m_samples):.4f}")

    # Chi-squared breakdown for best-fit
    best_idx = np.argmax(log_l)
    best_params = {name: points[best_idx, i] for i, name in enumerate(param_names)}
    best_logL = log_l[best_idx]
    chi2_total = -2.0 * best_logL
    n_params = len(param_names)
    ndof = n_data_total - n_params

    print(f"\nBest-fit chi^2:")
    print(f"  GC data points: {n_gc}")
    print(f"  GGL data points: {n_ggl}")
    print(f"  WL data points: {n_wl}")
    print(f"  Total data points: {n_data_total}")
    print(f"  Total chi^2 = {chi2_total:.2f} (ndof = {ndof})")
    print(f"  Reduced chi^2 = {chi2_total/ndof:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dark Emulator 3x2pt full sampling")
    parser.add_argument('--n_live', type=int, default=500,
                        help='Number of live points (default: 500)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    main(n_live=args.n_live, n_workers=args.n_workers, output_dir=output_dir)
