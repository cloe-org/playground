"""
Nautilus Sampler with cloelike: 3x2pt with selectable theory models
===================================================================
This script demonstrates how to:
1. Choose the intrinsic-alignment, galaxy-bias and baryonic-feedback models
2. Set up a 3x2pt likelihood from cloelike with that choice
3. Define a prior distribution with nautilus that follows the choice
4. Run Bayesian inference with the Nautilus sampler

Based on playground/scripts/sampling/nautilus_example.py, updated for
cloelike#169 (cloelib with TATT, nonlinear galaxy bias and baryonic boosts).

Data files expected in DATA_DIR: nz_example.fits, synthetic_coupled_cells.fits,
mixmats_example.fits, covmat_2D.npz.
"""

# ============================================================================
# 0. MODEL CHOICE
# ============================================================================

# Intrinsic alignments: "NLA", "TATT" or None (no IA)
IA_MODEL = "TATT"

# Galaxy bias: "poly" (b1(z) cubic polynomial), "per_bin" (one b1 per bin)
# or "nonlinear" (one-loop b1, b2, bs2, b3nl, bk2 per bin)
GALAXY_BIAS_MODEL = "nonlinear"

# Baryonic feedback: "HMcode" (log10TAGN), "FLAMINGO" (fgas_sigma,
# Mstar_sigma, jet_fraction) or None (dark matter only)
BARYON_MODEL = "HMcode"

# The synthetic data file was generated with one fixed model. With
# USE_MOCK_FROM_MODEL = True the data vector is replaced by the noise-free
# theory of the model chosen above at the fiducial parameters, so the
# posterior is centred on them. Ells, mixing matrices and covariance still
# come from the files.
USE_MOCK_FROM_MODEL = True

DATA_DIR = "."
N_LIVE = 4000

# ============================================================================
# IMPORTS
# ============================================================================

import os
import time

import numpy as np
from scipy import integrate

# Data reading
import euclidlib as el

# Sampler and prior
from nautilus import Prior, Sampler
from scipy.stats import norm

# Cosmology and theory
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)

# Likelihood
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# Suppress warnings
np.seterr(divide="ignore", over="ignore", invalid="ignore")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Load redshift distributions from data file
z_nz, nz_example = el.phz.redshift_distributions(
    os.path.join(DATA_DIR, "nz_example.fits")
)

# Create uniform redshift grid for interpolation
myz = np.linspace(1e-4, 3.0, 100)


def normalize_and_resample(nz_dict, z_grid, z_target):
    """Normalize and resample n(z) to a target redshift grid."""
    nz_array = np.vstack(
        [nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()]
    )
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])


my_dndz_pos_norm = normalize_and_resample(nz_example, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_example, z_nz, myz)
n_bins = my_dndz_pos_norm.shape[0]

# Load power spectra and mixing matrices
cells_data = el.le3.pk_wl.angular_power_spectra(
    os.path.join(DATA_DIR, "synthetic_coupled_cells.fits")
)
mixmat = el.le3.pk_wl.mixing_matrices(os.path.join(DATA_DIR, "mixmats_example.fits"))

# Load covariance matrix
full_cov = np.load(os.path.join(DATA_DIR, "covmat_2D.npz"))["Gauss"]

# ============================================================================
# 2. BUILD LIKELIHOOD DATA AND SETTINGS
# ============================================================================


def build_data(ell_key, cov, include_pos=False, include_she=False):
    """Organize data dictionary for likelihood."""
    data = {
        "cells": cells_data,
        "ells": cells_data[ell_key].ell,
        "z_arr": myz,
        "cov": cov,
        "mixmat": mixmat,
    }
    if include_pos:
        data["dndz_pos"] = my_dndz_pos_norm
    if include_she:
        data["dndz_she"] = my_dndz_she_norm
    return data


def build_settings(cls):
    """Define scale cuts for angular scales and the theory model."""
    scale_cuts = {}
    for key in cls:
        if key[:2] == ("POS", "POS"):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ("POS", "SHE"):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ("SHE", "SHE"):
            scale_cuts[key] = [10, 1500]

    settings = {
        "n_ell_bins": 32,
        "scale_cuts": scale_cuts,
        "ia_model": IA_MODEL,
        "galaxy_bias_model": GALAXY_BIAS_MODEL,
        # The nonlinear galaxy bias does not support RSD yet
        "include_rsd": False,
    }
    # Which sampled parameters go to the nonlinear perturbations
    if BARYON_MODEL == "HMcode":
        settings["nonlinear_param_keys"] = ("log10TAGN",)
    elif BARYON_MODEL == "FLAMINGO":
        settings["nonlinear_param_keys"] = ()
        settings["baryon_param_keys"] = ("fgas_sigma", "Mstar_sigma", "jet_fraction")
    elif BARYON_MODEL is None:
        settings["nonlinear_param_keys"] = ()
    else:
        raise ValueError(f"Unknown BARYON_MODEL {BARYON_MODEL!r}")
    return settings


def nonlinear_perturbations_class():
    """Nonlinear perturbations, with the FLAMINGO response on top if chosen."""
    if BARYON_MODEL == "FLAMINGO":
        # Needs a cloelib with the baryonic boosts (cloelib PR #257)
        from cloelib.cosmology.cosmology import with_baryon_boost
        from cloelib.cosmology.FlamingoBaryonResponseEmulator_cosmology import (
            FlamingoBaryonBoostMixin,
        )

        return with_baryon_boost(HMemuNonLinearPerturbations, FlamingoBaryonBoostMixin)
    return HMemuNonLinearPerturbations


# Prepare data for 3x2pt analysis
data_3x2pt = build_data(
    ("POS", "POS", 1, 1), full_cov, include_pos=True, include_she=True
)
settings_3x2pt = build_settings(cells_data)

# ============================================================================
# 3. FIDUCIAL PARAMETERS FOR THE CHOSEN MODEL
# ============================================================================

default_pars = {
    # Cosmological parameters
    "H0": 70, "Omega_cdm0": 0.25, "Omega_b0": 0.05,
    "ns": 0.96, "As": 2.0e-9, "w0": -1, "wa": 0,
    "Omega_k0": 0, "mnu": 0.06, "gamma_MG": 0.545, "N_mnu": 1,
    "alpha_s": 0.0,
}
# Magnification bias, photo-z shifts and multiplicative bias (one per bin)
for i in range(1, n_bins + 1):
    default_pars.update({
        f"magnification_bias_{i}": 0.0,
        f"dz_pos_{i}": 0.0,
        f"width_pos_{i}": 1.0,
        f"multiplicative_bias_{i}": 0.0,
        f"dz_shear_{i}": 0.0,
        f"width_shear_{i}": 1.0,
    })

# Intrinsic alignment parameters
if IA_MODEL in ("NLA", "TATT"):
    default_pars.update({"AIA": 1.72, "CIA": 0.0134, "EtaIA": -0.41})
if IA_MODEL == "TATT":
    default_pars.update({"A2IA": 0.0, "Eta2IA": 0.0, "bTA": 0.0, "z0IA": 0.62})

# Galaxy bias parameters (bins indexed from 0)
if GALAXY_BIAS_MODEL == "poly":
    default_pars.update({f"b1_photo_poly{i}": 0.0 for i in range(4)})
    default_pars["b1_photo_poly0"] = 1.0
elif GALAXY_BIAS_MODEL == "per_bin":
    default_pars.update({f"b1_photo_bin{i}": 1.0 for i in range(n_bins)})
elif GALAXY_BIAS_MODEL == "nonlinear":
    default_pars.update({f"b1_photo_nl_bin{i}": 1.0 for i in range(n_bins)})
    default_pars.update({f"b2_photo_nl_bin{i}": 0.0 for i in range(n_bins)})
    # bs2, b3nl and bk2 are left out, which drops them from the expansion

# Baryonic feedback parameters
if BARYON_MODEL == "HMcode":
    default_pars["log10TAGN"] = 7.8
elif BARYON_MODEL == "FLAMINGO":
    default_pars.update({"fgas_sigma": 0.0, "Mstar_sigma": 0.0, "jet_fraction": 0.0})

# ============================================================================
# 4. INITIALIZE LIKELIHOOD OBJECT
# ============================================================================

# Initialize likelihood instance with cosmology and perturbation modules
like_instance = EuclidLikelihood_3x2pt(
    data=data_3x2pt,
    settings=settings_3x2pt,
    Background=CAMBBackground,
    LinPerturbations=HMemuLinearPerturbations,
    NonLinPerturbations=nonlinear_perturbations_class(),
    mode="coupled",
)

print("=" * 60)
print(f"IA: {IA_MODEL}, galaxy bias: {GALAXY_BIAS_MODEL}, baryons: {BARYON_MODEL}")
print("=" * 60)

# First evaluation at a new cosmology, including any FAST-PT kernels
t_start = time.time()
fiducial_theory = np.asarray(like_instance.get_theory_vector_masked(default_pars))
print(f"Theory evaluation time (new cosmology): {time.time() - t_start:.2f} s")

if USE_MOCK_FROM_MODEL:
    # Noise-free mock: the theory of the chosen model at the fiducial point
    like_instance.get_data_vector_masked = lambda: fiducial_theory

# Test at fiducial; it should be ~0 with USE_MOCK_FROM_MODEL = True
fiducial_like = like_instance.loglike(default_pars)
print(f"Log-likelihood at fiducial: {fiducial_like:.4f}")
print()

# ============================================================================
# 5. DEFINE PRIOR DISTRIBUTION FOR NAUTILUS
# ============================================================================

print("Setting up prior distribution...")
prior = Prior()

# Add parameters to the prior
# Format: prior.add_parameter(name, dist=(min, max)) for uniform
#         prior.add_parameter(name, dist=norm(loc=mean, scale=std)) for Gaussian

# Cosmological parameters. The BBN-like ombh2 prior is centred on the
# fiducial Omega_b0 h^2 = 0.0245.
prior.add_parameter("ombh2", dist=norm(loc=0.0245, scale=0.00038))
prior.add_parameter("omch2", dist=(0.11, 0.13))
prior.add_parameter("logAs", dist=(np.log(1.7e-9 * 1e10), np.log(2.5e-9 * 1e10)))
prior.add_parameter("ns", dist=(0.6, 1.2))
prior.add_parameter("H0", dist=(50, 90))

# Intrinsic alignment parameters
if IA_MODEL in ("NLA", "TATT"):
    prior.add_parameter("AIA", dist=(-5, 5))
    prior.add_parameter("EtaIA", dist=(-5, 5))
if IA_MODEL == "TATT":
    prior.add_parameter("A2IA", dist=(-5, 5))
    prior.add_parameter("bTA", dist=(-5, 5))

# Galaxy bias parameters
if GALAXY_BIAS_MODEL == "poly":
    for i in range(4):
        prior.add_parameter(f"b1_photo_poly{i}", dist=(-2.0, 2.0))
elif GALAXY_BIAS_MODEL == "per_bin":
    for i in range(n_bins):
        prior.add_parameter(f"b1_photo_bin{i}", dist=(0.0, 3.0))
elif GALAXY_BIAS_MODEL == "nonlinear":
    for i in range(n_bins):
        prior.add_parameter(f"b1_photo_nl_bin{i}", dist=(0.0, 3.0))
        prior.add_parameter(f"b2_photo_nl_bin{i}", dist=(-5.0, 5.0))

# Baryonic feedback parameters
if BARYON_MODEL == "HMcode":
    prior.add_parameter("log10TAGN", dist=(7.6, 8.3))
elif BARYON_MODEL == "FLAMINGO":
    # Training ranges of the FLAMINGO emulator for jet_fraction = 0
    prior.add_parameter("fgas_sigma", dist=(-8.0, 2.0))
    prior.add_parameter("Mstar_sigma", dist=(-1.0, 0.0))

# Magnification bias
for i in range(1, n_bins + 1):
    prior.add_parameter(f"magnification_bias_{i}", dist=(-2.0, 2.0))

# Photo-z shifts and biases (Gaussian priors centered on 0)
for i in range(1, n_bins + 1):
    prior.add_parameter(f"dz_pos_{i}", norm(loc=0.0, scale=0.01))
    prior.add_parameter(f"multiplicative_bias_{i}", norm(loc=0.0, scale=0.01))
    prior.add_parameter(f"dz_shear_{i}", norm(loc=0.0, scale=0.01))

print(f"Total number of sampled parameters: {prior.dimensionality()}")
print()

# ============================================================================
# 6. DEFINE LIKELIHOOD WRAPPER FOR NAUTILUS
# ============================================================================


def like_Nautilus(param_dict):
    """
    Wrapper function connecting Nautilus sampler to cloelike likelihood.

    This function:
    1. Takes parameters from Nautilus sampler
    2. Converts sampling parameters to cloelib base parameters
    3. Evaluates the likelihood
    4. Returns log-likelihood for Nautilus

    Args:
        param_dict: Dictionary of sampled parameters from Nautilus

    Returns:
        float: log-likelihood value (or -inf if evaluation fails)
    """

    # Start with default parameters
    pars = default_pars.copy()

    # Update with sampled parameters
    pars.update(param_dict)

    # IMPORTANT: Convert physical parameters to cloelib base parameters
    # Nautilus samples in convenient physical units, but cloelib expects
    # certain base parameters. This is the KEY CONVERSION STEP.
    pars["Omega_cdm0"] = param_dict["omch2"] / (pars["H0"] / 100) ** 2
    pars["Omega_b0"] = param_dict["ombh2"] / (pars["H0"] / 100) ** 2
    pars["As"] = np.exp(param_dict["logAs"]) * 1e-10

    # Evaluate likelihood with cloelike
    try:
        log_likelihood = like_instance.loglike(pars)
    except (ValueError, RuntimeError):
        # If evaluation fails (e.g., unphysical parameters), return -inf
        log_likelihood = -np.inf

    return log_likelihood


# ====================================================================================
# 7. RUN NAUTILUS SAMPLER
#
# This will run the sampler and save results to 'chain_3x2pt_<model>.npz'
#
# Note: TATT and the nonlinear galaxy bias compute one-loop kernels with
# FAST-PT at every new cosmology, which makes each evaluation slower than with
# NLA and linear bias. Steps that only change nuisance parameters reuse them.
#
# For parallel runs, you can use the 'pool' argument in the Sampler to provide a multiprocessing pool.
# Check https://nautilus-sampler.readthedocs.io/en/latest/guides/parallelization.html
#
# ====================================================================================

run_name = f"3x2pt_{IA_MODEL}_{GALAXY_BIAS_MODEL}_{BARYON_MODEL}"

print("=" * 60)
print("Running Nautilus sampler...")
print("=" * 60)

# Initialize sampler with prior and likelihood
sampler = Sampler(
    prior,
    like_Nautilus,
    n_live=N_LIVE,                          # Number of live points
    filepath=f"checkpoint_{run_name}.hdf5",  # Checkpoint file
)

# Run the sampler
t_start = time.time()
sampler.run(verbose=True)
t_end = time.time()

print()
print(f"Sampling completed in {(t_end - t_start):.1f} seconds")
print()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

# Extract posterior samples
points, log_w, log_l = sampler.posterior()

# Save to compressed file
np.savez_compressed(
    f"chain_{run_name}.npz",
    chain=points,           # Posterior samples
    weights=log_w,          # Log-weights
    logl=log_l,             # Log-likelihood values
    names=np.array(prior.keys),
)

print(f"Results saved to 'chain_{run_name}.npz'")
