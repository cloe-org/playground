"""
Nautilus Sampler with cloelike using synthetic 3x2pt data
==========================================================
This script demonstrates how to:
1. Set up a likelihood function from cloelike
2. Define a prior distribution with nautilus
3. Connect them through a wrapper function
4. Run Bayesian inference with the Nautilus sampler
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
from scipy import integrate
from copy import deepcopy
import time

# Data reading
import euclidlib as el

# Sampler and prior
from nautilus import Prior, Sampler
from scipy.stats import norm

# Cosmology and theory
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations, 
    HMemuNonLinearPerturbations
)

# Likelihood
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# Suppress warnings
np.seterr(divide='ignore', over='ignore', invalid='ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Load redshift distributions from data file
z_nz, nz_example = el.phz.redshift_distributions('nz_example.fits')

# Create uniform redshift grid for interpolation
myz = np.linspace(1e-4, 3.0, 100)

def normalize_and_resample(nz_dict, z_grid, z_target):
    """Normalize and resample n(z) to a target redshift grid."""
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) 
                          for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_example, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_example, z_nz, myz)

# Load power spectra and mixing matrices
cells_data = el.le3.pk_wl.angular_power_spectra('synthetic_coupled_cells.fits')
mixmat = el.le3.pk_wl.mixing_matrices('mixmats_example.fits')

# Load covariance matrix
full_cov = np.load('covmat_2D.npz')['Gauss']

# ============================================================================
# 2. BUILD LIKELIHOOD DATA AND SETTINGS
# ============================================================================

def build_data(ell_key, cov, include_pos=False, include_she=False):
    """Organize data dictionary for likelihood."""
    data = {
        'cells': cells_data,
        'ells': cells_data[ell_key].ell,
        'z_arr': myz,
        'cov': cov,
        'mixmat': mixmat,
    }
    if include_pos:
        data['dndz_pos'] = my_dndz_pos_norm
    if include_she:
        data['dndz_she'] = my_dndz_she_norm
    return data

def build_settings(cls):
    """Define scale cuts for angular scales."""
    scale_cuts = {}
    for key in cls:
        if key[:2] == ('POS', 'POS'):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ('POS', 'SHE'):
            scale_cuts[key] = [10, 750]
        elif key[:2] == ('SHE', 'SHE'):
            scale_cuts[key] = [10, 1500]
    
    return {
        'n_ell_bins': 32,
        'scale_cuts': scale_cuts
    }

# Prepare data for 3x2pt analysis
data_3x2pt = build_data(('POS', 'POS', 1, 1), full_cov, 
                        include_pos=True, include_she=True)
settings_3x2pt = build_settings(cells_data)

# ============================================================================
# 3. INITIALIZE LIKELIHOOD OBJECT
# ============================================================================

# These are the fiducial parameters used to generate synthetic data
# The log-likelihood should be close to zero at these values
default_pars = {
    # Cosmological parameters
    'H0': 70, 'Omega_cdm0': 0.25, 'Omega_b0': 0.05,
    'ns': 0.96, 'As': 2.e-9, 'w0': -1, 'wa': 0,
    'Omega_k0': 0, 'mnu': 0.06, 'gamma_MG': 0.545, 'N_mnu': 1,
    # Intrinsic alignment parameters
    'log10TAGN': 7.8, 'AIA': 1.72, 'CIA': 0.0134, 'EtaIA': -0.41,
    # Galaxy bias parameters
    'b1_photo_poly0': 1., 'b1_photo_poly1': 0.,
    'b1_photo_poly2': 0.0, 'b1_photo_poly3': 0.,
    # Magnification bias (one per tomographic bin)
    'magnification_bias_1': 0.0, 'magnification_bias_2': 0.0,
    'magnification_bias_3': 0.0, 'magnification_bias_4': 0.0,
    'magnification_bias_5': 0.0, 'magnification_bias_6': 0.0,
    # Photo-z shifts for position (one per bin)
    'dz_pos_1': 0.0, 'dz_pos_2': 0.0, 'dz_pos_3': 0.0,
    'dz_pos_4': 0.0, 'dz_pos_5': 0.0, 'dz_pos_6': 0.0,
    # Multiplicative bias for shear (one per bin)
    'multiplicative_bias_1': 0.0, 'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0, 'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0, 'multiplicative_bias_6': 0.0,
    # Photo-z shifts for shear (one per bin)
    'dz_shear_1': 0.0, 'dz_shear_2': 0.0, 'dz_shear_3': 0.0,
    'dz_shear_4': 0.0, 'dz_shear_5': 0.0, 'dz_shear_6': 0.0,
}

# Initialize likelihood instance with cosmology and perturbation modules
like_instance = EuclidLikelihood_3x2pt(
    data=data_3x2pt,
    settings=settings_3x2pt,
    Background=CAMBBackground,
    LinPerturbations=HMemuLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
    mode='coupled'
)

# Test at fiducial
print("=" * 60)
print("Testing likelihood at fiducial parameters:")
print("=" * 60)
fiducial_like = like_instance.loglike(default_pars)
print(f"Log-likelihood: {fiducial_like:.4f}")
print()

# ============================================================================
# 4. DEFINE PRIOR DISTRIBUTION FOR NAUTILUS
# ============================================================================

print("Setting up prior distribution...")
prior = Prior()

# Add parameters to the prior
# Format: prior.add_parameter(name, dist=(min, max)) for uniform
#         prior.add_parameter(name, dist=norm(loc=mean, scale=std)) for Gaussian

# Cosmological parameters (wide priors)
prior.add_parameter('ombh2', dist=norm(loc=0.0227, scale=0.00038))
prior.add_parameter('omch2', dist=(0.11, 0.13))
prior.add_parameter('logAs', dist=(np.log(1.7e-9*1e10), np.log(2.5e-9*1e10)))
prior.add_parameter('ns', dist=(0.6, 1.2))
prior.add_parameter('H0', dist=(50, 90))

# Intrinsic alignment parameters
prior.add_parameter('AIA', dist=(-1, 1))
prior.add_parameter('EtaIA', dist=(-5, 5))

# Galaxy bias parameters
prior.add_parameter('b1_photo_poly0', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly1', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly2', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly3', dist=(-2.0, 2.0))

# Magnification bias
for i in range(1, 7):
    prior.add_parameter(f'magnification_bias_{i}', dist=(-2.0, 2.0))

# Photo-z shifts and biases (Gaussian priors centered on 0)
for i in range(1, 7):
    prior.add_parameter(f'dz_pos_{i}', norm(loc=0.0, scale=0.01))
    prior.add_parameter(f'multiplicative_bias_{i}', norm(loc=0.0, scale=0.01))
    prior.add_parameter(f'dz_shear_{i}', norm(loc=0.0, scale=0.01))

print(f"Total number of sampled parameters: {prior.dimensionality()}")
print()

# ============================================================================
# 5. DEFINE LIKELIHOOD WRAPPER FOR NAUTILUS
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
    pars['Omega_cdm0'] = param_dict['omch2'] / (pars['H0']/100)**2
    pars['Omega_b0'] = param_dict['ombh2'] / (pars['H0']/100)**2
    pars['As'] = np.exp(param_dict['logAs']) * 1e-10
    
    # Evaluate likelihood with cloelike
    try:
        log_likelihood = like_instance.loglike(pars)
    except (ValueError, RuntimeError):
        # If evaluation fails (e.g., unphysical parameters), return -inf
        log_likelihood = -np.inf
    
    return log_likelihood

# ====================================================================================
# 6. RUN NAUTILUS SAMPLER
#
# This will run the sampler and save results to 'chain_3x2pt.npz'
#
# Note: The sampling process may take several minutes depending on the number
# of live points and the complexity of the likelihood. You can adjust the
# number of live points in the Sampler initialization for faster testing.
#
# For parallel runs, you can use the 'pool' argument in the Sampler to provide a multiprocessing pool.
# Check https://nautilus-sampler.readthedocs.io/en/latest/guides/parallelization.html
#
# ====================================================================================

print("=" * 60)
print("Running Nautilus sampler...")
print("=" * 60)

# Initialize sampler with prior and likelihood
sampler = Sampler(
    prior,
    like_Nautilus,
    n_live=4000,                           # Number of live points
    filepath='checkpoint_3x2pt.hdf5'       # Checkpoint file
)

# Run the sampler
t_start = time.time()
sampler.run(verbose=True)
t_end = time.time()

print()
print(f"Sampling completed in {(t_end - t_start):.1f} seconds")
print()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

# Extract posterior samples
points, log_w, log_l = sampler.posterior()

# Save to compressed file
np.savez_compressed(
    "chain_3x2pt.npz",
    chain=points,           # Posterior samples
    weights=log_w,          # Log-weights
    logl=log_l              # Log-likelihood values
)

print("Results saved to 'chain_3x2pt.npz'")
