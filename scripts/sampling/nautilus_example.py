# General imports
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from copy import deepcopy
import time

# Import euclidlib for reading the Euclid data
import euclidlib as el

# Import cloelib for cosmology and theoretical predictions
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations

# Import cloelike for likelihoods
from cloelike.EuclidLikelihood_3x2pt_Cls import EuclidLikelihood_3x2pt_Cls

# Get n(z)
z_nz, nz_heracles = el.photo.redshift_distributions('nz_example.fits')

# Normalize and resample n(z) for both position and shear
myz = np.linspace(1e-4, 3.0, 100)
def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)

# We read with euclidlib v2025.2 (v2025.1 is also compatible)
cells_data = el.photo.angular_power_spectra('synth_cells_5000_binned.fits')
mixmat = el.photo.mixing_matrices('mixmat_identity_5000_binned.fits')

# This matrix copies the format of seaborne, that produces a full 3x2pt matrix
full_cov=np.load('cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy')

def build_data(ell_key, cov, include_pos=False, include_she=False):
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

def build_settings():
    scale_cuts = {key: [10, 1500] for key in cells_data}
    for key in cells_data:
        if key[:2] == ('SHE', 'SHE'):
            scale_cuts[key] = [scale_cuts[key], [0, 0]]
    return {'n_ell_bins': 32, 'scale_cuts': scale_cuts}

# Build each dataset with correct dndz components
data_3x2pt  = build_data(('POS', 'POS', 1, 1), full_cov,   include_pos=True, include_she=True)

# Settings (same for all, built from cells_data structure)
settings_3x2pt  = build_settings()

# These are the default parameters used to generate the synthetic data, therefore, the log-likelihood should be close to zero.
default_pars = {'H0':67,'Omega_cdm0':0.27,'Omega_b0':0.049,'ns':0.96,'As':2.1e-9,
                'w0':-1,'wa':0, 'Omega_k0':0,'mnu':0.0,'gamma_MG':0.545,
                'log10TAGN': 7.75,
                'AIA':0.16, 'EtaIA':1.66,
                'b1_photo_poly0': 1.33291, 'b1_photo_poly1': -0.72414,
                'b1_photo_poly2': 1.0183, 'b1_photo_poly3': -0.14913,
                'magnification_bias_1': 0.0, 'magnification_bias_2': 0.0,
                'magnification_bias_3': 0.0, 'magnification_bias_4': 0.0,
                'magnification_bias_5': 0.0, 'magnification_bias_6': 0.0,
                'dz_pos_1': 0.0, 'dz_pos_2': 0.0,
                'dz_pos_3': 0.0, 'dz_pos_4': 0.0,
                'dz_pos_5': 0.0, 'dz_pos_6': 0.0,
                'multiplicative_bias_1': 0.0, 'multiplicative_bias_2': 0.0,
                'multiplicative_bias_3': 0.0, 'multiplicative_bias_4': 0.0,
                'multiplicative_bias_5': 0.0, 'multiplicative_bias_6': 0.0,
                'dz_shear_1': 0.0, 'dz_shear_2': 0.0,
                'dz_shear_3': 0.0, 'dz_shear_4': 0.0,
                'dz_shear_5': 0.0, 'dz_shear_6': 0.0}


like_instance = EuclidLikelihood_3x2pt_Cls(
        data=data_3x2pt,
        settings=settings_3x2pt,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )

from nautilus import Prior
from nautilus import Sampler
from scipy.stats import norm
prior = Prior()
prior.add_parameter('ombh2', dist=norm(loc=0.0227, scale=0.00038))
prior.add_parameter('omch2', dist=(0.11, 0.13))
prior.add_parameter('logAs', dist=(np.log(1.7e-9*1e10), np.log(2.5e-9*1e10)))
prior.add_parameter('ns', dist=(0.6, 1.2))
prior.add_parameter('H0', dist=(50, 90))
prior.add_parameter('AIA', dist=(-1, 1))
prior.add_parameter('EtaIA', dist=(-5, 5))
#prior.add_parameter('w0', dist=(-2.0, -0.5))
#prior.add_parameter('wa', dist=(-1.0, 1.0))
prior.add_parameter('b1_photo_poly0', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly1', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly2', dist=(-2.0, 2.0))
prior.add_parameter('b1_photo_poly3', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_1', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_2', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_3', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_4', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_5', dist=(-2.0, 2.0))
prior.add_parameter('magnification_bias_6', dist=(-2.0, 2.0))
prior.add_parameter('dz_pos_1', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_pos_2', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_pos_3', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_pos_4', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_pos_5', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_pos_6', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_1', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_2', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_3', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_4', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_5', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_6', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_1', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_2', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_3', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_4', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_5', norm(loc=0.0, scale=0.01))
prior.add_parameter('dz_shear_6', norm(loc=0.0, scale=0.01))


def like_Naut_test(param_dict):

    pars_in = default_pars.copy()

    pars_in.update(param_dict)

    pars_in['Omega_cdm0'] = param_dict['omch2'] / (pars_in['H0']/100)**2
    pars_in['Omega_b0'] = param_dict['ombh2'] / (pars_in['H0']/100)**2
    pars_in['As'] = np.exp(param_dict['logAs'])*1e-10
    pars_in['ns'] = param_dict['ns']
    pars_in['H0'] = param_dict['H0']
    pars_in['AIA'] = param_dict['AIA']
    pars_in['EtaIA'] = param_dict['EtaIA']
    #pars_in['w0'] = param_dict['w0']
    #pars_in['wa'] = param_dict['wa']
    pars_in['b1_photo_poly0'] = param_dict['b1_photo_poly0']
    pars_in['b1_photo_poly1'] = param_dict['b1_photo_poly1']    
    pars_in['b1_photo_poly2'] = param_dict['b1_photo_poly2']
    pars_in['b1_photo_poly3'] = param_dict['b1_photo_poly3']
    pars_in['magnification_bias_1'] = param_dict['magnification_bias_1']
    pars_in['magnification_bias_2'] = param_dict['magnification_bias_2']
    pars_in['magnification_bias_3'] = param_dict['magnification_bias_3']
    pars_in['magnification_bias_4'] = param_dict['magnification_bias_4']
    pars_in['magnification_bias_5'] = param_dict['magnification_bias_5']
    pars_in['magnification_bias_6'] = param_dict['magnification_bias_6']
    pars_in['dz_pos_1'] = param_dict['dz_pos_1']
    pars_in['dz_pos_2'] = param_dict['dz_pos_2']
    pars_in['dz_pos_3'] = param_dict['dz_pos_3']
    pars_in['dz_pos_4'] = param_dict['dz_pos_4']
    pars_in['dz_pos_5'] = param_dict['dz_pos_5']
    pars_in['dz_pos_6'] = param_dict['dz_pos_6']
    pars_in['multiplicative_bias_1'] = param_dict['multiplicative_bias_1']
    pars_in['multiplicative_bias_2'] = param_dict['multiplicative_bias_2']
    pars_in['multiplicative_bias_3'] = param_dict['multiplicative_bias_3']
    pars_in['multiplicative_bias_4'] = param_dict['multiplicative_bias_4']
    pars_in['multiplicative_bias_5'] = param_dict['multiplicative_bias_5']
    pars_in['multiplicative_bias_6'] = param_dict['multiplicative_bias_6']
    pars_in['dz_shear_1'] = param_dict['dz_shear_1']
    pars_in['dz_shear_2'] = param_dict['dz_shear_2']
    pars_in['dz_shear_3'] = param_dict['dz_shear_3']
    pars_in['dz_shear_4'] = param_dict['dz_shear_4']
    pars_in['dz_shear_5'] = param_dict['dz_shear_5']
    pars_in['dz_shear_6'] = param_dict['dz_shear_6']
    
    try:
        like = like_instance.loglike(pars_in)
    except ValueError:
        like = -np.inf
    
    return like

sampler = Sampler(prior, like_Naut_test, n_live=4000, filepath='checkpoint_3x2pt.hdf5')
t_start = time.time()
sampler.run(verbose=True)
t_end = time.time()
print('Total time: {:.1f}s'.format(t_end - t_start))

points, log_w, log_l = sampler.posterior()
np.savez_compressed("chain_3x2pt.npz",
                    chain = points, weights=log_w, logl = log_l)
