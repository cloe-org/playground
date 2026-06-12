import numpy as np 
import matplotlib.pyplot as plt

from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.observables.photo_Weyl import PositionsTracer_Weyl_GC, PositionsTracer_Weyl_GGL
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.cosmology.Weyl_cosmology import Weyl_Perturbations

# Plot style
import seaborn as sns
sns.set_theme(style="ticks")
sns.set_palette(sns.color_palette("Paired"))

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rc('font',size=20)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=20)
plt.rc('lines', linewidth=3)
plt.rc('lines', markersize=6)
plt.rc('legend', fontsize=14)

def compute_zeffs(dndz, zs):
    dz = np.gradient(zs)
    # dndz shape: (nbins, nz)
    z_effs = np.sum(dndz * zs * dz[None, :], axis=1) / np.sum(dndz * dz[None, :], axis=1)
    # variance = <(z - z_mean)^2>
    #var = np.sum(dndz * (zs[None, :] - z_mean[:, None])**2 * dz[None, :], axis=1) / np.sum(dndz * dz[None, :], axis=1)
    #sigma = np.sqrt(var)
    return z_effs
    
def Omegam_sigma8_Jhat_vals(z_effs, background, perturbations):
    # Calcuate growth factors at z_eff, normalized to 1 at z=0.
    growth_factors_zeff = perturbations.growth_factor(z_effs,np.array([perturbations.k[0]]))[:,0] 
    #Omega_m at z_eff
    Omegam_vals = background.Omega_m(z_effs)
    #sigma8_0
    sigma_80 = perturbations.sigma8_0()
    #Jhat and sigma8 values
    Jhat_vals=Omegam_vals*sigma_80*growth_factors_zeff
    sigma8_vals = sigma_80 * growth_factors_zeff 
    return [Omegam_vals, sigma8_vals, Jhat_vals]

# If needed, this can be adjusted to allow even more flexibility
# i.e. passing different values of bias / magnification bias etc. to the init function

class Cells_calculation:
    def __init__(self, perturbations, dndz_pos, dndz_she, z, include_rsd = False):
        #self.background = background
        self.perturbations = perturbations
        self.dndz_pos = dndz_pos
        self.dndz_she = dndz_she
        self.z = z
        self.tracer_pos = PositionsTracer(perturbations = self.perturbations, 
                             dndz = self.dndz_pos,
                             z = self.z,
                             galaxy_bias_model='per_bin',
                             nuisance_params={'b1_photo_bin0': 1.0, 
                                              'b1_photo_bin1': 1.0, 'b1_photo_bin2': 1.0,
                                              'b1_photo_bin3': 1.0, 'b1_photo_bin4': 1.0, 'b1_photo_bin5': 1.0,
                                              'magnification_bias_1': 1.0, 'magnification_bias_2': 1.0,
                                              'magnification_bias_3': 1.0, 'magnification_bias_4': 1.0,
                                              'magnification_bias_5': 1.0, 'magnification_bias_6': 1.0,
                                              'width_pos_1': 1.0, 'width_pos_2': 1.0,
                                              'width_pos_3': 1.0, 'width_pos_4': 1.0,
                                              'width_pos_5': 1.0, 'width_pos_6': 1.0,
                                              'dz_pos_1': 0.000, 'dz_pos_2': 0.000,
                                              'dz_pos_3': 0.000, 'dz_pos_4': 0.000,
                                              'dz_pos_5': 0.000, 'dz_pos_6': 0.000},
                                          include_rsd = include_rsd
                                         )

        self.tracer_she = ShearTracer(perturbations = self.perturbations, 
                             dndz = self.dndz_she,
                             z = self.z,
                             nuisance_params={'AIA': 1.72, 'CIA': 0.0134, 'EtaIA':-0.41,
                                              'multiplicative_bias_1': 0.00, 'multiplicative_bias_2': 0.00,
                                              'multiplicative_bias_3': 0.00, 'multiplicative_bias_4': 0.00,
                                              'multiplicative_bias_5': 0.00, 'multiplicative_bias_6': 0.00,
                                              'width_shear_1': 1.0, 'width_shear_2': 1.0,
                                              'width_shear_3': 1.0, 'width_shear_4': 1.0,
                                              'width_shear_5': 1.0, 'width_shear_6': 1.0,
                                              'dz_shear_1': 0.000, 'dz_shear_2': 0.000,
                                              'dz_shear_3': 0.000, 'dz_shear_4': 0.000,
                                              'dz_shear_5': 0.000, 'dz_shear_6': 0.000})

    def calculate_Cells(self):
        nl = 100
        self.ells = np.logspace(1., np.log10(3000), nl)
        twopoint_pospos = AngularTwoPoint(self.tracer_pos, self.tracer_pos)
        twopoint_posshe = AngularTwoPoint(self.tracer_pos, self.tracer_she)
        self.cells ={**twopoint_pospos.get_Cl(self.ells, 0, self.perturbations.k), 
                     **twopoint_posshe.get_Cl(self.ells, 0, self.perturbations.k)}
        

    def plot_windows_she(self, ax=None, color_palette='mako', plot_tot = True, plot_IA = True, plot_lens = True):
        if ax is None:
            ax = plt.gca()

        colors = sns.color_palette(color_palette, 6)
    
        for i in range(0, 6):
            if plot_tot:
                ax.plot(self.z, 
                        self.tracer_she.get_window(self.z)[i, :], 
                        linewidth=2, 
                        color=colors[i], 
                        label='Total window' if i == 0 else "")

            if plot_IA:    
                ax.plot(self.z, 
                        self.tracer_she.get_window_IA(self.z)[i, :], 
                        linewidth=2, 
                        linestyle='--', 
                        color=colors[i], 
                        label='IA' if i == 0 else "")
                
            if plot_lens:
                ax.plot(self.z, 
                        self.tracer_she.get_window_lensing(self.z)[i, :], 
                        linewidth=2, 
                        linestyle=':', 
                        color=colors[i], 
                        label='Lensing' if i == 0 else "")
            
         # Label the axes
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$W(z)$')
        
        # Add the legend and grid
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

    def plot_windows_pos(self, ax = None, color_palette = 'hls', plot_tot = True, plot_mag = True, plot_pos = True):
        if ax is None:
            ax = plt.gca()
            
        colors = sns.color_palette(color_palette, 6)

        # Plot all three sets of data in one loop
        for i in range(0, 6):
            # Plot the window
            if plot_tot:
                ax.plot(self.z, 
                         self.tracer_pos.get_window(self.z)[i, :], 
                         linewidth=2, 
                         color=colors[i-1], 
                         label=f'Total window' if i == 1 else "")

            # Plot the intrinsic alignment window
            if plot_mag:
                ax.plot(self.z, 
                         self.tracer_pos.get_window_magnification(self.z)[i, :], 
                         linewidth=2, 
                         linestyle='--', 
                         color=colors[i-1], 
                         label=f'magnification' if i == 1 else "")
    
            # Plot the lensing window
            if plot_pos:
                ax.plot(self.z, 
                         self.tracer_pos.get_window_positions(self.z)[i, :], 
                         linewidth=2, 
                         linestyle=':', 
                         color=colors[i-1], 
                         label=f'Positions' if i == 1 else "")

        # Label the axes
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$W(z)$')

        # Add the legend and grid
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

    def plot_cells_GC(self, ax = None, color_palette = 'rocket'):
        if ax is None:
            ax = plt.gca()
            
        for i in range(0, 6):
            plt.loglog(self.ells, self.ells*(2*self.ells+1)*self.cells['POS', 'POS', i+1, i+1], label = '$n_{{{}}}$'.format(i+1), 
                       linewidth=2, color=sns.color_palette(color_palette, 6)[i])
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\ell (2\ell +1)C^{\rm gg}(\ell)$');
        ax.legend(fontsize=14)
        ax.grid(True)

    def plot_cells_GC_compare(self, other, ax = None, color_palette = 'rocket', in_percent = True, yscale = 'linear', 
                              linestyle='-'):
        if ax is None:
            ax = plt.gca()
        for i in range(0, 6):
            if in_percent:
                factor = 100/(self.cells['POS', 'POS', i+1, i+1][:]*self.ells*(2*self.ells+1))
            else:
                factor = 1
            plt.semilogx(self.ells, factor*self.ells*(2*self.ells+1)*(self.cells['POS', 'POS', i+1, i+1][:]-other.cells['POS', 'POS', i+1, i+1][:]) , 
                        label = '$n_{{{}}}$'.format(i+1), linestyle = linestyle, linewidth=2, color=sns.color_palette(color_palette, 6)[i])
        ax.set_xlabel(r'$\ell$')
        if in_percent:
            ax.set_ylabel(r'$\Delta C^{\rm gg}(\ell)$'+' in %')
        else:
            ax.set_ylabel(r'$\ell (2\ell +1)\Delta C^{\rm gg}(\ell)$')
        ax.set_yscale(yscale)
        ax.legend(fontsize=14)
        ax.grid(True)


        
    def plot_cells_GGL(self, ij_list, ax = None, color_palette = 'rocket'):    
        if ax is None:
            ax = plt.gca()
            
        for k in range(len(ij_list)):
            [i,j] = ij_list[k] 
            plt.loglog(self.ells, self.ells*(2*self.ells+1)*self.cells['POS', 'SHE', i, j][0], 
                       label = '$n_{{{}}}$'.format(i)+'-$n_{{{}}}$'.format(j), linewidth=2, color=sns.color_palette(color_palette, len(ij_list))[k])
            
        # Labels and legend
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\ell (2\ell +1)C^{\rm gE}(\ell)$')
        ax.legend(fontsize=14, loc=4)

        # Set y-axis to symmetric log scale to handle negative values
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.grid(True)

    def plot_cells_GGL_compare(self, other, ij_list, ax = None, color_palette = 'rocket', in_percent = True, yscale = 'linear', linestyle = '--'):
        if ax is None:
            ax = plt.gca()
        for k in range(len(ij_list)):
            [i,j] = ij_list[k] 
            if in_percent:
                factor = 100/(self.cells['POS', 'SHE', i, j][0][:]*self.ells*(2*self.ells+1))
            else:
                factor = 1
            plt.semilogx(self.ells, factor*self.ells*(2*self.ells+1)*np.abs(self.cells['POS', 'SHE', i, j][0][:]-other.cells['POS', 'SHE', i, j][0][:]), 
                       label = '$n_{{{}}}$'.format(i)+'-$n_{{{}}}$'.format(j), linewidth=2, linestyle = linestyle, color=sns.color_palette(color_palette, len(ij_list))[k])
        ax.set_xlabel(r'$\ell$')
        if in_percent:
            ax.set_ylabel(r'$\Delta C^{\rm gE}(\ell)$'+' in %')
        else:
            ax.set_ylabel(r'$\ell (2\ell +1)\Delta C^{\rm gE}(\ell)$')
        ax.set_yscale(yscale)
        ax.legend(fontsize=14)
        ax.grid(True)




# Again, this could be further generalized if we need some more flexibility
# Written as a child class of Cells_calculation to avoid repetitive code
class Cells_calculation_Weyl(Cells_calculation):
    def __init__(self, perturbations_nl, perturbations_lin, dndz_pos, dndz_she, z, z_ini=10, include_rsd = False):
        """
        Build a Weyl-perturbation-aware Cells_calculation by reusing the parent
        initialization and then replacing the position tracers with Weyl variants.
        """
        # set background and perturbations class
        self.background = perturbations_nl.background
        self.perturbations = Weyl_Perturbations(perturbations_nl, 
                                                perturbations_lin, 
                                                z, 
                                                z_ini, 
                                                )

        # store inputs
        self.dndz_pos = dndz_pos
        self.dndz_she = dndz_she
        self.z = z
        self.z_ini = z_ini

        # Call parent init to set defaults (this will create default tracers,
        # we'll immediately replace them with Weyl-specific ones below).
        # Parent __init__ expects (perturbations, dndz_pos, dndz_she, z)
        super().__init__(self.perturbations, dndz_pos, dndz_she, z)

        # compute lens n(z) effective redshifts and derived params for Weyl tracers
        self.z_effs = compute_zeffs(dndz_pos, z)
        self.Omegam_vals, self.sigma8_vals, self.Jhat_vals = Omegam_sigma8_Jhat_vals(
            self.z_effs, self.background, self.perturbations
        )
        
        self.tracer_pos_GGL = PositionsTracer_Weyl_GGL(
            perturbations=self.perturbations,
            dndz=self.dndz_pos,
            z=self.z,
            nuisance_params={
                'bhat_bin0': self.sigma8_vals[0], 'bhat_bin1': self.sigma8_vals[1],
                'bhat_bin2': self.sigma8_vals[2], 'bhat_bin3': self.sigma8_vals[3],
                'bhat_bin4': self.sigma8_vals[4], 'bhat_bin5': self.sigma8_vals[5],
                'magnification_bias_1': 1.0, 'magnification_bias_2': 1.0,
                'magnification_bias_3': 1.0, 'magnification_bias_4': 1.0,
                'magnification_bias_5': 1.0, 'magnification_bias_6': 1.0,
                'width_pos_1': 1.0, 'width_pos_2': 1.0,
                'width_pos_3': 1.0, 'width_pos_4': 1.0,
                'width_pos_5': 1.0, 'width_pos_6': 1.0,
                'dz_pos_1': 0.000, 'dz_pos_2': 0.000,
                'dz_pos_3': 0.000, 'dz_pos_4': 0.000,
                'dz_pos_5': 0.000, 'dz_pos_6': 0.000
            },
            Jhat_params={
                'Jhat_bin0': self.Jhat_vals[0], 'Jhat_bin1': self.Jhat_vals[1],
                'Jhat_bin2': self.Jhat_vals[2], 'Jhat_bin3': self.Jhat_vals[3],
                'Jhat_bin4': self.Jhat_vals[4], 'Jhat_bin5': self.Jhat_vals[5]
            },
            include_rsd = include_rsd,
        )

        self.tracer_pos_GC = PositionsTracer_Weyl_GC(
            perturbations=self.perturbations,
            dndz=self.dndz_pos,
            z=self.z,
            nuisance_params={
                'bhat_bin0': self.sigma8_vals[0], 'bhat_bin1': self.sigma8_vals[1],
                'bhat_bin2': self.sigma8_vals[2], 'bhat_bin3': self.sigma8_vals[3],
                'bhat_bin4': self.sigma8_vals[4], 'bhat_bin5': self.sigma8_vals[5],
                'magnification_bias_1': 1.0, 'magnification_bias_2': 1.0,
                'magnification_bias_3': 1.0, 'magnification_bias_4': 1.0,
                'magnification_bias_5': 1.0, 'magnification_bias_6': 1.0,
                'width_pos_1': 1.0, 'width_pos_2': 1.0,
                'width_pos_3': 1.0, 'width_pos_4': 1.0,
                'width_pos_5': 1.0, 'width_pos_6': 1.0,
                'dz_pos_1': 0.000, 'dz_pos_2': 0.000,
                'dz_pos_3': 0.000, 'dz_pos_4': 0.000,
                'dz_pos_5': 0.000, 'dz_pos_6': 0.000
            },
            include_rsd = include_rsd,
        )

        # For compatibility with inherited plotting methods that expect self.tracer_pos:
        # set self.tracer_pos to one of the Weyl tracers (choose whichever is more appropriate;
        # plotting methods that only use "tracer_pos" will then show the GC tracer).
        # When plotting GGL specifically, use the overridden plot_windows_pos below which accepts pos_type.
        self.tracer_pos = self.tracer_pos_GC

    def calculate_Cells(self):
        """
        Weyl-specific cell calculation: use GC and GGL Weyl tracers to build self.cells.
        """
        nl = 100
        self.ells = np.logspace(1., np.log10(3000.), nl)

        twopoint_pospos = AngularTwoPoint(self.tracer_pos_GC, self.tracer_pos_GC)
        twopoint_posshe = AngularTwoPoint(self.tracer_pos_GGL, self.tracer_she)

        # ensure we pass self.ells and the correct k array
        self.cells = {
            **twopoint_pospos.get_Cl(self.ells, 0, self.perturbations.k),
            **twopoint_posshe.get_Cl(self.ells, 0, self.perturbations.k)
        }
        
    # override plot_windows_pos to provide pos_type/rescale behavior specific to Weyl

    
    def plot_windows_pos(self, pos_type, ax=None, color_palette='hls', rescale=True, plot_tot = True, plot_mag = True, plot_pos = True):
        """
        pos_type: 'GC' or 'GGL' to choose which Weyl position tracer to plot.
        This method intentionally differs from the base-class plot_windows_pos signature.
        """
        if pos_type == 'GC':
            tracer_pos = self.tracer_pos_GC
        elif pos_type == 'GGL':
            tracer_pos = self.tracer_pos_GGL
        else:
            raise ValueError("pos_type must be 'GC' or 'GGL'")

        # compute rescaling factors if requested
        if rescale:
            growth = self.perturbations.growth_factor(self.z,np.array([self.perturbations.k[0]]))[:,0] / self.perturbations.growth_factor(np.array([self.z_ini]),np.array([self.perturbations.k[0]]))[0,0] 
            if pos_type == 'GC':
                # assume tracer_pos has attribute sigma8_ini as in your snippet
                rescale_by_pos = np.array(self.sigma8_vals) / tracer_pos.sigma8_ini
                rescale_by_mag = growth
            else:  # 'GGL'
                Omega_mz = self.background.Omega_m(self.z)
                # build list/array for per-bin rescaling; keep as numpy array for indexing
                rescale_by_pos = np.array([
                    (self.sigma8_vals[i] / tracer_pos.sigma8_ini)**2 * self.Omegam_vals[i] / Omega_mz
                    for i in range(6)
                ])
                rescale_by_mag = growth**2
        else:
            rescale_by_pos = np.ones(6)
            rescale_by_mag = 1

        if ax is None:
            ax = plt.gca()

        colors = sns.color_palette(color_palette, 6)

        for i in range(6):
            if plot_tot:
                ax.plot(self.z,
                        tracer_pos.get_window(self.z)[i, :] / rescale_by_pos[i],
                        linewidth=2, color=colors[i],
                        label='Total window' if i == 0 else "")

            if plot_mag:
                ax.plot(self.z,
                        tracer_pos.get_window_magnification(self.z)[i, :] / rescale_by_mag,
                        linewidth=2, linestyle='--', color=colors[i],
                        label='Magnification' if i == 0 else "")

            if plot_pos:
                ax.plot(self.z,
                        tracer_pos.get_window_positions(self.z)[i, :] / rescale_by_pos[i],
                        linewidth=2, linestyle=':', color=colors[i],
                        label='Positions' if i == 0 else "")

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$W(z)$')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)