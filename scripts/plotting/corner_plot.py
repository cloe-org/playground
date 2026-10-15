"""Corner Plot for Cosmological Parameter Constraints.

This script creates publication-quality corner plots (triangle plots) showing
1D and 2D posterior distributions for cosmological parameters like Ω_m, σ_8, and S_8.

Usage:
    python corner_plot.py [--chain CHAIN_FILE] [--output OUTPUT_FILE]

Example:
    python corner_plot.py --chain ../sampling/chain_darkemu_3x2pt.npz --output corner_cosmo.pdf

Requirements:
    - getdist (pip install getdist)
    - matplotlib
    - numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from pathlib import Path

# Configure matplotlib for publication quality
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['text.usetex'] = False  # Set to True if LaTeX is available


def load_chain(filepath: str, include_derived: bool = True) -> tuple:
    """Load MCMC chain from file.

    Parameters
    ----------
    filepath : str
        Path to chain file (.npz format from nautilus).
    include_derived : bool
        If True, include pre-computed derived parameters (Omega_m, sigma8, S8)
        if available in the file.

    Returns
    -------
    samples : np.ndarray
        Parameter samples, shape (n_samples, n_params).
    weights : np.ndarray
        Sample weights.
    param_names : list
        Parameter names.
    """
    data = np.load(filepath, allow_pickle=True)
    samples = data['chain']

    # Handle weights - could be raw or log weights
    if 'weights' in data:
        weights = data['weights']
        # Check if weights are log weights (typically have large negative values)
        if np.any(weights < 0):
            weights = np.exp(weights - weights.max())
    else:
        weights = np.ones(len(samples))

    param_names = list(data['param_names'])

    # Include pre-computed derived parameters if available
    if include_derived:
        derived_cols = []
        derived_names = []

        # Check for Omega_m, sigma8, S8 in the file
        for derived_name in ['Omega_m', 'sigma8', 'S8']:
            if derived_name in data and derived_name not in param_names:
                derived_cols.append(data[derived_name])
                derived_names.append(derived_name)

        if derived_cols:
            samples = np.column_stack([samples] + derived_cols)
            param_names = param_names + derived_names
            print(f"  Loaded derived parameters: {derived_names}")

    return samples, weights, param_names


def compute_derived_params(samples: np.ndarray, param_names: list) -> tuple:
    """Compute derived cosmological parameters (Omega_m, sigma_8, S_8).

    Parameters
    ----------
    samples : np.ndarray
        Parameter samples.
    param_names : list
        Parameter names.

    Returns
    -------
    derived_samples : np.ndarray
        Samples including derived parameters.
    derived_names : list
        Names including derived parameters.
    """
    n_samples = samples.shape[0]

    # Extract necessary parameters
    param_dict = {name: samples[:, i] for i, name in enumerate(param_names)}

    # Compute Omega_m
    if 'omch2' in param_dict and 'ombh2' in param_dict and 'H0' in param_dict:
        h = param_dict['H0'] / 100.0
        omega_m = (param_dict['omch2'] + param_dict['ombh2']) / h**2
    elif 'Omega_cdm0' in param_dict and 'Omega_b0' in param_dict:
        omega_m = param_dict['Omega_cdm0'] + param_dict['Omega_b0']
    else:
        omega_m = None

    # Compute sigma_8 from As (approximate relation)
    if 'logAs' in param_dict:
        As = np.exp(param_dict['logAs']) * 1e-10
        # Approximate sigma_8 scaling (simplified)
        sigma_8 = 0.8 * (As / 2.1e-9)**0.5
        if omega_m is not None:
            sigma_8 *= (omega_m / 0.3)**0.25
    elif 'sigma8' in param_dict:
        sigma_8 = param_dict['sigma8']
    else:
        sigma_8 = None

    # Compute S_8 = sigma_8 * (Omega_m / 0.3)^0.5
    if sigma_8 is not None and omega_m is not None:
        S_8 = sigma_8 * (omega_m / 0.3)**0.5
    else:
        S_8 = None

    # Build derived samples array
    derived_cols = []
    derived_names = list(param_names)

    if omega_m is not None and 'Omega_m' not in param_names:
        derived_cols.append(omega_m)
        derived_names.append('Omega_m')

    if sigma_8 is not None and 'sigma8' not in param_names:
        derived_cols.append(sigma_8)
        derived_names.append('sigma8')

    if S_8 is not None and 'S8' not in param_names:
        derived_cols.append(S_8)
        derived_names.append('S8')

    if derived_cols:
        derived_samples = np.column_stack([samples] + derived_cols)
    else:
        derived_samples = samples

    return derived_samples, derived_names


def create_corner_plot(
    samples_list: list,
    weights_list: list,
    param_names: list,
    labels: list = None,
    colors: list = None,
    linestyles: list = None,
    params_to_plot: list = None,
    param_labels: dict = None,
    filled: list = None,
    output_file: str = None,
    figsize: tuple = (8, 8),
    title: str = None,
):
    """Create a corner plot comparing multiple chains.

    Parameters
    ----------
    samples_list : list of np.ndarray
        List of sample arrays from different chains.
    weights_list : list of np.ndarray
        List of weight arrays.
    param_names : list
        Parameter names (common to all chains).
    labels : list, optional
        Legend labels for each chain.
    colors : list, optional
        Colors for each chain.
    linestyles : list, optional
        Line styles for each chain.
    params_to_plot : list, optional
        Subset of parameters to plot.
    param_labels : dict, optional
        LaTeX labels for parameters.
    filled : list of bool, optional
        Whether to fill contours for each chain.
    output_file : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    title : str, optional
        Figure title.
    """
    try:
        from getdist import MCSamples, plots
    except ImportError:
        print("getdist not installed. Using fallback corner plot.")
        _corner_plot_fallback(
            samples_list, weights_list, param_names, labels, colors,
            params_to_plot, param_labels, output_file, figsize
        )
        return

    # Default parameter labels (LaTeX)
    default_labels = {
        'Omega_m': r'\Omega_{\rm m}',
        'sigma8': r'\sigma_8',
        'S8': r'S_8',
        'H0': r'H_0',
        'ns': r'n_s',
        'ombh2': r'\omega_b',
        'omch2': r'\omega_c',
        'logAs': r'\ln(10^{10}A_s)',
        'logMmin': r'\log M_{\rm min}',
        'sigma_sq': r'\sigma^2',
        'logM1': r'\log M_1',
        'alpha': r'\alpha',
        'kappa': r'\kappa',
        'w0': r'w_0',
    }
    if param_labels:
        default_labels.update(param_labels)

    # Default settings
    if params_to_plot is None:
        params_to_plot = param_names

    if labels is None:
        labels = [f'Chain {i+1}' for i in range(len(samples_list))]

    if colors is None:
        colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']

    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', '-']

    if filled is None:
        filled = [True] + [False] * (len(samples_list) - 1)

    # Create MCSamples objects
    mc_samples_list = []
    for i, (samples, weights, label) in enumerate(zip(samples_list, weights_list, labels)):
        # Get indices for parameters to plot
        param_indices = [param_names.index(p) for p in params_to_plot if p in param_names]
        plot_names = [param_names[j] for j in param_indices]
        plot_labels = [default_labels.get(n, n) for n in plot_names]

        mc = MCSamples(
            samples=samples[:, param_indices],
            weights=weights,
            names=plot_names,
            labels=plot_labels,
            label=label,
        )
        mc_samples_list.append(mc)

    # Create triangle plot
    g = plots.get_subplot_plotter(width_inch=figsize[0])
    g.settings.figure_legend_frame = True
    g.settings.legend_fontsize = 11
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6

    # Plot with different styles for each chain
    g.triangle_plot(
        mc_samples_list,
        params=plot_names,
        filled=filled,
        contour_colors=colors[:len(mc_samples_list)],
        contour_ls=linestyles[:len(mc_samples_list)],
        legend_loc='upper right',
    )

    if title:
        plt.suptitle(title, y=1.02)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_file}")

    plt.show()


def _corner_plot_fallback(
    samples_list, weights_list, param_names, labels, colors,
    params_to_plot, param_labels, output_file, figsize
):
    """Fallback corner plot using matplotlib only (no getdist)."""
    try:
        import corner
    except ImportError:
        print("Neither getdist nor corner is installed.")
        print("Install with: pip install getdist  or  pip install corner")
        return

    if params_to_plot is None:
        params_to_plot = param_names

    param_indices = [param_names.index(p) for p in params_to_plot if p in param_names]

    default_labels = {
        'Omega_m': r'$\Omega_{\rm m}$',
        'sigma8': r'$\sigma_8$',
        'S8': r'$S_8$',
    }
    if param_labels:
        default_labels.update(param_labels)

    plot_labels = [default_labels.get(param_names[i], param_names[i]) for i in param_indices]

    if colors is None:
        colors = ['blue', 'red', 'green', 'orange']

    fig = None
    for i, (samples, weights, color) in enumerate(zip(samples_list, weights_list, colors)):
        fig = corner.corner(
            samples[:, param_indices],
            weights=weights,
            labels=plot_labels,
            color=color,
            fig=fig,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=(i == 0),
            levels=[0.68, 0.95],
            smooth=1.0,
        )

    if labels:
        # Add legend manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=c, linewidth=2, label=l)
            for c, l in zip(colors[:len(samples_list)], labels)
        ]
        fig.legend(handles=legend_elements, loc='upper right', frameon=True)

    if output_file:
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_file}")

    plt.show()


def create_hsc_style_plot(
    samples: np.ndarray,
    weights: np.ndarray,
    param_names: list,
    comparison_data: dict = None,
    output_file: str = None,
):
    """Create HSC-style corner plot for Omega_m, sigma_8, S_8.

    This produces a plot similar to the HSC-Y3 paper figure.

    Parameters
    ----------
    samples : np.ndarray
        Parameter samples including Omega_m, sigma8, S8.
    weights : np.ndarray
        Sample weights.
    param_names : list
        Parameter names.
    comparison_data : dict, optional
        Dictionary with comparison datasets:
        {'name': {'samples': array, 'weights': array, 'color': str, 'linestyle': str}}
    output_file : str, optional
        Output file path.
    """
    try:
        from getdist import MCSamples, plots
    except ImportError:
        print("getdist required for HSC-style plot. Install with: pip install getdist")
        return

    # Compute derived parameters if needed
    samples_full, names_full = compute_derived_params(samples, param_names)

    # Check required parameters
    required = ['Omega_m', 'sigma8', 'S8']
    if not all(p in names_full for p in required):
        print(f"Missing required parameters. Available: {names_full}")
        return

    # Main sample
    param_indices = [names_full.index(p) for p in required]
    mc_main = MCSamples(
        samples=samples_full[:, param_indices],
        weights=weights,
        names=required,
        labels=[r'\Omega_{\rm m}', r'\sigma_8', r'S_8'],
        label='This work',
    )

    samples_list = [mc_main]
    colors = ['#1f77b4']  # Blue
    linestyles = ['-']
    filled = [True]

    # Add comparison data
    if comparison_data:
        for name, data in comparison_data.items():
            comp_samples, comp_names = compute_derived_params(
                data['samples'], data.get('param_names', param_names)
            )
            indices = [comp_names.index(p) for p in required if p in comp_names]
            if len(indices) == len(required):
                mc_comp = MCSamples(
                    samples=comp_samples[:, indices],
                    weights=data.get('weights', np.ones(len(comp_samples))),
                    names=required,
                    labels=[r'\Omega_{\rm m}', r'\sigma_8', r'S_8'],
                    label=name,
                )
                samples_list.append(mc_comp)
                colors.append(data.get('color', 'red'))
                linestyles.append(data.get('linestyle', '--'))
                filled.append(data.get('filled', False))

    # Create plot
    g = plots.get_subplot_plotter(width_inch=7)
    g.settings.figure_legend_frame = True
    g.settings.legend_fontsize = 10
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 13
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6

    g.triangle_plot(
        samples_list,
        params=required,
        filled=filled,
        contour_colors=colors,
        contour_ls=linestyles,
        legend_loc='upper right',
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_file}")

    plt.show()


# =============================================================================
# Example with synthetic data
# =============================================================================

def generate_example_data():
    """Generate example data similar to HSC/DES/KiDS constraints."""
    np.random.seed(42)

    n_samples = 10000

    # HSC-Y3 like constraints (tighter S8)
    # Use truncated normal to avoid negative values
    omega_m_hsc = np.clip(np.random.normal(0.31, 0.05, n_samples), 0.1, 0.6)
    sigma8_hsc = np.clip(np.random.normal(0.76, 0.06, n_samples), 0.4, 1.2)
    # Add correlation
    sigma8_hsc = np.clip(sigma8_hsc - 0.4 * (omega_m_hsc - 0.31), 0.4, 1.2)
    S8_hsc = sigma8_hsc * np.sqrt(omega_m_hsc / 0.3)

    # DES-Y3 like constraints
    omega_m_des = np.clip(np.random.normal(0.34, 0.06, n_samples), 0.1, 0.6)
    sigma8_des = np.clip(np.random.normal(0.73, 0.07, n_samples), 0.4, 1.2)
    sigma8_des = np.clip(sigma8_des - 0.5 * (omega_m_des - 0.34), 0.4, 1.2)
    S8_des = sigma8_des * np.sqrt(omega_m_des / 0.3)

    # KiDS-1000 like constraints
    omega_m_kids = np.clip(np.random.normal(0.30, 0.07, n_samples), 0.1, 0.6)
    sigma8_kids = np.clip(np.random.normal(0.78, 0.08, n_samples), 0.4, 1.2)
    sigma8_kids = np.clip(sigma8_kids - 0.45 * (omega_m_kids - 0.30), 0.4, 1.2)
    S8_kids = sigma8_kids * np.sqrt(omega_m_kids / 0.3)

    # Planck 2018 like constraints (higher S8)
    omega_m_planck = np.clip(np.random.normal(0.315, 0.02, n_samples), 0.1, 0.6)
    sigma8_planck = np.clip(np.random.normal(0.81, 0.02, n_samples), 0.4, 1.2)
    sigma8_planck = np.clip(sigma8_planck + 0.3 * (omega_m_planck - 0.315), 0.4, 1.2)
    S8_planck = sigma8_planck * np.sqrt(omega_m_planck / 0.3)

    return {
        'HSC-Y3': {
            'samples': np.column_stack([omega_m_hsc, sigma8_hsc, S8_hsc]),
            'weights': np.ones(n_samples),
            'param_names': ['Omega_m', 'sigma8', 'S8'],
            'color': '#1f77b4',
            'linestyle': '-',
            'filled': True,
        },
        'DES-Y3': {
            'samples': np.column_stack([omega_m_des, sigma8_des, S8_des]),
            'weights': np.ones(n_samples),
            'param_names': ['Omega_m', 'sigma8', 'S8'],
            'color': '#2ca02c',
            'linestyle': '-.',
            'filled': False,
        },
        'KiDS-1000': {
            'samples': np.column_stack([omega_m_kids, sigma8_kids, S8_kids]),
            'weights': np.ones(n_samples),
            'param_names': ['Omega_m', 'sigma8', 'S8'],
            'color': '#d62728',
            'linestyle': '--',
            'filled': False,
        },
        'Planck 2018': {
            'samples': np.column_stack([omega_m_planck, sigma8_planck, S8_planck]),
            'weights': np.ones(n_samples),
            'param_names': ['Omega_m', 'sigma8', 'S8'],
            'color': '#ff7f0e',
            'linestyle': ':',
            'filled': False,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create corner plots for cosmological parameter constraints.'
    )
    parser.add_argument(
        '--chain', type=str, default=None,
        help='Path to chain file (.npz from nautilus sampler)'
    )
    parser.add_argument(
        '--output', type=str, default='corner_plot.png',
        help='Output file path (PNG recommended for faster viewing)'
    )
    parser.add_argument(
        '--example', action='store_true',
        help='Generate example plot with synthetic data'
    )
    parser.add_argument(
        '--params', type=str, nargs='+', default=['Omega_m', 'sigma8', 'S8'],
        help='Parameters to plot'
    )

    args = parser.parse_args()

    if args.example or args.chain is None:
        print("Generating example corner plot with synthetic data...")
        print("This demonstrates HSC-Y3 / DES-Y3 / KiDS-1000 / Planck 2018 style comparison")

        data = generate_example_data()

        # Extract main dataset
        main_key = 'HSC-Y3'
        main_data = data.pop(main_key)

        # Create comparison plot
        try:
            from getdist import MCSamples, plots

            # Build MCSamples list
            samples_list = []
            colors = []
            linestyles = []
            filled_list = []

            # Main sample first
            mc_main = MCSamples(
                samples=main_data['samples'],
                weights=main_data['weights'],
                names=['Omega_m', 'sigma8', 'S8'],
                labels=[r'\Omega_{\rm m}', r'\sigma_8', r'S_8'],
                label=f'{main_key} 3x2pt small scales',
            )
            samples_list.append(mc_main)
            colors.append(main_data['color'])
            linestyles.append(main_data['linestyle'])
            filled_list.append(True)

            # Add comparison datasets
            for name, d in data.items():
                mc = MCSamples(
                    samples=d['samples'],
                    weights=d['weights'],
                    names=['Omega_m', 'sigma8', 'S8'],
                    labels=[r'\Omega_{\rm m}', r'\sigma_8', r'S_8'],
                    label=name,
                )
                samples_list.append(mc)
                colors.append(d['color'])
                linestyles.append(d['linestyle'])
                filled_list.append(d['filled'])

            # Create triangle plot
            g = plots.get_subplot_plotter(width_inch=7)
            g.settings.figure_legend_frame = True
            g.settings.legend_fontsize = 9
            g.settings.axes_fontsize = 12
            g.settings.lab_fontsize = 14
            g.settings.alpha_filled_add = 0.5
            g.settings.solid_contour_palefactor = 0.6

            g.triangle_plot(
                samples_list,
                params=['Omega_m', 'sigma8', 'S8'],
                filled=filled_list,
                contour_colors=colors,
                contour_ls=linestyles,
                legend_loc='upper right',
            )

            plt.savefig(args.output, bbox_inches='tight', dpi=300)
            print(f"Example plot saved to {args.output}")
            # Also save PNG for preview
            png_output = args.output.replace('.pdf', '.png')
            plt.savefig(png_output, bbox_inches='tight', dpi=150)
            print(f"PNG preview saved to {png_output}")
            plt.show()

        except ImportError:
            print("getdist not installed. Install with: pip install getdist")
            return

    else:
        # Load real chain
        print(f"Loading chain from {args.chain}...")
        samples, weights, param_names = load_chain(args.chain)
        print(f"Loaded {len(samples)} samples with parameters: {param_names}")

        # Compute derived parameters
        samples_full, names_full = compute_derived_params(samples, param_names)
        print(f"Parameters with derived: {names_full}")

        # Create corner plot
        create_corner_plot(
            samples_list=[samples_full],
            weights_list=[weights],
            param_names=names_full,
            params_to_plot=args.params,
            output_file=args.output,
        )


if __name__ == '__main__':
    main()
