# Dark Emulator tutorials

Tutorials and fitting examples using [Dark Emulator](https://dark-emulator.readthedocs.io) with `cloelib` and `cloelike` for real-space observables (w_p, ΔΣ, 3x2pt).

**📹 Short overview (≈6 min):** [YouTube Unlisted VIDEO](https://youtu.be/tE1stIqNpXk)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `dark_emulator.ipynb` | Introduction to Dark Emulator in cloelib: cosmology, HOD, and computing observables. |
| `dark_emulator_observables_usage.ipynb` | How observables (w_p, ΔΣ) are obtained via the Dark Emulator pipeline in cloelib. |
| `fitting_gc_wp.ipynb` | Fit **galaxy clustering** w_p(R) with mock data; likelihood + `scipy.optimize.minimize` or Nautilus. |
| `fitting_ggl_delta_sigma.ipynb` | Fit **galaxy–galaxy lensing** ΔΣ(R) with mock data; likelihood + minimize or Nautilus. |
| `fitting_3x2pt.ipynb` | **3x2pt** (GC + GGL + WL) short demo in the notebook; uses mock data and Nautilus with small `n_live`. |

## Production 3x2pt run

For full 3x2pt sampling (larger `n_live`, sigma8/S8, checkpoint + chain), run from the **repository root**:

```bash
python playground/scripts/sampling/darkemu_3x2pt_full_sampling.py --n_live 500 --output_dir playground/results
```

Then create a corner plot from the saved chain:

```bash
python playground/scripts/plotting/corner_plot.py \
  --chain playground/results/chain_3x2pt_full.npz \
  --output playground/results/corner_3x2pt.pdf \
  --params Omega_m sigma8 S8
```

## Requirements

- `cloelib`, `cloelike`, and the `dark_emulator` package (see repository root).
- For sampling: `nautilus`. For corner plots: `getdist` or `corner`.


