# Playground Repository

Welcome to the **Playground** repository of the [`cloe` organisation](https://github.com/cloe-org)! 🚀

This repository serves as a sandbox for tutorials, exercises, and validation for various features, scripts, and models related to `cloelib` and `cloelike`. It provides an open space to learn quickly how to get around the `cloe` organisation.

To explore the contents of this repository, you may need to download synthetic example data available at [Zenodo – cloe-org Community](https://zenodo.org/communities/cloe-org/records).

The data can be read using the [`euclidlib`](https://euclidlib.readthedocs.io/en/latest/intro.html) library.

## 🔧 Features
- Tutorials on how to run `cloelib`

## 📂 Structure
The repository is organized as follows:

```
playground/
│── tutorials/       # Jupyter notebooks for cosmological codes, computing observables and evaluating the likelihood
│── validation/      # Jupyter notebooks for comparison of cosmological observables against `cloelib`
│── exercises/       # Jupyter notebooks with exercises that use `cloelib` for teaching purposes
│── protocols_test/  # Script for validating a new protocol interface
│── README.md        # This file
```

## 📦 Installation
To use this repository, clone it:

```bash
git clone https://github.com/cloe-org/playground.git
cd playground
```

It might require as dependencies `cloelib`, `cloelike`, `euclidlib` and others.

## 🚀 Usage
You can run the provided notebooks for experimentation!

```bash
jupyter notebook tutorials/observables/photo.ipynb
jupyter notebook tutorials/observables/spectro.ipynb
```

## 🤝 Contributing
Contributions are welcome! If you want to propose an experiment or new feature, feel free to open an issue or submit a pull request. Follow the general contribution guidelines of the organisation.

## 📬 Contact
For any questions or discussions, feel free to open an issue or reach out to the [cloe-maintainers](https://github.com/orgs/cloe-org/teams/cloe-maintainers).

Happy learning! 🎉
