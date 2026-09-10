<p align="center">
  <img src="https://github.com/user-attachments/assets/2dadf56f-9d51-4cda-83e7-97ac0ca1ba97" alt="playground banner" width="400">
</p>


**Playground repository of the [`cloe` organisation](https://github.com/cloe-org)! 🚀**

This repository serves as a sandbox for tutorials, exercises, and validation for various features, scripts, and models related to `cloelib` and `cloelike`. It provides an open space to learn quickly how to get around the `cloe` organisation.

To explore the contents of this repository, you may need to download synthetic example data available at [Zenodo – cloe-org Community](https://zenodo.org/communities/cloe-org/records).

The data can be read using the [`euclidlib`](https://euclidlib.readthedocs.io/en/latest) library.

Happy learning! 🎉

## 📦 Installation
To use this repository, clone it, no installation needed!

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
jupyter notebook tutorials/observables/bao.ipynb
```

For **Dark Emulator** (3x2pt, GC, GGL fitting), run the notebooks in `tutorials/dark_emulator/`; for production sampling use `python playground/scripts/sampling/darkemu_3x2pt_full_sampling.py` (from the repo root).

## 📬 Contact
For any questions or discussions, feel free to open an issue or reach out to the [cloe-maintainers](https://github.com/orgs/cloe-org/teams/cloe-maintainers).

## 📂 Structure
The repository is organized as follows:

<!-- REPO-STRUCTURE-START -->
```mermaid
graph LR
    root[🗂 playground]
    root__docs[docs/]
    root__exercises[exercises/]
    root__exercises__lensing_growth_factor_ipynb[lensing_growth_factor.ipynb]
    root__scripts[scripts/]
    root__scripts__sampling[sampling/]
    root__scripts__sampling__nautilus_example_py[nautilus_example.py]
    root__scripts__generate_structure_py[generate_structure.py]
    root__tutorials[tutorials/]
    root__tutorials__cosmology[cosmology/]
    root__tutorials__cosmology__cosmology_ipynb[cosmology.ipynb]
    root__tutorials__cosmology__neutrinos_ipynb[neutrinos.ipynb]
    root__tutorials__likelihood[likelihood/]
    root__tutorials__likelihood__likelihood_GCspectro_BAO_ipynb[likelihood_GCspectro_BAO.ipynb]
    root__tutorials__likelihood__likelihood_GCspectro_Pls_ipynb[likelihood_GCspectro_Pls.ipynb]
    root__tutorials__likelihood__likelihood_GCspectro_Pls_BAO_ipynb[likelihood_GCspectro_Pls_BAO.ipynb]
    root__tutorials__likelihood__photometric_likelihoods_ipynb[photometric_likelihoods.ipynb]
    root__tutorials__observables[observables/]
    root__tutorials__observables__bao_ipynb[bao.ipynb]
    root__tutorials__observables__photo_ipynb[photo.ipynb]
    root__tutorials__observables__spectro_ipynb[spectro.ipynb]
    root__tutorials__profiling[profiling/]
    root__tutorials__profiling__time_comparison_ipynb[time_comparison.ipynb]
    root__tutorials__profiling__time_profiling_ipynb[time_profiling.ipynb]
    root__validation[validation/]
    root__validation__validation_CosmoSIS_ipynb[validation_CosmoSIS.ipynb]
    root__validation__validation_ccl_ipynb[validation_ccl.ipynb]

    root --> root__docs
    root --> root__exercises
    root__exercises --> root__exercises__lensing_growth_factor_ipynb
    root --> root__scripts
    root__scripts --> root__scripts__sampling
    root__scripts__sampling --> root__scripts__sampling__nautilus_example_py
    root__scripts --> root__scripts__generate_structure_py
    root --> root__tutorials
    root__tutorials --> root__tutorials__cosmology
    root__tutorials__cosmology --> root__tutorials__cosmology__cosmology_ipynb
    root__tutorials__cosmology --> root__tutorials__cosmology__neutrinos_ipynb
    root__tutorials --> root__tutorials__likelihood
    root__tutorials__likelihood --> root__tutorials__likelihood__likelihood_GCspectro_BAO_ipynb
    root__tutorials__likelihood --> root__tutorials__likelihood__likelihood_GCspectro_Pls_ipynb
    root__tutorials__likelihood --> root__tutorials__likelihood__likelihood_GCspectro_Pls_BAO_ipynb
    root__tutorials__likelihood --> root__tutorials__likelihood__photometric_likelihoods_ipynb
    root__tutorials --> root__tutorials__observables
    root__tutorials__observables --> root__tutorials__observables__bao_ipynb
    root__tutorials__observables --> root__tutorials__observables__photo_ipynb
    root__tutorials__observables --> root__tutorials__observables__spectro_ipynb
    root__tutorials --> root__tutorials__profiling
    root__tutorials__profiling --> root__tutorials__profiling__time_comparison_ipynb
    root__tutorials__profiling --> root__tutorials__profiling__time_profiling_ipynb
    root --> root__validation
    root__validation --> root__validation__validation_CosmoSIS_ipynb
    root__validation --> root__validation__validation_ccl_ipynb
```
<!-- REPO-STRUCTURE-END -->

## 🤝 Contributing
This project follows the [all-contributors](https://allcontributors.org) specification. Contributions of any kind welcome!

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#review-gcanasherrera" title="Reviewed Pull Requests">👀</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#example-gcanasherrera" title="Examples">💡</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#tutorial-gcanasherrera" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chiaramoretti"><img src="https://avatars.githubusercontent.com/u/12472732?v=4?s=100" width="100px;" alt="Chiara Moretti"/><br /><sub><b>Chiara Moretti</b></sub></a><br /><a href="#code-chiaramoretti" title="Code">💻</a> <a href="#review-chiaramoretti" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-chiaramoretti" title="Maintenance">🚧</a> <a href="#tutorial-chiaramoretti" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#review-AndreaPezzotta" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#tutorial-AndreaPezzotta" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#review-PedroCarrilho" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#tutorial-PedroCarrilho" title="Tutorials">✅</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#code-santiagocasas" title="Code">💻</a> <a href="#review-santiagocasas" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josecolomanadal"><img src="https://avatars.githubusercontent.com/u/83759085?v=4?s=100" width="100px;" alt="Jose Coloma Nadal"/><br /><sub><b>Jose Coloma Nadal</b></sub></a><br /><a href="#code-josecolomanadal" title="Code">💻</a> <a href="#tutorial-josecolomanadal" title="Tutorials">✅</a> <a href="#bug-josecolomanadal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/llinke1"><img src="https://avatars.githubusercontent.com/u/42432333?v=4?s=100" width="100px;" alt="Laila Linke"/><br /><sub><b>Laila Linke</b></sub></a><br /><a href="#code-llinke1" title="Code">💻</a> <a href="#tutorial-llinke1" title="Tutorials">✅</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#code-itutusaus" title="Code">💻</a> <a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ippoppi"><img src="https://avatars.githubusercontent.com/u/50492103?v=4?s=100" width="100px;" alt="Filippo Oppizzi"/><br /><sub><b>Filippo Oppizzi</b></sub></a><br /><a href="#code-ippoppi" title="Code">💻</a> <a href="#ideas-ippoppi" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-ippoppi" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://marcobonici.github.io/"><img src="https://avatars.githubusercontent.com/u/58727599?v=4?s=100" width="100px;" alt="Marco Bonici"/><br /><sub><b>Marco Bonici</b></sub></a><br /><a href="#code-marcobonici" title="Code">💻</a> <a href="#ideas-marcobonici" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zahrabaghkhani"><img src="https://avatars.githubusercontent.com/u/47903409?v=4?s=100" width="100px;" alt="Zahra Baghkhani"/><br /><sub><b>Zahra Baghkhani</b></sub></a><br /><a href="#code-zahrabaghkhani" title="Code">💻</a> <a href="#doc-zahrabaghkhani" title="Documentation">📖</a> <a href="#ideas-zahrabaghkhani" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-zahrabaghkhani" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://arthurmloureiro.github.io"><img src="https://avatars.githubusercontent.com/u/6471279?v=4?s=100" width="100px;" alt="Arthur Loureiro"/><br /><sub><b>Arthur Loureiro</b></sub></a><br /><a href="#doc-arthurmloureiro" title="Documentation">📖</a> <a href="#code-arthurmloureiro" title="Code">💻</a> <a href="#ideas-arthurmloureiro" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KlaraBertmann"><img src="https://avatars.githubusercontent.com/u/153739278?v=4?s=100" width="100px;" alt="KlaraBertmann"/><br /><sub><b>KlaraBertmann</b></sub></a><br /><a href="#doc-KlaraBertmann" title="Documentation">📖</a> <a href="#ideas-KlaraBertmann" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jipdebuck"><img src="https://avatars.githubusercontent.com/u/236796982?v=4?s=100" width="100px;" alt="Jip de Buck"/><br /><sub><b>Jip de Buck</b></sub></a><br /><a href="#bug-jipdebuck" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://ntessore.page"><img src="https://avatars.githubusercontent.com/u/3993688?v=4?s=100" width="100px;" alt="Nicolas Tessore"/><br /><sub><b>Nicolas Tessore</b></sub></a><br /><a href="#bug-ntessore" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ivansladoljev"><img src="https://avatars.githubusercontent.com/u/144113061?v=4?s=100" width="100px;" alt="Ivan Sladoljev"/><br /><sub><b>Ivan Sladoljev</b></sub></a><br /><a href="#code-ivansladoljev" title="Code">💻</a> <a href="#ideas-ivansladoljev" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ytchang05"><img src="https://avatars.githubusercontent.com/u/67809413?v=4?s=100" width="100px;" alt="Yu-Ting"/><br /><sub><b>Yu-Ting</b></sub></a><br /><a href="#code-ytchang05" title="Code">💻</a> <a href="#bug-ytchang05" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

