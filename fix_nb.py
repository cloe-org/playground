import json

nb = json.load(open("tutorials/cosmology/baryon_boost.ipynb"))

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and "BACCOemuFLAMINGOPerturbations" in src:
        # Fix background= in FlamingoBaryonBoostMixin.__init__
        old = (
            "        FlamingoBaryonBoostMixin.__init__(\n"
            "            self,\n"
            "            background=background,\n"
            "            fgas_sigma=fgas_sigma,"
        )
        new = (
            "        FlamingoBaryonBoostMixin.__init__(\n"
            "            self,\n"
            "            fgas_sigma=fgas_sigma,"
        )
        src = src.replace(old, new)
        # Fix [0] on matter_power_spectrum result
        src = src.replace(
            "Pk_bacco_with_flamingo_A = pert_bacco_flamingo_A.matter_power_spectrum(z_ref, k_eval)[0]",
            "Pk_bacco_with_flamingo_A = pert_bacco_flamingo_A.matter_power_spectrum(z_ref, k_eval)",
        )
        cell["source"] = [src]
        print("Fixed markdown cell:", cell.get("id"))
        break
else:
    print("Target cell not found!")

json.dump(nb, open("tutorials/cosmology/baryon_boost.ipynb", "w"), indent=1)
print("Saved.")
