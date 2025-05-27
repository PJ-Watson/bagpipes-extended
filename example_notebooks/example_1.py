"""Example code."""

import os
from pathlib import Path

import yaml

# Or similar
root_dir = Path(os.getenv("ROOT_DIR"))
out_base_dir = root_dir / "2024_08_16_A2744_v4" / "glass_niriss_hires"

# Setup a separate directory for the binned data
bin_data_dir = out_base_dir / "binned_data"
root_name = "glass-a2744"

# Load the file containing all the info on photometric data
with open(out_base_dir / "conv_ancillary_data.yaml", "r") as file:
    info_dict = yaml.safe_load(file)

# Where the grism data can be found
grizli_extraction_dir = out_base_dir.parent / "grizli_home" / "MultiRegExtractions"
grizli_extraction_dir = out_base_dir.parent / "grizli_home" / "for_others"

from glass_niriss.isophotal import reproject_and_convolve

repr_seg_path = (
    out_base_dir / "PSF_matched_data" / f"repr_{root_name}_seg_map_forced.fits"
)

if not repr_seg_path.is_file():
    # Whichever mosaic we used as a reference for the photometry
    ref_mosaic = grizli_extraction_dir.parent / "Prep" / f"{root_name}-ir_drc_sci.fits"

    # The path of the original segmentation map
    orig_seg = grizli_extraction_dir / f"{root_name}-ir_seg_mod_1.fits"

    reproject_and_convolve(
        ref_path=ref_mosaic,
        orig_images=orig_seg,
        psfs=None,
        psf_target=None,
        out_dir=out_base_dir / "PSF_matched_data",
        new_names=f"repr_{root_name}_seg_map_forced.fits",
        reproject_image_kw={"method": "interp", "order": 0, "compress": False},
    )


# Create the [bag]pipes directory
pipes_dir = out_base_dir / "sed_fitting" / "pipes"
pipes_dir.mkdir(exist_ok=True, parents=True)

# Create the filter directory; populate as needed
filter_dir = pipes_dir / "filter_throughputs"
filter_dir.mkdir(exist_ok=True, parents=True)

# Create a list of the filters used in our data
filter_list = []
for key in info_dict.keys():
    filter_list.append(str(filter_dir / f"{key}.txt"))

# Create the atlas directory
atlas_dir = pipes_dir / "atlases"
atlas_dir.mkdir(exist_ok=True, parents=True)


obj_id = 1761
obj_z = 3.06
# obj_id = 497
# obj_z = 0.3033
obj_id = 1597
obj_z = 2.6724

# obj_id = 3311
# obj_z = 1.34
obj_id = 2606
obj_z = 0.296

# # obj_id = 497
# # obj_z = 0.30
# # obj_id = 2224
# # obj_z = 0.3064
# obj_id = 1742
# obj_z = 3.06
# obj_id = 908
# obj_z = 0.3033
# obj_id = 3278
# obj_z = 0.296
obj_id = 2074
obj_z = 1.364

# obj_id = 2328
# obj_z = 1.363
# obj_id = 2720
# obj_z = 3.04
# obj_id = 5021
# obj_z = 1.8868
# obj_id = 3137
# obj_z = 0.9384

# obj_id = 90001
# obj_z = 1.337
# obj_id = 3528
# obj_z = 0.318
# obj_id = 3112
# obj_z = 2.7184
# obj_id = 1333
# obj_z = 1.996
# obj_id = 1991
# obj_z = 2.178
obj_id = 732
obj_z = 0.2966

use_hex = False
bin_diameter = 3
target_sn = 20
sn_filter = "jwst-nircam-f150w"

from glass_niriss.sed import bin_and_save

binned_name = f"{obj_id}_{"hexbin" if use_hex else "vorbin"}_{bin_diameter}_{target_sn}_{sn_filter}"
binned_data_path = bin_data_dir / f"{binned_name}_data.fits"

if not binned_data_path.is_file():
    bin_and_save(
        obj_id=obj_id,
        out_dir=bin_data_dir,
        seg_map=repr_seg_path,
        info_dict=info_dict,
        sn_filter=sn_filter,
        target_sn=target_sn,
        bin_diameter=bin_diameter,
        use_hex=use_hex,
        overwrite=True,
    )

from bagpipes_extended.pipeline import generate_fit_params

# bagpipes_atlas_params = generate_fit_params(obj_z=obj_z, z_range=0.01, continuity=True, min_age_bin=30)
bagpipes_atlas_params = generate_fit_params(
    obj_z=obj_z, z_range=0.0, continuity=True, min_age_bin=30
)

print(bagpipes_atlas_params)

from bagpipes_extended.sed import AtlasGenerator

n_samples = 1e7
n_cores = 18

remake_atlas = False
# run_name = (
#     f"z_{bagpipes_atlas_params["redshift"][0]}_"
#     f"{bagpipes_atlas_params["redshift"][1]}_"
#     f"{n_samples:.2E}"
# )
run_name = (
    f"z_{bagpipes_atlas_params["redshift"]}_"
    f"{bagpipes_atlas_params["redshift"]}_"
    f"{n_samples:.2E}"
)
atlas_path = atlas_dir / f"{run_name}.hdf5"

if not atlas_path.is_file() or remake_atlas:

    atlas_gen = AtlasGenerator(
        fit_instructions=bagpipes_atlas_params,
        filt_list=filter_list,
        phot_units="ergscma",
    )

    atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores)

    atlas_gen.write_samples(filepath=atlas_path)


import os

# from glass_niriss.sed import AtlasFitter
from functools import partial

import numpy as np
from astropy.table import Table

# exit()
from bagpipes_extended.pipeline import load_photom_bagpipes
from bagpipes_extended.sed import AtlasFitter

os.chdir(pipes_dir)

overwrite = False

load_fn = partial(
    load_photom_bagpipes, phot_cat=binned_data_path, cat_hdu_index="PHOT_CAT"
)

fit = AtlasFitter(
    fit_instructions=bagpipes_atlas_params,
    atlas_path=atlas_path,
    out_path=pipes_dir.parent,
    overwrite=overwrite,
)

obs_table = Table.read(binned_data_path, hdu="PHOT_CAT")
cat_IDs = np.arange(len(obs_table))[:]

catalogue_out_path = fit.out_path / Path(f"{binned_name}_{run_name}.fits")
if (not catalogue_out_path.is_file()) or overwrite:

    fit.fit_catalogue(
        IDs=cat_IDs,
        load_data=load_fn,
        spectrum_exists=False,
        make_plots=False,
        cat_filt_list=filter_list,
        run=f"{binned_name}_{run_name}",
        parallel=16,
        # redshifts = np.zeros(11)+3.06,
        # redshifts=np.repeat(obj_z, len(cat_IDs)),
        # redshift_range=0.001,
        redshift_range=None,
        n_posterior=1000,
    )
    print(fit.cat)
else:
    fit.cat = Table.read(catalogue_out_path)

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from astropy.io import fits

fig, axs = plt.subplots(
    3, 1, sharex=True, sharey=True, constrained_layout=True, figsize=(6, 12)
)

# for a in axs:

seg_map = fits.getdata(binned_data_path, hdu="SEG_MAP")

plot_map = np.full_like(seg_map, np.nan, dtype=float)
for row in fit.cat:
    plot_map[seg_map == int(row["#ID"])] = (
        row[
            # "continuity:massformed_50"
            # "stellar_mass_50"
            # "ssfr_50"
            # "sfr_50"
            # "continuity:metallicity_50"
            # "mass_weighted_age_50"
            "dust:Av_50"
            # "dust:eta"
            # "nebular:logU_50"
            # "redshift_50"
        ]
        # *row[
        #     "dust:eta_50"
        # ]
        # /
        # -
        # np.log10(
        # (len((seg_map == int(row["#ID"])).nonzero()[0])
        # * ((0.04 * 4.63) ** 2))
        # )
    )
plot_map[seg_map == 0] = np.nan
im = axs[1].imshow(
    plot_map,
    # np.log10(plot_map),
    origin="lower",
    # vmin=3,
    # vmax=9,
    # vmin=-4,
    vmax=0.2,
    # vmin=-12,
    # vmax=-8,
    # vmin=-8,
    # vmax=-3,
    # cmap="plasma",
    # vmin=0,
    cmap="rainbow",
    # cmap = cmc.lajolla
)
axs[1].set_facecolor("k")
# axs[1].colorbar(im)
plt.colorbar(im, ax=axs[1], label="Av")


seg_map = fits.getdata(binned_data_path, hdu="SEG_MAP")

plot_map = np.full_like(seg_map, np.nan, dtype=float)
for row in fit.cat:
    plot_map[seg_map == int(row["#ID"])] = row["stellar_mass_50"] - np.log10(
        (len((seg_map == int(row["#ID"])).nonzero()[0]) * ((0.04 * 4.63) ** 2))
    )
plot_map[seg_map == 0] = np.nan
im = axs[0].imshow(
    plot_map,
    # np.log10(plot_map),
    origin="lower",
    # vmin=3,
    # vmax=9,
    # vmin=-4,
    # vmax=1,
    # vmin=-12,
    # vmax=-8,
    # vmin=-8,
    # vmax=-3,
    cmap="plasma",
    # vmin=0,
    # cmap="rainbow"
    # cmap = cmc.lajolla
)
axs[0].set_facecolor("k")
plt.colorbar(im, ax=axs[0], label="M_*/kpc2")


seg_map = fits.getdata(binned_data_path, hdu="SEG_MAP")

plot_map = np.full_like(seg_map, np.nan, dtype=float)
for row in fit.cat:
    plot_map[seg_map == int(row["#ID"])] = (
        row[
            # "continuity:massformed_50"
            # "stellar_mass_50"
            # "ssfr_50"
            # "sfr_50"
            # "continuity:metallicity_50"
            "mass_weighted_age_50"
            # "dust:Av_50"
            # "dust:eta"
            # "nebular:logU_50"
            # "redshift_50"
        ]
        # *row[
        #     "dust:eta_50"
        # ]
        # /
        # -
        # np.log10(
        # (len((seg_map == int(row["#ID"])).nonzero()[0])
        # * ((0.04 * 4.63) ** 2))
        # )
    )
plot_map[seg_map == 0] = np.nan
im = axs[2].imshow(
    plot_map,
    # np.log10(plot_map),
    origin="lower",
    # vmin=3,
    # vmax=9,
    # vmin=-4,
    # vmax=1,
    # vmin=-12,
    # vmax=-8,
    # vmin=-8,
    # vmax=-3,
    # cmap="plasma",
    # vmin=0,
    cmap="rainbow",
    # cmap = cmc.lajolla
)
axs[2].set_facecolor("k")
plt.colorbar(im, ax=axs[2], label="Age_MW")
import pathlib

print(pathlib.Path().cwd())
plt.savefig("1e7_ID732.pdf", dpi=1200)

plt.show()
