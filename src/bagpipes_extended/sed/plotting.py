"""Additional plotting scripts to make mujy summary plots."""

from os import PathLike
from pathlib import Path

import bagpipes
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

aanda_columnwidth = 256.0748 / 72.27
aanda_textwidth = 523.5307 / 72.27


def setup_aanda_style(dark: bool = False):
    """
    A helper function to setup the A&A style.

    Parameters
    ----------
    dark : bool, optional
        Use a dark plotting style, by default `False`.
    """

    rc_fonts = {
        "font.family": "serif",
        "font.size": 7,
        "figure.figsize": (aanda_columnwidth, 3),
        "text.usetex": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.labelsize": 7,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.labelsize": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1,
        "image.interpolation": "none",
        "text.latex.preamble": (
            r"""
        \usepackage{amsmath}
        \usepackage{txfonts}
        \usepackage{siunitx}
        %
        \DeclareMathAlphabet{\mathsc}{OT1}{cmr}{m}{sc}
        \def\testbx{bx}%
        \DeclareRobustCommand{\ion}[2]{%
        \relax\ifmmode
        \ifx\testbx\f@series
        {\mathbf{#1\,\mathsc{#2}}}\else
        {\mathrm{#1\,\mathsc{#2}}}\fi
        \else\textup{#1\,{\mdseries\textsc{#2}}}%
        \fi}
        %
        """
        ),
    }

    if dark:
        rc_fonts |= {
            "text.color": "white",
            "axes.facecolor": "0C1C23",  # axes background color
            "axes.edgecolor": "#e1e9ec",  # axes edge color
            "axes.labelcolor": "#e1e9ec",
            "grid.color": "#e1e9ec",
            "legend.edgecolor": "#e1e9ec",
            "legend.facecolor": "inherit",
            "legend.labelcolor": "#e1e9ec",
            "xtick.color": "#e1e9ec",
            "ytick.color": "#e1e9ec",
            "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
            "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
        }

    plt.rcdefaults()
    mpl.rcParams.update(rc_fonts)

    return


class modLogFormatter(mpl.ticker.LogFormatter):
    """
    A modification of the default LogFormatter for ticks.
    """

    def _num_to_string(self, x, vmin, vmax):

        if x > 10000:
            s = "%1.0e" % x
        elif x < 1 and x >= 0.001:
            s = "%g" % x
        elif x < 0.001:
            s = "%1.0e" % x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s


def mscatter(
    ax: plt.Axes, x: ArrayLike, y: ArrayLike, m: ArrayLike | None = None, **kw
) -> mpl.collections.PathCollection:
    """
    Allow for multiple types of markers in one scatter plot.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which the points will be plotted.
    x : ArrayLike
        The x-coordinates of the data.
    y : ArrayLike
        The y-coordinates of the data.
    m : ArrayLike | None, optional
        The marker style, by default `None`. Heterogeneous arrays can be
        passed, unlike the default `plt.scatter()`.
    **kw : dict, optional
        Any other keywords.

    Returns
    -------
    mpl.collections.PathCollection
        The plotted points.
    """

    import matplotlib.markers as mmarkers

    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def mujy_summary_plot(
    fit: bagpipes.fitting.fit.fit,
    skip_no_obs: bool = True,
    log_lam: bool = False,
    lw: float = 1.0,
    color: str = "blue",
    ptsize: float = 30,
    label: str | None = None,
    color1: str | None = None,
    color2: str | None = None,
    background_spectrum: bool = True,
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Create a summary plot for bagpipes fits.

    The spectrum is plotted in `"muJy"`.

    Parameters
    ----------
    fit : bagpipes.fitting.fit.fit
        The fit object from running `bagpipes`.
    skip_no_obs : bool, optional
        Do not plot points masked from the fit, by default `True`.
    log_lam : bool, optional
        Plot wavelengths in logarithmic units, by default `False`.
    lw : float, optional
        The linewidth used for plotting, by default `1.0`.
    color : str, optional
        The colour used for the observed photometry, by default `"blue"`.
    ptsize : float, optional
        The size of the markers for the observed photometry, by
        default `30`.
    label : str | None, optional
        A label to assign to the observed photometry, by default `None`.
    color1 : str | None, optional
        The colour to use for the full model spectrum, by default `None`.
    color2 : str | None, optional
        The colour to use for the model photometry, by default `None`.
    background_spectrum : bool, optional
        Show the full model spectrum in the background, by default `True`.
    save : bool, optional
        Save the figure to the `pipes/plots` directory, by default `True`.
    show : bool, optional
        Show the figure, by default `False`.
    """

    setup_aanda_style()

    fig = plt.figure(figsize=(aanda_textwidth, 6), constrained_layout=True)

    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.75, 1])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_sfh = fig.add_subplot(gs[1, 0:2])
    ax_mass = fig.add_subplot(gs[1, 2])
    ax_sfr = fig.add_subplot(gs[1, 3])

    if "redshift" in fit.fitted_model.params:
        z_ref = np.median(fit.posterior.samples["redshift"])

    else:
        z_ref = fit.fitted_model.model_components["redshift"]

    galaxy = fit.galaxy
    photometry = np.copy(galaxy.photometry)
    filter_list = np.copy(galaxy.filter_set.filt_list)

    if skip_no_obs:
        mask = photometry[:, 1] != 0.0
        photometry = photometry[mask, :]
        filter_list = filter_list[mask]

    markers = []
    for f, p in zip(filter_list, photometry[:, 1]):
        f_name = Path(f).stem.lower()
        if "niriss" in f_name:
            markers.append("s")
        elif ("hst" in f_name) or ("jwst" in f_name):
            markers.append("D")
        else:
            markers.append("o")

    if log_lam:
        ax_spec.set_xlim(
            (np.log10(galaxy.filter_set.eff_wavs.min()) - 0.025),
            (np.log10(galaxy.filter_set.eff_wavs.max()) + 0.025),
        )
    else:
        ax_spec.set_xlim(
            10 ** (np.log10(galaxy.filter_set.eff_wavs.min()) - 0.05 - 4),
            10 ** (np.log10(galaxy.filter_set.eff_wavs.max()) + 0.05 - 4),
        )

    # Convert from ergscma to mujy
    conversion = 10**-29 * 2.9979 * 10**18 / (photometry[:, 0] ** 2)

    photometry[:, 1] /= conversion
    photometry[:, 2] /= conversion

    mask = photometry[:, 1] > 0.0
    ymax = 1.1 * np.nanmax((photometry[:, 1])[mask])

    ax_spec.set_ylim(0.0, ymax)
    ax_spec.tick_params(axis="x", which="both", top=False)

    if log_lam:
        x_vals = np.log10(photometry[:, 0])
    else:
        x_vals = photometry[:, 0] / 1e4

    ax_spec.errorbar(
        x_vals,
        photometry[:, 1],
        yerr=photometry[:, 2],
        lw=lw,
        linestyle=" ",
        capsize=3,
        capthick=1,
        zorder=3,
    )

    mscatter(
        ax_spec,
        x_vals,
        photometry[:, 1],
        color=color,
        s=ptsize,
        zorder=4,
        linewidth=lw,
        facecolor=color,
        label=label,
        m=markers,
        edgecolor=color,
    )

    for marker_style, label_marker in zip(
        ["s", "D", "o"], ["NIRISS", "JWST/HST", "Other"]
    ):

        ax_spec.scatter(
            ax_spec.get_xlim()[0],
            -3,
            color=color,
            s=ptsize,
            zorder=4,
            linewidth=lw,
            facecolor=color,
            label=label_marker,
            marker=marker_style,
            edgecolor=color,
        )

    # dummy_markers
    ax_spec.legend(borderaxespad=1.5)

    if color1 == None:
        color1 = "darkorange"

    if color2 == None:
        color2 = "darkorange"

    if log_lam:
        # Plot the posterior photometry and full spectrum.
        log_wavs = np.log10(fit.posterior.model_galaxy.wavelengths * (1.0 + z_ref))
        log_eff_wavs = np.log10(fit.galaxy.filter_set.eff_wavs)
    else:
        log_wavs = fit.posterior.model_galaxy.wavelengths * (1.0 + z_ref) / 1e4
        log_eff_wavs = fit.galaxy.filter_set.eff_wavs / 1e4

    if background_spectrum:
        spec_post = np.percentile(
            fit.posterior.samples["spectrum_full"],
            (16, 84),
            axis=0,
        ).T

        spec_post = spec_post.astype(float)  # fixes weird isfinite error

        conversion = (
            10**-29
            * 2.9979
            * 10**18
            / (fit.posterior.model_galaxy.wavelengths * (1.0 + z_ref)) ** 2
        )

        ax_spec.fill_between(
            log_wavs,
            spec_post[:, 0] / conversion,
            spec_post[:, 1] / conversion,
            zorder=1,
            facecolor=color1,
            edgecolor=color1,
            linewidth=0.5,
            alpha=0.5,
        )

    phot_post = (
        np.percentile(fit.posterior.samples["photometry"], (16, 84), axis=0).T
        / np.array(
            [
                10**-29 * 2.9979 * 10**18 / (fit.galaxy.filter_set.eff_wavs) ** 2,
                10**-29 * 2.9979 * 10**18 / (fit.galaxy.filter_set.eff_wavs) ** 2,
            ]
        ).T
    )

    for j in range(fit.galaxy.photometry.shape[0]):

        if skip_no_obs and fit.galaxy.photometry[j, 1] == 0.0:
            continue

        phot_band = fit.posterior.samples["photometry"][:, j] / (
            10**-29 * 2.9979 * 10**18 / (fit.galaxy.filter_set.eff_wavs)[j] ** 2
        )
        mask = (phot_band > phot_post[j, 0]) & (phot_band < phot_post[j, 1])

        phot_1sig = phot_band[mask]
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]

        if phot_1sig.min() < ymax:
            ax_spec.scatter(
                wav_array,
                phot_1sig,
                color=color2,
                zorder=2,
                alpha=0.05,
                s=30,
                rasterized=True,
            )

    def _obs_to_restframe(x):
        return x / (1 + z_ref)

    def _restframe_to_obs(x):
        return x * (1 + z_ref)

    ax_spec.semilogx()

    ax_spec.get_xaxis().set_major_formatter(
        modLogFormatter(labelOnlyBase=False, minor_thresholds=(100, 0.4))
    )
    ax_spec.get_xaxis().set_minor_formatter(
        modLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
    )
    ax_spec.set_xlabel(rf"$\lambda_{{\rm{{observed}}}}\,$[\textmu m]")
    ax_spec.set_ylabel(rf"$F_{{\nu}}\,$[\textmu Jy]")

    ax_rf = ax_spec.secondary_xaxis(
        "top", functions=(_obs_to_restframe, _restframe_to_obs)
    )
    ax_rf.set_xlabel(rf"$\lambda_{{\rm{{rest-frame}}}}\,$[\textmu m]; $z={z_ref:.3f}$")

    ax_rf.get_xaxis().set_major_formatter(
        modLogFormatter(labelOnlyBase=False, minor_thresholds=(100, 0.4))
    )
    ax_rf.get_xaxis().set_minor_formatter(
        modLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
    )

    age_of_universe = np.interp(z_ref, bagpipes.utils.z_array, bagpipes.utils.age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    # Plot the SFH
    x = fit.posterior.sfh.ages * 10**-9

    post_mask = np.all(post > 0.0, axis=-1)
    post = post[post_mask]
    x = x[post_mask]

    ax_sfh.plot(x, np.log10(post[:, 1]), color="purple", zorder=2)
    ax_sfh.fill_between(
        x,
        np.log10(post[:, 0]),
        np.log10(post[:, 2]),
        color="purple",
        alpha=0.5,
        zorder=1,
        lw=0,
        label=label,
    )

    ax_sfh.set_ylim(
        np.log10(0.9 * np.nanmin(post[:, 1][np.argwhere(x < age_of_universe)[:-10]])),
        np.log10(np.max([ax_sfh.get_ylim()[1], 1.25 * np.max(post[:, 2])])),
    )
    ax_sfh.set_xlim(1e-3, 1.05 * age_of_universe)

    ax_sfh.semilogx()
    ax_sfh.get_xaxis().set_major_formatter(
        modLogFormatter(labelOnlyBase=False, minor_thresholds=(100, 0.4))
    )
    ax_sfh.set_ylabel(
        r"$ \log_{10} \left(\rm{SFR}\, /\, M_{\odot} \rm{yr}^{-1}\right)$"
    )
    ax_sfh.set_xlabel(r"Age\,\,[Gyr]")

    # zvals = [0, 0.5, 1, 2, 4, 10]
    # zvals = [0.,0.5,1.,1.5,2,2.5,3,4,5,6,10]
    zvals = [0.0, 0.25, 0.5, 1.0, 1.5, 2, 3, 4, 6, 10]
    ax_z = ax_sfh.twiny()
    ax_z.set_xticks(
        np.interp(
            zvals, bagpipes.utils.z_array, age_of_universe - bagpipes.utils.age_at_z
        )
    )
    ax_z.set_xticklabels(["$" + str(z) + "$" for z in zvals])
    ax_z.set_xlim(ax_sfh.get_xlim())
    ax_z.set_xlabel(r"Redshift")
    ax_z.xaxis.minorticks_off()

    mass_samples = fit.posterior.samples["stellar_mass"]

    _hist1d(
        mass_samples[np.invert(np.isnan(mass_samples))],
        ax_mass,
        smooth=True,
    )

    ax_mass.set_xlabel(r"$\log_{10}\left(M_*/ \rm{M}_{\odot}\right)$")
    mass_low, mass_med, mass_high = np.nanpercentile(mass_samples, [16, 50, 84])
    ax_mass.set_title(
        rf"$\log_{{10}}\left(M_*/ \rm{{M}}_{{\odot}}\right)"
        rf" = {mass_med:.2f}^{{+{mass_high-mass_med:.2f}}}"
        rf"_{{-{mass_med-mass_low:.2f}}}$"
    )

    sfr_samples = np.log10(fit.posterior.samples["sfr"])

    _hist1d(
        sfr_samples[np.invert(np.isnan(sfr_samples))],
        ax_sfr,
        smooth=True,
    )

    ax_sfr.set_xlabel(
        r"$ \log_{10} \left(\rm{SFR}_{100}\, /\, M_{\odot} \rm{yr}^{-1}\right)$"
    )
    sfr_low, sfr_med, sfr_high = np.nanpercentile(sfr_samples, [16, 50, 84])
    ax_sfr.set_title(
        r"$ \log_{10} \left(\rm{SFR}_{100}\right)"
        rf" = {sfr_med:.2f}^{{+{sfr_high-sfr_med:.2f}}}"
        rf"_{{-{sfr_med-sfr_low:.2f}}}$"
    )

    chisq_phot = np.nanmin(fit.posterior.samples["chisq_phot"])

    n_bands = np.sum(fit.galaxy.photometry[:, 1] != 0.0)

    fig.suptitle(
        rf"\texttt{{{fit.run.replace("_","\\_")}}}: ID {fit.galaxy.ID}"
        rf"\quad\quad $\chi^2_{{\rm{{phot}}}}="
        rf"{chisq_phot:.2f} / {n_bands:0d}="
        rf"{chisq_phot / n_bands:.2f}$"
    )

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_mujy_summary.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)


def _hist1d(
    samples,
    ax,
    smooth=False,
    label=None,
    color="orange",
    percentiles=True,
    zorder=4,
    bins=50,
    lw=2,
    color1=None,
    color2=None,
    alpha=None,
):
    """Modified from bagpipes code."""

    if color == "orange":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.7

    if color == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4

    if color == "blue":
        color1 = "blue"
        color2 = "dodgerblue"
        alpha = 0.6

    if color == "gray":
        color1 = "black"
        color2 = "gray"
        alpha = 0.7

    if label is not None:
        x_label = fix_param_names([label])
        ax.set_xlabel(x_label)

    width = samples.max() - np.max([samples.min(), -99.0])
    x_range = (
        np.max([samples.min(), -99.0]) - width / 10.0,
        samples.max() + width / 10.0,
    )

    y, x = np.histogram(samples, bins=bins, density=True, range=x_range)

    y = gaussian_filter(y, 1.5)

    if smooth:
        x_midp = (x[:-1] + x[1:]) / 2.0
        ax.plot(x_midp, y, color=color1, zorder=zorder - 1)
        ax.fill_between(
            x_midp, np.zeros_like(y), y, color=color2, alpha=alpha, zorder=zorder - 2
        )
        ax.plot(
            [x_midp[0], x_midp[0]], [0, y[0]], color=color1, zorder=zorder - 1, lw=lw
        )

        ax.plot(
            [x_midp[-1], x_midp[-1]], [0, y[-1]], color=color1, zorder=zorder - 1, lw=lw
        )

    else:
        x_hist, y_hist = make_hist_arrays(x, y)
        ax.plot(x_hist, y_hist, color="black")

    if percentiles:
        for percentile in [16, 50, 84]:
            ax.axvline(
                np.percentile(samples, percentile),
                linestyle="--",
                color="black",
                zorder=zorder,
                lw=1,
            )

    ax.set_ylim(bottom=0)
    ax.set_xlim(x_range)
    plt.setp(ax.get_yticklabels(), visible=False)
