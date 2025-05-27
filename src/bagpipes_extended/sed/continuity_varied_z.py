"""An implementation of the continuity SFH varying with redshift."""

import numpy as np
from numpy.typing import ArrayLike


def continuity_varied_z(self, sfr: ArrayLike, param: dict):
    """
    A redshift-varying continuity SFH.

    Parameters
    ----------
    sfr : ArrayLike
        The SFH of the current model galaxy.
    param : dict
        The parameter dictionary used to update the current SFH.
    """

    bin_edges_low = np.array(param.get("bin_edges_low", [0]))
    bin_edges_high = np.array(
        param.get("bin_edges_high", [np.floor(self.age_of_universe) * 10 ** (-6)])
    )
    n_bins = param.get("n_bins", 7)

    bin_edges = (
        np.concatenate(
            [
                bin_edges_low,
                np.geomspace(
                    bin_edges_low[-1],
                    bin_edges_high[0],
                    num=n_bins - len(bin_edges_low) - len(bin_edges_high) + 3,
                )[1:-1],
                bin_edges_high,
            ]
        )[::-1].astype(int)
        * 1e6
    )

    dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

    for i in range(1, n_bins + 1):
        mask = (self.ages < bin_edges[i - 1]) & (self.ages > bin_edges[i])
        sfr[mask] += 10 ** np.sum(dsfrs[: i - 1])
