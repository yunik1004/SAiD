"""Compute the WInD (Wasserstein Inception Distance)
"""
from dataclasses import dataclass
from typing import List
from cvxopt import matrix, solvers, spmatrix
import numpy as np
from scipy import sparse as sp
from sklearn.mixture import GaussianMixture
from .frechet_distance import frechet_distance


@dataclass
class StatisticGMM:
    """Dataclass for the statistic of each modal of GMM"""

    mean: np.ndarray
    cov: np.ndarray
    weight: float


def get_statistic_gmm(data: List[np.ndarray], num_clusters: int) -> List[StatisticGMM]:
    """Compute the statistics of the data based on GMM

    Parameters
    ----------
    data: List[np.ndarray]
        (z_dim,), List of the numpy 1-d arrays
    num_clusters: int
        The number of clusters in GMM

    Returns
    -------
    List[Statistic]
        List of mean, covariance of the data
    """
    gm = GaussianMixture(n_components=num_clusters).fit(data)

    means = gm.means_
    covs = gm.covariances_
    weights = gm.weights_

    return [
        StatisticGMM(mean=means[cdx], cov=covs[cdx], weight=weights[cdx])
        for cdx in range(num_clusters)
    ]


def wind(stats1: List[StatisticGMM], stats2: List[StatisticGMM]) -> float:
    """Compute the WInD between two GMM
    X1 ~ stats1 and X2 ~ stats2

    Parameters
    ----------
    stats1: List[StatisticGMM]
        Statistics of features 1
    stats2: List[StatisticGMM]
        Statistics of features 2

    Returns
    -------
    float
        WInD
    """
    num_clusters = len(stats1)

    d = np.zeros((num_clusters, num_clusters))
    for jdx in range(num_clusters):
        for kdx in range(num_clusters):
            d[jdx, kdx] = frechet_distance(
                stats1[jdx].mean, stats1[jdx].cov, stats2[kdx].mean, stats2[kdx].cov
            )

    c = matrix(d.reshape(-1, 1))
    h = matrix(
        np.array(
            [stat.weight for stat in stats1]
            + [stat.weight for stat in stats2]
            + [0 for _ in range(num_clusters * num_clusters)]
        ).reshape(-1, 1)
    )
    A = matrix(np.ones((1, num_clusters * num_clusters)))
    b = matrix(np.ones((1, 1)))

    # Compute G
    coeff_ineq1 = sp.block_diag(
        [[[1 for _ in range(num_clusters)]] for _ in range(num_clusters)], format="coo"
    )
    eye = sp.identity(num_clusters, dtype="int", format="coo")
    eye_large = sp.identity(num_clusters * num_clusters, dtype="int", format="coo")
    coeff_ineq2 = sp.bmat(
        [[eye for _ in range(num_clusters)]], dtype="int", format="coo"
    )
    coeff_G = sp.bmat(
        [[coeff_ineq1], [coeff_ineq2], [-eye_large]], dtype="int", format="coo"
    )
    G = spmatrix(
        coeff_G.data.tolist(),
        coeff_G.row.tolist(),
        coeff_G.col.tolist(),
        size=coeff_G.shape,
    )

    # Solve the LP
    sol = solvers.lp(
        c=c,
        G=G,
        h=h,
        A=A,
        b=b,
        solver="glpk",
        options={"glpk": {"msg_lev": "GLP_MSG_OFF"}},
    )

    return sol["primal objective"]
