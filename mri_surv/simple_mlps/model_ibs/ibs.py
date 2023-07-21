from scipy import interpolate
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score, cumulative_dynamic_auc
import numpy as np


def retrieve_brier_scores(bins, preds_raw, train_struc, test_struc):
    """
    retrieve_brier_scores

    Gets IBS

    Parameters
    ----------
    bins : Input time bins for the data, should always basically be 0, 24, 48, 108 or as 
    appropriate for CPH and Weibull models

    preds_raw : 
    model predictions. These should be the marginal probabilities
    np array of dims N x t where N is sample and t is time bin

    train_struc : 
        a numpy datastructure to estimate censoring weights. Plz use function below
        to generate (generally, ADNI data, though you COULD use NACC data)

    test_struc : 
        the target to evaluate preds_raw w.r.t. (use ADNI test or whole NACC)

    Returns
    -------
    Tuple
        brier scores for array input and interpolator
    """
    bins = bins.copy()
    new_max = min(float(max(test_struc['time'])),108)
    truncated_bins = np.concatenate([bins[:-1],[new_max-1]], axis=-1)
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    preds_brier = interp(truncated_bins)
    brier_scores = integrated_brier_score(train_struc, test_struc, preds_brier, truncated_bins)
    return brier_scores, interp

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(hits == 1, obss)], dtype=[('hit',bool),('time',float)])
