# Metrics specified are mutual information and jaccard coefficient
import numpy as np
from sklearn.metrics import jaccard_similarity_score


def mutual_information(slice1, slice2, n_bins=20):
    """ Mutual information for joint histogram
    Taken from [Mutual Information as an Image Metric]
    (https://matthew-brett.github.io/teaching/mutual_information.html)
    """
    hgram, _, _ = np.histogram2d(
                        slice1.ravel(),
                        slice2.ravel(),
                        bins=n_bins)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def jaccard_coeff(slice1, slice2):
    """ Calculates the intersection of union (iou) of
    the two image slices 
    """
    slice1 = slice1.ravel()
    slice2 = slice2.ravel()
    return jaccard_similarity_score(slice1, slice2)
