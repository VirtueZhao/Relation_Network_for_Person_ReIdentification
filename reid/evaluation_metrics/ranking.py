import numpy as np
from ..utils import to_numpy
from sklearn.metrics import average_precision_score


def cmc():
    pass


def mean_ap(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    aps = []
    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No Valid Query")
    return np.mean(aps)
