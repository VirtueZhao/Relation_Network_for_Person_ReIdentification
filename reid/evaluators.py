import torch
from evaluation_metrics import cmc, mean_ap


def pairwise_distance(features, query=None, gallery=None, metric=None):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def evaluate_all(distmat, query=None, gallery=None, cmc_topk=(1, 5, 10), dataset=None, top1=True):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print("Mean AP: {:4.1%}".format(mAP))


