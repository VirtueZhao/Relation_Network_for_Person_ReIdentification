import torch
from reid.evaluation_metrics import cmc, mean_ap


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
    query_ids = [pid for _, pid, _ in query]
    gallery_ids = [pid for _, pid, _ in gallery]
    query_cams = [cam for _, _, cam in query]
    gallery_cams = [cam for _, _, cam in gallery]

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print("Mean AP: {:4.1%}".format(mAP))

    cmc_configs = {
        'market1501': dict(
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True
        )
    }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, **params) for name, params in cmc_configs.items()}
    print('CMC Scores{:>12}'.format('Market1501'))
    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'][0], mAP



