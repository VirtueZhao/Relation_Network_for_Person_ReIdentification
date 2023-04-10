import os
import torch
import random
import argparse
import numpy as np
from tabulate import tabulate
from torch.nn import CrossEntropyLoss

from utils import build_data_loader
from triplet import TripletSemiHardLoss
from Relation_final_ver_last_multi_scale_large_losses import RelationModel as Model

def print_args(args):
    args_table = [
        ["Args", "Value"],
        ["GPU", args.gpu],
        ["Seed", args.seed],
        ["Max Epoch", args.max_epoch],
        ["Batch Size", args.batch_size],
        ["LR", args.lr],
        ["Backbone LR", args.backbone_lr],
        ["LR Schedule", args.decay_schedule],
        ["LR Decay Factor", args.lr_decay_factor],
        ["SGD Momentum", args.sgd_momentum],
        ["SGD Weight Decay", args.sgd_weight_decay],
        ["Num Workers", args.num_workers],
        ["Split", args.split],
        ["Num Individuals", args.num_individuals],
        ["Image Height", args.h],
        ["Image Width", args.w],
        ["Margin", args.margin],
        ["Dataset", args.dataset],
        ["Dataset Path", args.dataset_path],
        ["Combine TrainVal", args.combine_trainval],
        ["Log Steps", args.steps_per_log],
        ["Output Dir", args.output_dir]
    ]

    print(tabulate(args_table))


def main(args):
    print_args(args)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.decay_schedule is None:
        decay_schedule = (40, 60)
    else:
        decay_schedule = tuple(args.decay_schedule)

    log_directory = os.path.join(args.output_dir + "_" + args.dataset)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    np_ratio = args.num_individuals - 1

    # Build Data Loader
    dataset, train_loader, val_loader, test_loader = build_data_loader(args.dataset, args.split, args.dataset_path,
                                                                       args.h, args.w, args.batch_size,
                                                                       args.num_workers, args.combine_trainval, np_ratio
                                                                       )
    # Build Model
    model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256, num_classes=dataset.num_trainval_ids)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Losses and Optimizer
    cross_entropy_loss = CrossEntropyLoss()
    triplet_loss = TripletSemiHardLoss(device=device, margin=args.margin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="specify GPU")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--max_epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--backbone_lr", type=float, default=0.001, help="initial learning rate for resnet50")
    parser.add_argument("--decay_schedule", nargs='+', type=int, help="learning rate decay schedule")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="decaying coefficient")
    parser.add_argument("--sgd_momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0005, help="SGD weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    parser.add_argument("--split", type=int, default=0, help="split")
    parser.add_argument("--num_individuals", type=int, default=4, help="number of individuals in each batch")
    parser.add_argument("--h", type=int, default=384, help="height of input images")
    parser.add_argument("--w", type=int, default=128, help="width of input images")
    parser.add_argument("--margin", type=float, default=1.2, help="margin of triplet loss")
    parser.add_argument("--dataset", type=str, default="market1501", help="dataset: [market1501, dukemtmc, cuhk03]")
    parser.add_argument("--dataset_path", type=str, default="./datasets/", help="path to datasets")
    parser.add_argument("--combine_trainval", action="store_true", default=False, help="select train or trainval")
    parser.add_argument("--steps_per_log", type=int, default=100, help="frequency of printing")
    parser.add_argument("--output_dir", type=str, default="log", help="directory of log")

    args = parser.parse_args()
    main(args)

