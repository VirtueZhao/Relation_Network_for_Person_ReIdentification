import os
import time
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
import reid.evaluators as evaluators
from tabulate import tabulate
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from triplet import TripletSemiHardLoss
from reid.utils.meters import AverageMeter
from reid.evaluation_metrics import accuracy
from utils import get_data, adjust_lr_staircase
from Relation_final_ver_last_multi_scale_large_losses import RelationModel as Model

# import torch.nn as nn
# from torch.nn.modules import loss
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torch.nn.parallel import DataParallel



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
    dataset, train_loader, val_loader, test_loader = get_data(args.dataset, args.split, args.dataset_path,
                                                                       args.h, args.w, args.batch_size,
                                                                       args.num_workers, args.combine_trainval, np_ratio
                                                                       )

    # Build Model
    model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256, num_classes=dataset.num_trainval_ids)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Losses and Optimizer
    cross_entropy_loss = CrossEntropyLoss()
    triplet_loss = TripletSemiHardLoss(margin=args.margin)

    finetuned_params = list(model.base.parameters())

    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]

    param_groups = [{'params': finetuned_params, 'lr': args.lr * 0.1},
                    {'params': new_params, 'lr': args.lr}]

    optimizer = optim.SGD(param_groups, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay)

    for epoch in range(1, args.max_epoch + 1):
        print("Current Epoch: {}".format(epoch))

        adjust_lr_staircase(
            optimizer.param_groups,
            [args.backbone_lr, args.lr],
            epoch,
            decay_schedule,
            args.lr_decay_factor)

        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)

            (imgs, _, labels, _) = inputs
            inputs = Variable(imgs).float().cuda(device)
            labels = Variable(labels).cuda(device)

            optimizer.zero_grad()
            final_feat_list, logits_local_rest_list, logits_local_list, logits_rest_list, logits_global_list = model(inputs)
            T_loss = torch.sum(torch.stack([triplet_loss(output, labels) for output in final_feat_list]))

            C_loss_local = torch.sum(torch.stack(
                [cross_entropy_loss(output, labels) for output in logits_local_list]
            ), dim=0)

            C_loss_local_rest = torch.sum(torch.stack(
                [cross_entropy_loss(output, labels) for output in logits_local_rest_list]
            ), dim=0)

            C_loss_rest = torch.sum(torch.stack(
                [cross_entropy_loss(output, labels) for output in logits_rest_list]
            ), dim=0)

            C_loss_global = torch.sum(torch.stack(
                [cross_entropy_loss(output, labels) for output in logits_global_list]
            ), dim=0)

            C_loss = C_loss_local_rest + C_loss_global + C_loss_local + C_loss_rest
            loss = T_loss + 2 * C_loss

            losses.update(loss.data.item(), labels.size(0))
            prec1 = (
                sum([accuracy(output.data, labels.data)[0].item() for output in logits_local_rest_list]) +
                sum([accuracy(output.data, labels.data)[0].item() for output in logits_global_list]) +
                sum([accuracy(output.data, labels.data)[0].item() for output in logits_local_list]) +
                sum([accuracy(output.data, labels.data)[0].item() for output in logits_rest_list])
            ) / (12 + 12 + 12 + 9)

            precisions.update(prec1, labels.size(0))
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.steps_per_log == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, args.steps_per_log * batch_time.avg,
                              data_time.val, args.steps_per_log * data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    torch.save(model, os.path.join(log_directory, 'model.pth'))

    print("Evaluate Model")
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()

    print('Extracting Features for Test Datasets')
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(test_loader):
            data_time.update(time.time() - end)

            imgs_flip = torch.flip(imgs, [3])
            final_feat_list, _, _, _, _, = model(Variable(imgs).cuda(device))
            final_feat_list_flip, _, _, _, _ = model(Variable(imgs_flip).cuda(device))

            for j in range(len(final_feat_list)):
                if j == 0:
                    outputs = (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu()) / 2
                else:
                    outputs = torch.cat((outputs, (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu()) / 2), 1)
            outputs = F.normalize(outputs, p=2, dim=1)

            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(i + 1, len(test_loader),
                                                      batch_time.val, batch_time.avg,
                                                      data_time.val, data_time.avg))

    # Evaluating Distance Matrix
    distmat = evaluators.pairwise_distance(features, dataset.query, dataset.gallery)
    evaluators.evaluate_all(distmat, dataset.query, dataset.gallery, dataset=args.dataset, top1=True)


def evaluate(args):
    print("Evaluate Model")
    log_directory = os.path.join(args.output_dir + "_" + args.dataset)
    np_ratio = args.num_individuals - 1
    # Build Data Loader
    dataset, train_loader, val_loader, test_loader = get_data(args.dataset, args.split, args.dataset_path,
                                                                       args.h, args.w, args.batch_size,
                                                                       args.num_workers, args.combine_trainval, np_ratio
                                                                       )

    # model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(log_directory, 'model.pth'))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    print('Extracting Features for Test Datasets')
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(test_loader):
            data_time.update(time.time() - end)

            imgs_flip = torch.flip(imgs, [3])
            final_feat_list, _, _, _, _, = model(Variable(imgs).cuda(device))
            final_feat_list_flip, _, _, _, _ = model(Variable(imgs_flip).cuda(device))

            for j in range(len(final_feat_list)):
                if j == 0:
                    outputs = (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu()) / 2
                else:
                    outputs = torch.cat((outputs, (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu()) / 2), 1)
            outputs = F.normalize(outputs, p=2, dim=1)

            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(i + 1, len(test_loader),
                                                      batch_time.val, batch_time.avg,
                                                      data_time.val, data_time.avg))

    # Evaluating Distance Matrix
    distmat = evaluators.pairwise_distance(features, dataset.query, dataset.gallery)
    evaluators.evaluate_all(distmat, dataset.query, dataset.gallery, dataset=args.dataset, top1=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="specify GPU")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--max_epoch", type=int, default=3)
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
    # main(args)
    evaluate(args)

