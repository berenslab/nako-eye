import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from nako_eye.classification.train import predict
from nako_eye.models import models
from nako_eye.utils import nako


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation of classifiers from retinal images.'
    )
    parser.add_argument(
        '--experiments',
        default=['clf_resnet18_224_finetune_1'],
        nargs='+',
        type=str,
        help='experiment name',
    )
    parser.add_argument(
        '--batchsize', default=64, type=int, help='batch size for training'
    )
    parser.add_argument(
        '--numworkers',
        default=8,
        type=int,
        help='number of subprocesses to use for dataloading',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help='device in which training will take place',
    )

    args = parser.parse_args()
    return args


def add_experiment_args(experiment_name, args):
    all_args = experiment_name.split('_')
    args.backbone = all_args[1]
    args.imagesize = (int(all_args[2]), int(all_args[2]))
    args.method = all_args[3]
    args.kfold = int(all_args[4])

    args.backbone_name = args.backbone
    args.backbone = models.MODEL_DICT.get(args.backbone, args.backbone)
    return args


def get_dataloader(args, feature_name):
    nako_paths = nako.get_nako_paths(dataset='590', image_res=str(args.imagesize[0]))
    transform = models.get_augmentations(
        args.imagesize, normalization=nako.NORMALIZATION
    )['test']
    dataset_val = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform,
        split='val',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.numworkers,
        pin_memory=True,
    )

    n_classes = len(dataset_val.mapping)

    return dataloader_val, n_classes


def get_model(args, n_classes, model_file):
    if args.backbone_name in ['xception', 'vit', 'swin']:
        model = models.TimmwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=n_classes,
            linear=True,
        )
    else:
        model = models.CNNwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=n_classes,
            linear=True,
        )
    model.load_state_dict(
        torch.load(model_file, weights_only=True, map_location=torch.device('cpu'))
    )
    return model


if __name__ == '__main__':
    args = get_args()

    # Where results will be saved
    project_dir = Path.cwd()
    models_dir = project_dir.joinpath('checkpoints')

    val_acc, val_auc = [], []
    for experiment_name in args.experiments:
        start = time.perf_counter()
        print(f'----------{experiment_name}-----------')
        args = add_experiment_args(experiment_name, args)
        model_file = models_dir.joinpath(f'{experiment_name}.pt')
        results_file = models_dir.joinpath(f'{experiment_name}.json')

        # Load evaluation objects
        feature_name = 'basis_sex'
        dataloader_val, n_classes = get_dataloader(args, feature_name)
        model = get_model(args, n_classes, model_file)

        # Check for dead convolutional layers
        dead_layer_count = 0
        for name, parameters in model.named_parameters():
            if 'conv' in name:
                max_weight = parameters.flatten().abs().max()

                if max_weight <= 1e-4:
                    dead_layer_count += 1

        print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

        # Inference on val set
        preds_val, probs_val, targets_val, _ = predict(
            model, dataloader_val, args.device
        )

        # Metrics
        acc_val = accuracy_score(targets_val, preds_val)
        print(f'Accuracy (val) = {acc_val:.2f}')

        probs_val = probs_val[:, 1] if n_classes == 2 else probs_val
        auc_val = roc_auc_score(targets_val, probs_val)
        print(f'AUC (val) = {auc_val:.2f}')

        val_acc.append(acc_val)
        val_auc.append(auc_val)

        # Trained epochs
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f'Trained for {len(results["loss_val"])} epochs.')

        print(f'Total validation time: {time.perf_counter() - start:.1f} s')

    print('Best model from validation accuracy: ', np.array(val_acc).argmax() + 1)
    print('Best model from validation AUC: ', np.array(val_auc).argmax() + 1)
