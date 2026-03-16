import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from models import models
from regression_multitask.train import predict
from utils import nako


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation of classifiers from retinal images.'
    )
    parser.add_argument(
        '--experiments',
        default=['regmulti_resnet18_224_finetune_none_1'],
        nargs='+',
        type=str,
        help='experiment name',
    )
    parser.add_argument(
        '--batchsize', default=128, type=int, help='batch size for training'
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
    args.normalize = all_args[4] == 'standard'
    args.kfold = int(all_args[5])

    args.backbone_name = args.backbone
    args.backbone = models.MODEL_DICT.get(args.backbone, args.backbone)
    return args


def get_dataloader(args, feature_name):
    nako_paths = nako.get_nako_paths(dataset='590', image_res=str(args.imagesize[0]))
    transform = models.get_augmentations(
        args.imagesize, normalization=nako.NORMALIZATION
    )['test']

    dataset_train = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform,
        split='train',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )
    dataset_test = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform,
        split='test',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )
    ids_test = dataset_test.ids
    types_test = np.array([p.split(sep='/')[-2] for p in dataset_test.image_paths])

    scaler = None
    if args.normalize:
        scaler = models.Scaler()
        dataset_train.labels = scaler.fit_transform(dataset_train.labels)
        dataset_test.labels = scaler.transform(dataset_test.labels)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.numworkers,
        pin_memory=True,
    )

    return dataloader_test, ids_test, types_test, scaler


def get_model(args, model_file):
    if args.backbone_name in ['xception', 'vit', 'swin']:
        model = models.TimmMultitask(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=[1, 1],
            linear=True,
        )
    else:
        model = models.CNNMultitask(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=[1, 1],
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

    for experiment_name in args.experiments:
        start = time.time()
        print(f'----------{experiment_name}-----------')
        args = add_experiment_args(experiment_name, args)
        model_file = models_dir.joinpath(f'{experiment_name}.pt')
        results_file = models_dir.joinpath(f'{experiment_name}.json')

        # Load evaluation objects
        feature_name = ['a_rr_sys', 'a_rr_dia']
        dataloader_test, ids_test, types_test, scaler = get_dataloader(
            args, feature_name
        )
        model = get_model(args, model_file)

        # Check for dead convolutional layers
        dead_layer_count = 0
        for name, parameters in model.named_parameters():
            if 'conv' in name:
                max_weight = parameters.flatten().abs().max()

                if max_weight <= 1e-4:
                    dead_layer_count += 1

        print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

        # Inference on test set
        preds_test, targets_test, embeddings_test = predict(
            model, dataloader_test, args.device
        )

        if args.normalize:
            preds_test, targets_test = (
                scaler.inverse_transform(preds_test),
                scaler.inverse_transform(targets_test),
            )

        # Metrics
        mae_test = mean_absolute_error(targets_test[:, 0], preds_test[:, 0])
        print(f'Mean Absolute Error (test) = {mae_test:.2f}')

        r2_test = r2_score(targets_test[:, 0], preds_test[:, 0])
        print(f'R2 (test) = {r2_test:.2f}')

        var_test = np.var(preds_test[:, 0])
        print(f'Variance (test) = {var_test:.2f}')

        # Load results, to save loss curves with the other values
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Save results
        results = {
            'loss_train': results['loss_train'],
            'loss_val': results['loss_val'],
            'ID': ids_test.tolist(),
            'type': types_test.tolist(),
            'targets': targets_test.tolist(),
            'preds': preds_test.tolist(),
        }

        with open(results_file, 'w') as f:
            json.dump(results, f)

        print(f'Total testing time: {time.time() - start:.1f} s')
