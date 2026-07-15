import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from nako_eye.models import models
from nako_eye.regression.train import predict
from nako_eye.utils import nako


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation of regressors from retinal images.'
    )
    parser.add_argument(
        '--experiments',
        default=['reg_resnet18_224_finetune_none_1'],
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
    args.normalize = all_args[4] == 'norm'
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
    dataset_val = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform,
        split='val',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )

    scaler = None
    if args.normalize:
        scaler = models.Scaler()
        dataset_train.labels = scaler.fit_transform(dataset_train.labels)
        dataset_val.labels = scaler.transform(dataset_val.labels)

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.numworkers,
        pin_memory=True,
    )

    return dataloader_val, scaler


def get_model(args, model_file):
    if args.backbone_name in ['xception', 'vit', 'swin']:
        model = models.TimmwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=1,
            linear=True,
        )
    else:
        model = models.CNNwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights='DEFAULT',
            n_classes=1,
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

    val_mae, val_r2 = [], []
    for experiment_name in args.experiments:
        start = time.time()
        print(f'----------{experiment_name}-----------')
        args = add_experiment_args(experiment_name, args)
        model_file = models_dir.joinpath(f'{experiment_name}.pt')
        results_file = models_dir.joinpath(f'{experiment_name}.json')

        # Load evaluation objects
        feature_name = 'basis_age'
        model = get_model(args, model_file)
        dataloader_val, scaler = get_dataloader(args, feature_name)

        # Check for dead convolutional layers
        dead_layer_count = 0
        for name, parameters in model.named_parameters():
            if 'conv' in name:
                max_weight = parameters.flatten().abs().max()

                if max_weight <= 1e-4:
                    dead_layer_count += 1

        print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

        # Inference on val set
        preds_val, targets_val, embeddings_val = predict(
            model, dataloader_val, args.device
        )

        if args.normalize:
            preds_val, targets_val = (
                scaler.inverse_transform(preds_val),
                scaler.inverse_transform(targets_val),
            )

        # Metrics
        mae_val = mean_absolute_error(targets_val, preds_val)
        print(f'Mean Absolute Error (val) = {mae_val:.2f}')

        r2_val = r2_score(targets_val, preds_val)
        print(f'R2 (val) = {r2_val:.2f}')

        val_mae.append(mae_val)
        val_r2.append(r2_val)

        # Trained epochs
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f'Trained for {len(results["loss_val"])} epochs.')

        print(f'Total validation time: {time.time() - start:.1f} s')

    print('Best model from validation MAE: ', np.array(val_mae).argmin() + 1)
    print('Best model from validation R2: ', np.array(val_r2).argmax() + 1)
