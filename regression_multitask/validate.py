import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

import models.models as models
import regression_multitask as reg
import utils.nako as nako


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

    val_mae, val_r2 = [], []
    for experiment_name in args.experiments:
        start = time.time()
        print(f'----------{experiment_name}-----------')
        args = add_experiment_args(experiment_name, args)
        model_file = models_dir.joinpath(f'{experiment_name}.pt')
        results_file = models_dir.joinpath(f'{experiment_name}.json')

        # Load evaluation objects
        feature_name = ['a_rr_sys', 'a_rr_dia']
        dataloader_val, scaler = get_dataloader(args, feature_name)
        model = get_model(args, model_file)

        # Check for dead convolutional layers
        dead_layer_count = 0
        for name, parameters in model.named_parameters():
            if 'conv' in name:
                max_weight = parameters.flatten().abs().max()

                if max_weight <= 1e-4:
                    dead_layer_count += 1

        print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

        # Inference on val set
        preds_val, targets_val, _ = reg.predict(model, dataloader_val, args.device)

        if args.normalize:
            preds_val, targets_val = (
                scaler.inverse_transform(preds_val),
                scaler.inverse_transform(targets_val),
            )

        # Metrics
        mae_val_sys = mean_absolute_error(targets_val[:, 0], preds_val[:, 0])
        print(f'Mean Absolute Error systolic (val) = {mae_val_sys:.2f}')
        r2_val_sys = r2_score(targets_val[:, 0], preds_val[:, 0])
        print(f'R2 systolic (val) = {r2_val_sys:.2f}')

        mae_val_dias = mean_absolute_error(targets_val[:, 1], preds_val[:, 1])
        print(f'Mean Absolute Error diastolic (val) = {mae_val_dias:.2f}')
        r2_val_dias = r2_score(targets_val[:, 1], preds_val[:, 1])
        print(f'R2 diastolic (val) = {r2_val_dias:.2f}')

        val_mae.append(mae_val_sys + r2_val_dias)
        val_r2.append(r2_val_sys + r2_val_dias)

        print(f'Total validation time: {time.time() - start:.1f} s')

    print('Best model from validation MAE: ', np.array(val_mae).argmin() + 1)
    print('Best model from validation R2: ', np.array(val_r2).argmax() + 1)
