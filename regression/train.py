import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.models as models
import utils.nako as nako


def get_args():
    parser = argparse.ArgumentParser(
        description='Supervised training for regression from retinal images'
    )
    parser.add_argument(
        '--backbone',
        default='resnet18',
        type=str,
        help='backbone model',
        choices=[
            'resnet18',
            'resnet50',
            'inceptionv3',
            'efficientnetv2',
            'xception',
            'vit',
            'swin',
        ],
    )
    parser.add_argument(
        '--method',
        default='transfer',
        type=str,
        help='transfer learning (freeze backbone) or finetune or train from scratch',
        choices=['transfer', 'finetune', 'scratch'],
    )
    parser.add_argument(
        '--normalize',
        default='none',
        type=str,
        help='normalize target',
        choices=['none', 'standard'],
    )
    parser.add_argument(
        '--imagesize',
        default=224,
        type=int,
        help='image size, only square images are supported',
    )
    parser.add_argument(
        '--batchsize', default=128, type=int, help='batch size for training'
    )
    parser.add_argument(
        '--epochs', default=10, type=int, help='number of epochs for training'
    )
    parser.add_argument(
        '--kfold',
        default=0,
        type=int,
        help='index of 5-fold split to train, if 0 the 5-fold split is not used',
        choices=[0, 1, 2, 3, 4, 5],
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
    args.backbone_name = args.backbone
    args.backbone = models.MODEL_DICT.get(args.backbone, args.backbone)
    args.imagesize = (args.imagesize, args.imagesize)
    return args


def get_dataloader(args, feature_name):
    nako_paths = nako.get_nako_paths(dataset='590', image_res=str(args.imagesize[0]))
    transform = models.get_augmentations(
        args.imagesize, normalization=nako.NORMALIZATION
    )

    dataset_train = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform['train'],
        split='train',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )
    dataset_val = nako.NakoDataset(
        nako_paths,
        nako.IMAGE_TYPES,
        transform=transform['test'],
        split='val',
        kfold=args.kfold,
        feature_name=feature_name,
        drop_nan=True,
    )

    if args.normalize == 'normalize':
        scaler = models.Scaler()
        dataset_train.labels = scaler.fit_transform(dataset_train.labels)
        dataset_val.labels = scaler.transform(dataset_val.labels)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.numworkers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.numworkers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader_train, dataloader_val


def get_train_objs(args):
    freeze = args.method == 'transfer'
    weights = None if args.method == 'scratch' else 'DEFAULT'

    if args.backbone_name in ['xception', 'vit', 'swin']:
        model = models.TimmwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights=weights,
            freeze=freeze,
            n_classes=1,
            linear=True,
        )
    else:
        model = models.CNNwithPredictionHead(
            img_size=args.imagesize,
            backbone=args.backbone,
            weights=weights,
            freeze=freeze,
            n_classes=1,
            linear=True,
        )

    uncompiled_model = model
    model = torch.compile(uncompiled_model)  # Set drop_last=True in the dataloader

    base_lr = (
        1e-4 * args.batchsize / 256
        if args.method != 'transfer'
        else 1e-3 * args.batchsize / 256
    )
    params = model.parameters() if args.method != 'transfer' else model.fc.parameters()
    optimizer = AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_fn = nn.MSELoss()

    return model, uncompiled_model, optimizer, loss_fn


def train(model, dataloader, loss_fn, optimizer, scaler, device):
    model.to(device)
    model.train()
    epoch_loss = 0.0
    for b, (imgs, labels) in enumerate(dataloader):
        # print(f'Batch {b + 1} out of {len(dataloader_train)}...')
        imgs, labels = imgs.to(device), labels.unsqueeze(1).to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)

    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()
    val_loss = 0.0
    for b, (img, labels) in enumerate(dataloader):
        img, labels = img.to(device), labels.unsqueeze(1).to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(img)
                loss = loss_fn(outputs, labels)

        val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    return val_loss


def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    preds, targets, embeddings = [], [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                z, y = model(img, return_z=True)

        preds.extend(y.squeeze().tolist())
        targets.extend(labels.tolist())
        embeddings.extend(z.tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    embeddings = np.array(embeddings)

    return preds, targets, embeddings


if __name__ == '__main__':
    args = get_args()
    start = time.perf_counter()

    ##################### TRAINING PARAMS ######################
    # utils.set_seed(nako.SEED)
    experiment_name = f'reg_{args.backbone_name}_{args.imagesize[0]}_{args.method}_{args.normalize}_{args.kfold}'
    print(experiment_name)

    # Where results will be saved
    project_dir = Path.cwd()
    models_dir = project_dir.joinpath('checkpoints')
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir.joinpath(f'{experiment_name}.pt')
    results_file = models_dir.joinpath(f'{experiment_name}.json')

    # Training objects
    feature_name = 'basis_age'
    dataloader_train, dataloader_val = get_dataloader(args, feature_name)
    model, uncompiled_model, optimizer, loss_fn = get_train_objs(args)

    ###################### TRAINING LOOP #########################
    patience = 10 if args.method == 'scratch' else 5
    early_stopping = models.EarlyStopping(
        patience=patience, min_delta=0.1, model_file=model_file, verbose=True
    )
    scaler = torch.amp.GradScaler()
    loss_train, loss_val = [], []
    for epoch in range(args.epochs):
        start_time = time.perf_counter()

        # Training
        train_loss = train(
            model, dataloader_train, loss_fn, optimizer, scaler, args.device
        )
        loss_train.append(train_loss)

        # Validation
        val_loss = validate(model, dataloader_val, loss_fn, args.device)
        loss_val.append(val_loss)
        # scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss, uncompiled_model.state_dict())
        if early_stopping.stop_training:
            break

        end_time = time.perf_counter()
        print(
            f'Epoch {epoch + 1}, train loss {train_loss:.3f}, val loss {val_loss:.3f}, {end_time - start_time:.1f} s'
        )

    ###################### TRAINING CURVE #########################
    loss = {'loss_train': loss_train, 'loss_val': loss_val}
    with open(results_file, 'w') as f:
        json.dump(loss, f)

    print(f'Total training time: {time.perf_counter() - start:.1f} s')
