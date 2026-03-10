import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from torchvision.models import get_model

IMAGENET_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
MODEL_DICT = {
    'inceptionv3': 'inception_v3',
    'efficientnetv2': 'efficientnet_v2_s',
    'vit': 'vit_base_patch16_224',
    'swin': 'swin_base_patch4_window7_224',
}


def get_augmentations(img_size, normalization=None, imagenet=False):
    if imagenet:
        transform = {
            # Regular augmentations for imagenet
            'train': transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                ]
            ),
        }
    else:
        # Augmentations for fundus images
        transform = {
            'train': transforms.Compose(
                [
                    # transforms.RandomResizedCrop(size=img_size, scale=[0.9, 1.1], ratio=[0.9, 1.1]),
                    transforms.RandomRotation(degrees=(-15, 15)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.2, 0.2, 0, 0)], p=0.5
                    ),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose([transforms.ToTensor()]),
        }

    if normalization:
        normalize = transforms.Normalize(normalization['mean'], normalization['std'])
        _ = [transform[k].transforms.append(normalize) for k in transform.keys()]

    return transform


class CNNwithPredictionHead(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='resnet18',
        weights='DEFAULT',
        freeze=True,
        n_classes=2,
        linear=True,
    ):
        # n_classes = 1 for regression
        super().__init__()
        self.img_size = img_size
        self.backbone_id = backbone
        self.n_classes = n_classes
        self.linear = linear

        self.backbone = get_model(backbone, weights=weights)

        if hasattr(self.backbone, 'classifier'):
            # For efficientnet
            self.n_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        if 'inception' in backbone:
            self.backbone.aux_logits = False
            self.backbone.AuxLogits = None

        # Freeze backbone
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add fully connected layer and output
        self.init_fc()

    def init_fc(self):
        if self.linear:
            self.fc = nn.Linear(self.n_features, self.n_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.BatchNorm1d(self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_classes),
            )

    def forward(self, x, return_z=False):
        z = self.backbone(x)
        y = self.fc(z)
        return (z, y) if return_z else y


class CNNMultitask(CNNwithPredictionHead):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='resnet18',
        weights='DEFAULT',
        freeze=True,
        n_classes=[1, 1],
        linear=True,
    ):
        # n_classes = 1 for regression
        super().__init__(img_size, backbone, weights, freeze, n_classes[0], linear)
        self.n_classes = n_classes

        # Add fully connected layer
        self.init_fcs()

    def init_fcs(self):
        self.fc = nn.ModuleList([self.fc])

        for self.n_classes in self.n_classes[1:]:
            if self.linear:
                self.fc.append(nn.Linear(self.n_features, self.n_classes))
            else:
                self.fc.append(
                    nn.Sequential(
                        nn.Linear(self.n_features, self.n_features),
                        nn.BatchNorm1d(self.n_features),
                        nn.ReLU(),
                        nn.Linear(self.n_features, self.n_classes),
                    )
                )

    def forward(self, x, return_z=False):
        z = self.backbone(x)
        y = [fc(z) for fc in self.fc]
        return (z, y) if return_z else y


class TimmwithPredictionHead(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='vit_base_patch16_224',
        weights='DEFAULT',
        freeze=True,
        n_classes=2,
        linear=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone_id = backbone
        self.n_classes = n_classes
        self.linear = linear

        # Load backbone without classification head: vit_base_patch16_224, swin_base_patch4_window7_224
        pretrained = weights == 'DEFAULT'  # To mimic pytorch get_model
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained
        )  # model.default_cfg
        self.n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)

        # Turn off the last LayerNorm for regression
        if self.n_classes == 1:
            self.backbone.norm = nn.Identity()

        # Freeze backbone
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add fully connected layer
        self.init_fc()

    def init_fc(self):
        if self.linear:
            self.fc = nn.Linear(self.n_features, self.n_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.BatchNorm1d(self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_classes),
            )

    def forward(self, x, return_z=False):
        z = self.backbone(x)
        y = self.fc(z)
        return (z, y) if return_z else y


class TimmMultitask(TimmwithPredictionHead):
    def __init__(
        self,
        img_size=(224, 224),
        backbone='vit_base_patch16_224',
        weights='DEFAULT',
        freeze=True,
        n_classes=[1, 1],
        linear=True,
    ):
        super().__init__(img_size, backbone, weights, freeze, n_classes[0], linear)
        self.n_classes = n_classes

        # Add fully connected layers
        self.init_fcs()

    def init_fcs(self):
        self.fc = nn.ModuleList([self.fc])

        for self.n_classes in self.n_classes[1:]:
            if self.linear:
                self.fc.append(nn.Linear(self.n_features, self.n_classes))
            else:
                self.fc.append(
                    nn.Sequential(
                        nn.Linear(self.n_features, self.n_features),
                        nn.BatchNorm1d(self.n_features),
                        nn.ReLU(),
                        nn.Linear(self.n_features, self.n_classes),
                    )
                )

    def forward(self, x, return_z=False):
        z = self.backbone(x)
        y = [fc(z) for fc in self.fc]
        return (z, y) if return_z else y


class Scaler:
    """Normalize array by column statistics."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, arr):
        arr = self.scaler.fit_transform(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
        return arr.squeeze() if arr.shape[1] == 1 else arr

    def transform(self, arr):
        arr = self.scaler.transform(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
        return arr.squeeze() if arr.shape[1] == 1 else arr

    def inverse_transform(self, arr):
        arr = self.scaler.inverse_transform(
            arr.reshape(-1, 1) if arr.ndim == 1 else arr
        )
        return arr.squeeze() if arr.shape[1] == 1 else arr


class EarlyStopping:
    """
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        model_file (str): filename to save the best model.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    """

    def __init__(self, patience=5, min_delta=0.1, model_file='model.pt', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.model_file = model_file
        self.verbose = verbose
        self.best_loss = np.inf
        self.no_improvement_count = 0
        self.stop_training = False

    def __call__(self, val_loss, checkpoint):
        if (self.best_loss is None) or (val_loss < self.best_loss - self.min_delta):
            self.save_checkpoint(val_loss, checkpoint)
            self.no_improvement_count = 0
            self.best_loss = val_loss
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(
                        f'Stopping early after {self.patience} epochs with no improvement.'
                    )

    def save_checkpoint(self, val_loss, checkpoint):
        """Save checkpoint when validation loss decreases."""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_loss:.3f} → {val_loss:.3f}). Saving checkpoint...'
            )

        torch.save(checkpoint, self.model_file)


if __name__ == '__main__':
    print('models.py')
