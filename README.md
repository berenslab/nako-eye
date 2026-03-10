# NAKO Ophthalmology Dataset
This repository contains the code for the paper "The German National Cohort Ophthalmology Dataset: Design Methods, Baseline Profile and Potential for AI-based Eye Research".

### Installation
A conda environment file is included to install the R and python packages used in this project.

```bash
conda env create -f environment.yml
```

The R packages itsadug and ggthemes ar not included in conda r-essentials but can be installed after activating the conda environment with the following commands:

```bash
R -e "install.packages('itsadug', repos='https://cloud.r-project.org')"
R -e "install.packages('ggthemes', repos='https://cloud.r-project.org')"
```

### Usage
To load any of the models from the paper for sex classification use:

```python
import models.models as models

# Possible choices: 'resnet18', 'resnet50', 'inceptionv3', 'efficientnetv2', 'xception', 'vit', 'swin'
backbone = 'swin'
n_classes = 2
model_file = 'path/to/checkpoint'

backbone = models.MODEL_DICT.get(backbone_name, backbone_name)
if args.backbone_name in ['xception', 'vit', 'swin']:
    model = models.TimmwithPredictionHead(backbone=backbone, n_classes=n_classes)
else:
    model = models.CNNwithPredictionHead(backbone=backbone, n_classes=n_classes)

model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))

```

To load the age regression models, just change n_classes to 1. For the multitask systolic and diastolic blood pressure prediction use:

```python
import models.models as models

# Possible choices: 'resnet18', 'resnet50', 'inceptionv3', 'efficientnetv2', 'xception', 'vit', 'swin'
backbone = 'swin'
n_classes = [1, 1]
model_file = 'path/to/checkpoint'

if args.backbone_name in ['xception', 'vit', 'swin']:
    model = models.TimmMultitask(backbone=backbone, n_classes=n_classes)
else:
    model = models.CNNMultitask(backbone=backbone, n_classes=n_classes)

model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))

```

### Weights download
The model weights corresponding to the experiments reported in the paper are available for download at [zenodo](https://zenodo.org/records/18607263).

They can also be downloaded and extracted using the following commands:
```bash
wget https://zenodo.org/records/18607263/files/nako-eye.zip?download=1
unzip nako-eye.zip -d <destination_directory>
```
