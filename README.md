# NAKO Ophthalmology Dataset
This repository contains the code for the paper "The German National Cohort: Ophthalmological Assessment, Baseline Profile 
and Potential for AI-based Eye Research".

## Installation
A conda environment file is included to install the R and python packages used in this project.

```bash
conda env create -f environment.yml
```

The R packages itsadug and ggthemes are not included in conda r-essentials but must be installed manually after activating the conda environment with the following commands:

```bash
R -e "install.packages('itsadug', repos='https://cloud.r-project.org')"
R -e "install.packages('ggthemes', repos='https://cloud.r-project.org')"
```

## Repository structure
```
nako-eye/
├── classification/            # Sex classification (Male, Female)
├── regression/                # Age regression
├── regression_multitask/      # Blood pressure regression (systolic and diastolic, multitask model)
├── preprocess/                # Validate images and perform quality grading
├── models/                    # CNN and Vision Transformer architectures
├── utils/                     # Dataset loaders, helpers, plotting
├── mplstyles/                 # Matplotlib styles for paper figures
├── example_scripts/           # Example training commands
├── explore_dataset.ipynb      # Dataset exploration
└── explore_measurements.ipynb # AutoMorph measurements exploration
```

## Usage
See `example_scripts/training_example.sh` for an example of training and evaluating a sex classifier.

Use the `-h` flag to see all available arguments for any script, e.g.:
```bash
python -m classification.train -h
```

## Supported Architectures
All architectures can be trained with ImageNet initialization or from scratch.
|           Backbone           	| Type 	|
|:----------------------------:	|:----:	|
| resnet18, resnet50           	| CNN  	|
| inceptionv3                  	| CNN  	|
| efficientnetv2_s             	| CNN  	|
| xception                     	| CNN  	|
| vit_base_patch16_224         	| ViT  	|
| swin_base_patch4_window7_224 	| Swin 	|

## License
This project is licensed under the [MIT License](LICENSE).
