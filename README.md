# NAKO Ophthalmology Dataset
This repository contains the code for the paper ["The German National Cohort: Ophthalmological Assessment, Baseline Profile 
and Potential for AI-based Eye Research"](https://www.medrxiv.org/content/10.64898/2026.05.04.26352019v1).

## Installation
A conda environment file is included to install the R and python packages used in this project.

```bash
conda env create -f environment.yaml
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

## Pretrained Models
Pretrained model checkpoints are not yet publicly available.

## Citation
If you use this code, please cite:
```bibtex
@article{Roa2026.05.04.26352019,
  author    = {Roa, Camila and Beuse, Ansgar and Schweig, Alexandra and Mueller, Sarah and Berger, Klaus and Brandl, Caroline and Brinker, Titus and Elbrecht, Anne and Finger, Robert and Geerling, Gerd and Greiser, Karin Halina and Grohmann, Carsten and Guenther, Kathrin and Heid, Iris and Karch, Andre and Keil, Thomas and Krepel, Jessica and Leitzmann, Michael and Meinke-Franze, Claudia and Peters, Annette and Schipf, Sabine and Schulz, Matthias and Schuster, Alexander K. and Willich, Stefan N. and Leitritz, Martin A. and Ueffing, Marius and Berens, Philipp},
  title     = {The German National Cohort: Ophthalmological Assessment, Baseline Profile and Potential for AI-based Eye Research},
  journal   = {medRxiv},
  year      = {2026},
  doi       = {10.64898/2026.05.04.26352019},
  publisher = {Cold Spring Harbor Laboratory Press},
  URL       = {https://www.medrxiv.org/content/early/2026/05/10/2026.05.04.26352019}
}
```

## License
This project is licensed under the [MIT License](LICENSE).
