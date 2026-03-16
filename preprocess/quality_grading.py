import argparse
import json
import os
import time
from pathlib import Path

import fundus_image_toolbox as fit
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import nako


def get_args():
    parser = argparse.ArgumentParser(description='Validate images from NAKO.')
    parser.add_argument(
        '--dataset',
        default='590',
        type=str,
        help='version of the NAKO dataset 590 or 810, or path to image folder.',
    )
    parser.add_argument(
        '--baseline',
        default='baseline',
        type=str,
        help='for NAKO 810 baseline and followup 1 are available.',
        choices=['baseline', 'followup1'],
    )
    parser.add_argument(
        '--batchsize', default=128, type=int, help='batch size for inference'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help='device in which training will take place',
    )
    args = parser.parse_args()
    return args


def quality_dataset(args, dataset, csv_file, threshold=0.25, checkpoint_iter=100):
    """
    Compute quality grading for an image dataset and save results in a csv file.

    Args:
        args (dict): contains the batchsize and device.
        dataset (torch.utils.data.Dataset): image dataset.
        csv_file (str, Path): path of the output csv file.
        threshold (float): for quality prediction.
            Values below this threshold are considered as bad quality.
        checkpoint_iter (int): number of batches after which to save a checkpoint of the quality grading.
    """

    model = fit.load_quality_ensemble(device=args.device)
    df = pd.DataFrame(columns=['ID', 'confidence', 'label'])
    df.to_csv(csv_file, index=False, float_format='%.3f')

    dataloader = DataLoader(
        dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True
    )
    for b, (image_batch, id_batch) in enumerate(tqdm(dataloader)):
        # print(f'Batch {b + 1} out of {len(image_dataloader)}...')
        c, l = fit.ensemble_predict_quality(model, image_batch, threshold=threshold)

        df_append = pd.DataFrame({'ID': id_batch, 'confidence': c, 'label': l})
        df = pd.concat([df.astype(df_append.dtypes), df_append], ignore_index=True)

        # Write results to csv file
        if (b % checkpoint_iter == 0) or (b == (len(dataloader) - 1)):
            df.to_csv(
                csv_file, mode='a', index=False, header=False, float_format='%.3f'
            )
            df = pd.DataFrame(columns=['ID', 'confidence', 'label'])


def quality_nako(args, nako_paths, unusable_images, quality_dir, transform, threshold):
    # Check quality on all folders of the dataset
    for image_type in nako.IMAGE_TYPES:
        for i in range(3):
            image_dir = os.path.join(
                nako_paths['image_dir'], image_type + '_' + str(i + 1)
            )
            image_dataset = nako.NakoImageDataset(
                image_dir,
                nako_paths,
                unusable_images=unusable_images,
                transform=transform,
            )

            # Where results of quality grading will be saved
            quality_file = quality_dir.joinpath(os.path.basename(image_dir) + '.csv')

            print(
                f'Computing quality grading on folder: {os.path.basename(image_dir)} ...'
            )
            quality_dataset(
                args,
                image_dataset,
                quality_file,
                threshold=threshold,
                checkpoint_iter=100,
            )

    # Save all the paths of ungradable images as a json file
    ungradable_images = []
    for image_type in nako.IMAGE_TYPES:
        for i in range(3):
            image_dir = os.path.join(
                nako_paths['image_dir'], image_type + '_' + str(i + 1)
            )
            quality_file = quality_dir.joinpath(os.path.basename(image_dir) + '.csv')
            df = pd.read_csv(quality_file, dtype={'ID': str})
            df = df[df['label'] == 0]
            ungradable = [
                os.path.join(image_dir, idx + '.jpg') for idx in df['ID'].tolist()
            ]
            ungradable_images.extend(ungradable)  # Get only ungradable images

    # Get only relative path (image type + folder index / *.jpg)
    ungradable_images = [
        '/'.join(item.split(sep='/')[-2:]) for item in ungradable_images
    ]

    ungradable_file = nako_paths['quality_dir'].joinpath('ungradable_images.json')
    with open(ungradable_file, 'w') as f:
        json.dump(ungradable_images, f)


class ImageDataset(Dataset):
    """
    Loads images and filenames from an image folder.
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.labels, self.image_paths = self.get_image_paths(image_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def get_image_paths(self, image_dir):
        image_dir = Path(image_dir)
        image_paths = [p for p in image_dir.rglob('*') if p.is_file()]
        image_names = [os.path.relpath(p, image_dir) for p in image_paths]

        return image_names, image_paths


if __name__ == '__main__':
    start = time.time()
    args = get_args()

    # Threshold for the quality prediction: values below this threshold are considered as bad quality
    threshold = 0.25
    # transform = transforms.ToTensor() # Feed in same sized images of at least 350x350 px
    transform = transforms.Compose(
        [transforms.CenterCrop((512, 512)), transforms.ToTensor()]
    )

    if args.dataset in ['590', '810']:
        # Specific to the NAKO dataset
        nako_paths = nako.get_nako_paths(
            args.dataset, args.baseline, 'images_lowres', '512_raw'
        )
        corrupted_images, lowres_images, _ = nako.get_unusable_images(
            nako_paths['quality_dir']
        )
        unusable_images = corrupted_images + lowres_images

        # Where results will be saved
        quality_dir = nako_paths['quality_dir'].joinpath('fundus_image_toolbox')
        quality_dir.mkdir(parents=True, exist_ok=True)

        quality_nako(
            args,
            nako_paths,
            unusable_images,
            quality_dir,
            transform=transform,
            threshold=threshold,
        )

    else:
        # Image dataset
        quality_dir = Path.cwd().joinpath('quality_results', 'other_dataset')
        quality_dir.mkdir(parents=True, exist_ok=True)
        quality_file = quality_dir.joinpath('quality_fit.csv')
        image_dataset = ImageDataset(args.dataset, transform=transform)
        quality_dataset(
            args, image_dataset, quality_file, threshold=threshold, checkpoint_iter=100
        )

    print(f'Total execution time: {time.time() - start:.1f} s')
