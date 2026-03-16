import argparse
import glob
import os
import time
from pathlib import Path

import pandas as pd
import timm
import torch
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


def quality_dataset(args, dataset, csv_file, checkpoint_iter=100):
    """
    Compute quality grading for an image dataset and save results in a csv file.

    Args:
        args (dict): contains the batchsize and device.
        dataset (torch.utils.data.Dataset): image dataset.
        csv_file (str, Path): path of the output csv file.
        checkpoint_iter (int): number of batches after which to save a checkpoint of the quality grading.
    """

    # Quality filter
    # download the weights, e.g. with
    # wget https://github.com/justinengelmann/RetiRatey/releases/download/v1/retiratey_v1_statedict.pth
    retiratey = timm.create_model(
        'mobilenetv4_conv_small',
        num_classes=3,
        checkpoint_path='models/retiratey_v1_statedict.pth',
    )
    retiratey.to(args.device)
    retiratey.eval()

    df = pd.DataFrame(columns=['ID', 'disc', 'macula', 'vessels'])
    df.to_csv(quality_file, index=False, float_format='%.3f')

    dataloader = DataLoader(
        dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True
    )
    for b, (image_batch, id_batch) in enumerate(tqdm(dataloader)):
        # print(f'Batch {b + 1} out of {len(dataloader)}...')
        image_batch = image_batch.to(args.device)
        with torch.no_grad():
            outputs = retiratey(image_batch).cpu().numpy()
            outputs = 1 + (outputs * 3)

        df_append = pd.DataFrame(
            {
                'ID': id_batch,
                'disc': outputs[:, 0],
                'macula': outputs[:, 1],
                'vessels': outputs[:, 2],
            }
        )
        df = pd.concat([df.astype(df_append.dtypes), df_append], ignore_index=True)

        # Write results to csv file
        if (b % checkpoint_iter == 0) or (b == (len(dataloader) - 1)):
            df.to_csv(
                csv_file, mode='a', index=False, header=False, float_format='%.3f'
            )
            df = pd.DataFrame(columns=['ID', 'disc', 'macula', 'vessels'])


def quality_nako(args, nako_paths, unusable_images, quality_dir, transform):
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

            # Where results of quality filter will be saved
            quality_file = quality_dir.joinpath(os.path.basename(image_dir) + '.csv')

            print(
                f'Computing quality grading on folder: {os.path.basename(image_dir)} ...'
            )
            quality_dataset(args, image_dataset, quality_file, checkpoint_iter=100)


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
        image_paths = glob.glob(os.path.join(image_dir, '*'))
        image_names = [os.path.basename(p) for p in image_paths]

        return image_names, image_paths


if __name__ == '__main__':
    start = time.time()
    args = get_args()

    # Transform from Retiratey
    img_size = (384, 384)
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Check quality on all folders of the dataset
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
        quality_dir = nako_paths['quality_dir'].joinpath('retiratey')
        quality_dir.mkdir(parents=True, exist_ok=True)

        quality_nako(
            args, nako_paths, unusable_images, quality_dir, transform=transform
        )

    else:
        # Image dataset
        quality_dir = Path.cwd().joinpath('quality_results', 'other_dataset')
        quality_dir.mkdir(parents=True, exist_ok=True)
        quality_file = quality_dir.joinpath('quality_retiratey.csv')
        image_dataset = ImageDataset(args.dataset, transform=transform)
        quality_dataset(args, image_dataset, quality_file, checkpoint_iter=100)

    print(f'Total execution time: {time.time() - start:.1f} s')
