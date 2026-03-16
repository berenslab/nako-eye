import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from utils import nako


def get_args():
    parser = argparse.ArgumentParser(description='Validate images from NAKO.')
    parser.add_argument(
        '--dataset',
        default='590',
        type=str,
        help='version of the NAKO dataset 590 or 810.',
        choices=['810', '590'],
    )
    parser.add_argument(
        '--baseline',
        default='baseline',
        type=str,
        help='for NAKO 810 baseline and followup 1 are available.',
        choices=['baseline', 'followup1'],
    )
    args = parser.parse_args()
    return args


def get_corrupted_files(image_dir):
    """Get a list of paths of corrupted images."""
    corrupted = []
    for image_file in glob.glob(os.path.join(image_dir, '*.jpg')):
        try:
            with Image.open(image_file) as img:
                img.verify()  # Verify the image file integrity

            # Check image is not truncated
            Image.open(image_file).load()
        except (IOError, SyntaxError, OSError):
            corrupted.append(image_file)

    return corrupted


def get_lowres_images(image_dir, img_shape, nako_paths, corrupted_images):
    """Get a list of paths of images of lower resolution than the default."""
    dataset = nako.NakoImageDataset(
        image_dir, nako_paths, unusable_images=corrupted_images
    )
    shapes = [dataset[i][0].size for i in range(len(dataset))]
    idx_different_shape = np.unique(np.argwhere(np.array(shapes) != img_shape)[:, 0])

    unique_shapes, unique_counts = np.unique(
        np.array(shapes), axis=0, return_counts=True
    )
    for unique_shape, unique_count in zip(unique_shapes, unique_counts):
        print(
            f'Image Shape = ({unique_shape[0]}, {unique_shape[1]}), Number of Images = {unique_count}'
        )

    ids = dataset.ids[idx_different_shape]
    lowres = [os.path.join(f'{image_dir}', f'{id}.jpg') for id in ids]
    return lowres


if __name__ == '__main__':
    args = get_args()

    # Specific to the NAKO dataset
    nako_paths = nako.get_nako_paths(args.dataset, args.baseline, 'images')
    image_shape = (2592, 1944)  # Regular image shape

    # Where results are stored
    quality_dir = nako_paths['quality_dir']
    corrupted_file = quality_dir.joinpath('corrupted_images.json')
    lowres_file = quality_dir.joinpath('lowres_images.json')

    # Get corrupted images
    corrupted_images, lowres_images = [], []
    for image_type in nako.IMAGE_TYPES:
        for i in range(3):
            image_dir = os.path.join(
                nako_paths['image_dir'], image_type + '_' + str(i + 1)
            )
            print(
                f'Scanning {os.path.basename(image_dir)} directory to find corrupted / truncated images...'
            )
            corrupted_images.extend(get_corrupted_files(image_dir))

    # Get only relative path (image type + folder index / *.jpg)
    corrupted_images = ['/'.join(item.split(sep='/')[-2:]) for item in corrupted_images]
    with open(corrupted_file, 'w') as f:
        json.dump(corrupted_images, f)

    # Get lowres images
    for image_type in nako.IMAGE_TYPES:
        for i in range(3):
            image_dir = os.path.join(
                nako_paths['image_dir'], image_type + '_' + str(i + 1)
            )
            print(
                f'Scanning {os.path.basename(image_dir)} directory to find lowres images...'
            )
            lowres_images.extend(
                get_lowres_images(image_dir, image_shape, nako_paths, corrupted_images)
            )

    lowres_images = ['/'.join(item.split(sep='/')[-2:]) for item in lowres_images]
    with open(lowres_file, 'w') as f:
        json.dump(lowres_images, f)

    print('Number of corrupted images = ', len(corrupted_images))
    print('Number of low resolution = ', len(lowres_images))
