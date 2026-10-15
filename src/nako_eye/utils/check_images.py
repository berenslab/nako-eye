"""Check which images exist for each participant.

For every image folder (image type + retake index, e.g. rt_leftcentral_1) the script
looks for a file named <ID>.<ext> and writes a CSV with one row per participant and
one boolean column per folder.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

IMAGE_TYPES = ['rt_leftcentral', 'rt_leftnasal', 'rt_rightcentral', 'rt_rightnasal']
RETAKES = [1, 2, 3]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def get_args():
    parser = argparse.ArgumentParser(
        description='Check which images exist per participant.'
    )
    parser.add_argument('ids_csv', type=Path, help='CSV with all participant IDs.')
    parser.add_argument(
        'image_dir',
        type=Path,
        help='directory containing the rt_*_1/2/3 image folders.',
    )
    parser.add_argument('output_csv', type=Path, help='path of the output CSV.')
    parser.add_argument('--id-column', default='ID', help='name of the ID column.')
    return parser.parse_args()


def image_ids(folder):
    """IDs (file names without extension) of all images in a folder."""
    if not folder.is_dir():
        print(f'Warning: folder not found: {folder}')
        return set()
    with os.scandir(folder) as entries:
        return {
            os.path.splitext(e.name)[0]
            for e in entries
            if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS
        }


def check_images(ids, image_dir):
    df = pd.DataFrame({'ID': ids})
    for image_type in IMAGE_TYPES:
        for i in RETAKES:
            folder = f'{image_type}_{i}'
            found = image_ids(image_dir / folder)
            df[folder] = df['ID'].isin(found)
    return df


if __name__ == '__main__':
    args = get_args()
    ids = pd.read_csv(args.ids_csv, dtype={args.id_column: str})[args.id_column]
    ids = ids.str.strip()

    df = check_images(ids, args.image_dir)
    df.to_csv(args.output_csv, index=False)

    print(f'Participants: {len(df)}')
    print(df.drop(columns='ID').sum().rename('images found').to_string())
    print(
        f'Participants without any image: {(~df.drop(columns="ID").any(axis=1)).sum()}'
    )
