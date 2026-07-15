import argparse
import csv
import os
from pathlib import Path

import fundus_image_toolbox as fit
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Center crop fundus images.')
    parser.add_argument('--srcdir', type=str, help='image directory for the dataset.')
    parser.add_argument(
        '--destdir', type=str, help='directory where results will be saved.'
    )
    parser.add_argument(
        '--imagesize', default=512, type=int, help='size of the processed images.'
    )
    parser.add_argument(
        '--logsdir',
        default='.',
        type=str,
        help='directory to save the filenames of the images that could not be processed.',
    )
    args = parser.parse_args()
    args.imagesize = (args.imagesize, args.imagesize)
    return args


def fundus_circle_crop(src_dir, dest_dir, image_size=(512, 512)):
    """Crop fundus images from the AREDS dataset using the fundus image toolbox."""
    transform = transforms.ToPILImage()

    # For the progress bar
    file_count = sum(len(files) for _, _, files in os.walk(src_dir))
    failures = []

    with tqdm(total=file_count) as pbar:
        for subdir, _, filenames in os.walk(src_dir):
            dest_subdir = os.path.join(dest_dir, os.path.relpath(subdir, src_dir))
            Path(dest_subdir).mkdir(parents=True, exist_ok=True)

            for filename in filenames:
                pbar.update(1)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        image = Image.open(os.path.join(subdir, filename))
                        outputs = fit.crop(image, size=image_size)
                        image = transform(outputs)
                        image.save(os.path.join(dest_subdir, filename))
                    except Exception as e:
                        failures.append((os.path.join(subdir, filename), str(e)))
    return failures


if __name__ == '__main__':
    args = get_args()
    failures = fundus_circle_crop(args.srcdir, args.destdir, args.imagesize)

    log_path = os.path.join(args.logsdir, 'center_crop_logs.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'reason'])
        writer.writerows(failures)
