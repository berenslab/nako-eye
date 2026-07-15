import argparse
import random
import time
from pathlib import Path

import fundus_image_toolbox as fit
import numpy as np
import pandas as pd
import torch
from fundus_image_toolbox.quality_prediction.scripts.transforms import get_transforms
from openTSNE import TSNE
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from an image folder for the FIT quality ensemble.'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        nargs='+',
        type=str,
        help='Path(s) to image folder(s).',
    )
    parser.add_argument(
        '--names',
        default=None,
        nargs='+',
        type=str,
        help='Names for each dataset. Defaults to the folder name. Must match --dataset count if provided.',
    )
    parser.add_argument(
        '--output',
        default=None,
        type=str,
        help='Output directory for parquet files. Defaults to ./data/quality_embeddings.',
    )
    parser.add_argument(
        '--batchsize', default=64, type=int, help='Batch size for inference.'
    )
    parser.add_argument(
        '--device', default='cuda:0', type=str, help='Device for inference.'
    )
    parser.add_argument(
        '--threshold',
        default=0.25,
        type=float,
        help='Confidence threshold for gradable/ungradable label.',
    )
    parser.add_argument(
        '--subset',
        default=None,
        type=int,
        help='Randomly sample this many images per dataset. Uses all images if not set.',
    )
    parser.add_argument(
        '--tsne',
        action='store_true',
        help='Compute t-SNE embeddings (X_2d) per model and store them in the parquet.',
    )
    return parser.parse_args()


class ImageDataset(Dataset):
    """Loads images, dataset name, and relative paths from an image folder."""

    def __init__(self, image_dir, transform=None, name=None, subset=None):
        self.image_dir = Path(image_dir)
        self.name = name or self.image_dir.name
        self.transform = transform
        all_paths = sorted(p for p in self.image_dir.rglob('*') if p.is_file())
        self.image_paths = random.sample(all_paths, subset) if subset else all_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.name, str(path.relative_to(self.image_dir))


def compute_tsne_embeddings(X):
    # Compute t-sne embeddings
    tsne = TSNE(
        perplexity=30,
        initialization='pca',
        metric='cosine',
        random_state=42,
        verbose=True,
    )

    print('Fitting t-SNE...')
    tsne_embeddings = tsne.fit(X)
    X_2d = np.array(tsne_embeddings)
    return X_2d


@torch.no_grad()
def extract_embeddings(ensemble, dataloader, threshold, device, tsne=False):
    """
    Run all ensemble models over the dataloader. For each model, capture the
    avgpool embedding (via a forward hook) alongside the confidence and label.

    Returns:
        DataFrame with columns: dataset, model, image_path, X, confidence, label
    """
    results = {}

    for i, wrapper in enumerate(ensemble):
        model_key = f'model_{i}_{wrapper.config.model_type}'
        model = wrapper.model
        model.eval()

        embedding_cache = []

        def _hook(_module, _input, output):
            # Get embeddings before fc layer
            embedding_cache.append(output.flatten(1).detach().cpu())

        hook = model.avgpool.register_forward_hook(_hook)
        dataset_names, image_paths, embeddings_list, confidences_list, labels_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for image_batch, name_batch, path_batch in tqdm(dataloader, desc=model_key):
            embedding_cache.clear()
            image_batch = image_batch.to(device)

            logits = model(image_batch).squeeze(1)
            confidences = torch.sigmoid(logits).cpu()
            labels = (confidences >= threshold).int()
            embeddings = embedding_cache[0]  # (B, C)

            dataset_names.extend(name_batch)
            image_paths.extend(path_batch)
            embeddings_list.append(embeddings.numpy())
            confidences_list.append(confidences.numpy())
            labels_list.append(labels.numpy())

        hook.remove()
        X = np.concatenate(embeddings_list, axis=0)
        df = pd.DataFrame(
            {
                'dataset': dataset_names,
                'model': model_key,
                'image_path': image_paths,
                'X': list(X),
                'confidence': np.concatenate(confidences_list).round(4),
                'label': np.concatenate(labels_list),
            }
        )
        if tsne:
            df['X_2d'] = list(compute_tsne_embeddings(X))
        results[model_key] = df

    return pd.concat(results.values(), ignore_index=True)


if __name__ == '__main__':
    start = time.time()
    args = get_args()

    output_dir = Path(args.output) if args.output else Path('./data/quality_embeddings')
    output_dir.mkdir(parents=True, exist_ok=True)

    names = args.names or [Path(p).name for p in args.dataset]
    assert len(names) == len(args.dataset), '--names count must match --dataset count'

    transform = get_transforms(split='test')
    datasets = [
        ImageDataset(p, transform=transform, name=n, subset=args.subset)
        for p, n in zip(args.dataset, names)
    ]
    combined = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined,
        batch_size=args.batchsize,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    print(f'Loaded {len(combined)} images from {len(datasets)} dataset(s)')

    ensemble = fit.load_quality_ensemble(device=args.device)
    results = extract_embeddings(
        ensemble, dataloader, args.threshold, args.device, tsne=args.tsne
    )

    output_path = output_dir / 'quality_embeddings.parquet'
    results.to_parquet(output_path, index=False)

    print(f'Saved embeddings to {output_path}')
    print(f'Total time: {time.time() - start:.1f} s')
