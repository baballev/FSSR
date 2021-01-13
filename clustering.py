import os
from math import ceil
import random
from argparse import ArgumentParser
from statistics import mean
from math import floor
from itertools import product

from tqdm import tqdm
import numpy as np
from PIL import Image

parser = ArgumentParser()

parser.add_argument('mode', choices=['tile', 'embed', 'pca-fit', 'pca-transform', 'cluster'])
parser.add_argument('--path', type=str, default='../data/DIV2K/DIV2K_valid')
parser.add_argument('--pca', type=str, help='File containing the pickled PCA.')
parser.add_argument('--out', type=str)
parser.add_argument('--sample', type=float, default=0.1, help='Propotion of data points used.')
parser.add_argument('--n-components', type=int, default=100, help='Number of components in the PCA.')
parser.add_argument('--n-clusters', type=int, default=80)
parser.add_argument('--min-size', type=int, default=10, help='Minimum cluster size.')
parser.add_argument('--max-size', type=int, default=20, help='Maximum cluster size.')
parser.add_argument('--tile-size', type=int, default=192, help='Size of patches used to tile the images.')

opt = parser.parse_args()


if opt.mode == 'tile':
    assert opt.path and opt.out

    filenames = os.listdir(opt.path)
    d = opt.tile_size
    total = 0
    for f in tqdm(filenames):
        name, ext = os.path.splitext(f)
        img = Image.open(os.path.join(opt.path, f))
        w, h = img.size

        grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
        for i, j in grid:
            box = (j, i, j+d, i+d)
            out = os.path.join(opt.out, f'{name}_{i}_{j}{ext}')
            img.crop(box).save(out)
        total += len(grid)
    print(f'created {total} image patches from {len(filenames)} images')


elif opt.mode == 'embed': # use @torch.no_grad()
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T

    assert os.path.isdir(opt.path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model.eval().to(device)

    filenames = os.listdir(opt.path)
    samples = random.sample(filenames, int(len(filenames)*opt.sample))
    embs = {}

    with torch.no_grad():
        for f in tqdm(samples):
            name, _ = os.path.splitext(f)
            img = Image.open(os.path.join(opt.path, f))
            x = T.ToTensor()(img).unsqueeze(0).to(device)

            emb = model(x).detach().cpu().numpy()
            embs[f] = emb

    np.save(f'{opt.path}_{emb.shape[1]}emb_{len(samples)}', embs)
    print(f'embedded {len(samples)} of the {len(filenames)} images found')


elif opt.mode == 'pca-fit':
    # will fit on the entire content of opt.fit
    from sklearn.decomposition import PCA
    import json

    assert os.path.isfile(opt.path)

    name, _ = os.path.splitext(opt.path)

    embs = np.load(opt.path, allow_pickle=True).item()
    X = np.concatenate([emb for emb in embs.values()])

    pca = PCA(n_components=opt.n_components)
    pca.fit(X)

    np.save(f'{name}_{opt.n_components}pca', pca)

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    json.dump(str(list(cumsum_variance)), 
        open(f'{name}_{opt.n_components}pca-var.txt', 'w'))
    print(f'PCA fit with {opt.n_components} components on {len(X)} vectors ended')


elif opt.mode == 'pca-transform':
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    assert os.path.isdir(opt.path) and os.path.isfile(opt.pca)

    name, _ = os.path.splitext(opt.path)
    pca = np.load(opt.pca, allow_pickle=True).item()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model.eval().to(device)

    filenames = os.listdir(opt.path)
    projections = {}
    with torch.no_grad():
        for f in tqdm(filenames):
            #name, _ = os.path.splitext(f)
            img = Image.open(os.path.join(opt.path, f))
            x = T.ToTensor()(img).unsqueeze(0).to(device)

            emb = model(x).detach().cpu().numpy()
            projected = pca.transform(emb)
            projections[f] = projected

    np.save(f'{name}_{pca.n_components}pca', projections)
    print(f'embedded and projected w/ PCA {len(filenames)} images')


elif opt.mode == 'cluster':
    from k_means_constrained import KMeansConstrained
    assert os.path.isfile(opt.path)
    
    embs = np.load(opt.path, allow_pickle=True).item()
    keys = embs.keys() # extract once, items()

    kmeans = KMeansConstrained(n_clusters=opt.n_clusters, size_min=opt.min_size, size_max=opt.max_size)
   
    print('fitting with %i images on %i clusters (w/ %i < size < %i)' % (len(keys), opt.n_clusters, opt.min_size, opt.max_size))
    samples = random.sample(keys, k=len(keys))
    x = np.stack([embs[s] for s in samples])
    kmeans.fit(x)

    y = np.stack(list(embs.values()))
    clusters = kmeans.predict(y)
    
    groups = []
    for i in range(opt.n_clusters):
        neighbors = np.array(list(keys))[clusters == i]
        # print('cluster_%i' % i, np.sort(neighbors))
        groups.append(neighbors)

    print([len(g) for g in groups])
     
    dest = opt.path.replace('_emb.npy', '_cluster')
    np.save(dest, np.asarray(groups, dtype=object))
    print('saved clusters to %s.npy' % dest)
