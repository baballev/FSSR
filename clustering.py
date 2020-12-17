import os
from math import ceil
import random
from argparse import ArgumentParser
from statistics import mean

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

parser = ArgumentParser()

parser.add_argument('mode', choices=['embed', 'cluster'])
parser.add_argument('--path', type=str, default='../data/DIV2K/DIV2K_valid')
parser.add_argument('--n-clusters', type=int, default=80)
parser.add_argument('--min-size', type=int, default=10, help='Minimum cluster size.')
parser.add_argument('--max-size', type=int, default=20, help='Maximum cluster size.')

opt = parser.parse_args()

if opt.mode == 'embed':    
    assert os.path.isdir(opt.path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model.eval().to(device)

    fps = os.listdir(opt.path)

    embs = {}
    for fp in tqdm(fps):
        img = Image.open(os.sep.join((opt.path, fp))).convert('RGB')
        x = T.ToTensor()(img).unsqueeze(0).to(device)
    
        emb = model(x).squeeze(0).detach().cpu().numpy()
        embs[fp] = emb

    dest = '%s_emb' % opt.path
    np.save(dest, embs)
    print('saved embeddings to %s.npy' % dest)


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
