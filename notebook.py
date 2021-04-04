
"""
@Author: Alex Nguyen
Gettysburg College
"""
# %%

from __future__ import print_function, division
import os
import gc
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

from core.data import ShopeeDataset

DATASET_PATH = "./dataset/shopee-product-matching"
ds = ShopeeDataset(DATASET_PATH)

# Dataset shortcut
# All data column is string excpet for label_group 
# is integer
posting_id = "posting_id"
image_path = "image"
phash = "image_phash"
title = "title"
label_group = "label_group"

def get_text_embeddings(df, max_features = 15000, n_components = 5000, verbose: bool=False):
    model = TfidfVectorizer(binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df['title']).toarray()
    pca = PCA(n_components = n_components)
    text_embeddings = pca.fit_transform(text_embeddings).get()
    if verbose:
        print(f'Our title text embedding shape is {text_embeddings.shape}')
    del model, pca
    gc.collect()
    return text_embeddings

# %%

ds[label_group]

# %%


ds.df.head()



