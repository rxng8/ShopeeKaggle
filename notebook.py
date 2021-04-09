
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

import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
print('TF',tf.__version__)

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

IMG_FOLDER = os.path.join(DATASET_PATH, "train_images")

def displayDF(train, random=False, COLS=6, ROWS=4, path=IMG_FOLDER):
    for k in range(ROWS):
        plt.figure(figsize=(20,5))
        for j in range(COLS):
            if random: row = np.random.randint(0,len(train))
            else: row = COLS*k + j
            name = train.iloc[row,1]
            title = train.iloc[row,3]
            title_with_return = ""
            for i,ch in enumerate(title):
                title_with_return += ch
                if (i!=0)&(i%20==0): title_with_return += '\n'
            img = cv2.imread(os.path.join(path, name))
            plt.subplot(1,COLS,j+1)
            plt.title(title_with_return)
            plt.axis('off')
            plt.title(title_with_return)
            plt.axis('off')
            plt.imshow(img)
        plt.show()

# %%

# Display counts

groups = ds.df.label_group.value_counts()
plt.figure(figsize=(20,5))
plt.plot(np.arange(len(groups)),groups.values)
plt.ylabel('Duplicate Count',size=14)
plt.xlabel('Index of Unique Item',size=14)
plt.title('Duplicate Count vs. Unique Item Count',size=16)
plt.show()

plt.figure(figsize=(20,5))
plt.bar(groups.index.values[:50].astype('str'),groups.values[:50])
plt.xticks(rotation = 45)
plt.ylabel('Duplicate Count',size=14)
plt.xlabel('Label Group',size=14)
plt.title('Top 50 Duplicated Items',size=16)
plt.show()

for k in range(5):
    print('#'*40)
    print('### TOP %i DUPLICATED ITEM:'%(k+1),groups.index[k])
    print('#'*40)
    top = ds.df.loc[ds.df.label_group==groups.index[k]]
    displayDF(top, random=False, ROWS=2, COLS=4)

# %%

displayDF(ds.df, random=True)

# %%

ds.df.head()

# %%

ds[label_group]
ds.df.head()

# %%

"""
Reference: https://www.kaggle.com/cdeotte/rapids-cuml-tfidfvectorizer-and-knn

"""

# %%


model = TfidfVectorizer(stop_words='english', binary=True)
text_embeddings = model.fit_transform(ds.df.title).toarray()
print('text embeddings shape is',text_embeddings.shape)

# %%

from sklearn.neighbors import NearestNeighbors
KNN = 50
model = NearestNeighbors(n_neighbors=KNN)
model.fit(text_embeddings)
distances, indices = model.kneighbors(text_embeddings)

# %%


