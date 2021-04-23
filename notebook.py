
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
from torchvision import utils
from torchvision import transforms as T

import faiss

import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
print('TF',tf.__version__)

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

from core.data import ShopeeDataset, ShopeeTorch

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

dim = text_embeddings.shape[1]
n = text_embeddings.shape[0]


xb = np.random.normal(size=(n, dim)) # normal distribution with mean 0 std 1

index = faiss.IndexFlatL2(dim)   # build the index, d=size of vectors 
# here we assume xb contains a n-by-d numpy matrix of type float32
index.add(xb)                  # add vectors to the index
print(index.ntotal)


# %%

# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# %%

from sklearn.neighbors import NearestNeighbors
KNN = 50
model = NearestNeighbors(n_neighbors=KNN)
model.fit(text_embeddings)
distances, indices = model.kneighbors(text_embeddings)


# %%

knn_model_folder = "./models/knn"
np.save(os.path.join(knn_model_folder, "distances_50_centroids.npy"), distances)
np.save(os.path.join(knn_model_folder, "indices_50_centroids.npy"), indices)


# %%

###############  Image training Pytorch

# configs:

torch_train_dataset = ShopeeTorch()

# Number of classes in the dataset
num_classes = torch_train_dataset.n_labels()

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True



data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def data_transforms_2(image_size: int) -> (object, object):
    '''
        Return transformations to be applied.
        Input:
            image_size: int
        Output:
            train_transformations: transformations to be applied on the training set
            valid_tfms: transformations to be applied on the validation or test set
    '''

    # imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # https://www.kaggle.com/ronaldokun/multilabel-stratification-cv-and-ensemble
    
    mean = torch.tensor([0.05438065, 0.05291743, 0.07920227])
    std = torch.tensor([0.39414383, 0.33547948, 0.38544176])
    
    train_trans = [           
        T.Resize(image_size + 4),
        T.CenterCrop(image_size),
        T.RandomRotation(40),
        T.RandomAffine(
            degrees=10,
            translate=(0.01, 0.12),
            shear=(0.01, 0.03),
        ),
        T.RandomHorizontalFlip(), 
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True), 
        T.RandomErasing(inplace=True)
    ]

    val_trans = [
        T.Resize(image_size), 
        T.ToTensor(), 
        T.Normalize(mean, std, inplace=True)
    ]

    train_transformations = T.Compose(train_trans)
    valid_tfms = T.Compose(val_trans)

    return train_transformations, valid_tfms


# pretrained models
from efficientnet_pytorch import EfficientNet

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Transfer learning model
models = {
#     "EfficientNet-b0": EfficientNet.from_pretrained('efficientnet-b0'),
    "EfficientNet-b1": EfficientNet.from_pretrained('efficientnet-b1'), 
#     "EfficientNet-b2": EfficientNet.from_pretrained('efficientnet-b2'),
#     "EfficientNet-b3": EfficientNet.from_pretrained('efficientnet-b3'),
}

# There are these model [resnet, alexnet, vgg, squeezenet, densenet, inception]
image_sizes = {
    "EfficientNet-b0": 224,
    "EfficientNet-b1": 240,
    "EfficientNet-b2": 260,
    'EfficientNet-b3': 300,
    'EfficientNet-b4': 380,
}

batch_sizes = {
    "EfficientNet-b0": 150,
    "EfficientNet-b1": 100,
    "EfficientNet-b2": 64,
    'EfficientNet-b3': 50,
    'EfficientNet-b4': 20
}

# %%


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# %%



