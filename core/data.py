from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class ShopeeDataset:

    """ ShopeeDataset class
    Documentation to be implemented.
    """

    class Listing:
        """ Deprecated
        """
        def __init__(self, 
            posting_id: str,
            image: str,
            image_phash: str,
            title: str,
            label_group: str
        ) -> None:
            self.posting_id = posting_id
            self.image = image
            self.image_phash = image_phash
            self.title = title
            self.label_group = label_group

    def __init__(self, root) -> None:
        self.root = root
        self.train_path_csv = os.path.join(root, "train.csv")
        self.test_path_csv = os.path.join(root, "test.csv")
        # self.df = pd.read_csv(self.train_path_csv)
        self.df = self.read_dataset(self.train_path_csv)

    def read_dataset(self, path: str, GET_CV: bool=True, CHECK_SUB: bool=True):
        if GET_CV:
            df = pd.read_csv(path)
            tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
            df['matches'] = df['label_group'].map(tmp)
            df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
            if CHECK_SUB:
                df = pd.concat([df, df], axis = 0)
                df.reset_index(drop = True, inplace = True)
        else:
            df = pd.read_csv(path)            
        return df

    def __getitem__(self, id):
        if isinstance(id, str):
            return self.df[id]
        return self.df.iloc[id]

    def __getslice__(self,start, end):
        return self.df.iloc[start, end]

    def __str__(self) -> str:
        return self.train_dataset

    def head(self, n: int=0):
        return self.train_dataset.head(n)
        
class ShopeeTorch(ShopeeDataset, torch.utils.data.Dataset):
    def __init__(self, root) -> None:
        super().__init__(root)
    
    def __getitem__(self, id):
        pass

    def __len__(self) -> int:
        return len(self.df)