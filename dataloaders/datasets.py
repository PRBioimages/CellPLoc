import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from utils.randaugment import RandAugment
from utils.transforms import RandomResizedCropAndInterpolation
from batchgenerators.utilities.file_and_folder_operations import *
from torchvision.transforms import (ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip)


LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]


def all_Img_df():
    meta_dir = './Meta_files/'
    files = ['hpa_multi_bbox_meta', 'hpa_single_bbox_meta', 'train_single_bbox_meta', 'train_multi_bbox_meta', 'SCV_notest', 'Extra_meta']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell = df_allcell.drop_duplicates('ID', keep='last', ignore_index=True).reset_index(drop=True)
    return df_allcell


class RANZERDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None):
        self.df_img = all_Img_df()
        self.df = df.reset_index(drop=True)
        self.IDs = df.ID.unique()
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cols = LBL_NAMES
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'
        else:
            self.cell_path = cfg.data.cell
        self.celllabel = cfg.data.celllabel
        print('[ ! ] using', cfg.data.celllabel)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        row = self.df[self.df.ID.isin([ID])].reset_index(drop=True)
        cnt = self.cfg.experiment.count
        if len(row) > cnt:
            selected = random.sample([i for i in range(len(row))], cnt)
        else:
            selected = [i for i in range(len(row))]
        batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        mask = np.zeros((cnt))
        label = np.zeros((cnt, 19))
        img_label = self.df_img[self.df_img.ID.isin([ID])].reset_index(drop=True)
        img_label = np.max(img_label.loc[0:, self.cols].values.astype(np.float), axis=0)
        for idx, s in enumerate(selected):
            path = '/path/to/your/IF_single_cell_images/data' + f'/{row.loc[s, "ID_idx"]}.npy'  # Path of segmented single cell image data
            img = np.load(path)
            if self.transform is not None:
                res = self.transform(image=img)
                img = res['image']
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            img = self.tensor_tfms(img)
            batch[idx, :, :, :] = img
            mask[idx] = 1
            if self.celllabel != 'imagelabel':
                label[idx] = row.loc[s, self.cols].values.astype(np.float)
            else:
                label[idx] = img_label

        if self.cfg.experiment.smoothing == 0:
            return batch, mask, label, img_label
        else:
            return batch, mask, 0.9 * label + 0.1 / 19, 0.9 * img_label + 0.1 / 19


class RANZERDataset_Mix_transform_strong(Dataset):
    def __init__(self, df, tfms=None, cfg=None):
        self.df_img = all_Img_df()
        self.df = df.reset_index(drop=True)
        self.IDs = df.ID.unique()
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.tensor_tfms_strong = Compose([
            Resize(224),
            RandomResizedCropAndInterpolation((224, 224)),
            RandomHorizontalFlip(),
            RandAugment(3, 10),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cols = LBL_NAMES
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'
        else:
            self.cell_path = cfg.data.cell
        self.celllabel = cfg.data.celllabel
        print('[ ! ] using', cfg.data.celllabel)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        row = self.df[self.df.ID.isin([ID])].reset_index(drop=True)
        cnt = self.cfg.experiment.count
        if len(row) > cnt:
            selected = random.sample([i for i in range(len(row))], cnt)
        else:
            selected = [i for i in range(len(row))]
        batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        batch_strong = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        mask = np.zeros((cnt))
        label = np.zeros((cnt, 19))
        bib_label = np.zeros((cnt, 19))
        img_label = self.df_img[self.df_img.ID.isin([ID])].reset_index(drop=True)
        img_label = np.max(img_label.loc[0:, self.cols].values.astype(float), axis=0)
        for idx, s in enumerate(selected):
            path = '/path/to/your/IF_single_cell_images/data' + f'/{row.loc[s, "ID_idx"]}.npy'  # Path of segmented single cell image data
            img_old = np.load(path)  # ndarray
            if self.transform is not None:
                res = self.transform(image=img_old)
                img = res['image']  # ndarray, (224, 224, 4)
            if self.tensor_tfms_strong is not None:
                img_pil = Image.fromarray(img_old.astype('uint8'))
                img_strong = self.tensor_tfms_strong(img_pil)
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            img = self.tensor_tfms(img)
            batch[idx, :, :, :] = img
            batch_strong[idx, :, :, :] = img_strong
            mask[idx] = 1
            if self.celllabel != 'imagelabel':
                label[idx] = img_label
                bib_label[idx] = row.loc[s, self.cols].values.astype(float)  # BiB Label
            else:
                label[idx] = img_label

        if self.cfg.experiment.smoothing == 0:
            if self.celllabel != 'imagelabel':
                return batch, batch_strong, mask, label, img_label, bib_label
            else:
                return batch, batch_strong, mask, label, img_label
        else:
            if self.celllabel != 'imagelabel':
                return batch, batch_strong, mask, 0.9 * label + 0.1 / 19, 0.9 * img_label + 0.1 / 19, 0.9 * bib_label + 0.1 / 19
            else:
                return batch, batch_strong, mask, 0.9 * label + 0.1 / 19, 0.9 * img_label + 0.1 / 19


def HPAv23_Img_df():
    meta_dir = './Meta_files/'
    files = ['subcellular_images_19classes_v23_unique_with_label-num']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell = df_allcell.drop_duplicates('image_id', keep='last', ignore_index=True).reset_index(drop=True)
    return df_allcell


class HPAv23Dataset_Mix_transform_strong(Dataset):
    def __init__(self, df, tfms=None, cfg=None):
        self.df_img = HPAv23_Img_df()
        self.df = df.reset_index(drop=True)
        self.IDs = df.image_id.unique()
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.tensor_tfms_strong = Compose([
            Resize(224),
            RandomResizedCropAndInterpolation((224, 224)),
            RandomHorizontalFlip(),
            RandAugment(3, 10),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cols = LBL_NAMES
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'
        else:
            self.cell_path = cfg.data.cell
        self.celllabel = cfg.data.celllabel
        print('[ ! ] using', cfg.data.celllabel)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        row = self.df[self.df.image_id.isin([ID])].reset_index(drop=True)
        cnt = self.cfg.experiment.count
        if len(row) > cnt:
            selected = random.sample([i for i in range(len(row))], cnt)
        else:
            selected = [i for i in range(len(row))]
        batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        batch_strong = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        mask = np.zeros((cnt))
        label = np.zeros((cnt, 19))
        bib_label = np.zeros((cnt, 19))
        img_label = self.df_img[self.df_img.image_id.isin([ID])].reset_index(drop=True)
        img_label = np.max(img_label.loc[0:, self.cols].values.astype(float), axis=0)
        for idx, s in enumerate(selected):
            id_idx = row.loc[s, "ID_idx"]
            path = '/path/to/your/IF_single_cell_images/data' + f'/{id_idx}.npy'  # Path of segmented HPAv23 single cell image data
            img_old = np.load(path)  # ndarray
            if self.transform is not None:
                res = self.transform(image=img_old)
                img = res['image']  # ndarray, (224, 224, 4)
            if self.tensor_tfms_strong is not None:
                img_pil = Image.fromarray(img_old.astype('uint8'))
                img_strong = self.tensor_tfms_strong(img_pil)
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            img = self.tensor_tfms(img)
            batch[idx, :, :, :] = img
            batch_strong[idx, :, :, :] = img_strong
            mask[idx] = 1
            if self.celllabel != 'imagelabel':
                label[idx] = img_label
                bib_label[idx] = row.loc[s, self.cols].values.astype(float)  # BiB Label
            else:
                label[idx] = img_label

        if self.cfg.experiment.smoothing == 0:
            if self.celllabel != 'imagelabel':
                return batch, batch_strong, mask, label, img_label, bib_label   # 经常选这个
            else:
                return batch, batch_strong, mask, label, img_label
        else:
            if self.celllabel != 'imagelabel':
                return batch, batch_strong, mask, 0.9 * label + 0.1 / 19, 0.9 * img_label + 0.1 / 19, 0.9 * bib_label + 0.1 / 19
            else:
                return batch, batch_strong, mask, 0.9 * label + 0.1 / 19, 0.9 * img_label + 0.1 / 19

