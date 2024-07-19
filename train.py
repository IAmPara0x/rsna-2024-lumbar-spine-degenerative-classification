#!/usr/bin/python3

import albumentations as A

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import pickle


class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        x = np.zeros((IN_CHANS, 512, 512), dtype=np.uint8)

        patient_id = self.df.iloc[idx]["study_id"]
        label = self.df.iloc[idx][1:].values.astype(np.uint8)
        patient_info_path = f"./processed-dataset/{patient_id}.pkl"

        with open(patient_info_path, "rb") as f:
            data = pickle.load(f)


        scans_used = []

        for series_info in data["series"].values():

            scan_type = series_info["series_description"]

            if scan_type in scans_used:
                continue

            scans = series_info["images"]

            if scan_type == "Sagittal T2/STIR":

                for i in range(len(scans)):
                    x[i] = scans[i]["img"].astype(np.uint8)

            elif scan_type == "Sagittal T1":

                for i in range(len(scans)):
                    x[i+10] = scans[i]["img"].astype(np.uint8)

            elif scan_type == "Axial T2":

                for i in range(min(5,len(scans))):
                    x[i+20] = scans[i]["img"].astype(np.uint8)
                
            else:
                raise ValueError(f"unknown series_description: {series_info["series_description"]}")

            scans_used.append(scan_type)

        if self.transform is not None:
            x = self.transform(image=x)['image']

        return x, label


if __name__ == "__main__":

    AUG_PROB = 0.75
    IMG_SIZE = [512, 512]
    IN_CHANS = 25


    df = pd.read_csv('train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
    df = df.replace(label2id)


    transforms_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=AUG_PROB),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=AUG_PROB),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
        A.Normalize(mean=0.5, std=0.5)
    ])

    transforms_val = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])

    tmp_ds = RSNA24Dataset(df, phase='train', transform=None)
