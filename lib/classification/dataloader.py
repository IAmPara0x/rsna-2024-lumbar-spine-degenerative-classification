

import random
from abc import abstractmethod, ABC
from typing import Callable
from pandas import DataFrame
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
import pickle

from lib.patientInfo import PatientInfo, Scan, DiscLevel, Condition, Location

@dataclass
class RSNA24DF:
    train_df: DataFrame
    train_label_coordinates_df: DataFrame
    train_series_descriptions_df: DataFrame
    img_dir: str

@dataclass
class DiscLevelLocs:
    disc_pixel_loc: np.ndarray
    img_idxs: np.ndarray
    img_type: Scan
    disc_loc_mm: np.ndarray
    patient_id: int

class ClassificationDataLoader(Dataset, ABC):
    def __init__(self
                 , patient_ids: list[int]
                 , rsna24DF: RSNA24DF
                 , disc_level_locs_dir: str
                 , depth: int
                 , transformations = None
                 , phase = "train"
                 ) -> None:

        self.patient_ids = patient_ids
        self.rsna24DF = rsna24DF
        self.transformations = transformations
        self.depth = depth
        self.disc_level_locs_dir = disc_level_locs_dir 
        self.phase = phase

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        # print(f"{self.phase == 'train'=}")
        patient_info = PatientInfo.from_df(self.patient_ids[idx], self.rsna24DF.train_df, self.rsna24DF.train_series_descriptions_df, self.rsna24DF.train_label_coordinates_df, self.rsna24DF.img_dir, patient_type=self.phase)

        with open(f"{self.disc_level_locs_dir}/{patient_info.patient_id}.pkl", "rb") as f:
            disc_level_locs = pickle.load(f)


        if self.phase == "train":
            disc_level = DiscLevel.from_int(random.choice([*range(5)]))

            axial_t2_imgs    = self._axial_t2_imgs(disc_level, patient_info, disc_level_locs)
            sagittal_t1_imgs = self._sagittal_t1_imgs(disc_level, patient_info, disc_level_locs)
            sagittal_t2_imgs = self._sagittal_t2_stir_imgs(disc_level, patient_info, disc_level_locs)

            y = np.zeros(len(Condition.all_conditions()))

            for idx, condition in enumerate(Condition.all_conditions()):
                loc = Location(disc_level=disc_level, condition=condition)
                y[idx] = patient_info.conditions[loc.to_str()].to_int()

            if self.transformations != None:
                axial_t2_imgs = self.transformations(image=axial_t2_imgs)["image"]
                sagittal_t1_imgs = self.transformations(image=sagittal_t1_imgs)["image"]
                sagittal_t2_imgs = self.transformations(image=sagittal_t2_imgs)["image"]

            x = np.vstack( [ axial_t2_imgs.transpose(2,0,1) , sagittal_t1_imgs.transpose(2,0,1) , sagittal_t2_imgs.transpose(2,0,1) ])
            return x, y
        # else:
        #     for i in range(5):
        #         disc_level = DiscLevel.from_int(i)
        #         axial_t2_imgs    = self._axial_t2_imgs(disc_level, patient_info, disc_level_locs)
        #         sagittal_t1_imgs = self._sagittal_t1_imgs(disc_level, patient_info, disc_level_locs)
        #         sagittal_t2_imgs = self._sagittal_t2_stir_imgs(disc_level, patient_info, disc_level_locs)
        #
        #         if self.transformations != None:
        #             axial_t2_imgs = self.transformations(image=axial_t2_imgs)["image"]
        #             sagittal_t1_imgs = self.transformations(image=sagittal_t1_imgs)["image"]
        #             sagittal_t2_imgs = self.transformations(image=sagittal_t2_imgs)["image"]
        #
        #         x = np.vstack( [ axial_t2_imgs.transpose(2,0,1) , sagittal_t1_imgs.transpose(2,0,1) , sagittal_t2_imgs.transpose(2,0,1) ])
        #         yield x


    @abstractmethod
    def _axial_t2_imgs(self, disc_level: DiscLevel, patient_info: PatientInfo, disc_level_locs: DiscLevelLocs) -> np.ndarray:
        raise NotImplementedError("_axial_t2_imgs is not implemented.")

    @abstractmethod
    def _sagittal_t2_stir_imgs(self, disc_level: DiscLevel, patient_info: PatientInfo, disc_level_locs: DiscLevelLocs) -> np.ndarray:
        raise NotImplementedError("_sagittal_t2_stir_imgs is not implemented.")

    @abstractmethod
    def _sagittal_t1_imgs(self, disc_level: DiscLevel, patient_info: PatientInfo, disc_level_locs: DiscLevelLocs) -> np.ndarray:
        raise NotImplementedError("_sagittal_t1_imgs is not implemented.")
