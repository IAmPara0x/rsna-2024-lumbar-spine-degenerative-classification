
from abc import abstractmethod, ABC
from typing import Callable
from pandas import DataFrame
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
import random

from lib.patientInfo import PatientInfo, Img
from lib.model import DetectionModel

@dataclass
class RSNA24DF:
    train_df: DataFrame
    train_label_coordinates_df: DataFrame
    train_series_descriptions_df: DataFrame
    img_dir: str


class DetectionDataLoader(Dataset, ABC):
    def __init__(self
                 , patient_ids: list[int]
                 , rsna24DF: RSNA24DF
                 , transformations: list[Callable[ [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] ]]
                 , height: int
                 , width: int
                 , window_width: int
                 , window_center: int
                 , positive_negative_ratio: float
                 , positive_augment_prob: float
                 , negative_augment_prob: float
                 , phase = "train"
                 , coord_dim = 2
                 , change_window = True
                 , normalise = 1.0
                 ) -> None:

        self.patient_ids = patient_ids
        self.rsna24DF = rsna24DF
        self.transformations = transformations
        self.height = height
        self.width = width
        self.positive_negative_ratio = positive_negative_ratio
        self.positive_augment_prob = positive_augment_prob
        self.negative_augment_prob = negative_augment_prob
        self.phase = phase
        self.coord_dim = coord_dim
        self.window_width = window_width
        self.window_center = window_center
        self.change_window = change_window
        self.normalise = normalise


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_info = PatientInfo.from_df(self.patient_ids[idx], self.rsna24DF.train_df, self.rsna24DF.train_series_descriptions_df, self.rsna24DF.train_label_coordinates_df, self.rsna24DF.img_dir,
                                           patient_type=self.phase)


        if self.phase == "train":
            return self._train_dataloader(patient_info)
        elif self.phase == "valid":
            return self._validation_dataloader(patient_info)
        elif self.phase == "pred":
            return self._prediction_dataloader(patient_info)

    def _train_dataloader(self, patient_info: PatientInfo):

        scans = self._get_patient_scans(patient_info)

        if random.random() <= self.positive_negative_ratio:
            imgs = [img for img in scans if img.has_label()]
            has_sampled_postive_label = True
        else:
            imgs = [img for img in scans if not img.has_label()]
            has_sampled_postive_label = False

        img = random.choice(imgs) if len(imgs) != 0 else random.choice(scans)
        x,labels = img.resize(self.height, self.width)
        x = x.astype(np.float32)
        if self.change_window:
            x = self.apply_window(x) / self.normalise
        y_class, y_loc = self._mk_target_array(x,labels)

        aug_prob = self.positive_augment_prob if has_sampled_postive_label else self.negative_augment_prob
        x,y_loc = self.transform(x, y_class, y_loc, aug_prob)

        x = np.expand_dims(x, 0)
        return x, (y_class, y_loc)

    def apply_window(self, image):
        if self.change_window:
            img_max = self.window_center + self.window_width // 2
            img_min = self.window_center - self.window_width // 2
            return np.clip(image, img_min, img_max)
        else:
            return image

    def transform(self, x, y_class, y_loc, aug_prob):
        for transformation in self.transformations:
            if random.random() <= aug_prob:
                x,y_loc = transformation(x,y_loc)
                y_loc = np.expand_dims(y_class, 1) * y_loc
        return x, y_loc

    @abstractmethod
    def _mk_target_array(self, _x, _labels) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("_mk_labels is not implemented.")

    @abstractmethod
    def _get_patient_scans(self, _patient_info) -> list[Img]:
        raise NotImplementedError("_mk_labels is not implemented.")

    def _validation_dataloader(self, patient_info: PatientInfo):
        imgs = self._get_patient_scans(patient_info)
        num_imgs = len(imgs)
        xs = []
        y_classes = []
        y_locs = []

        for img in imgs:
            x,labels = img.resize(self.height, self.width)
            x = x.astype(np.float32)
            if self.change_window:
                x = self.apply_window(x) / self.normalise
            y_class, y_loc = self._mk_target_array(x,labels)

            xs.append(x)
            y_classes.append(y_class)
            y_locs.append(y_loc)

        xs = np.vstack(xs).reshape(num_imgs, 1, self.height, self.width)
        y_classes = np.vstack(y_classes)
        y_locs = np.vstack(y_locs).reshape(num_imgs, -1, self.coord_dim)

        return xs, (y_classes, y_locs)

    def _prediction_dataloader(self, patient_info: PatientInfo):
        imgs = self._get_patient_scans(patient_info)
        num_imgs = len(imgs)
        xs = []

        for img in imgs:
            x,_ = img.resize(self.height, self.width)
            x = x.astype(np.float32)
            if self.change_window:
                x = self.apply_window(x) / self.normalise

            xs.append(x)

        xs = np.vstack(xs).reshape(num_imgs, 1, self.height, self.width)

        return xs, imgs
