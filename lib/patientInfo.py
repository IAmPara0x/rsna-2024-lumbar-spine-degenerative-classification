from dataclasses import dataclass
from strenum import StrEnum
from typing import Dict, Callable
from pydicom import FileDataset
import pandas as pd
import glob
import pydicom
import numpy as np
import cv2
import copy

class Condition(StrEnum):
    SCS = "spinal_canal_stenosis"
    LNFN = "left_neural_foraminal_narrowing"
    RNFN = "right_neural_foraminal_narrowing"
    LSS = "left_subarticular_stenosis"
    RSS = "right_subarticular_stenosis"

    @staticmethod
    def all_conditions() -> "list[Condition]":
        return [Condition.SCS, Condition.LNFN, Condition.RNFN, Condition.LSS, Condition.RSS]

class Severity(StrEnum):
    Normal = "Normal/Mild"
    Moderate = "Moderate"
    Severe = "Severe"

    @staticmethod
    def all_severity() -> "list[Severity]":
        return [Severity.Normal, Severity.Moderate, Severity.Severe]


    def to_int(self) -> int:
        if self == Severity.Normal:
            return 0
        elif self == Severity.Moderate:
            return 1
        elif self == Severity.Severe:
            return 2
        else:
            raise RuntimeError(f"Unexpected Severity: {self}")



class DiscLevel(StrEnum):
    l1_l2 = "l1_l2"
    l2_l3 = "l2_l3"
    l3_l4 = "l3_l4"
    l4_l5 = "l4_l5"
    l5_s1 = "l5_s1"

    @staticmethod
    def from_int(level) -> "DiscLevel":
        if level >= 5: raise ValueError(f"level must be <= 5 but got {level}")
        if level == 4:
            return DiscLevel.l5_s1
        return DiscLevel(f"l{level+1}_l{level+2}")

    def to_int(self) -> int:
        if self == DiscLevel.l1_l2:
            return 0
        elif self == DiscLevel.l2_l3:
            return 1
        elif self == DiscLevel.l3_l4:
            return 2
        elif self == DiscLevel.l4_l5:
            return 3
        elif self == DiscLevel.l5_s1:
            return 4
        else:
            raise RuntimeError(f"{self} is not in DiscLevel")

class Scan(StrEnum):
    AxialT2 = "Axial T2"
    SagittalT2_STIR = "Sagittal T2/STIR"
    SagittalT1 = "Sagittal T1"

    @staticmethod
    def all_scans() -> "list[Scan]":
        return [Scan.AxialT2, Scan.SagittalT2_STIR, Scan.SagittalT1]

@dataclass
class Location:
    disc_level: DiscLevel
    condition: Condition

    def to_str(self):
        return f"{self.condition}_{self.disc_level}"

    @staticmethod
    def from_str(loc: str) -> "Location":
        parts = loc.split("_")
        condition,disc_level = Condition('_'.join(parts[:-2])), DiscLevel('_'.join(parts[-2:]))
        return Location(disc_level, condition)
    
@dataclass
class ImgLabel:
    x: float
    y: float
    location: Location

@dataclass
class Img:
    labels: list[ImgLabel]
    dicom: FileDataset
    SOPInstanceUID: int

    def find_label(self, f: Callable[["ImgLabel"], bool]) -> ImgLabel | None:
        r = [*filter(f, self.labels)]

        if len(r) == 0:
            return None
        return r[0]

    def has_label(self) -> bool:
        return len(self.labels) != 0

    def resize(self, height, width) -> tuple[np.ndarray, list[ImgLabel]]:

        if (img := self.dicom.pixel_array).shape == (height, width):
            return img, self.labels

        img = cv2.resize(self.dicom.pixel_array, (height, width),interpolation=cv2.INTER_CUBIC)
        orig_height, orig_width = self.dicom.pixel_array.shape

        r_height, r_width = height/orig_height, width/orig_width

        labels = copy.deepcopy(self.labels)

        for label in labels:
            label.x = label.x * r_width
            label.y = label.y * r_height
        return img, labels

@dataclass
class PatientInfo:

    patient_id: int
    conditions: Dict[str, Severity]
    scans: Dict[Scan, list[Img]]


    @staticmethod
    def from_df(patient_id, df, series_df, label_coord_df, img_dir, patient_type="train") -> "PatientInfo":

        patient_img_dir = f"{img_dir}/{patient_id}"

        # This is for condition labels
        conditions = {}

        if patient_type == "train":

            patient_df = df[df["study_id"] == patient_id]
            for col in df.columns:

                if col != "study_id":
                    if pd.isna(x := patient_df[col].item()):
                        conditions[col] = Severity.Normal
                    else:
                        conditions[col] = Severity(x)


        # columns ['study_id', 'series_id', 'series_description']
        patient_series = series_df[series_df["study_id"] == patient_id]


        scans = {}

        for _, series in patient_series.iterrows():

            series_id = series['series_id']
            series_img_path = f"{patient_img_dir}/{series_id}"

            # Get all the images of the series
            images_path = glob.glob(f"{series_img_path}/*.dcm")



            if patient_type == "train":

                # columns ['study_id', 'series_id', 'instance_number', 'condition', 'level', 'x', 'y', 'series_description']
                patient_labels = label_coord_df[label_coord_df["study_id"] == patient_id]

                # Get all the labels of the series of a particular patient
                series_labels = patient_labels[patient_labels["series_id"] == series_id]


            imgs = []
            for j in sorted(images_path, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):

                uid = int(j.split('/')[-1].replace('.dcm', ''))
                img = Img(SOPInstanceUID=uid, dicom=pydicom.dcmread(j), labels=[])

                if patient_type == "train":
                    # G et all the labels belonging to a particular image of a series
                    img_labels = series_labels[series_labels["instance_number"] == uid]

                    for _, label in img_labels.iterrows():
                        location = "_".join([label["condition"].lower().replace(" ", "_"), label["level"].lower().replace("/","_")])
                        x,y = float(label["x"]), float(label["y"])
                        img.labels.append(ImgLabel(x=x,y=y,location=Location.from_str(location)))

                imgs.append(img)

            if (desc := series["series_description"]) in scans:
                scans[desc].extend(imgs)
            else:
                scans[desc] = imgs

        return PatientInfo(conditions=conditions, scans=scans, patient_id=patient_id)

    def get_scans(self, scan: Scan):
        if scan in self.scans:
            return self.scans[scan]
        else:
            return []

    def find_img(self, scan: Scan, f: Callable[[Img], bool]) -> Img | None:
        r = [*filter(f, self.scans[scan])]

        if len(r) == 0:
            return None
        return r[0]


def condition_all_locations(condition: Condition) -> list[Location]:

    locs = []

    for i in range(5):
        locs.append(Location(DiscLevel.from_int(i), condition=condition))
    return locs
