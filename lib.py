from dataclasses import dataclass
from enum import StrEnum
from typing import Dict
from pydicom import FileDataset

class Condition(StrEnum):
    SCS = "spinal_canal_stenosis"
    LNFN = "left_neural_foraminal_narrowing"
    RNFN = "right_neural_foraminal_narrowing"
    LSS = "left_subarticular_stenosis"
    RSS = "right_subarticular_stenosis"
    
@dataclass
class ImgLabel:
    x: float
    y: float
    condition: Condition
    level: int
        
    def location(self):
        if self.level > 5: raise ValueError(f"level must be <= 5 but got {self.level}")
        if self.level == 5:
            return f"{self.condition}_l5_s1"
        return f"{self.condition}_l{self.level}_l{self.level+1}"


# def get_patient_info(patient_id):
#
#     patient_info = {}
#     patient_img_dir = get_patient_img_dir(patient_id)
#     
#
#     tmp = train_df[train_df["study_id"] == patient_id]
#     for col in train_df.columns:
#         patient_info[col] = tmp[col].item()
#     
#     patient_info["series"] = {}
#     
#     for _, row in train_series_descriptions_df[train_series_descriptions_df["study_id"] == patient_id].iterrows():
#         patient_info["series"][row["series_id"]] = {"series_description": row["series_description"], "images": []}
#         
#     for series_id, series_info in patient_info["series"].items():
#         
#         series_image_path = f"{patient_img_dir}/{series_id}"
#         images_path = glob.glob(f"{series_image_path}/*.dcm")
#         for j in sorted(images_path, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
#         
#             series_info["images"].append({
#                 'SOPInstanceUID': int(j.split('/')[-1].replace('.dcm', '')), 
#                 'dicom': pydicom.dcmread(j),
#                 "labels": []
#             })
#             
#     for _, row in train_label_coordinates_df[train_label_coordinates_df["study_id"] == patient_id].iterrows():
#         
#         for img in patient_info["series"][row["series_id"]]["images"]:
#    
#             if img["SOPInstanceUID"] == row["instance_number"]:
#
#                 condition = None
#                 raw_condition = row["condition"].lower().replace(" ", "_")
#                 if raw_condition == Condition.SCS:
#                     condition = Condition.SCS
#                 elif raw_condition == Condition.LNFN:
#                     condition = Condition.LNFN
#                 elif raw_condition == Condition.RNFN:
#                     condition = Condition.RNFN
#                 elif raw_condition == Condition.LSS:
#                     condition = Condition.LSS
#                 elif raw_condition == Condition.RSS:
#                     condition = Condition.RSS
#                 else:
#                     raise Exception(f"unknown condition: {row['condition']}")
#                 level = int(row["level"][1])
#                 
#                 img["labels"].append(ImgLabel(x=float(row["x"]), y=float(row["y"]), condition=condition, level=level))
#                 break
#
#         
#     return patient_info
