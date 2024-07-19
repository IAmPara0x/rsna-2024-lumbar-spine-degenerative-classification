#!/usr/bin/python

import pickle
import os
from tqdm import tqdm
import glob
from lib import *



if __name__ == "__main__":


    PATH = f"./processed-data/sagittal_t2_stir_segmentation/"
    patient_infos = glob.glob(f"{PATH}/*.pkl")

    print(os.path.basename(patient_infos[0]))


    # for patient_info in tqdm(patient_infos):
    #     with open(patient_info, "rb") as f:
    #         data = pickle.load(f)
    #
    #         sagittal_t2_stir = [*data["series"].values()]
    #
    #     if len(sagittal_t2_stir) == 0:
    #         os.remove(patient_info)
