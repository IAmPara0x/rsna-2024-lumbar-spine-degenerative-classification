#!/usr/bin/python


import os
import pickle
from tqdm import tqdm
from lib import *


if __name__ == "__main__":

    PATH = os.getcwd() + "/processed-dataset"
    files = os.listdir(PATH)

    for file in tqdm(files):
        with open(f"{PATH}/{file}", "rb+") as f:
            patient_info = pickle.load(f)

            for series_info in patient_info["series"].values():
                for img in series_info["images"]:
                    img["position_patient"] = list(img["position_patient"])
            pickle.dump(patient_info, f)




