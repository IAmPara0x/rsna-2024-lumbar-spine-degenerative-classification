
import numpy as np
import cv2

def vertical_flip(img: np.ndarray, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_coord = coord.copy()
    img = cv2.flip(img, 0)  # 0 for vertical flip
    new_coord[:, 1] = img.shape[1] - new_coord[:, 1]  # Adjusting the y-coordinates for vertical flip
    return img, new_coord

def horizontal_flip(img: np.ndarray, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_coord = coord.copy()
    img = cv2.flip(img, 1)  # Corrected to 1 for horizontal flip
    new_coord[:, 0] = img.shape[0] - new_coord[:, 0]  # Adjusting the x-coordinates for horizontal flip
    return img, new_coord
