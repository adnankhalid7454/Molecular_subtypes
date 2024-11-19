import skimage.io as io
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, skif
from scipy.spatial import distance
from scipy.stats import kurtosis, skew
import shutil

def extract_kinetic_features(roi):
    mean_intensity = np.mean(roi)
    variance = np.var(roi)
    max_intensity = np.max(roi)
    min_intensity = np.min(roi)
    skewness = skew(roi.flatten())
    kurt = kurtosis(roi.flatten())
    entropy = -np.sum(roi * np.log2(roi + 1e-10))
    return {
        "Mean Intensity": mean_intensity,
        "Variance": variance,
        "Max Intensity": max_intensity,
        "Min Intensity": min_intensity,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Entropy": entropy
    }

# GLCM
def extract_glcm_features(roi):
    glcm = skif.graycomatrix(roi, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = skif.graycoprops(glcm, 'contrast').mean()
    correlation = skif.graycoprops(glcm, 'correlation').mean()
    energy = skif.graycoprops(glcm, 'energy').mean()
    homogeneity = skif.graycoprops(glcm, 'homogeneity').mean()
    return {
        "Contrast": contrast,
        "Correlation": correlation,
        "Energy": energy,
        "Homogeneity": homogeneity
    }

# Function to extract texture features using LBP
def extract_lbp_features(roi):
    lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return {f"LBP_{i}": hist[i] for i in range(n_bins)}

# data path and excel file to return features. 
def texture_feature_extraction(data_path, excel_file):
    results = []
    images_path = os.path.join(data_path, 'path/imgs')
    masks_path = os.path.join(data_path, 'path/msks')
    class_info = pd.read_excel(os.path.join(data_path, excel_file))

    images = os.listdir(images_path)
    masks = os.listdir(masks_path)
    for j in range(len(images)):
        mammogram_image = cv2.imread(os.path.join(train_data_path, images[j]),0)
        img_mask = cv2.imread(os.path.join(train_mask_path, masks[j]), 0)
       
        # Isolate the tumor region
        tumor_region = cv2.bitwise_and(mammogram_image, mammogram_image, mask=img_mask)
        dict_list = {"Patient_Information": j}
        # Extract kinetic features
        kinetic_features = extract_kinetic_features(tumor_region)
        dict_list.update(kinetic_features)

        # Extract GLCM texture features
        glcm_features = extract_glcm_features(tumor_region)
        dict_list.update(glcm_features)
        # Extract LBP texture features
        lbp_features = extract_lbp_features(tumor_region)
        dict_list.update(lbp_features)

        # Append the result to the results list
        results.append(dict_list)

    return results


